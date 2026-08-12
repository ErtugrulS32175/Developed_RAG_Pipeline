"""The disposable acceptance mirror. PACKAGE B2B-C1.

WHY A THIRD TREE. The acceptance commands are the only thing in this
loop that RUNS candidate code, and a command is free to write wherever
its process can reach. It may therefore not run in the operator's
checkout, and it may not run in the IMPLEMENTER tree either: that tree is
the evidence the change set was derived from, and a command that edits it
destroys the only record of what the model did. It may not run in the
REFERENCE tree for the same reason, one step worse -- that copy is what
the work is measured against.

So a THIRD tree is built, in a temp root this package owns, and thrown
away when the commands are done. Nothing downstream of it survives.

HOW IT IS BUILT, and why not with `copytree`. The mirror is materialised
from the baseline's RAW GIT OBJECTS -- the same seam `flat_workspace`
uses, verifying every blob's id against the bytes that actually arrived
-- and then patched with the FRESH change set. Copying the implementer
tree wholesale would carry whatever is in it, including objects the
evidence model refuses to represent, and would make the mirror's content
a claim rather than a derivation.

EVERY CANDIDATE BYTE IS READ THROUGH A DESCRIPTOR BOUND TO THE LISTING
THAT NAMED IT. `open(path)` after an `lstat(path)` is two questions, and
D2 measured the answer changing between them three times. The transport
opens the root once and every child relative to the object that listed
it; a symlink, a junction, a FIFO or a device is refused rather than
followed.

WHAT MAY LEAVE. Fixed sentences and closed reasons. Never a candidate's
bytes, never a corpus fragment, never an absolute path, never the
operating system's own error text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import (changes, contract, flat_workspace, fs_evidence,
                              fs_transport, git_objects,
                              state as state_module, workspace_evidence)

# A NAMED subdirectory of the system temp root, never the temp root
# itself: containment against a directory that holds everything transient
# on the machine means almost nothing. Distinct from the flat workspace's
# root, because the two lifecycles are owned by different packages and a
# shared root makes one package's residue look like the other's.
ROOT_DIRNAME = "agent-loop-acceptance"
TEMP_PREFIX = "agent-loop-acc-"

MARKER_NAME = "acceptance-owner.json"
MARKER_VERSION = 1
TREE_DIRNAME = "tree"
HOME_DIRNAME = "home"
SCRATCH_DIRNAME = "scratch"
HOOKS_DIRNAME = "hooks"
GIT_CONFIG_NAME = "gitconfig"

# The two roots of the operator's checkout the frozen `leak_scan` command
# reads. Snapshotted only when a task actually names that command.
CORPUS_DIRNAMES = ("data", "output")

ACCEPTANCE_ID = re.compile(r"^[0-9a-f]{32}$")
_TIMESTAMP = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"

# The one fixed sentence a cleanup failure may add. Free text next to a
# filesystem is a path leak, so there is exactly this and nothing else.
CLEANUP_NOTE = "kabul aynasi artigi temizlenemedi"

MARKER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "marker_version", "acceptance_id",
                 "repo_id", "run_id", "workspace_id", "baseline_sha",
                 "created_at"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "marker_version": {"const": MARKER_VERSION},
        "acceptance_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        # identity, never a path: an absolute path in a marker is an
        # absolute path in whatever prints the marker
        "repo_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "run_id": {"type": "string", "pattern": contract.IDENTIFIER_PATTERN},
        "workspace_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "created_at": {"type": "string", "pattern": _TIMESTAMP},
    },
}
_MARKER_VALIDATOR = Draft202012Validator(MARKER_SCHEMA)


class MirrorError(RuntimeError):
    """A refusal from the mirror layer. Fixed text, closed reason."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


class MirrorContainment(MirrorError):
    """The mirror would have sat somewhere it is not allowed to sit, or
    the candidate carries something this model cannot represent."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PATH_NOT_ALLOWED)


class MirrorCleanupFailed(MirrorError):
    """Worse than a failed acceptance: a directory of ours is still on
    the machine and this package could not prove otherwise.

    A SEPARATE TYPE on purpose. Folding it into the ordinary failure
    would let "the gate went red" and "there is residue nobody owns"
    arrive as the same event, and only one of them needs a human."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PATH_NOT_ALLOWED)


@dataclass(frozen=True, slots=True)
class AcceptanceMirror:
    """Where a command may run, and the four call-owned directories that
    keep it from reaching anything else."""

    acceptance_id: str
    holder: Path
    tree: Path
    home: Path
    scratch: Path
    hooks: Path
    git_config: Path


# ---------------------------------------------------------------------
# identity and containment
# ---------------------------------------------------------------------

def mirror_temp_root() -> Path:
    root = Path(tempfile.gettempdir()).resolve() / ROOT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def holder_for(acceptance_id) -> Path:
    """The single directory a given id can name.

    The id is checked against a strict 32-hex pattern before it is
    joined, so no id can denote a path outside the root this package
    owns -- `..` is not an id."""
    if not ACCEPTANCE_ID.match(str(acceptance_id)):
        raise MirrorError("gecersiz kabul aynasi kimligi")
    return mirror_temp_root() / f"{TEMP_PREFIX}{acceptance_id}"


def _comparable(path) -> str:
    text = str(Path(path).resolve())
    return text.casefold() if os.name == "nt" else text


def _inside(candidate: str, root: str) -> bool:
    return candidate == root or candidate.startswith(root + os.sep)


def _assert_outside(holder: Path, roots) -> None:
    """The mirror may not be, contain, or live inside any authorised
    root. Checked in BOTH directions: a mirror under the repository is a
    command writing into the repository, and a mirror ABOVE one is a
    cleanup that would delete it."""
    mine = _comparable(holder)
    for root in roots:
        theirs = _comparable(root)
        if _inside(mine, theirs) or _inside(theirs, mine):
            raise MirrorContainment("kabul aynasi yetkili bir kokun icinde")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------
# talking to the transport
# ---------------------------------------------------------------------

def _transport(call, *args, message):
    """EVERY transport call goes through here.

    A `TransportError` already carries a fixed sentence; anything else --
    above all a raw `OSError`, whose text names absolute paths -- is
    replaced by one. The cleanup FLAG crosses the boundary; the lower
    layer's notes deliberately do not, because free text written closer
    to the filesystem is exactly where a path rides out."""
    try:
        return call(*args)
    except fs_transport.TransportError as exc:
        yeni = MirrorContainment(str(exc))
        if fs_transport.cleanup_failed(exc):
            fs_transport.mark_cleanup_failed(yeni)
        raise yeni from None
    except OSError:
        raise MirrorContainment(message) from None


def _close_quietly(directory) -> bool:
    try:
        fs_transport.close_directory(directory)
        return True
    except (fs_transport.TransportError, OSError):
        return False


def _close_descriptor_quietly(descriptor: int) -> bool:
    try:
        os.close(descriptor)
        return True
    except OSError:
        return False


def _fold(exc: BaseException, ok: bool) -> None:
    """Consume a cleanup result on a failure path: it may not replace the
    error being raised, and it may not disappear."""
    if not ok:
        fs_transport.mark_cleanup_failed(exc)


def _read_descriptor(descriptor: int, ceiling: int) -> bytes:
    """Bounded WHILE reading. A ceiling applied after the bytes are in
    the process is a report, not a bound."""
    chunks, read = [], 0
    while True:
        try:
            block = os.read(descriptor, 1 << 20)
        except OSError:
            raise MirrorError("aday dosyasi okunamadi") from None
        if not block:
            break
        read += len(block)
        if read > ceiling:
            raise MirrorError("aday dosyasi sozlesme tavanini asiyor")
        chunks.append(block)
    return b"".join(chunks)


def _close_frames(frames) -> bool:
    """Close what is left, innermost first, and attempt EVERY one: a
    cleanup that stops at the first problem leaks everything behind it."""
    kalan = 0
    while frames:
        if not _close_quietly(frames.pop()):
            kalan += 1
    return kalan == 0


def _entry_bytes(root, parts, ceiling):
    """One file's bytes, opened relative to the objects that listed it.

    There are NO PATHS after the root is opened: each component is looked
    up in the listing of the directory currently held open, and the next
    one is opened relative to that object. A parent swapped mid-walk is
    refused by the transport rather than followed."""
    frames = []
    try:
        frames.append(_transport(fs_transport.open_root, root,
                                 message="aday kok dizini acilamadi"))
        for index, name in enumerate(parts):
            records = _transport(fs_transport.list_directory, frames[-1],
                                 message="aday dizini listelenemedi")
            record = next((item for item in records if item.name == name),
                          None)
            if record is None:
                raise MirrorError("aday girdisi bulunamadi")
            if record.reparse_tag or record.kind == "link":
                raise MirrorContainment("aday agacinda ayrisma noktasi")
            if index == len(parts) - 1:
                if record.kind != "file":
                    raise MirrorContainment(
                        "aday girdisi siradan bir dosya degil")
                descriptor = _transport(fs_transport.open_child_file,
                                        frames[-1], record,
                                        message="aday dosyasi acilamadi")
                try:
                    data = _read_descriptor(descriptor, ceiling)
                except BaseException as birincil:
                    _fold(birincil, _close_descriptor_quietly(descriptor))
                    raise
                if not _close_descriptor_quietly(descriptor):
                    raise MirrorError("aday tanimlayicisi kapatilamadi")
                break
            if record.kind != "dir":
                raise MirrorContainment("aday girdisi siradan bir dizin degil")
            frames.append(_transport(fs_transport.open_child_directory,
                                     frames[-1], record,
                                     message="aday dizini acilamadi"))
        else:
            raise MirrorError("aday girdisi bulunamadi")
    except BaseException as birincil:
        _fold(birincil, _close_frames(frames))
        raise
    if not _close_frames(frames):
        raise MirrorError("aday dizini kapatilamadi")
    return data


# ---------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------

def _write_bytes(root: Path, relative: str, data: bytes) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # a fresh file every time: never a hard link, never a reuse, so a
    # write in the mirror can never be a write in the tree it came from
    with open(target, "wb") as stream:
        stream.write(data)
    return target


def _apply_mode(target: Path, mode: str) -> None:
    """POSIX gets the recorded permission bits back.

    Windows has no such bits to restore, and the semantic record keeps
    the mode there instead -- the walker reports `0o0` on that platform,
    so applying anything would be inventing a fact."""
    if os.name == "nt" or not mode.startswith("0o"):
        return
    try:
        bits = int(mode, 8)
    except ValueError:
        raise MirrorError("aday modu cozumlenemedi") from None
    if bits:
        os.chmod(target, bits)


def _materialise_baseline(repo, tree: Path, baseline_sha: str) -> None:
    """The baseline, from RAW GIT OBJECTS, through the seam that already
    recomputes every object id from the bytes that arrived."""
    limits = flat_workspace.DEFAULT_LIMITS
    toplam = 0
    try:
        entries = git_objects._read_tree(repo, baseline_sha, limits)
        for yol, mode, oid in entries:
            data = git_objects._object_bytes(repo, oid,
                                             limit=limits.max_file_bytes)
            if type(data) is not bytes:
                raise MirrorError("nesne baytlari okunamadi")
            if git_objects.blob_object_id(data) != oid:
                raise MirrorError("nesne kimligi baytlarla uyusmuyor")
            toplam += len(data)
            if toplam > limits.max_total_bytes:
                raise MirrorError("toplam boyut sozlesme tavanini asiyor")
            flat_workspace._write_blob(tree, yol, mode, data)
    except git_objects.FlatWorkspaceError as refused:
        # the lower layer's sentences are fixed and carry no path
        raise MirrorError(str(refused)) from None


def _apply_changes(tree: Path, implementer_root: Path, candidate) -> None:
    """The FRESH change set, applied one entry at a time.

    Every added or modified file's bytes are read from the candidate
    through the no-follow transport and then required to hash to the
    digest the evidence recorded. A file that moved between the scan and
    this read is refused, not written."""
    ceiling = fs_evidence.Limits().max_content_file_bytes
    for change in candidate.changes:
        target = tree / change.path
        if change.kind == changes.DELETED:
            try:
                target.unlink()
            except OSError:
                raise MirrorError("aday silmesi aynada uygulanamadi") from None
            continue
        data = _entry_bytes(implementer_root, tuple(change.path.split("/")),
                            ceiling)
        if hashlib.sha256(data).hexdigest() != change.sha256:
            raise MirrorContainment("aday dosyasi kanitla ayni degil")
        _apply_mode(_write_bytes(tree, change.path, data), change.mode)


def _copy_corpus(source: Path, destination: Path, budget) -> None:
    """One git-ignored root of the operator's checkout, copied as REAL
    independent files.

    No hard link, no symlink, no junction and no reparse point is created
    or followed: the frozen scanner reads this copy, and a link back into
    the operator's tree would put the command one write away from it."""
    frames = []
    try:
        frames.append((_transport(fs_transport.open_root, source,
                                  message="korpus koku acilamadi"), ()))
        while frames:
            directory, parts = frames[-1]
            records = _transport(fs_transport.list_directory, directory,
                                 message="korpus dizini listelenemedi")
            alt = []
            try:
                for record in records:
                    if record.reparse_tag or record.kind == "link":
                        raise MirrorContainment(
                            "korpus girdisi temsil edilemiyor")
                    child = parts + (fs_transport.validate_child_name(
                        record.name),)
                    if record.kind == "dir":
                        (destination.joinpath(*child)).mkdir(parents=True,
                                                             exist_ok=True)
                        alt.append((
                            _transport(fs_transport.open_child_directory,
                                       directory, record,
                                       message="korpus dizini acilamadi"),
                            child))
                        continue
                    if record.kind != "file":
                        raise MirrorContainment(
                            "korpus girdisi temsil edilemiyor")
                    budget["entries"] += 1
                    if budget["entries"] > budget["max_entries"]:
                        raise MirrorError(
                            "korpus giris sayisi sozlesme tavanini asiyor")
                    descriptor = _transport(fs_transport.open_child_file,
                                            directory, record,
                                            message="korpus dosyasi acilamadi")
                    try:
                        data = _read_descriptor(descriptor,
                                                budget["max_file_bytes"])
                    except BaseException as birincil:
                        _fold(birincil,
                              _close_descriptor_quietly(descriptor))
                        raise
                    if not _close_descriptor_quietly(descriptor):
                        raise MirrorError("korpus tanimlayicisi kapatilamadi")
                    budget["bytes"] += len(data)
                    if budget["bytes"] > budget["max_total_bytes"]:
                        raise MirrorError(
                            "korpus toplami sozlesme tavanini asiyor")
                    _write_bytes(destination, "/".join(child), data)
            except BaseException as birincil:
                _fold(birincil, _close_frames([opened for opened, _ in alt]))
                raise
            frames.pop()
            if not _close_quietly(directory):
                raise MirrorError("korpus dizini kapatilamadi")
            frames.extend(alt)
    except BaseException as birincil:
        _fold(birincil, _close_frames([opened for opened, _ in frames]))
        raise


def _corpus_evidence(repo: Path, key: bytes):
    """What the operator's two git-ignored roots ARE, right now.

    Read on BOTH sides of the copy: a snapshot taken while something else
    is writing describes neither version, and a mirror built from that is
    a scan of a moment that never existed."""
    fs_evidence.quiesce()
    seen = {}
    for name in CORPUS_DIRNAMES:
        root = repo / name
        if not root.is_dir():
            continue
        try:
            manifest = fs_evidence.scan(root, key=key,
                                        limits=fs_evidence.Limits())
        except fs_evidence.EvidenceError as refused:
            raise MirrorError(str(refused)) from None
        except OSError:
            raise MirrorError("korpus kaniti alinamadi") from None
        seen[name] = (manifest.root_identity, manifest.digest)
    return seen


# ---------------------------------------------------------------------
# the ownership marker
# ---------------------------------------------------------------------

def _marker_payload(acceptance_id, repo, run_id, workspace_id, baseline_sha):
    return {"protocol_version": contract.PROTOCOL_VERSION,
            "marker_version": MARKER_VERSION,
            "acceptance_id": acceptance_id,
            "repo_id": state_module.repo_identity(repo),
            "run_id": run_id, "workspace_id": workspace_id,
            "baseline_sha": baseline_sha, "created_at": _now()}


def read_marker(holder: Path):
    """The marker, read back through the HOLDER HANDLE that listed it.

    The holder is opened with the transport's no-follow root open, so a
    holder replaced by a symlink or a junction is refused before a single
    child is looked at -- which is what makes this safe to consult right
    before a delete."""
    kok = _transport(fs_transport.open_root, holder,
                     message="kabul aynasi dizini acilamadi")
    try:
        records = _transport(fs_transport.list_directory, kok,
                             message="kabul aynasi listelenemedi")
        record = next((item for item in records
                       if item.name == MARKER_NAME), None)
        if record is None:
            raise MirrorError("kabul aynasi sahiplik isareti yok")
        if record.kind != "file" or record.reparse_tag:
            raise MirrorContainment(
                "sahiplik isareti siradan bir dosya degil")
        descriptor = _transport(fs_transport.open_child_file, kok, record,
                                message="sahiplik isareti acilamadi")
        try:
            data = _read_descriptor(descriptor, 4096)
        finally:
            if not _close_descriptor_quietly(descriptor):
                raise MirrorError("sahiplik isareti kapatilamadi")
    except BaseException as birincil:
        _fold(birincil, _close_quietly(kok))
        raise
    if not _close_quietly(kok):
        raise MirrorError("kabul aynasi dizini kapatilamadi")
    try:
        marker = json.loads(data.decode("utf-8"))
        _MARKER_VALIDATOR.validate(marker)
    except (UnicodeDecodeError, ValueError, ValidationError):
        raise MirrorError(
            "kabul aynasi sahiplik isareti sozlesmeye uymuyor") from None
    return marker


# ---------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------

def _remove_tree(holder: Path) -> bool:
    """Isolated behind one name so a test can make exactly this fail and
    prove the refusal is a strongly typed cleanup failure."""
    try:
        shutil.rmtree(holder, onexc=_clear_readonly)
    except TypeError:                              # pragma: no cover -- <3.12
        try:
            shutil.rmtree(holder, onerror=_clear_readonly_legacy)
        except OSError:
            return False
    except OSError:
        return False
    return not _present(holder)


def _clear_readonly(function, path, error):
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _clear_readonly_legacy(function, path, info):   # pragma: no cover
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _present(path) -> bool:
    """`lstat`, not `exists()`: `exists()` follows a link and answers
    False for a broken one, and a dangling link is still residue."""
    try:
        os.lstat(path)
        return True
    except OSError:
        return False


def create(*, repo, workspace, run_id, baseline_sha, candidate,
           snapshot_corpus) -> AcceptanceMirror:
    """Build the mirror, or leave nothing behind.

    WRITE-AHEAD OWNERSHIP. The marker exists, and has been read back
    through a handle, before a single candidate byte is written -- so a
    process that dies during materialisation still leaves a directory
    that says whose it is, and `remove` has something to check against."""
    repo_path = Path(repo)
    acceptance_id = secrets.token_hex(16)
    holder = holder_for(acceptance_id)
    _assert_outside(holder, (repo_path, workspace.reference_root,
                             workspace.implementer_root,
                             Path(workspace.reference_root).parent))

    try:
        holder.mkdir()
    except FileExistsError:
        # a collision might BE another call, and the only safe answer is
        # to refuse rather than make room
        raise MirrorError("kabul aynasi dizini zaten var") from None
    except OSError:
        raise MirrorError("kabul aynasi dizini yaratilamadi") from None

    mirror = AcceptanceMirror(
        acceptance_id=acceptance_id, holder=holder,
        tree=holder / TREE_DIRNAME, home=holder / HOME_DIRNAME,
        scratch=holder / SCRATCH_DIRNAME, hooks=holder / HOOKS_DIRNAME,
        git_config=holder / GIT_CONFIG_NAME)
    try:
        payload = _marker_payload(acceptance_id, repo_path, run_id,
                                  workspace.workspace_id, baseline_sha)
        try:
            state_module.write_json_atomically(
                holder / MARKER_NAME, payload, MARKER_SCHEMA,
                "kabul aynasi sahiplik isareti")
        except (state_module.StateError, OSError):
            raise MirrorError("sahiplik isareti yazilamadi") from None
        written = read_marker(holder)
        if written != payload:
            raise MirrorError("sahiplik isareti geri okunamadi")

        for directory in (mirror.tree, mirror.home, mirror.scratch,
                          mirror.hooks):
            directory.mkdir()
        # an EMPTY call-owned file, so the isolation variables point at
        # something that exists and can never be the operator's own
        mirror.git_config.write_bytes(b"")

        _materialise_baseline(repo_path, mirror.tree, baseline_sha)
        _apply_changes(mirror.tree, workspace.implementer_root, candidate)
        # THE PROJECTION IS THE GATE. Two independent copies differ by
        # root identity, file identity and timestamps by definition, so
        # the semantic content is what must agree -- and it must agree
        # BEFORE the corpus and the git metadata make the two trees
        # legitimately different.
        try:
            workspace_evidence.compare_roots(workspace.implementer_root,
                                             mirror.tree)
        except git_objects.FlatWorkspaceError:
            raise MirrorContainment("ayna aday agacla ayni degil") from None

        if snapshot_corpus:
            key = secrets.token_bytes(fs_evidence.KEY_BYTES)
            before = _corpus_evidence(repo_path, key)
            budget = {"entries": 0, "bytes": 0,
                      "max_entries": fs_evidence.Limits().max_entries,
                      "max_file_bytes":
                          fs_evidence.Limits().max_content_file_bytes,
                      "max_total_bytes":
                          fs_evidence.Limits().max_content_total_bytes}
            for name in CORPUS_DIRNAMES:
                if name not in before:
                    continue
                (mirror.tree / name).mkdir(parents=True, exist_ok=True)
                _copy_corpus(repo_path / name, mirror.tree / name, budget)
            if _corpus_evidence(repo_path, key) != before:
                raise MirrorContainment("korpus kaynagi degistirildi")
    except BaseException as birincil:
        if not _remove_tree(holder):
            try:
                birincil.add_note(CLEANUP_NOTE)
            except AttributeError:                 # pragma: no cover -- <3.11
                pass
        raise
    return mirror


def assert_git_metadata(mirror: AcceptanceMirror) -> None:
    """The mirror's git metadata is a REAL directory of its own.

    A `.git` FILE is a gitdir pointer, which is precisely how a
    disposable tree gets attached to somebody else's object database."""
    target = mirror.tree / ".git"
    try:
        info = os.lstat(target)
    except OSError:
        raise MirrorError("kabul aynasi git ust verisi yok") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) \
            or getattr(info, "st_reparse_tag", 0):
        raise MirrorContainment("kabul aynasi git ust verisi siradan degil")


def remove(mirror: AcceptanceMirror) -> None:
    """Delete the holder THIS CALL owns, and nothing else.

    The name, the parent and the marker all have to agree before a single
    byte is unlinked: a directory sitting in this root that does not
    carry our own acceptance id belongs to something else, and this
    function is not allowed to have an opinion about it."""
    holder = holder_for(mirror.acceptance_id)
    if _comparable(holder) != _comparable(mirror.holder):
        raise MirrorContainment("kabul aynasi beklenen yerde degil")
    if holder.name != f"{TEMP_PREFIX}{mirror.acceptance_id}":
        raise MirrorContainment("hedef beklenen ayna adi degil")
    if _comparable(holder.parent) != _comparable(mirror_temp_root()):
        raise MirrorContainment("hedef ayna kokunun icinde degil")
    if not _present(holder):
        return
    info = os.lstat(holder)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) or \
            (getattr(info, "st_file_attributes", 0) & 0x400):
        raise MirrorContainment("hedef bir baglanti ya da ayrisma noktasi")
    marker = read_marker(holder)
    if marker.get("acceptance_id") != mirror.acceptance_id:
        raise MirrorContainment("kabul aynasi bu cagriya ait degil")
    if not _remove_tree(holder):
        raise MirrorCleanupFailed("kabul aynasi silinemedi")
