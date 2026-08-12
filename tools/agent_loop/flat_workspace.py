"""The git-less flat workspace. PACKAGE B2B-A-D3A.

WHY NOT A DISPOSABLE WORKTREE. Git was the evidence authority twice and
lost twice, both times to state the model could reach: the per-repository
index, where `skip-worktree` hid a modified control-plane file, and
shared metadata, where a clean filter declared in `.git/config` hid a
change AND executed a model-supplied command during verification. A
worktree carries a `.git` link straight into the model's reach. So the
workspace this module builds has no `.git` at all.

WHAT IS BUILT. Two independent plain trees -- a REFERENCE the model
never sees and an IMPLEMENTER tree it works in -- materialised from the
raw blob bytes of one commit. They start byte-identical and are not
hard links, so a write in one cannot reach the other.

WHY RAW OBJECTS AND NOT `git checkout`. Checkout runs clean/smudge
filters, applies end-of-line conversion and fires hooks: three places
where the repository's own configuration gets a vote over the copy.
`git cat-file` on a blob returns the stored bytes and nothing else. The
same reasoning bars `git checkout-index`, `git archive`, and copying
from the working tree -- the working tree is not the commit.

GIT'S ANSWER IS VERIFIED, NOT TRUSTED. Every blob's object id is
recomputed here from the bytes that actually arrived --
`sha1(b"blob " + length + b"\\0" + data)` -- and compared with the id
`ls-tree` reported. A truncated read, a replaced object or a confused
pipe therefore fails instead of being written out.

WHERE GIT STOPS. `create()` is the only function that may run it. After
`create()` returns, `assert_binding`, `remove` and `find_orphans` touch
nothing but the filesystem and the ledger, so verification of the next
model call never has to ask a repository anything.

WHAT MAY LEAVE. Fixed sentences, closed reasons, ids and digests. Never
a blob's bytes, never an absolute path, never git's own stderr -- that
text carries paths, remotes and credential-helper complaints.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tools.agent_loop import contract
from tools.agent_loop import git_objects
from tools.agent_loop import state as state_module
from tools.agent_loop import workspace_evidence

# ONE error class for the package, defined in the lower module so
# both layers can raise it without an import cycle.
FlatWorkspaceError = git_objects.FlatWorkspaceError
baseline_digest = git_objects.baseline_digest
blob_object_id = git_objects.blob_object_id
object_id = git_objects.object_id
MODE_FILE = git_objects.MODE_FILE
MODE_EXEC = git_objects.MODE_EXEC
MODE_TREE = git_objects.MODE_TREE

TEMP_PREFIX = "agent-loop-flat-"
ROOT_DIRNAME = "agent-loop-flat"
REGISTRY_DIRNAME = "flat-workspaces"

# named where the evidence layer names them, so the walker and the
# lifecycle can never disagree about which directory is which
REFERENCE_DIRNAME = workspace_evidence.REFERENCE_DIRNAME
IMPLEMENTER_DIRNAME = workspace_evidence.IMPLEMENTER_DIRNAME
MARKER_NAME = workspace_evidence.MARKER_NAME
MARKER_VERSION = workspace_evidence.MARKER_VERSION

STATUS_REGISTERED = "registered"
STATUS_MATERIALIZING = "materializing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
_STATUSES = (STATUS_REGISTERED, STATUS_MATERIALIZING, STATUS_READY,
             STATUS_FAILED)

# THE WHOLE LIFECYCLE, written down. Without it every guard downstream
# was reading a status that could have arrived from anywhere -- a record
# that jumped straight to READY described a workspace nobody built.
# READY and FAILED are terminal: a finished run does not change its mind.
_ALLOWED_TRANSITIONS = {
    STATUS_REGISTERED: (STATUS_MATERIALIZING,),
    STATUS_MATERIALIZING: (STATUS_READY, STATUS_FAILED),
    STATUS_READY: (),
    STATUS_FAILED: (),
}

# Who the record belongs to. A status update may never touch these: the
# one thing a ledger entry is for is saying whose residue this is, and a
# transition that could rewrite that could hand a holder to another run.
_OWNERSHIP_FIELDS = ("protocol_version", "workspace_id", "repo_id",
                     "run_id", "baseline_sha", "created_at")

# The one fixed sentence a cleanup failure may add. Free text next to a
# filesystem is a path leak, so there is exactly this and nothing else.
CLEANUP_NOTE = "calisma alani artigi temizlenemedi"

WORKSPACE_ID = re.compile(r"^[0-9a-f]{32}$")
_TIMESTAMP = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"

RECORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "workspace_id", "repo_id", "run_id",
                 "baseline_sha", "status", "created_at"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "workspace_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        # identity, never a path: an absolute path in a state file is an
        # absolute path in a report
        "repo_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "run_id": {"type": "string", "pattern": contract.IDENTIFIER_PATTERN},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "status": {"enum": list(_STATUSES)},
        "baseline_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        # a CLOSED shape, and the runner is the only thing that writes
        # it: a timestamp a caller supplies is a timestamp a caller can
        # choose, and recovery decisions are made by reading it
        "created_at": {"type": "string", "pattern": _TIMESTAMP},
        # R1B.3: what the filesystem said, not what the writer intended
        "evidence_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "reference_identity": {"type": "string", "minLength": 1,
                               "maxLength": 128},
        "implementer_identity": {"type": "string", "minLength": 1,
                                 "maxLength": 128},
        "marker_version": {"const": MARKER_VERSION},
    },
    # READY is the status every downstream guard trusts, so it is the
    # one that has to carry the whole proof. A record cannot arrive
    # there without the evidence digest, both root identities and the
    # marker version -- the fields `assert_binding` compares against.
    "allOf": [{
        "if": {"properties": {"status": {"const": STATUS_READY}},
               "required": ["status"]},
        "then": {"required": ["baseline_digest", "evidence_digest",
                              "reference_identity", "implementer_identity",
                              "marker_version"]},
    }],
}


@dataclass(frozen=True, slots=True)
class Limits:
    """Refusals, never truncations.

    The ceilings describe a source tree, not the protected roots D2
    measures: this materialises a commit's tracked content, which was
    measured at 162 files and 1.81 MB. The headroom is generous and
    deliberate -- a ceiling that has to be raised every month stops
    being read."""

    max_entries: int = 400000
    # a D3-specific MEASURED decision, not a guess: the largest tracked
    # blob at the baseline is 98,162 bytes, so this is roughly 680x
    # headroom -- and it is also the transport's read ceiling, which is
    # why a blob past it is refused before it reaches memory
    max_file_bytes: int = 64 << 20
    max_total_bytes: int = 1024 << 20


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class FlatWorkspace:
    workspace_id: str
    baseline_sha: str
    baseline_digest: str
    reference_root: Path
    implementer_root: Path


# ---------------------------------------------------------------------
# identity and containment
# ---------------------------------------------------------------------

def _mint_id() -> str:
    return secrets.token_hex(16)


def _now() -> str:
    """The runner's clock, in one closed shape. Never a caller's."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def runner_temp_root() -> Path:
    """The one directory a flat workspace may live in.

    A NAMED subdirectory, never the system temp directory itself:
    containment against a root that holds everything transient on the
    machine means almost nothing."""
    root = Path(tempfile.gettempdir()).resolve() / ROOT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def holder_for(workspace_id) -> Path:
    """The single directory a given id can name.

    The id is checked against a strict 32-hex pattern before it is
    joined, so no id can denote a path outside the runner-owned root --
    `..` is not an id. This is containment ONLY; it says nothing about
    who owns the directory, which is what the record is for."""
    if not WORKSPACE_ID.match(str(workspace_id)):
        raise FlatWorkspaceError("gecersiz calisma alani kimligi")
    return runner_temp_root() / f"{TEMP_PREFIX}{workspace_id}"


def _comparable(path) -> str:
    """One spelling for a path. Case folded only where the filesystem
    folds -- the standard `os.name` assumption, not a per-directory
    measurement."""
    text = str(Path(path))
    return text.casefold() if os.name == "nt" else text


# ---------------------------------------------------------------------
# git, and only inside `create`
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# names
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# the baseline manifest
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# the ledger
# ---------------------------------------------------------------------

def _record_path(state_dir, workspace_id) -> Path:
    holder_for(workspace_id)                       # validates the id
    return Path(state_dir) / REGISTRY_DIRNAME / f"{workspace_id}.json"


def _write_record(state_dir, record):
    """Isolated so a test can make exactly this fail and prove nothing
    was created on disk before it."""
    try:
        state_module.write_json_atomically(
            _record_path(state_dir, record["workspace_id"]), record,
            RECORD_SCHEMA, "duz calisma alani kaydi")
    except state_module.StateError:
        raise FlatWorkspaceError("calisma alani kaydi yazilamadi") from None
    except OSError:
        raise FlatWorkspaceError("calisma alani kaydi yazilamadi") from None


def read_record(state_dir, workspace_id):
    """The ownership record, or None. Never guesses."""
    try:
        return state_module.read_json_checked(
            _record_path(state_dir, workspace_id), RECORD_SCHEMA,
            "duz calisma alani kaydi")
    except state_module.StateError:
        return None
    except OSError:
        return None


def _load_record(state_dir, workspace_id):
    """The record, or a refusal that says WHICH kind of nothing it is.

    Absent and malformed are different situations -- one is a workspace
    that was never registered, the other is a ledger entry somebody or
    something damaged -- and collapsing them hides the second."""
    if not _record_path(state_dir, workspace_id).is_file():
        raise FlatWorkspaceError("calisma alani kaydi yok")
    record = read_record(state_dir, workspace_id)
    if record is None:
        raise FlatWorkspaceError("calisma alani kaydi bozuk")
    return record


def _set_status(state_dir, workspace_id, status, **changes):
    """ONE legal step along the lifecycle, or a refusal.

    Every guard downstream reads this status, so a write that could put
    any value there is a write that could satisfy any guard: a record
    that arrived at READY without ever materialising describes a
    workspace nobody built."""
    if status not in _STATUSES:
        raise FlatWorkspaceError("bilinmeyen calisma alani durumu")
    for alan in changes:
        if alan in _OWNERSHIP_FIELDS:
            raise FlatWorkspaceError(
                "durum degisikligi sahiplik alanina dokunuyor")
    record = _load_record(state_dir, workspace_id)
    simdiki = record["status"]
    if simdiki not in _ALLOWED_TRANSITIONS:
        raise FlatWorkspaceError("bilinmeyen calisma alani durumu")
    if status not in _ALLOWED_TRANSITIONS[simdiki]:
        raise FlatWorkspaceError("izinsiz durum gecisi")
    record.update(changes)
    record["status"] = status
    _write_record(state_dir, record)
    return record


def _authorised_record(repo, state_dir, workspace_id):
    """The record, checked against the repository asking to use it.

    An id alone authorising a delete is exactly the hole an earlier
    audit walked through in the worktree module."""
    record = _load_record(state_dir, workspace_id)
    if record["repo_id"] != state_module.repo_identity(repo):
        raise FlatWorkspaceError("kayit bu depoya ait degil")
    return record


# ---------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------

def _write_blob(root: Path, relative: str, mode: str, data: bytes):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # a fresh file every time: never a hard link, never a reuse, so a
    # write in one tree cannot be a write in the other
    with open(target, "wb") as stream:
        stream.write(data)
    if mode == MODE_EXEC and os.name != "nt":
        current = os.stat(target).st_mode
        os.chmod(target, current | stat.S_IXUSR | stat.S_IXGRP |
                 stat.S_IXOTH)


def _materialise(repo, holder: Path, girisler, limits: Limits):
    reference = holder / REFERENCE_DIRNAME
    implementer = holder / IMPLEMENTER_DIRNAME
    reference.mkdir(parents=True)
    implementer.mkdir(parents=True)

    manifest = []
    toplam = 0
    for yol, mode, oid in girisler:
        data = git_objects._object_bytes(
            repo, oid, limit=limits.max_file_bytes)
        if type(data) is not bytes:
            raise FlatWorkspaceError("nesne baytlari okunamadi")
        # git's answer, verified: a truncated read or a substituted
        # object fails here rather than being written out
        if blob_object_id(data) != oid:
            raise FlatWorkspaceError("nesne kimligi baytlarla uyusmuyor")
        if len(data) > limits.max_file_bytes:
            raise FlatWorkspaceError("dosya sozlesme tavanini asiyor")
        toplam += len(data)
        if toplam > limits.max_total_bytes:
            raise FlatWorkspaceError("toplam boyut sozlesme tavanini asiyor")
        # one source, two destinations
        _write_blob(reference, yol, mode, data)
        _write_blob(implementer, yol, mode, data)
        manifest.append((yol, mode, oid, len(data)))
    return reference, implementer, manifest


def _remove_tree_quietly(path: Path) -> bool:
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        return False


def _present(path) -> bool:
    """Is anything there. `lstat`, not `exists()`: `exists()` follows a
    link and answers False for a broken one, and a dangling link at the
    holder path is still residue."""
    try:
        os.lstat(path)
        return True
    except OSError:
        return False


def _remove_record_quietly(state_dir, workspace_id) -> bool:
    try:
        _record_path(state_dir, workspace_id).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _fail_quietly(state_dir, workspace_id) -> bool:
    try:
        _set_status(state_dir, workspace_id, STATUS_FAILED)
        return True
    except (FlatWorkspaceError, OSError):
        return False


def _mark(error):
    """The fixed marker, and only the fixed marker."""
    try:
        error.add_note(CLEANUP_NOTE)
    except AttributeError:                         # pragma: no cover -- <3.11
        pass


def _after_failure(state_dir, workspace_id, holder, ours, primary):
    """Decide what the ledger says once materialisation has failed.

    THE QUESTION IS NOT "DID WE CREATE IT" BUT "IS ANYTHING THERE NOW".
    A record was kept unconditionally before, so every failure left a
    FAILED entry -- including the ones that cleaned up perfectly, which
    made `find_orphans` report residue that did not exist and trained
    whoever read it to ignore the answer.

    The holder is removed only when it is OURS: a directory that was
    already at that path belongs to something else, and this function
    is not allowed to have an opinion about it.

    Whatever failed first still wins. Cleanup only ever adds the one
    fixed marker."""
    cleaned = _remove_tree_quietly(holder) if ours else True
    if _present(holder):
        cleaned = False
    if cleaned:
        # nothing of ours is on disk, so the record describes nothing
        if _remove_record_quietly(state_dir, workspace_id):
            return
    else:
        # residue may remain: it has to stay findable, with its
        # ownership intact, or recovery has no way back to it
        _fail_quietly(state_dir, workspace_id)
    _mark(primary)


def create(repo, *, state_dir, run_id, baseline_sha) -> FlatWorkspace:
    """Build the two trees. The ONLY function here that runs git.

    Write-ahead: the record exists before a single directory does, so a
    process that dies during materialisation still leaves something
    that names the repository and the run."""
    limits = DEFAULT_LIMITS
    girisler = git_objects._read_tree(repo, baseline_sha, limits)

    record = {"protocol_version": contract.PROTOCOL_VERSION,
              "workspace_id": _mint_id(),
              "repo_id": state_module.repo_identity(repo),
              "run_id": run_id, "baseline_sha": baseline_sha,
              "status": STATUS_REGISTERED, "created_at": _now()}
    _write_record(state_dir, record)
    workspace_id = record["workspace_id"]
    holder = holder_for(workspace_id)

    ours = False
    try:
        _set_status(state_dir, workspace_id, STATUS_MATERIALIZING)
        try:
            # never `exist_ok`: a collision might BE another run, and
            # the only safe answer is to refuse rather than make room
            holder.mkdir()
        except FileExistsError:
            raise FlatWorkspaceError(
                "calisma alani dizini zaten var") from None
        except OSError:
            raise FlatWorkspaceError(
                "calisma alani dizini yaratilamadi") from None
        # only NOW may anything here delete that directory
        ours = True
        reference, implementer, manifest = _materialise(
            repo, holder, girisler, limits)
        digest = baseline_digest(manifest)
        # THE WRITES RETURNING IS NOT EVIDENCE. Both trees are read back
        # through the handle-bound walker and required to agree, because
        # every later comparison between them is meaningless if they did
        # not start equal.
        kanit, ref_kimlik, imp_kimlik = workspace_evidence.compare_roots(
            reference, implementer)
        # the marker exists BEFORE READY, and is read back through a
        # handle rather than believed because it was just written
        workspace_evidence.write_marker(holder, record)
        workspace_evidence.inspect_holder(holder)
    except BaseException as primary:
        _after_failure(state_dir, workspace_id, holder, ours, primary)
        raise

    _set_status(state_dir, workspace_id, STATUS_READY,
                baseline_digest=digest, evidence_digest=kanit,
                reference_identity=ref_kimlik,
                implementer_identity=imp_kimlik,
                marker_version=MARKER_VERSION)
    return FlatWorkspace(workspace_id=workspace_id, baseline_sha=baseline_sha,
                         baseline_digest=digest, reference_root=reference,
                         implementer_root=implementer)


# ---------------------------------------------------------------------
# the git-less surface
# ---------------------------------------------------------------------

def _roots_of(workspace_id):
    holder = holder_for(workspace_id)
    return holder, holder / REFERENCE_DIRNAME, holder / IMPLEMENTER_DIRNAME


def assert_binding(repo, *, state_dir, run_id, workspace_id,
                   baseline_sha) -> FlatWorkspace:
    """All four identities AND the objects themselves, or nothing. No git.

    A record alone is not authority: it has to name THIS repository,
    THIS run, THIS workspace and THIS baseline. Neither is a path --
    the holder is opened once and the marker and both roots are opened
    relative to that OBJECT, so a directory swapped for a link, or for
    another directory with the same contents, is refused rather than
    described."""
    record = _authorised_record(repo, state_dir, workspace_id)
    if record["status"] != STATUS_READY:
        raise FlatWorkspaceError("calisma alani hazir degil")
    if record["run_id"] != run_id:
        raise FlatWorkspaceError("kayit bu kosuya ait degil")
    if record["baseline_sha"] != baseline_sha:
        raise FlatWorkspaceError("kayit bu taban surumune ait degil",
                                 reason=contract.StopReason.BASELINE_MISMATCH)
    digest = record.get("baseline_digest")
    if type(digest) is not str or len(digest) != 64:
        raise FlatWorkspaceError("taban ozeti kayitta yok")

    holder, reference, implementer = _roots_of(workspace_id)
    marker, ref_kimlik, imp_kimlik = workspace_evidence.inspect_holder(holder)
    # the marker is a SECOND statement of the same identities, so it has
    # to agree with the ledger field by field -- it never replaces it
    for alan in ("workspace_id", "repo_id", "run_id", "baseline_sha"):
        if marker.get(alan) != record[alan]:
            raise FlatWorkspaceError("sahiplik isareti kayitla uyusmuyor")
    if marker.get("marker_version") != record.get("marker_version"):
        raise FlatWorkspaceError("sahiplik isareti kayitla uyusmuyor")
    # identity comes from the OPENED object, never from a second look at
    # the path that named it
    if ref_kimlik != record.get("reference_identity") or \
            imp_kimlik != record.get("implementer_identity"):
        raise FlatWorkspaceError("calisma alani kokleri degistirilmis",
                                 reason=contract.StopReason.PATH_NOT_ALLOWED)
    return FlatWorkspace(workspace_id=workspace_id, baseline_sha=baseline_sha,
                         baseline_digest=digest, reference_root=reference,
                         implementer_root=implementer)


def _assert_removable(repo, state_dir, workspace_id) -> Path:
    """Everything that has to be true before anything is deleted."""
    _authorised_record(repo, state_dir, workspace_id)
    holder = holder_for(workspace_id)
    kok = runner_temp_root()

    if holder.name != f"{TEMP_PREFIX}{workspace_id}":
        raise FlatWorkspaceError("hedef beklenen holder adi degil")
    ebeveyn = _comparable(holder.parent)
    if ebeveyn != _comparable(kok):
        raise FlatWorkspaceError("hedef runner kokunun icinde degil")
    if _comparable(holder) in (_comparable(kok), _comparable(state_dir),
                               _comparable(Path(repo))):
        raise FlatWorkspaceError("hedef genis bir dizin")
    if not holder.exists():
        raise FlatWorkspaceError("calisma alani dizini yok")
    info = os.lstat(holder)
    if stat.S_ISLNK(info.st_mode) or \
            getattr(info, "st_reparse_tag", 0) or \
            (getattr(info, "st_file_attributes", 0) & 0x400):
        raise FlatWorkspaceError("hedef bir baglanti ya da ayrisma noktasi")
    if not holder.is_dir():
        raise FlatWorkspaceError("hedef bir dizin degil")
    return holder


def remove(repo, *, state_dir, workspace_id) -> None:
    """Delete the holder this RECORD authorises. No path is accepted
    and no git runs.

    `shutil.rmtree` does not follow a directory symlink or a junction:
    the link is unlinked and whatever it points at is untouched."""
    holder = _assert_removable(repo, state_dir, workspace_id)
    try:
        shutil.rmtree(holder, onexc=_clear_readonly)
    except TypeError:                              # pragma: no cover - <3.12
        shutil.rmtree(holder, onerror=_clear_readonly_legacy)
    except OSError:
        raise FlatWorkspaceError("calisma alani silinemedi") from None
    if _present(holder):
        raise FlatWorkspaceError("calisma alani silinemedi")
    # the record goes ONLY after the holder is proven gone, and a
    # record that cannot be removed is its own refusal rather than a
    # silent success -- the entry left behind would name a holder that
    # is not there
    try:
        _record_path(state_dir, workspace_id).unlink(missing_ok=True)
    except OSError:
        raise FlatWorkspaceError("calisma alani kaydi silinemedi") from None


def _clear_readonly(function, path, error):
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _clear_readonly_legacy(function, path, info):   # pragma: no cover
    os.chmod(path, stat.S_IWRITE)
    function(path)


def find_orphans(repo, *, state_dir, run_id=None) -> tuple:
    """Records whose holder is missing or unfinished. No git.

    Recovery does not have to be told an id: the ledger is scanned, so
    residue from a process that died anywhere is findable."""
    kok = Path(state_dir) / REGISTRY_DIRNAME
    if not kok.is_dir():
        return ()
    repo_id = state_module.repo_identity(repo)
    yetimler = []
    for kayit_yolu in sorted(kok.glob("*.json")):
        workspace_id = kayit_yolu.stem
        if not WORKSPACE_ID.match(workspace_id):
            continue
        record = read_record(state_dir, workspace_id)
        if record is None or record["repo_id"] != repo_id:
            continue
        if run_id is not None and record["run_id"] != run_id:
            continue
        holder, reference, implementer = _roots_of(workspace_id)
        eksik = (not holder.is_dir() or not reference.is_dir()
                 or not implementer.is_dir())
        if eksik or record["status"] != STATUS_READY:
            yetimler.append({"workspace_id": workspace_id,
                             "run_id": record["run_id"],
                             "status": record["status"],
                             "holder_exists": holder.is_dir()})
    return tuple(yetimler)
