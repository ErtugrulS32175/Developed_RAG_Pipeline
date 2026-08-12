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
from dataclasses import dataclass
from pathlib import Path

from tools.agent_loop import contract
from tools.agent_loop import git_objects
from tools.agent_loop import state as state_module

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
REFERENCE_DIRNAME = "reference"
IMPLEMENTER_DIRNAME = "implementer"
REGISTRY_DIRNAME = "flat-workspaces"

STATUS_REGISTERED = "registered"
STATUS_MATERIALIZING = "materializing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
_STATUSES = (STATUS_REGISTERED, STATUS_MATERIALIZING, STATUS_READY,
             STATUS_FAILED)

WORKSPACE_ID = re.compile(r"^[0-9a-f]{32}$")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# The only two blob modes this version represents. A symlink, a gitlink
# and an unknown mode each mean something different, and inventing a
# representation for one is how an unreviewed thing gets written to disk.
MODE_FILE = "100644"
MODE_EXEC = "100755"
MODE_TREE = "40000"
_ALLOWED_BLOB_MODES = (MODE_FILE, MODE_EXEC)
_MAX_TREE_DEPTH = 64

_GIT_TIMEOUT_SECONDS = 120
_FORBIDDEN_CATEGORIES = ("Cc", "Cf", "Zl", "Zp")
_ORDINARY_SPACE = " "

RECORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "workspace_id", "repo_id", "run_id",
                 "baseline_sha", "status"],
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
        "created_at": {"type": "string"},
    },
}


@dataclass(frozen=True, slots=True)
class Limits:
    """Refusals, never truncations.

    The ceilings describe a source tree, not the protected roots D2
    measures: this materialises a commit's tracked content, which was
    measured at 162 files and 1.81 MB. The headroom is generous and
    deliberate -- a ceiling that has to be raised every month stops
    being read."""

    max_entries: int = 50000
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


def _set_status(state_dir, workspace_id, status, **changes):
    if status not in _STATUSES:
        raise FlatWorkspaceError("bilinmeyen calisma alani durumu")
    record = read_record(state_dir, workspace_id)
    if record is None:
        raise FlatWorkspaceError("calisma alani kaydi yok")
    record.update(changes)
    record["status"] = status
    _write_record(state_dir, record)
    return record


def _authorised_record(repo, state_dir, workspace_id):
    """The record, checked against the repository asking to use it.

    An id alone authorising a delete is exactly the hole an earlier
    audit walked through in the worktree module."""
    record = read_record(state_dir, workspace_id)
    if record is None:
        raise FlatWorkspaceError("calisma alani kaydi yok")
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
              "status": STATUS_REGISTERED}
    _write_record(state_dir, record)
    workspace_id = record["workspace_id"]

    holder = holder_for(workspace_id)
    try:
        # never `exist_ok`: a collision might BE another run, and the
        # only safe answer is to refuse rather than to make room
        holder.mkdir()
    except FileExistsError:
        _set_status(state_dir, workspace_id, STATUS_FAILED)
        raise FlatWorkspaceError("calisma alani dizini zaten var") from None
    except OSError:
        _set_status(state_dir, workspace_id, STATUS_FAILED)
        raise FlatWorkspaceError("calisma alani dizini yaratilamadi") from None

    _set_status(state_dir, workspace_id, STATUS_MATERIALIZING)
    try:
        reference, implementer, manifest = _materialise(
            repo, holder, girisler, limits)
        digest = baseline_digest(manifest)
    except BaseException:
        # FAILED either way and the record is NEVER dropped: if the
        # holder could not be removed the residue has to stay findable,
        # and a cleanup problem must not replace the error that got us
        # here.
        _remove_tree_quietly(holder)
        _set_status(state_dir, workspace_id, STATUS_FAILED)
        raise

    _set_status(state_dir, workspace_id, STATUS_READY, baseline_digest=digest)
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
    """All four identities, or nothing. No git.

    A record alone is not authority: it has to name THIS repository,
    THIS run, THIS workspace and THIS baseline, and the directories it
    describes have to actually be there and finished."""
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
    for yol in (holder, reference, implementer):
        if not yol.is_dir():
            raise FlatWorkspaceError("calisma alani dizini yok")
    for kok in (reference, implementer):
        if (kok / ".git").exists():
            raise FlatWorkspaceError("calisma alaninda git denetim duzlemi var")
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
    if holder.exists():
        raise FlatWorkspaceError("calisma alani silinemedi")
    # the record goes ONLY after the holder is proven gone
    _record_path(state_dir, workspace_id).unlink(missing_ok=True)


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
