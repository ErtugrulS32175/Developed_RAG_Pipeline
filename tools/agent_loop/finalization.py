"""Archive a terminal run, then reset the repository. PACKAGE B6-R2.

WHAT THIS MODULE REPLACES. Closing a run has been a hand-run sequence:
read the identities, copy nine or ten documents somewhere, hash every
copy, remove the workspace through its own seam, unlink the state files
one exact name at a time, and check the tree afterwards. Done by hand it
worked; done by hand it is also a sequence whose ORDER carries the whole
safety argument, and an operator in a hurry reorders it. So it becomes one
call whose order is not negotiable.

THE ORDER IS THE CONTRACT, and it has exactly one shape:

    prove who this run is  ->  prove where the archive may go  ->
    copy every byte and READ IT BACK  ->  record completion durably  ->
    and ONLY THEN remove anything

Nothing is deleted on the strength of a copy having been written. A copy
is trusted after it has been read back off the disk and its digest has
matched, and every source is measured AGAIN in the instant before it is
unlinked -- because between the archive and the removal there is a window,
and a file that changed inside it is not the file that was proven.

THERE IS ONE DELETE AUTHORITY, and it is not here. A workspace goes
through `flat_workspace.remove`, which is the only thing that checks the
ledger record authorises it. State documents go one exact name at a time
from a closed allowlist. There is no `rmtree` in this module, no glob, no
recursive walk, and a test pins their absence from the source -- because
"clean up the run directory" is precisely the instruction that, written
generously once, deletes something nobody meant.

MEASURED, and every one of these shaped a rule:

  approved  state, backup, binding, events, receipt, findings; the ledger
            is EMPTY and there is no holder, because the runner releases
            the workspace on success. So "no workspace" is an ordinary
            outcome, not a missing file.
  failed    the same set MINUS findings.json, with the ledger record and
            the holder present. So findings is genuinely optional, and a
            failed run holds the ONLY copy of the candidate.
  blocked   at preflight, only `run.lock` exists -- a preflight failure
            writes no state document at all. So there is nothing to
            finalize, and that is a refusal rather than an empty success.

WHAT IS NOT DECIDED HERE. Whether the run was any good, whether the
commit is the right commit, and whether CI passed. `shipped_commit` and
`ci_run_id` are RECORDED, and the shipment claim is checked against the
repository -- but no network call is made and no verdict is invented. A
test pins that this module contains no HTTP path at all.

WHAT MAY LEAVE. Closed codes, counts, identities and exact file names.
Never a prompt, never model prose, never stdout or stderr, never an
absolute path, never a username, never an environment value.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import application
from tools.agent_loop import application_transport as transport
from tools.agent_loop import changes, contract, flat_workspace, locking
from tools.agent_loop import runner_events
from tools.agent_loop import state as state_module

ARCHIVE_VERSION = 1

MANIFEST_NAME = "archive-manifest.json"
# WRITTEN LAST, AND ON ITS OWN. The manifest can be complete and correct
# while the copy loop is only half done; this file exists so "is this
# archive finished" is a question with a durable answer rather than an
# inference from a directory listing.
COMPLETE_NAME = "archive-complete.json"

STATE_SUBDIR = "state"
WORKSPACE_SUBDIR = "workspace"
CANDIDATE_SUBDIR = "candidate"
TASK_SUBDIR = "task"

LEDGER_DIRNAME = "flat-workspaces"
OWNER_NAME = "workspace-owner.json"

# A state document that MUST be there. Only one: a run that got past
# preflight has a state document, and everything else about it is
# conditional.
MANDATORY_STATE_FILES = (state_module.STATE_FILENAME,)
# Present or absent, and either is a fact about the run.
OPTIONAL_STATE_FILES = (
    f"{state_module.STATE_FILENAME}.backup",
    state_module.BINDING_FILENAME,
    runner_events.EVENTS_FILENAME,
    "acceptance-receipt.json",
    runner_events.FINDINGS_FILENAME,
)
# Archived as a snapshot and NEVER removed: the lock is this repository's,
# not this run's, and the directory has to survive for the next one.
KEPT_STATE_FILES = (locking.LOCK_FILENAME,)
# THE EXACT-NAME ALLOWLIST for removal. Nothing outside it is ever
# unlinked, and it deliberately excludes `run.lock`.
REMOVABLE_STATE_FILES = MANDATORY_STATE_FILES + OPTIONAL_STATE_FILES

ARCHIVED_STATE_FILES = (MANDATORY_STATE_FILES + OPTIONAL_STATE_FILES
                        + KEPT_STATE_FILES)

_RUN_ID = re.compile(r"^kosu-[0-9a-f]{24}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ARCHIVE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,120}$")

# The read ceiling for any single archived document, borrowed from the
# write transport rather than invented here.
MAX_FILE_BYTES = transport.MAX_FILE_BYTES


class FinalizeStatus:
    """What this call did. Closed, and there is no third answer."""

    FINALIZED = "finalized"
    ALREADY_FINALIZED = "already_finalized"


class Shipment:
    """Whether the run's work is on the branch everybody reads.

    `unshipped` exists so an APPROVED run whose candidate has not been
    committed is not filed as shipped -- the archive would then carry a
    claim the repository does not support."""

    SHIPPED = "shipped"
    UNSHIPPED = "unshipped"
    NOT_APPLICABLE = "not_applicable"


ALL_STATUSES = (FinalizeStatus.FINALIZED, FinalizeStatus.ALREADY_FINALIZED)
ALL_SHIPMENTS = (Shipment.SHIPPED, Shipment.UNSHIPPED,
                 Shipment.NOT_APPLICABLE)


# ---------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------

class FinalizeRefused(RuntimeError):
    """A finalize that will not proceed.

    Fixed sentence, closed reason. Never a path, never a byte of a file,
    never the operating system's own error text. A caller that sees any
    subclass of this has been told that NOTHING was removed."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


class RunNotTerminal(FinalizeRefused):
    """The run is still in flight, or there is no run at all."""


class StateBindingFailed(FinalizeRefused):
    """The documents that describe this run do not agree with each other,
    or with the identity the caller named."""


class PendingApplication(FinalizeRefused):
    """An application has not finished, so the checkout is in a state
    nobody has described. Not something to archive over."""


class ArchiveContainmentFailed(FinalizeRefused):
    """The destination is inside, above, or the same as something this
    call is about to read or remove."""


class ArchiveWriteFailed(FinalizeRefused):
    """The archive could not be written. Nothing was removed."""


class ArchiveVerificationFailed(FinalizeRefused):
    """A copy did not read back as the bytes it was made from, or a
    manifest does not satisfy its closed schema."""


class PartialArchivePresent(FinalizeRefused):
    """An archive directory exists without a completion record. A half
    archive is not evidence, and it is not overwritten either."""


class TaskManifestMismatch(FinalizeRefused):
    """The task file is not the one this run was issued, or it is not the
    caller's to remove."""


class ShipmentMismatch(FinalizeRefused):
    """The repository does not support the shipment claim."""


class CandidateScopeRefused(FinalizeRefused):
    """The candidate's changed files could not be derived through the
    change authority, so which bytes are evidence is unknown."""


class WorkspaceCleanupFailed(FinalizeRefused):
    """The workspace could not be released. The archive stands; the
    holder does too."""


class StateCleanupFailed(FinalizeRefused):
    """A state document could not be removed, or drifted between being
    archived and being removed."""


# ---------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FinalizeResult:
    """Closed fields only. No absolute path, no prose, no counts of
    anything a caller could reconstruct a document from."""

    status: str
    run_id: str
    terminal_state: str
    stop_reason: str
    shipment: str
    archive_name: str
    archived_file_count: int
    removed_workspace: bool
    removed_state_files: tuple
    removed_task_manifest: bool
    ready_for_new_task: bool
    recovered_partial_archive: bool
    pending_applications: tuple


# ---------------------------------------------------------------------
# the archive manifest, and its closed schema
# ---------------------------------------------------------------------

MANIFEST_FIELDS = (
    "archive_version", "protocol_version", "run_id", "terminal_state",
    "stop_reason", "baseline_sha", "manifest_digest", "workspace_id",
    "receipt_id", "receipt_status", "candidate_fingerprint",
    "evaluator_status", "evaluator_finding_count", "shipment",
    "shipped_commit", "ci_run_id", "files", "absent", "file_count",
)

_FILE_ENTRY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "size", "sha256"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 512},
        "size": {"type": "integer", "minimum": 0},
        "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
    },
}

ARCHIVE_MANIFEST_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["archive_version", "protocol_version", "run_id",
                 "terminal_state", "stop_reason", "shipment", "files",
                 "absent", "file_count"],
    "properties": {
        "archive_version": {"const": ARCHIVE_VERSION},
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "run_id": {"type": "string", "pattern": r"^kosu-[0-9a-f]{24}$"},
        "terminal_state": {"enum": list(contract.TERMINAL_STATES)},
        "stop_reason": {"enum": list(contract.ALL_STOP_REASONS)},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "manifest_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "workspace_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "receipt_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "receipt_status": {"type": "string", "maxLength": 32},
        "candidate_fingerprint": {"type": "string",
                                  "pattern": r"^[0-9a-f]{64}$"},
        "evaluator_status": {"type": "string", "maxLength": 32},
        "evaluator_finding_count": {"type": "integer", "minimum": 0},
        "shipment": {"enum": list(ALL_SHIPMENTS)},
        "shipped_commit": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "ci_run_id": {"type": "string", "pattern": r"^[0-9]{1,32}$"},
        "files": {"type": "array", "items": _FILE_ENTRY_SCHEMA,
                  "maxItems": 4096},
        "absent": {"type": "array", "maxItems": 64,
                   "items": {"type": "string", "minLength": 1,
                             "maxLength": 64}},
        "file_count": {"type": "integer", "minimum": 0},
    },
}


def assert_manifest(payload) -> None:
    """The manifest, against its closed schema.

    `additionalProperties: False` is the load-bearing half: a field
    nobody declared is how prose gets into a document that outlives the
    run."""
    try:
        Draft202012Validator(ARCHIVE_MANIFEST_SCHEMA).validate(payload)
    except ValidationError:
        # the validator's message quotes the instance, which is exactly
        # what must not travel
        raise ArchiveVerificationFailed("arsiv manifesti sema disi") from None
    if payload["file_count"] != len(payload["files"]):
        raise ArchiveVerificationFailed("arsiv manifesti dosya sayisiyla "
                                        "uyusmuyor")


# ---------------------------------------------------------------------
# the inputs
# ---------------------------------------------------------------------

def _exact_path(value, what: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise FinalizeRefused(f"{what} bir yol degil")
    text = os.fspath(value)
    if type(text) is not str or not text:
        raise FinalizeRefused(f"{what} bir yol degil")
    return text


def _exact_match(value, pattern, what: str, error=None) -> str:
    if type(value) is not str or not pattern.match(value):
        raise (error or FinalizeRefused)(f"{what} beklenen bicimde degil")
    return value


def archive_name_for(run_id: str, terminal_state: str,
                     shipped_commit=None) -> str:
    """The archive directory's name, from CLOSED FACTS ONLY.

    DETERMINISTIC ON PURPOSE, and a timestamp would ruin it: a second call
    has to land on the same name, or a crash between the archive and the
    cleanup strands the first copy and starts a second one. Nothing here
    can carry a username, a path or a prompt."""
    _exact_match(run_id, _RUN_ID, "kosu kimligi")
    if terminal_state not in contract.TERMINAL_STATES:
        raise FinalizeRefused("terminal durum sozlesme disi")
    name = f"{terminal_state}-{run_id}"
    if shipped_commit is not None:
        _exact_match(shipped_commit, _SHA1, "gonderilen commit",
                     ShipmentMismatch)
        name = f"{name}-{shipped_commit[:12]}"
    if not _ARCHIVE_NAME.match(name):
        raise FinalizeRefused("arsiv adi kanonik degil")
    return name


# ---------------------------------------------------------------------
# reading the run
# ---------------------------------------------------------------------

def _read_state(state_dir: Path):
    """The state document, or `None` when there is no run here.

    Absence is an ANSWER, not a failure: MEASURED, a preflight failure
    writes no state document at all, and a run this call already finalized
    has had its own removed."""
    if not (state_dir / state_module.STATE_FILENAME).exists():
        return None
    try:
        return state_module.read_state(state_dir)
    except state_module.StateError:
        raise StateBindingFailed("durum belgesi okunamadi") from None


def _read_binding(state_dir: Path):
    if not (state_dir / state_module.BINDING_FILENAME).exists():
        return None
    try:
        return state_module.read_binding(state_dir)
    except state_module.StateError:
        raise StateBindingFailed("kosu baglamasi okunamadi") from None


def _bind_run(repo_path: Path, state_dir: Path, state, expected_run_id):
    """The state document and the binding, required to agree with each
    other and with the identity the CALLER named.

    `expected_run_id` is a gate rather than a label. This call deletes
    things, so an operator who points it at the wrong checkout gets a
    refusal instead of somebody else's run archived."""
    terminal = state.get("state")
    if terminal not in contract.TERMINAL_STATES:
        raise RunNotTerminal("kosu terminal durumda degil")
    run_id = _exact_match(state.get("run_id"), _RUN_ID, "kosu kimligi",
                          StateBindingFailed)
    if run_id != expected_run_id:
        raise StateBindingFailed("durum belgesi beklenen kosuyu adlamiyor")

    binding = _read_binding(state_dir)
    workspace_id = None
    if binding is not None:
        if binding.get("run_id") != run_id:
            raise StateBindingFailed("kosu baglamasi baska bir kosuyu adliyor")
        if binding.get("repo_id") != state_module.repo_identity(repo_path):
            raise StateBindingFailed("kosu baglamasi baska bir depoyu adliyor")
        workspace_id = binding.get("workspace_id")
    return terminal, run_id, binding, workspace_id


def _bind_workspace(repo_path: Path, state_dir: Path, run_id, binding,
                    workspace_id):
    """The ledger record, the holder and the owner marker -- or a PROVEN
    absence.

    MEASURED: an APPROVED run has already released its workspace, so "no
    holder" is the ordinary case there rather than a missing file. What is
    refused is a HALF presence -- a holder with no record, a record with no
    holder, or a binding the workspace authority does not accept."""
    ledger = state_dir / LEDGER_DIRNAME
    records = sorted(item.name for item in ledger.iterdir()
                     if item.is_file()) if ledger.is_dir() else []
    if workspace_id is None:
        if records:
            raise StateBindingFailed("baglama olmadan calisma alani kaydi var")
        return None, None

    record_name = f"{workspace_id}.json"
    holder = flat_workspace.holder_for(workspace_id)
    if record_name not in records:
        if holder.exists():
            raise StateBindingFailed("calisma alani kaydi yok ama tutucu var")
        return None, None
    if not holder.is_dir():
        raise StateBindingFailed("calisma alani kaydi var ama tutucu yok")
    if not (holder / OWNER_NAME).is_file():
        raise StateBindingFailed("calisma alani sahiplik isareti yok")
    try:
        flat_workspace.assert_binding(
            repo_path, state_dir=state_dir, run_id=run_id,
            workspace_id=workspace_id,
            baseline_sha=binding.get("baseline_sha"))
    except flat_workspace.FlatWorkspaceError:
        raise StateBindingFailed("calisma alani baglamasi uyusmuyor") from None
    return holder, record_name


# ---------------------------------------------------------------------
# git, asked read-only through the transport that already bounds it
# ---------------------------------------------------------------------

def _git_lines(repo_path: Path, *args, error):
    """One read-only git question, through the ALREADY CONTAINED transport.

    No shell, bounded output, a checked return code and a container proven
    empty. This module adds no second way to run git, and it never asks
    git to CHANGE anything."""
    from tools.agent_loop import git_transport
    try:
        raw = git_transport.git_bytes(repo_path, *args, stdout_limit=64 << 10)
    except git_transport.FlatWorkspaceError:
        raise error("depo durumu olculemedi") from None
    return [line for line in raw.decode("utf-8", "replace").splitlines()
            if line]


def _git_one(repo_path: Path, *args, error):
    lines = _git_lines(repo_path, *args, error=error)
    return lines[0] if lines else ""


# ---------------------------------------------------------------------
# the task manifest
# ---------------------------------------------------------------------

def _bind_task(repo_path: Path, task_path, binding):
    """The manifest bytes, and the FOUR gates that make it removable.

    Removing the operator's own input is the most dangerous thing this
    call does, so none of these is folded into another: inside the
    repository, an ordinary file, carrying the exact digest this run was
    issued, and unknown to git in either the index or the tree."""
    if task_path is None:
        return None, None, None
    path = Path(_exact_path(task_path, "gorev dosyasi yolu"))
    try:
        entry = os.lstat(path)
    except OSError:
        raise TaskManifestMismatch("gorev dosyasi yok") from None
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) \
            or getattr(entry, "st_reparse_tag", 0):
        raise TaskManifestMismatch("gorev dosyasi siradan bir dosya degil")
    try:
        relative = path.resolve(strict=True).relative_to(
            repo_path.resolve(strict=True))
    except (OSError, ValueError):
        raise TaskManifestMismatch("gorev dosyasi depo agacinin disinda") \
            from None
    posix = relative.as_posix()

    data = path.read_bytes()
    computed = hashlib.sha256(data).hexdigest()
    if binding is None or computed != binding.get("manifest_digest"):
        raise TaskManifestMismatch("gorev dosyasi bu kosuya ait degil")
    if _git_lines(repo_path, "ls-files", "--", posix,
                  error=TaskManifestMismatch):
        raise TaskManifestMismatch("gorev dosyasi git tarafindan izleniyor")
    if _git_lines(repo_path, "diff", "--cached", "--name-only", "--", posix,
                  error=TaskManifestMismatch):
        raise TaskManifestMismatch("gorev dosyasi indekse alinmis")
    return path, posix, computed


# ---------------------------------------------------------------------
# the shipment claim
# ---------------------------------------------------------------------

def _shipment(repo_path: Path, terminal, shipped_commit, task_relative):
    """What the repository actually supports.

    `shipped` is a claim about the branch everybody else reads, so HEAD
    alone will not do: `origin/main` has to be the same commit, nothing may
    be staged, and the only file allowed to be dirty is the task manifest
    this call is about to remove. An APPROVED run with no claim is
    `unshipped` -- a real outcome, not a downgrade."""
    if shipped_commit is None:
        return (Shipment.UNSHIPPED if terminal == contract.State.APPROVED
                else Shipment.NOT_APPLICABLE)
    _exact_match(shipped_commit, _SHA1, "gonderilen commit", ShipmentMismatch)
    if terminal != contract.State.APPROVED:
        raise ShipmentMismatch("gonderilen commit yalniz onaylanmis kosu icin")
    if _git_one(repo_path, "rev-parse", "HEAD",
                error=ShipmentMismatch) != shipped_commit:
        raise ShipmentMismatch("HEAD gonderilen commit degil")
    if _git_one(repo_path, "rev-parse", "origin/main",
                error=ShipmentMismatch) != shipped_commit:
        raise ShipmentMismatch("origin/main gonderilen commit degil")
    if _git_lines(repo_path, "diff", "--cached", "--name-only",
                  error=ShipmentMismatch):
        raise ShipmentMismatch("indekste degisiklik var")
    # BOTH KINDS OF DIRTY, asked as two precise questions rather than
    # parsed out of `status --porcelain` -- whose prefixes, rename arrows
    # and quoting are a parser waiting to be wrong. Untracked counts: the
    # only file allowed to be there is the task manifest this call is about
    # to remove, and it is untracked BY DESIGN.
    dirty = _git_lines(repo_path, "diff", "--name-only",
                       error=ShipmentMismatch)
    dirty += _git_lines(repo_path, "ls-files", "--others",
                        "--exclude-standard", error=ShipmentMismatch)
    if [name for name in dirty if name != task_relative]:
        raise ShipmentMismatch("calisma agacinda gorev dosyasi disinda "
                               "degisiklik var")
    return Shipment.SHIPPED


# ---------------------------------------------------------------------
# where the archive may go
# ---------------------------------------------------------------------

def _comparable(path: Path) -> str:
    """A path in the form the local filesystem compares by.

    Windows folds case, so two spellings of one directory are the same
    directory there -- and a containment check that missed it would let
    the archive be written into the tree it is about to clean up."""
    text = os.path.normcase(os.path.normpath(str(path)))
    return text.rstrip("\\/") or text


def _overlaps(left: Path, right: Path) -> bool:
    """Same, inside, or ABOVE. All three, because an archive above the
    repository owns the repository."""
    one, two = _comparable(left), _comparable(right)
    if one == two:
        return True
    return one.startswith(two + os.sep) or two.startswith(one + os.sep)


def _assert_archive_root(root_path: Path, repo_path: Path, state_dir: Path,
                         holder):
    """The destination, checked before a byte is written.

    A reparse point HERE sends the operator's evidence somewhere they did
    not name, so it is refused as itself rather than resolved and
    followed. Then containment, in BOTH directions, against everything
    this call is about to read or remove."""
    try:
        entry = os.lstat(root_path)
    except OSError:
        raise ArchiveContainmentFailed("arsiv koku yok") from None
    if stat.S_ISLNK(entry.st_mode) or getattr(entry, "st_reparse_tag", 0):
        raise ArchiveContainmentFailed("arsiv koku yeniden ayrisma noktasi")
    if not stat.S_ISDIR(entry.st_mode):
        raise ArchiveContainmentFailed("arsiv koku bir dizin degil")

    try:
        resolved = root_path.resolve(strict=True)
    except OSError:
        raise ArchiveContainmentFailed("arsiv koku cozumlenemedi") from None

    forbidden = [repo_path, state_dir, application.apply_root_for(repo_path),
                 Path(flat_workspace.runner_temp_root())]
    if holder is not None:
        forbidden.append(holder)
    for other in forbidden:
        try:
            candidate = Path(other).resolve()
        except OSError:                       # pragma: no cover - unlikely
            candidate = Path(other)
        if _overlaps(resolved, candidate):
            raise ArchiveContainmentFailed("arsiv koku kosu agaclariyla "
                                           "cakisiyor")
    return resolved


# ---------------------------------------------------------------------
# the archive, written through handles and PROVEN by reading it back
# ---------------------------------------------------------------------

def _child_directory(parent, name: str):
    """An archive subdirectory, created exclusively or opened as itself."""
    transport.validate_child_name(name)
    try:
        entry = transport.child_entry(parent, name)
    except transport.TransportError:
        raise ArchiveWriteFailed("arsiv dizini sorgulanamadi") from None
    if entry is not None and (entry.kind != "dir" or entry.reparse_tag):
        raise ArchiveContainmentFailed("arsiv icinde siradan olmayan dizin")
    if entry is None:
        try:
            transport.create_child_directory(parent, name)
        except transport.TransportError:
            raise ArchiveWriteFailed("arsiv dizini yaratilamadi") from None
    try:
        return transport.open_child_directory(parent, name)
    except transport.TransportError:
        raise ArchiveWriteFailed("arsiv dizini acilamadi") from None


def _read_back(directory, name: str) -> bytes:
    """The bytes that are ACTUALLY on the disk under that name.

    Isolated behind one name so a test can corrupt exactly this, which is
    the only way to prove the verification is load-bearing."""
    try:
        handle = transport.open_child_file(directory, name)
    except transport.TransportError:
        raise ArchiveVerificationFailed("arsiv kopyasi geri okunamadi") \
            from None
    try:
        return transport.read_all(handle, MAX_FILE_BYTES)
    except transport.TransportError:
        raise ArchiveVerificationFailed("arsiv kopyasi geri okunamadi") \
            from None
    finally:
        transport.close_handle_quietly(handle)


def _copy_verified(data: bytes, directory, name: str):
    """Write one file, make it durable, and READ IT BACK.

    The read-back is not decoration. A write that returned, a handle that
    closed and a digest computed from the bytes still in memory prove
    nothing about what reached the disk -- and this archive is about to
    become the only copy."""
    transport.validate_child_name(name)
    try:
        handle = transport.create_child_file(directory, name)
    except transport.AlreadyExists:
        raise ArchiveWriteFailed("arsiv dosyasi zaten var") from None
    except transport.TransportError:
        raise ArchiveWriteFailed("arsiv dosyasi yaratilamadi") from None
    try:
        transport.write_all(handle, data)
        transport.fsync_handle(handle)
    except transport.TransportError:
        transport.close_handle_quietly(handle)
        raise ArchiveWriteFailed("arsiv dosyasi yazilamadi") from None
    if not transport.close_handle_quietly(handle):
        raise ArchiveWriteFailed("arsiv dosyasi kapatilamadi")
    if _read_back(directory, name) != data:
        raise ArchiveVerificationFailed("arsiv kopyasi kaynakla ayni degil")
    return len(data), hashlib.sha256(data).hexdigest()


def _source_bytes(path: Path, what: str) -> bytes:
    """One source document, read as ITSELF.

    `lstat` first, so a name swapped for a link is a refusal rather than a
    read of whatever it points at."""
    try:
        entry = os.lstat(path)
    except OSError:
        raise ArchiveWriteFailed(f"{what} okunamadi") from None
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) \
            or getattr(entry, "st_reparse_tag", 0):
        raise ArchiveContainmentFailed(f"{what} siradan bir dosya degil")
    if entry.st_size > MAX_FILE_BYTES:
        raise ArchiveWriteFailed(f"{what} sozlesme tavanini asiyor")
    try:
        return path.read_bytes()
    except OSError:
        raise ArchiveWriteFailed(f"{what} okunamadi") from None


def _write_json(directory, name: str, payload):
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return _copy_verified(data, directory, name)


def _write_completion(directory, payload):
    """The LAST thing written, and the reason a half archive is knowable.

    Isolated behind one name so a test can interrupt exactly here, which
    is the crash window that matters: everything copied, nothing yet
    allowed to be removed."""
    _write_json(directory, COMPLETE_NAME, payload)
    transport.fsync_directory(directory)


def _verify_sources_unchanged(entries):
    """Every source, measured AGAIN now that the copying has finished.

    THE WINDOW THIS CLOSES. Copying many files takes time, and a file
    rewritten while the loop was elsewhere would leave an archive that is
    not a snapshot of any single moment. Cheaper alternatives -- size,
    mtime -- are what a same-size rewrite with a restored timestamp
    defeats, so this is the digest again."""
    for path, name, expected in entries:
        if hashlib.sha256(_source_bytes(path, "arsiv kaynagi")).hexdigest() \
                != expected:
            raise ArchiveVerificationFailed("arsiv kaynagi kopyalama "
                                            "sirasinda degisti")


# ---------------------------------------------------------------------
# cleanup -- one exact name at a time, and never before the proof
# ---------------------------------------------------------------------

def _unlink_verified(directory, name: str, expected: str, *, path: Path):
    """Remove ONE file, and only after proving it is still what was
    archived.

    Two gates, and neither is redundant. The entry has to be an ordinary
    file through the held directory handle -- a name swapped for a link is
    refused rather than unlinked blindly -- and its digest has to still
    match the archived copy, because between the archive and here there is
    a window a rewrite fits in."""
    try:
        entry = transport.child_entry(directory, name)
    except transport.TransportError:
        raise StateCleanupFailed("durum belgesi sorgulanamadi") from None
    if entry is None:
        return False
    if entry.kind != "file" or entry.reparse_tag:
        raise StateCleanupFailed("durum belgesi siradan bir dosya degil")
    if hashlib.sha256(_read_back(directory, name)).hexdigest() != expected:
        raise StateCleanupFailed("durum belgesi arsivlenenden farkli")
    path.unlink()
    return True


def _cleanup(repo_path: Path, state_dir: Path, holder, workspace_id, task,
             digests):
    """The removals, in the only order that is safe.

    THE WORKSPACE FIRST, and through `flat_workspace.remove` -- the only
    authority that checks the ledger record permits it. That call also
    removes the record, which is why the record was archived before any of
    this ran. Then the state documents, one exact name from the closed
    allowlist at a time. `run.lock` and the ledger DIRECTORY stay: the
    lock is this repository's, not this run's."""
    removed_workspace = False
    if holder is not None:
        try:
            flat_workspace.remove(repo_path, state_dir=state_dir,
                                  workspace_id=workspace_id)
        except flat_workspace.FlatWorkspaceError:
            raise WorkspaceCleanupFailed("calisma alani birakilamadi") \
                from None
        removed_workspace = True

    removed = []
    try:
        root = transport.open_root(state_dir)
    except transport.TransportError:
        raise StateCleanupFailed("durum dizini acilamadi") from None
    try:
        for name in REMOVABLE_STATE_FILES:
            expected = digests.get(f"{STATE_SUBDIR}/{name}")
            if expected is None:
                continue
            try:
                gone = _unlink_verified(root, name, expected,
                                        path=state_dir / name)
            except OSError:
                raise StateCleanupFailed("durum belgesi kaldirilamadi") \
                    from None
            if gone:
                removed.append(name)
    finally:
        transport.close_directory_quietly(root)

    removed_task = False
    if task is not None:
        path, _posix, expected = task
        if hashlib.sha256(_source_bytes(path, "gorev dosyasi")).hexdigest() \
                != expected:
            raise TaskManifestMismatch("gorev dosyasi kaldirilmadan once "
                                       "degisti")
        try:
            path.unlink()
        except OSError:
            raise StateCleanupFailed("gorev dosyasi kaldirilamadi") from None
        removed_task = True
    return removed_workspace, tuple(removed), removed_task


# ---------------------------------------------------------------------
# the candidate's evidence, from the change authority and nowhere else
# ---------------------------------------------------------------------

def _candidate_paths(repo_path: Path, state_dir: Path, task, binding, holder):
    """Which files in the workspace ARE the candidate.

    NOT a copy of the whole holder. The change authority is what knows
    which paths the manifest allowed, which ones the control plane
    forbids, and which of them actually differ from the baseline -- and it
    needs the exact manifest bytes to answer. So without a `task_path`
    there is no way to derive the answer, and a workspace that holds
    changes we cannot enumerate is a REFUSAL rather than a blind copy."""
    if holder is None:
        return ()
    if task is None:
        raise CandidateScopeRefused("aday kaniti gorev dosyasi olmadan "
                                    "turetilemez")
    path, _posix, digest = task
    try:
        candidate = changes.derive_candidate_changes(
            repo=repo_path, state_dir=state_dir, task_path=path,
            manifest_digest=digest, run_id=binding["run_id"],
            workspace_id=binding["workspace_id"],
            baseline_sha=binding["baseline_sha"])
    except changes.ChangeSetError:
        raise CandidateScopeRefused("aday kaniti turetilemedi") from None
    # DELETED entries have no bytes in the implementer tree; the fact that
    # they were deleted lives in the change set, not in a file.
    return tuple(change.path for change in candidate.changes
                 if change.kind != changes.DELETED)


# ---------------------------------------------------------------------
# THE PUBLIC SEAM
# ---------------------------------------------------------------------

def _find_finalized(root_path: Path, expected_run_id: str):
    """A COMPLETE archive already naming this run, if there is one."""
    if not root_path.is_dir():
        return None
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if not (child / COMPLETE_NAME).is_file():
            continue
        manifest = child / MANIFEST_NAME
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("run_id") == expected_run_id:
            assert_manifest(payload)
            return child.name, payload
    return None


def _ready(repo_path: Path, state_dir: Path) -> tuple:
    """Whether a new task could start here, and what is still pending."""
    pending = application.find_pending_applications(repo_path)
    names = sorted(item.name for item in state_dir.iterdir()) \
        if state_dir.is_dir() else []
    ledger = state_dir / LEDGER_DIRNAME
    records = [item for item in ledger.iterdir()] if ledger.is_dir() else []
    ready = (not pending and not records
             and set(names) <= {LEDGER_DIRNAME} | set(KEPT_STATE_FILES))
    return ready, tuple(pending)


def finalize(*, repo, archive_root, expected_run_id, task_path=None,
             shipped_commit=None, ci_run_id=None) -> FinalizeResult:
    """Archive a terminal run byte-exact, then reset the repository.

    ONE call, and the single-instance lock is its OUTERMOST boundary -- so
    a loop starting in another process cannot be halfway through creating
    the workspace this one is halfway through deleting.

    `expected_run_id` is mandatory and is a GATE. `task_path` is optional,
    but without it the candidate's changed files cannot be derived through
    the change authority, so a run whose workspace still exists is refused
    rather than guessed at. `shipped_commit` and `ci_run_id` are RECORDED;
    no network call is made and no CI verdict is invented.

    Nothing is removed until the archive has been read back off the disk,
    every source has been measured a second time, and a completion record
    is durable."""
    repo_path = Path(_exact_path(repo, "depo yolu"))
    root_path = Path(_exact_path(archive_root, "arsiv koku"))
    expected_run_id = _exact_match(expected_run_id, _RUN_ID,
                                  "beklenen kosu kimligi", StateBindingFailed)
    if ci_run_id is not None:
        _exact_match(ci_run_id, re.compile(r"^[0-9]{1,32}$"), "CI kosu no")
    state_dir = repo_path / contract.STATE_DIR_NAME

    with locking.single_instance_lock(state_dir):
        return _finalize_locked(repo_path, root_path, state_dir,
                                expected_run_id, task_path, shipped_commit,
                                ci_run_id)


def _finalize_locked(repo_path, root_path, state_dir, expected_run_id,
                     task_path, shipped_commit, ci_run_id):
    """Everything from the first read to the last unlink, in order."""
    state = _read_state(state_dir)
    if state is None:
        # No run here. Either this call already finished -- in which case a
        # complete archive says so -- or there is nothing to finalize, and
        # inventing a success would be the worst answer available.
        found = _find_finalized(root_path, expected_run_id)
        if found is None:
            raise RunNotTerminal("sonlandirilacak kosu yok")
        name, manifest = found
        ready, pending = _ready(repo_path, state_dir)
        return FinalizeResult(
            status=FinalizeStatus.ALREADY_FINALIZED,
            run_id=manifest["run_id"],
            terminal_state=manifest["terminal_state"],
            stop_reason=manifest["stop_reason"],
            shipment=manifest["shipment"], archive_name=name,
            archived_file_count=manifest["file_count"],
            removed_workspace=False, removed_state_files=(),
            removed_task_manifest=False, ready_for_new_task=ready,
            recovered_partial_archive=False, pending_applications=pending)

    terminal, run_id, binding, workspace_id = _bind_run(
        repo_path, state_dir, state, expected_run_id)

    # BEFORE ANY WRITE. An unfinished application means the checkout is in
    # a state nobody has described, and filing the run as closed over it
    # would lose the only record that it is not.
    pending = application.find_pending_applications(repo_path)
    if pending:
        raise PendingApplication("bitmemis uygulama var")

    holder, record_name = _bind_workspace(repo_path, state_dir, run_id,
                                          binding, workspace_id)
    task = _bind_task(repo_path, task_path, binding)
    task_relative = task[1] if task[0] is not None else None
    shipment = _shipment(repo_path, terminal, shipped_commit, task_relative)
    resolved_root = _assert_archive_root(root_path, repo_path, state_dir,
                                         holder)
    candidate_paths = _candidate_paths(repo_path, state_dir,
                                       task if task[0] is not None else None,
                                       binding, holder)

    name = archive_name_for(run_id, terminal, shipped_commit)
    existing = resolved_root / name
    recovered = False
    if existing.exists():
        if not (existing / COMPLETE_NAME).is_file():
            raise PartialArchivePresent("tamamlanmamis arsiv dizini var")
        manifest = _verify_existing(existing, run_id)
        recovered = True
        digests = {entry["name"]: entry["sha256"]
                   for entry in manifest["files"]}
        file_count = manifest["file_count"]
    else:
        manifest, digests = _build_archive(
            resolved_root, name, repo_path, state_dir, state, binding,
            terminal, run_id, holder, record_name, workspace_id,
            task if task[0] is not None else None, candidate_paths,
            shipment, shipped_commit, ci_run_id)
        file_count = manifest["file_count"]

    removed_workspace, removed_state, removed_task = _cleanup(
        repo_path, state_dir, holder, workspace_id,
        task if task[0] is not None else None, digests)

    ready, still_pending = _ready(repo_path, state_dir)
    return FinalizeResult(
        status=FinalizeStatus.FINALIZED, run_id=run_id,
        terminal_state=terminal,
        stop_reason=state.get("stop_reason") or contract.StopReason.COMPLETED,
        shipment=shipment, archive_name=name,
        archived_file_count=file_count,
        removed_workspace=removed_workspace,
        removed_state_files=removed_state,
        removed_task_manifest=removed_task, ready_for_new_task=ready,
        recovered_partial_archive=recovered,
        pending_applications=still_pending)


def _verify_existing(directory: Path, run_id: str):
    """A completed archive from an earlier call, re-proven before it is
    trusted enough to clean up against."""
    try:
        manifest = json.loads(
            (directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ArchiveVerificationFailed("arsiv manifesti okunamadi") from None
    assert_manifest(manifest)
    if manifest["run_id"] != run_id:
        raise ArchiveContainmentFailed("arsiv baska bir kosuya ait")
    for entry in manifest["files"]:
        copy = directory / entry["name"]
        if not copy.is_file():
            raise ArchiveVerificationFailed("arsiv dosyasi eksik")
        data = copy.read_bytes()
        if len(data) != entry["size"] or \
                hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ArchiveVerificationFailed("arsiv dosyasi bozulmus")
    return manifest


def _build_archive(resolved_root, name, repo_path, state_dir, state, binding,
                   terminal, run_id, holder, record_name, workspace_id, task,
                   candidate_paths, shipment, shipped_commit, ci_run_id):
    """Copy every byte, prove it, and record completion durably.

    Returns `(manifest, digests)`. Raises without having removed anything:
    a failure here leaves the state directory, the workspace and the task
    manifest byte-identical to how they were found."""
    sources = []                       # (source path, archive name)
    absent = []
    for filename in ARCHIVED_STATE_FILES:
        path = state_dir / filename
        if path.exists():
            sources.append((path, f"{STATE_SUBDIR}/{filename}"))
        elif filename in MANDATORY_STATE_FILES:
            raise StateBindingFailed("zorunlu durum belgesi yok")
        else:
            absent.append(filename)
    if record_name is not None:
        sources.append((state_dir / LEDGER_DIRNAME / record_name,
                        f"{STATE_SUBDIR}/{LEDGER_DIRNAME}/{record_name}"))
    else:
        absent.append(f"{LEDGER_DIRNAME}/kayit")
    if holder is not None:
        sources.append((holder / OWNER_NAME,
                        f"{WORKSPACE_SUBDIR}/{OWNER_NAME}"))
        for relative in candidate_paths:
            sources.append((holder / flat_workspace.IMPLEMENTER_DIRNAME
                            / Path(relative),
                            f"{CANDIDATE_SUBDIR}/{relative}"))
    else:
        absent.append(OWNER_NAME)
    if task is not None:
        sources.append((task[0], f"{TASK_SUBDIR}/{task[0].name}"))
    else:
        absent.append("gorev-dosyasi")

    root = transport.open_root(resolved_root)
    try:
        base = _child_directory(root, name)
    except Exception:
        transport.close_directory_quietly(root)
        raise
    directories = {(): base}
    digests, entries, verify = {}, [], []
    try:
        for path, archive_name in sources:
            parts = tuple(archive_name.split("/"))
            parent = _ensure_chain(base, directories, parts[:-1])
            data = _source_bytes(path, "arsiv kaynagi")
            size, digest = _copy_verified(data, parent, parts[-1])
            digests[archive_name] = digest
            entries.append({"name": archive_name, "size": size,
                            "sha256": digest})
            verify.append((path, archive_name, digest))
        # EVERY SOURCE AGAIN, now that the copying has finished
        _verify_sources_unchanged(verify)
        manifest = _manifest_payload(state, binding, terminal, run_id,
                                     workspace_id, entries, absent, shipment,
                                     shipped_commit, ci_run_id)
        assert_manifest(manifest)
        _write_json(base, MANIFEST_NAME, manifest)
        _write_completion(base, {"archive_version": ARCHIVE_VERSION,
                                 "run_id": run_id,
                                 "file_count": manifest["file_count"]})
    finally:
        for directory in reversed(list(directories.values())):
            transport.close_directory_quietly(directory)
        transport.close_directory_quietly(root)
    return manifest, digests


def _ensure_chain(base, directories, parts):
    """Open or create the archive subdirectory chain, by handle."""
    walked = ()
    current = base
    for part in parts:
        walked = walked + (part,)
        if walked not in directories:
            directories[walked] = _child_directory(current, part)
        current = directories[walked]
    return current


def _manifest_payload(state, binding, terminal, run_id, workspace_id, entries,
                      absent, shipment, shipped_commit, ci_run_id):
    """Closed facts only, and every optional field omitted rather than
    filled with a placeholder."""
    payload = {
        "archive_version": ARCHIVE_VERSION,
        "protocol_version": contract.PROTOCOL_VERSION,
        "run_id": run_id, "terminal_state": terminal,
        "stop_reason": state.get("stop_reason")
        or contract.StopReason.COMPLETED,
        "shipment": shipment,
        "files": sorted(entries, key=lambda item: item["name"]),
        "absent": sorted(absent),
        "file_count": len(entries),
    }
    if binding is not None:
        for field in ("baseline_sha", "manifest_digest"):
            if binding.get(field):
                payload[field] = binding[field]
    if workspace_id:
        payload["workspace_id"] = workspace_id
    if shipped_commit is not None:
        payload["shipped_commit"] = shipped_commit
    if ci_run_id is not None:
        payload["ci_run_id"] = ci_run_id
    for field, key in (("evaluator_status", "evaluator_status"),
                       ("evaluator_finding_count", "finding_count")):
        value = state.get(key)
        if isinstance(value, str) and value:
            payload[field] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            payload[field] = value
    return payload
