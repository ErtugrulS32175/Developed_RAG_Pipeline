"""Transactional application to the main checkout. PACKAGE B2B-C2.

ONE primitive: take a candidate that has already been VERIFIED and
already been ACCEPTED, prove it is still exactly that by deriving it from
the filesystem again, and move it into the operator's working tree behind
a write-ahead journal -- or leave that tree exactly as it was found.

WHAT THIS IS NOT. It does not decide whether the candidate is good: the
evaluator does that, later, and this seam is only reached after it says
yes. It runs no command, calls no model, advances no state machine and
touches git at no point -- no index, no HEAD, no `add`, no `commit`, no
`push`. A working tree is changed; nothing is recorded as history.

WHAT IT WILL NOT CLAIM. A multi-file change is not one atomic operation
and nothing here calls it one. What is true is narrower and provable:

    * a write-ahead journal, durable before the first repository write;
    * PER-FILE atomic moves, none of which can replace an existing name;
    * an exact post-verification, before any success is reported;
    * a VERIFIED rollback on every failure path, interrupts included;
    * a findable recovery record if the process dies mid-flight.

An outside observer CAN see an intermediate state -- half the files
moved -- and that is a stated limit rather than an oversight. What such
an observer can never see is a file of theirs overwritten, a file of
theirs deleted, or a half-applied change set left behind by a failure
this package returned from.

THE RECEIPT IS NOT AN AUTHORITY. `AcceptanceReport` says which candidate
was tested, and that is all it is used for: every field it carries is
compared against evidence derived fresh here, through the modules that
own each question. Nothing is copied out of it and acted on.

WHY EVERY OPERATION IS A MOVE. There is no `write` into the operator's
tree and no `delete` of the operator's bytes. New content is staged
inside this call's own holder, verified there, and RENAMED into place;
displaced content is RENAMED into the holder. So the undo of any step is
the same step backwards, and nothing that was on disk when this call
started stops existing until the holder is removed.

A DECLARED LIMIT, MEASURED WHILE BUILDING THIS. "Has the operator drifted
from the baseline" is answered by comparing their file to the MATERIALISED
BASELINE, byte for byte. A checkout whose working tree does not agree with
its own blobs -- the ordinary result of `core.autocrlf=true` on Windows,
which is Git for Windows' installed default -- therefore differs from the
baseline in every text file, and every MODIFIED or DELETED target is
refused with `MainCheckoutMismatch`.

That is the safe direction and it is deliberate. This package has no git
and cannot ask which differences a filter would have normalised away, so
the alternative is to treat SOME byte differences as no difference --
which is exactly how a real edit of the operator's would get discarded.
A loud refusal costs a run; the other answer costs their work.

WHAT MAY LEAVE. Normalised repo-relative paths, counts, closed contract
codes and an opaque application id. Never a byte of a file, never a
patch, never a staging or backup location, never an absolute path, never
the operating system's own error text.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import (acceptance, application_transport as transport,
                              changes, cli, contract, flat_workspace,
                              fs_evidence, preflight, schemas,
                              state as state_module)

# A SIBLING of the repository, on the same volume by construction: a
# cross-volume rename is not atomic and is not a move at all, and every
# operation here is a move. Never inside the repository (a holder there
# is a file the change set would have to describe), never inside the
# state directory and never inside the flat roots (three lifecycles, one
# root, and one package's residue starts looking like another's).
APPLY_ROOT_DIRNAME = ".agent-loop-apply"
HOLDER_PREFIX = "apply-"
JOURNAL_NAME = "apply-journal.json"
SLOTS_DIRNAME = "slots"
JOURNAL_VERSION = 1
JOURNAL_CEILING = 1 << 22

APPLICATION_ID = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(contract.IDENTIFIER_PATTERN)
_TIMESTAMP = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
_SLOT = re.compile(r"^[sb][0-9]{4}$")

REPO_SIDE, HOLD_SIDE = "repo", "hold"

# The one fixed sentence a cleanup failure may add. Free text next to a
# filesystem is a path leak, so there is exactly this and nothing else.
CLEANUP_NOTE = "uygulama artigi temizlenemedi"


class JournalState:
    """Where an application got to. A closed set, because recovery reads
    this and has to know every value it can meet."""

    REGISTERED = "registered"
    PREPARED = "prepared"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


ALL_JOURNAL_STATES = (JournalState.REGISTERED, JournalState.PREPARED,
                      JournalState.APPLYING, JournalState.APPLIED,
                      JournalState.ROLLING_BACK, JournalState.ROLLED_BACK,
                      JournalState.FAILED)

# APPLIED and ROLLED_BACK are finished. FAILED is terminal for the
# JOURNAL and NOT for the machine: it means a rollback could not be
# proven, so it stays on disk, keeps its backups and is the one state a
# later automatic apply must refuse to run alongside.
TERMINAL_JOURNAL_STATES = (JournalState.APPLIED, JournalState.ROLLED_BACK,
                           JournalState.FAILED)

ALLOWED_JOURNAL_TRANSITIONS = {
    JournalState.REGISTERED: (JournalState.PREPARED, JournalState.ROLLING_BACK,
                              JournalState.FAILED),
    JournalState.PREPARED: (JournalState.APPLYING, JournalState.ROLLING_BACK,
                            JournalState.FAILED),
    JournalState.APPLYING: (JournalState.APPLIED, JournalState.ROLLING_BACK,
                            JournalState.FAILED),
    JournalState.ROLLING_BACK: (JournalState.ROLLED_BACK, JournalState.FAILED),
    JournalState.APPLIED: (),
    JournalState.ROLLED_BACK: (),
    JournalState.FAILED: (),
}

_MOVE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["index", "from_side", "from_name", "to_side", "to_name",
                 "sha256", "mode"],
    "properties": {
        "index": {"type": "integer", "minimum": 0},
        "from_side": {"enum": [REPO_SIDE, HOLD_SIDE]},
        "to_side": {"enum": [REPO_SIDE, HOLD_SIDE]},
        # a normalised repo-relative path or a slot name, never anything
        # a reader could resolve against a machine
        "from_name": {"type": "string", "minLength": 1, "maxLength": 1024},
        "to_name": {"type": "string", "minLength": 1, "maxLength": 1024},
        "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "mode": {"type": "string", "maxLength": 16},
    },
}

JOURNAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "journal_version", "application_id",
                 "repo_id", "run_id", "workspace_id", "baseline_sha",
                 "manifest_digest", "candidate_fingerprint", "state",
                 "created_at", "moves", "directories", "applied_index"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "journal_version": {"const": JOURNAL_VERSION},
        "application_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        # identity, never a location: an absolute path in a journal is an
        # absolute path in whatever prints the journal
        "repo_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "run_id": {"type": "string", "pattern": contract.IDENTIFIER_PATTERN},
        "workspace_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "manifest_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "candidate_fingerprint": {"type": "string",
                                  "pattern": r"^[0-9a-f]{64}$"},
        "state": {"enum": list(ALL_JOURNAL_STATES)},
        "created_at": {"type": "string", "pattern": _TIMESTAMP},
        "moves": {"type": "array", "items": _MOVE_SCHEMA, "maxItems": 100000},
        "directories": {"type": "array", "maxItems": 100000,
                        "items": {"type": "string", "minLength": 1,
                                  "maxLength": 1024}},
        # how far APPLY got: the number of moves whose execution has been
        # journalled, which is what rollback and recovery walk back from
        "applied_index": {"type": "integer", "minimum": 0},
    },
}
_JOURNAL_VALIDATOR = Draft202012Validator(JOURNAL_SCHEMA)


# ---------------------------------------------------------------------
# refusals -- closed types, fixed sentences
# ---------------------------------------------------------------------

class ApplicationError(RuntimeError):
    """A refused or failed application. Carries a fixed sentence chosen
    here and a closed contract reason. Never a path, never a byte of a
    file, never an OS message."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


class ApplicationRefused(ApplicationError):
    """An input, a bound or a binding this application may not proceed
    on. Nothing was written."""


class CandidateNotAccepted(ApplicationError):
    """The candidate on disk is not the one the acceptance report
    describes, or the report is not one this run may act on."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PATH_NOT_ALLOWED)


class MainCheckoutMismatch(ApplicationError):
    """A target in the operator's checkout is not in the shape the change
    set says it should be: an ADDED name is taken, or a MODIFIED or
    DELETED target has drifted from the baseline.

    Never a repair to attempt: the operator's own work is the thing that
    disagrees, and guessing which of the two to keep is not this
    package's decision."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PATH_NOT_ALLOWED)


class ApplicationContainment(ApplicationError):
    """A write would have left the repository, or the tree carries
    something this model cannot represent."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.PATH_NOT_ALLOWED)


class ConcurrentMainChange(ApplicationError):
    """Something outside this call moved while it was running."""

    def __init__(self, message):
        super().__init__(message, reason=contract.StopReason.DIRTY_WORKTREE)


class RollbackFailed(ApplicationError):
    """The strongest failure this package has, and the only one that
    outranks whatever went wrong first.

    An ordinary failure means the tree is as it was found. This one means
    NOBODY KNOWS what the tree is, so it is never folded into an ordinary
    red: the journal stays on disk as FAILED, the backups are not
    deleted, and a human is required before another apply runs."""

    def __init__(self, message):
        super().__init__(message,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED)


class ApplicationCleanupFailed(ApplicationError):
    """The application itself succeeded and a directory of ours is still
    on the machine. A different event from a red gate, and only one of
    them needs a human."""

    def __init__(self, message):
        super().__init__(message,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED)


# ---------------------------------------------------------------------
# what leaves
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ApplicationReport:
    """What happened, as identities, counts and closed codes."""

    run_id: str
    workspace_id: str
    baseline_sha: str
    manifest_digest: str
    candidate_fingerprint: str
    application_id: str
    applied_files: tuple
    added: int
    modified: int
    deleted: int
    rollback_performed: bool
    event: str


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """What a recovery pass did to ONE crashed application."""

    application_id: str
    state: str
    rolled_back: bool
    event: str


@dataclass(frozen=True, slots=True)
class _Move:
    """One atomic step. Its undo is itself, backwards."""

    index: int
    from_side: str
    from_name: str
    to_side: str
    to_name: str
    sha256: str
    mode: str


# ---------------------------------------------------------------------
# input canonicalization
# ---------------------------------------------------------------------

def _exact_text(value, what: str) -> str:
    try:
        return cli.exact_text(value, what=what)
    except cli.UnsafeInvocation:
        raise ApplicationRefused(f"{what} tam bir metin degil") from None


def _exact_match(value, pattern, what: str) -> str:
    if type(value) is not str or not pattern.match(value):
        raise ApplicationRefused(f"{what} beklenen bicimde degil")
    return value


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _comparable(path) -> str:
    text = str(Path(path).resolve())
    return text.casefold() if os.name == "nt" else text


def _inside(candidate: str, root: str) -> bool:
    return candidate == root or candidate.startswith(root + os.sep)


# ---------------------------------------------------------------------
# the holder
# ---------------------------------------------------------------------

def apply_root_for(repo) -> Path:
    """The ONE directory a given repository's applications may live in.

    A sibling, so a rename into the tree is a rename on one volume, and
    so nothing this package creates is ever a file the change set would
    have to describe."""
    return Path(repo).resolve().parent / APPLY_ROOT_DIRNAME


def holder_for(repo, application_id) -> Path:
    """The single directory a given id can name.

    The id is matched against a strict 32-hex pattern before it is
    joined, so no caller-supplied value can denote a directory outside
    the root this package owns -- `..` is not an id, and there is no
    parameter anywhere in this module that takes a holder path."""
    if not APPLICATION_ID.match(str(application_id)):
        raise ApplicationRefused("gecersiz uygulama kimligi")
    return apply_root_for(repo) / f"{HOLDER_PREFIX}{application_id}"


def _assert_holder_outside(holder: Path, roots) -> None:
    """Checked in BOTH directions: a holder under one of these roots is a
    write into it, and a holder ABOVE one is a cleanup that would delete
    it."""
    mine = _comparable(holder)
    for root in roots:
        theirs = _comparable(root)
        if _inside(mine, theirs) or _inside(theirs, mine):
            raise ApplicationContainment("uygulama dizini yetkili bir kokun "
                                         "icinde")


def _present(path) -> bool:
    """`lstat`, not `exists()`: `exists()` follows a link and answers
    False for a broken one, and a dangling link is still residue."""
    try:
        os.lstat(path)
        return True
    except OSError:
        return False


def _clear_readonly(function, path, error):
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _clear_readonly_legacy(function, path, info):   # pragma: no cover -- <3.12
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _remove_holder(holder: Path) -> bool:
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


# ---------------------------------------------------------------------
# the journal
# ---------------------------------------------------------------------

def _write_journal(holder: Path, payload) -> None:
    """Atomic, validated and fsynced, every single time.

    A journal that is half-written is a journal that describes nothing,
    and this is the document a recovery pass has to believe."""
    try:
        state_module.write_json_atomically(holder / JOURNAL_NAME, payload,
                                           JOURNAL_SCHEMA, "uygulama gunlugu")
    except (state_module.StateError, OSError):
        raise ApplicationRefused("uygulama gunlugu yazilamadi") from None


def _advance_journal(holder: Path, payload, target: str):
    """One legal step, or nothing. The transition table is closed for the
    same reason the run's own is: a journal that can reach any state from
    any state cannot tell a crash from a bug."""
    current = payload["state"]
    allowed = ALLOWED_JOURNAL_TRANSITIONS.get(current, ())
    if target not in allowed:
        raise ApplicationRefused("uygulama gunlugu izinsiz gecis")
    payload = {**payload, "state": target}
    _write_journal(holder, payload)
    return payload


def _read_journal(holder: Path):
    """The journal, read back THROUGH A NO-FOLLOW HANDLE on the holder.

    Opening the holder with the write transport's root open is what makes
    this safe to consult right before a delete: a holder replaced by a
    link or a junction is refused before a single child is looked at."""
    try:
        root = transport.open_root(holder)
    except transport.TransportError:
        raise ApplicationContainment("uygulama dizini acilamadi") from None
    try:
        entry = transport.child_entry(root, JOURNAL_NAME)
        if entry is None:
            raise ApplicationRefused("uygulama gunlugu yok")
        if entry.kind != "file" or entry.reparse_tag:
            raise ApplicationContainment(
                "uygulama gunlugu siradan bir dosya degil")
        handle = transport.open_child_file(root, JOURNAL_NAME)
        try:
            data = transport.read_all(handle, JOURNAL_CEILING)
        finally:
            if not transport.close_handle_quietly(handle):
                raise ApplicationCleanupFailed(
                    "uygulama gunlugu kapatilamadi")
    except transport.TransportError:
        raise ApplicationContainment("uygulama gunlugu okunamadi") from None
    finally:
        transport.close_directory_quietly(root)
    try:
        payload = json.loads(data.decode("utf-8"))
        _JOURNAL_VALIDATOR.validate(payload)
    except (UnicodeDecodeError, ValueError, ValidationError):
        raise ApplicationRefused(
            "uygulama gunlugu sozlesmeye uymuyor") from None
    return payload


def journal_state_of(repo, application_id) -> str:
    """Where one application got to. Identity in, closed code out."""
    return _read_journal(holder_for(repo, application_id))["state"]


def journal_progress_of(repo, application_id) -> int:
    """How many moves have been journalled as executed."""
    return _read_journal(holder_for(repo, application_id))["applied_index"]


# ---------------------------------------------------------------------
# navigating the tree BY HANDLE
# ---------------------------------------------------------------------

class _Tree:
    """Every directory this call has opened, held OPEN for its lifetime.

    That is the whole containment argument, and it is stronger than any
    check: a parent cannot be swapped out from under an operation because
    the operation does not name a parent, it holds one. A path is
    resolved exactly once, when its handle is first taken, and every
    later question about that directory goes to the OBJECT."""

    __slots__ = ("root", "frames", "created")

    def __init__(self, root_path):
        try:
            self.root = transport.open_root(root_path)
        except transport.TransportError:
            raise ApplicationContainment("kok dizin acilamadi") from None
        self.frames = {(): self.root}
        self.created = []

    def directory(self, parts, *, create=False, missing_ok=False):
        """The handle for a directory, opening or creating the chain.

        THREE OUTCOMES, KEPT APART. A component that is simply ABSENT is
        an answer -- an ADDED file's parent legitimately does not exist
        yet -- and collapsing it into a refusal made every such target
        look like a containment failure. A component that is present but
        is a LINK, a junction or a file is a refusal and always was.
        Anything else is a refusal too.

        `create` is only ever true for the parents of an ADDED file, and
        each level it actually makes is recorded so rollback can remove
        exactly those and nothing else."""
        walked = ()
        for name in parts:
            walked = walked + (name,)
            if walked in self.frames:
                continue
            parent = self.frames[walked[:-1]]
            try:
                entry = transport.child_entry(parent, name)
            except transport.TransportError:
                raise ApplicationContainment("ust dizin sorgulanamadi") \
                    from None
            if entry is not None and (entry.kind != "dir" or
                                      entry.reparse_tag):
                # a junction, a symlink or a file where a directory has
                # to be: the write-through this package exists to refuse
                raise ApplicationContainment(
                    "ust dizin siradan bir dizin degil")
            if entry is None:
                if not create:
                    if missing_ok:
                        return None
                    raise ApplicationContainment("hedef ust dizini yok")
                try:
                    transport.create_child_directory(parent, name)
                    self.created.append("/".join(walked))
                except transport.AlreadyExists:
                    # somebody else made it between the question and the
                    # create: it is not ours, so it is not recorded as
                    # ours and rollback will not remove it
                    pass
                except transport.TransportError:
                    raise ApplicationContainment(
                        "hedef ust dizini olusturulamadi") from None
            try:
                self.frames[walked] = transport.open_child_directory(parent,
                                                                     name)
            except transport.TransportError:
                raise ApplicationContainment(
                    "hedef ust dizini acilamadi") from None
        return self.frames[walked]

    def close(self) -> bool:
        """Release everything, deepest first, and attempt EVERY one: a
        cleanup that stops at the first problem leaks all the rest."""
        stuck = 0
        for key in sorted(self.frames, key=len, reverse=True):
            if not transport.close_directory_quietly(self.frames.pop(key)):
                stuck += 1
        return stuck == 0


def _split(relative: str):
    """A normalised repo-relative path as name components.

    The spelling comes from the change-set module, which is the one
    authority for it; every name is then validated by the transport
    before it is used to open anything."""
    parts = tuple(part for part in str(relative).replace("\\", "/").split("/")
                  if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise ApplicationContainment("hedef yolu kanonik degil")
    for part in parts:
        transport.validate_child_name(part)
    return parts


def _file_evidence(tree: _Tree, relative: str, *, ceiling):
    """A file's bytes, hash and mode, read through the handle chain.

    `None` when the name is free. An entry that is there but is not an
    ordinary file is a REFUSAL rather than an answer: a link, a junction,
    a FIFO or a device at a target is the write-through this whole
    package exists to prevent."""
    parts = _split(relative)
    directory = tree.directory(parts[:-1], missing_ok=True)
    if directory is None:
        # the parent chain does not exist yet, so neither does the file:
        # an ADDED target under a new package is the ordinary case
        return None
    try:
        entry = transport.child_entry(directory, parts[-1])
    except transport.TransportError:
        raise ApplicationContainment("hedef sorgulanamadi") from None
    if entry is None:
        return None
    if entry.kind != "file" or entry.reparse_tag:
        raise ApplicationContainment(
            "hedef siradan bir dosya degil")
    try:
        handle = transport.open_child_file(directory, parts[-1])
    except transport.TransportError:
        raise ApplicationContainment("hedef acilamadi") from None
    try:
        data = transport.read_all(handle, ceiling)
        mode = transport.handle_mode(handle)
    except transport.TransportError:
        transport.close_handle_quietly(handle)
        raise ApplicationContainment("hedef okunamadi") from None
    if not transport.close_handle_quietly(handle):
        raise ApplicationCleanupFailed("hedef kapatilamadi")
    return hashlib.sha256(data).hexdigest(), mode, data


# ---------------------------------------------------------------------
# the bindings, in order
# ---------------------------------------------------------------------

def _assert_report(report, candidate, *, run_id, workspace_id, baseline_sha,
                   manifest_digest):
    """The receipt, compared to the fresh derivation, field by field.

    EXACT TYPE FIRST, and not `isinstance`. A subclass answers every
    comparison below while being free to lie about any of them through a
    property, and this object is the whole authority for "these bytes
    were tested"."""
    if type(report) is not acceptance.AcceptanceReport:
        raise CandidateNotAccepted("kabul raporu beklenen turde degil")
    if report.passed is not True:
        raise CandidateNotAccepted("kabul raporu gecmis bir kosuyu adlamiyor")
    for field, expected in (("run_id", run_id),
                            ("workspace_id", workspace_id),
                            ("baseline_sha", baseline_sha)):
        if getattr(report, field) != expected:
            raise CandidateNotAccepted("kabul raporu bu cagriya ait degil")
    if report.manifest_digest != manifest_digest or \
            report.manifest_digest != candidate.task_digest:
        raise CandidateNotAccepted("kabul raporu baska bir gorevi adliyor")
    if report.candidate_fingerprint != candidate.fingerprint:
        # THE GATE THIS PACKAGE EXISTS FOR. A candidate edited after the
        # commands went green is a different candidate, and the receipt
        # is the only thing that would otherwise still say green.
        raise CandidateNotAccepted("kabul raporu baska bir adayi adliyor")
    if report.command_plan_digest != acceptance.command_plan_digest(
            candidate.acceptance_commands):
        raise CandidateNotAccepted("kabul raporu baska bir komut planini "
                                   "adliyor")
    results = report.command_results
    if type(results) is not tuple or len(results) != len(
            candidate.acceptance_commands):
        raise CandidateNotAccepted("kabul raporu eksik komut sonucu tasiyor")
    for result, (command_id, _) in zip(results, candidate.acceptance_commands):
        if type(result) is not acceptance.AcceptanceCommandResult:
            raise CandidateNotAccepted(
                "kabul komut sonucu beklenen turde degil")
        if result.command_id != command_id or result.passed is not True:
            # ORDER included: a plan is a sequence, and a set comparison
            # would accept the gates having run in some other one
            raise CandidateNotAccepted(
                "kabul komut sonuclari plani karsilamiyor")


def _assert_receipt(report, candidate, state_dir, binding) -> None:
    """THE AUTHORITY. Everything above this is a transport object.

    `_assert_report` checks that the receipt object is exactly an
    `AcceptanceReport` and that its digests agree with fresh evidence.
    That was measured insufficient on the commit before this one: the
    class is public, so is its constructor, and a report built by hand
    with every digest honestly derived applied a candidate to the
    operator's checkout while ZERO acceptance commands had run and ZERO
    mirrors had been created. Proving the CLASS proves nothing about the
    RUN.

    What proves the run is a file only `run_acceptance` writes, in the
    runner-owned state directory, under a name nobody can choose. It is
    read back through a handle, validated against its closed schema, and
    then required to agree -- field by field -- with the identities this
    call derived for itself. `pending` is refused, which is what makes a
    crashed or interrupted acceptance fail closed instead of leaving the
    previous green result standing.

    This runs BEFORE the holder, the journal and the first repository
    write, so a forged report costs the caller one refusal and nothing
    else."""
    try:
        receipt = acceptance.read_receipt(state_dir)
    except acceptance.AcceptanceError as refused:
        # the lower layer's sentences are fixed and carry no path
        raise CandidateNotAccepted(str(refused)) from None
    if receipt["status"] != acceptance.STATUS_PASSED:
        # `pending` is a run that started and never finished; `failed` is
        # one that finished red. Neither is permission to apply anything.
        raise CandidateNotAccepted("kabul makbuzu gecmis bir kosuyu adlamiyor")
    expected = {**binding,
                "candidate_fingerprint": candidate.fingerprint,
                "command_plan_digest": acceptance.command_plan_digest(
                    candidate.acceptance_commands),
                "command_count": len(candidate.acceptance_commands)}
    for field, value in expected.items():
        if receipt[field] != value:
            raise CandidateNotAccepted("kabul makbuzu bu adayi adlamiyor")
    if type(report.receipt_id) is not str or \
            receipt["receipt_id"] != report.receipt_id:
        # the one field that ties the object in hand to the run on disk
        raise CandidateNotAccepted("kabul raporu makbuzla eslesmiyor")


def _bind_task(task_path, repo: Path, manifest_digest: str, baseline_sha: str):
    """The exact manifest bytes, and the run they were issued for."""
    path = Path(_exact_text(task_path, "gorev dosyasi yolu"))
    try:
        entry = os.lstat(path)
    except OSError:
        raise ApplicationRefused("gorev dosyasi yok") from None
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) \
            or getattr(entry, "st_reparse_tag", 0):
        raise ApplicationContainment("gorev dosyasi siradan bir dosya degil")
    try:
        snapshot = preflight.snapshot_manifest(path)
    except (OSError, ValueError):
        raise ApplicationRefused("gorev dosyasi okunamadi") from None
    if snapshot.digest != manifest_digest:
        raise ApplicationRefused("gorev dosyasi bu kosuya ait degil")
    try:
        Draft202012Validator(schemas.TASK_SCHEMA).validate(snapshot.task)
    except ValidationError:
        raise ApplicationRefused("gorev dosyasi sema disi") from None
    try:
        path.resolve(strict=True).relative_to(repo.resolve(strict=True))
    except (OSError, ValueError):
        raise ApplicationContainment("gorev dosyasi depo agacinin disinda")
    if snapshot.task["baseline_sha"] != baseline_sha:
        raise ApplicationRefused("gorev dosyasi baska bir taban surumu adliyor")
    return path, snapshot


def _bind_state(state_dir: Path, repo: Path, run_id, baseline_sha,
                manifest_digest, workspace_id) -> None:
    try:
        binding = state_module.assert_binding(
            state_dir, repo_id=state_module.repo_identity(repo),
            baseline_sha=baseline_sha, manifest_digest=manifest_digest,
            workspace_id=workspace_id)
    except state_module.StateError:
        raise ApplicationRefused("kosu baglamasi bu cagriyla uyusmuyor") \
            from None
    if binding.get("run_id") != run_id:
        raise ApplicationRefused("kosu baglamasi bu cagriyla uyusmuyor")


def _bind_workspace(repo: Path, state_dir: Path, run_id, workspace_id,
                    baseline_sha):
    try:
        return flat_workspace.assert_binding(
            repo, state_dir=state_dir, run_id=run_id,
            workspace_id=workspace_id, baseline_sha=baseline_sha)
    except flat_workspace.FlatWorkspaceError as refused:
        # the lower layer's sentences are fixed and carry no path; its
        # reason is closed and is not re-decided here
        raise ApplicationRefused(str(refused),
                                 reason=getattr(
                                     refused, "reason",
                                     contract.StopReason.PREFLIGHT_FAILED)
                                 ) from None


# ---------------------------------------------------------------------
# preconditions and the move plan
# ---------------------------------------------------------------------

def _plan(candidate, main: _Tree, reference: _Tree, *, ceiling):
    """What every target must be right now, and the moves that follow.

    THE BASELINE IS THE REFERENCE TREE, not a record. A MODIFIED or
    DELETED entry describes a transformation OF THE BASELINE, so the
    operator's copy has to still BE the baseline -- byte for byte and
    mode for mode -- or applying it silently discards whatever they did
    instead. That comparison is made against the materialised baseline
    itself, because a digest recorded earlier is a claim and the tree is
    evidence.

    Nothing is written here. A precondition that fails costs the caller
    exactly one refusal and leaves the checkout untouched."""
    moves, staged, directories = [], [], []
    for slot, change in enumerate(candidate.changes):
        parts = _split(change.path)
        current = _file_evidence(main, change.path, ceiling=ceiling)
        if change.kind == changes.ADDED:
            if current is not None:
                # theirs: an untracked note, a local scratch file. The
                # only safe answer is to refuse the whole application.
                raise MainCheckoutMismatch("eklenecek hedef zaten var")
            directories.append(parts[:-1])
            moves.append(_Move(index=len(moves), from_side=HOLD_SIDE,
                               from_name=f"s{slot:04d}", to_side=REPO_SIDE,
                               to_name=change.path, sha256=change.sha256,
                               mode=change.mode))
            staged.append((f"s{slot:04d}", change))
            continue

        baseline = _file_evidence(reference, change.path, ceiling=ceiling)
        if baseline is None:
            raise MainCheckoutMismatch("taban surumde hedef yok")
        if current is None:
            raise MainCheckoutMismatch("degistirilecek hedef yok")
        if current[0] != baseline[0] or current[1] != baseline[1]:
            raise MainCheckoutMismatch("hedef taban surumden sapmis")
        moves.append(_Move(index=len(moves), from_side=REPO_SIDE,
                           from_name=change.path, to_side=HOLD_SIDE,
                           to_name=f"b{slot:04d}", sha256=baseline[0],
                           mode=baseline[1]))
        if change.kind == changes.MODIFIED:
            moves.append(_Move(index=len(moves), from_side=HOLD_SIDE,
                               from_name=f"s{slot:04d}", to_side=REPO_SIDE,
                               to_name=change.path, sha256=change.sha256,
                               mode=change.mode))
            staged.append((f"s{slot:04d}", change))
    return tuple(moves), tuple(staged), tuple(directories)


def _missing_directories(main: _Tree, wanted):
    """Which parents of an ADDED file do not exist yet, outermost first.

    Asked through the same held handles the moves will use, so a
    component that is a link is a REFUSAL here rather than a directory
    somebody creates inside later."""
    missing, asked = [], set()
    for parts in wanted:
        walked = ()
        for name in parts:
            walked = walked + (name,)
            relative = "/".join(walked)
            if relative in asked:
                continue
            asked.add(relative)
            if "/".join(walked[:-1]) in missing:
                # its parent is already known absent, so this is too --
                # and asking the filesystem would mean opening a
                # directory that does not exist
                missing.append(relative)
                continue
            parent = main.directory(walked[:-1])
            try:
                entry = transport.child_entry(parent, name)
            except transport.TransportError:
                raise ApplicationContainment("ust dizin sorgulanamadi") \
                    from None
            if entry is None:
                missing.append(relative)
                continue
            if entry.kind != "dir" or entry.reparse_tag:
                raise ApplicationContainment(
                    "ust dizin siradan bir dizin degil")
    return tuple(missing)


def _stage(slots, implementer: _Tree, staged, *, ceiling) -> None:
    """The candidate's bytes, copied into this call's own holder and
    proven there before anything in the repository moves.

    Read through the no-follow transport, hashed against the evidence
    that named them, written, flushed and READ BACK. A file that moved
    between the scan and this read is refused rather than staged."""
    for name, change in staged:
        source = _file_evidence(implementer, change.path, ceiling=ceiling)
        if source is None:
            raise ApplicationContainment("aday dosyasi bulunamadi")
        if source[0] != change.sha256:
            raise ApplicationContainment("aday dosyasi kanitla ayni degil")
        try:
            handle = transport.create_child_file(slots, name)
        except transport.TransportError:
            raise ApplicationRefused("aday dosyasi hazirlanamadi") from None
        try:
            transport.write_all(handle, source[2])
            transport.fsync_handle(handle)
            transport.set_handle_mode(handle, change.mode)
        except transport.TransportError:
            transport.close_handle_quietly(handle)
            raise ApplicationRefused("aday dosyasi yazilamadi") from None
        if not transport.close_handle_quietly(handle):
            raise ApplicationCleanupFailed("aday dosyasi kapatilamadi")
        written = _file_evidence(_SlotTree(slots), name, ceiling=ceiling)
        if written is None or written[0] != change.sha256:
            raise ApplicationRefused("hazirlanan dosya geri okunamadi")


class _SlotTree:
    """The slot directory, wearing the `_Tree` shape.

    A one-level tree whose handle is already open, so `_file_evidence`
    can read a staged file back through exactly the same code path that
    reads a repository target -- rather than through a second one that
    could disagree with it."""

    __slots__ = ("slots",)

    def __init__(self, slots):
        self.slots = slots

    def directory(self, parts, *, create=False, missing_ok=False):
        if parts:
            raise ApplicationContainment("yuva agacinda alt dizin yok")
        return self.slots


# ---------------------------------------------------------------------
# executing and undoing moves
# ---------------------------------------------------------------------

def _endpoint(move_side: str, name: str, main: _Tree, slots):
    """One end of a move, as `(directory handle, child name)`.

    The repository end walks the held handles; the holder end is a single
    flat directory, which is why no slot name can ever denote a path."""
    if move_side == HOLD_SIDE:
        if not _SLOT.match(name):
            raise ApplicationContainment("gecersiz yuva adi")
        return slots, name
    parts = _split(name)
    return main.directory(parts[:-1]), parts[-1]


def _evidence_of(side: str, name: str, main: _Tree, slots, *, ceiling):
    """What is at one end of a move, right now. `None` when nothing is."""
    if side == HOLD_SIDE:
        if not _SLOT.match(name):
            raise ApplicationContainment("gecersiz yuva adi")
        return _file_evidence(_SlotTree(slots), name, ceiling=ceiling)
    return _file_evidence(main, name, ceiling=ceiling)


def _reverse(move: _Move) -> _Move:
    return _Move(index=move.index, from_side=move.to_side,
                 from_name=move.to_name, to_side=move.from_side,
                 to_name=move.from_name, sha256=move.sha256, mode=move.mode)


def _execute(move: _Move, main: _Tree, slots, *, ceiling) -> None:
    """One move, with its SOURCE proven first.

    The source's bytes are hashed against the journal before it is
    touched: for a repository source that is "is this still the baseline
    object", and for a holder source "is this still what we staged". The
    TARGET is deliberately not checked here -- the transport refuses an
    occupied name in the kernel, which no check-then-act can match."""
    evidence = _evidence_of(move.from_side, move.from_name, main, slots,
                            ceiling=ceiling)
    if evidence is None:
        raise ConcurrentMainChange("tasinacak nesne artik yok")
    if evidence[0] != move.sha256:
        raise ConcurrentMainChange("tasinacak nesne beklenen icerik degil")
    source_directory, source_name = _endpoint(move.from_side, move.from_name,
                                              main, slots)
    target_directory, target_name = _endpoint(move.to_side, move.to_name,
                                              main, slots)
    try:
        transport.rename_child(source_directory, source_name,
                               target_directory, target_name)
    except transport.AlreadyExists:
        raise ConcurrentMainChange("hedef adi baskasi tarafindan alinmis") \
            from None
    except (transport.TransportError, OSError):
        raise ApplicationContainment("nesne tasinamadi") from None


def _undo(move: _Move, main: _Tree, slots, *, ceiling) -> None:
    """The same move, backwards -- AFTER establishing whether it happened.

    THE JOURNAL RECORDS INTENT, NOT COMPLETION. That is what write-ahead
    means: `applied_index` is raised BEFORE the move, so the highest
    recorded step is precisely the one whose outcome is unknown. Reading
    it as "done" made rollback try to undo a move that had not happened,
    which fails on every single first-operation failure -- the most
    ordinary case there is.

    So the object is LOCATED before anything is moved. Exactly two
    placements are consistent with this journal, and each has one honest
    answer:

        at the TARGET with the expected content -> the move happened,
                                                   so undo it;
        at the SOURCE with the expected content -> it never happened, or
                                                   has already been
                                                   undone, so do nothing.

    Anything else -- the object missing from both ends, or sitting at the
    target with content this call did not put there -- means somebody
    else has been in the tree. That is NOT overwritten and NOT deleted:
    it is a rollback that cannot be completed, which is the strongest
    failure this package has."""
    backwards = _reverse(move)
    landed = _evidence_of(backwards.from_side, backwards.from_name, main,
                          slots, ceiling=ceiling)
    if landed is not None and landed[0] == move.sha256:
        _execute(backwards, main, slots, ceiling=ceiling)
        return
    still = _evidence_of(move.from_side, move.from_name, main, slots,
                         ceiling=ceiling)
    if still is not None and still[0] == move.sha256:
        return
    raise RollbackFailed("geri alinacak nesne iki ucta da bulunamadi")


def _rollback(payload, holder: Path, main: _Tree, slots, moves, *, ceiling):
    """Undo everything this call did, in reverse, and PROVE it.

    Returns the journal payload. Raises `RollbackFailed` -- which
    outranks whatever failed first -- when any step cannot be completed,
    leaving the journal FAILED on disk with its backups intact."""
    payload = _advance_journal(holder, payload, JournalState.ROLLING_BACK)
    done = payload["applied_index"]
    try:
        for move in reversed(moves[:done]):
            _undo(move, main, slots, ceiling=ceiling)
            payload = {**payload, "applied_index": move.index}
            _write_journal(holder, payload)
        for relative in reversed(payload["directories"]):
            parts = _split(relative)
            parent = main.directory(parts[:-1])
            try:
                transport.remove_child_directory(parent, parts[-1])
            except transport.NotEmpty:
                # somebody put something in it, so it is no longer only
                # ours and removing it would delete their work
                continue
            except transport.TransportError:
                raise RollbackFailed("geri alma dizini kaldiramadi") from None
    except BaseException:
        _write_journal(holder, {**payload, "state": JournalState.FAILED})
        raise RollbackFailed("geri alma dogrulanamadi") from None
    return _advance_journal(holder, payload, JournalState.ROLLED_BACK)


# ---------------------------------------------------------------------
# THE PUBLIC SEAM
# ---------------------------------------------------------------------

def apply_accepted_candidate(*, repo, state_dir, task_path, manifest_digest,
                             run_id, workspace_id, baseline_sha,
                             verified_changes,
                             acceptance_report) -> ApplicationReport:
    """Move an accepted candidate into the operator's checkout.

    The caller passes IDENTITIES, a verified change set and an acceptance
    receipt. It does not pass a workspace path, a holder path, a file
    list, a patch or a target: the permissions come from the exact
    manifest bytes whose digest this run was issued, the candidate comes
    from the change-set module's own fresh derivation, and the holder is
    minted here. There is nothing for a caller to substitute.

    ORDER OF THE GATES IS THE CONTRACT. Every binding, every fingerprint
    and every target precondition is settled before a journal exists, and
    the journal is durable before a single repository object moves."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    state_path = Path(_exact_text(state_dir, "durum dizini"))
    manifest_digest = _exact_match(manifest_digest, _HEX64, "gorev ozeti")
    run_id = _exact_match(run_id, _RUN_ID, "kosu kimligi")
    workspace_id = _exact_match(workspace_id, flat_workspace.WORKSPACE_ID,
                                "calisma alani kimligi")
    baseline_sha = _exact_match(baseline_sha, _SHA1, "taban surum")

    identity = {"repo": repo_path, "state_dir": state_path,
                "task_path": task_path, "manifest_digest": manifest_digest,
                "run_id": run_id, "workspace_id": workspace_id,
                "baseline_sha": baseline_sha}
    task_file, snapshot = _bind_task(task_path, repo_path, manifest_digest,
                                     baseline_sha)
    _bind_state(state_path, repo_path, run_id, baseline_sha, manifest_digest,
                workspace_id)
    workspace = _bind_workspace(repo_path, state_path, run_id, workspace_id,
                                baseline_sha)
    # THE FRESH DERIVATION IS THE GATE, through the module that owns the
    # manifest binding, both walkers, the classifier and the
    # authorisation order. There is no second opinion here.
    try:
        candidate = changes.derive_candidate_changes(**identity)
    except changes.ChangeSetError as refused:
        raise ApplicationRefused(str(refused), reason=refused.reason) from None
    try:
        acceptance.assert_candidate(verified_changes, candidate,
                                    run_id=run_id, workspace_id=workspace_id,
                                    baseline_sha=baseline_sha)
    except acceptance.AcceptanceError as refused:
        raise CandidateNotAccepted(str(refused)) from None
    _assert_report(acceptance_report, candidate, run_id=run_id,
                   workspace_id=workspace_id, baseline_sha=baseline_sha,
                   manifest_digest=manifest_digest)
    binding = {"repo_id": state_module.repo_identity(repo_path),
               "run_id": run_id, "workspace_id": workspace_id,
               "baseline_sha": baseline_sha,
               "manifest_digest": manifest_digest}
    _assert_receipt(acceptance_report, candidate, state_path, binding)
    if not candidate.changes:
        raise ApplicationRefused("uygulanacak degisiklik yok")

    # THE SECOND READ of the manifest, compared to the first. The
    # document that chose everything above must not have moved while it
    # was being used.
    task_before = snapshot.digest
    if preflight.snapshot_manifest(task_file).digest != task_before:
        raise ConcurrentMainChange("gorev dosyasi degistirildi")

    return _apply(candidate, repo_path=repo_path, workspace=workspace,
                  task_file=task_file, task_before=task_before,
                  identity=identity, run_id=run_id, workspace_id=workspace_id,
                  baseline_sha=baseline_sha, manifest_digest=manifest_digest)


def _apply(candidate, *, repo_path, workspace, task_file, task_before,
           identity, run_id, workspace_id, baseline_sha, manifest_digest):
    """Everything from the first handle to the last, with exactly one way
    out of each phase."""
    ceiling = fs_evidence.Limits().max_content_file_bytes
    application_id = secrets.token_hex(16)
    holder = holder_for(repo_path, application_id)
    _assert_holder_outside(holder, (repo_path, workspace.reference_root,
                                    workspace.implementer_root,
                                    Path(workspace.reference_root).parent))

    # ONE call-local key and ONE frozen policy for every reading of the
    # operator's checkout, so the before and the after are answers to the
    # same question -- and a root created during the call cannot claim
    # the reduced evidence class for itself.
    policy = changes.freeze_main_policy(repo_path)
    key = secrets.token_bytes(fs_evidence.KEY_BYTES)
    main_before = changes.main_projection(repo_path, key=key, policy=policy)

    main = _Tree(repo_path)
    reference = _Tree(workspace.reference_root)
    implementer = _Tree(workspace.implementer_root)
    slots_tree = None
    payload = None
    moves = ()
    rolled_back = False
    try:
        moves, staged, wanted = _plan(candidate, main, reference,
                                      ceiling=ceiling)
        directories = _missing_directories(main, wanted)

        try:
            apply_root_for(repo_path).mkdir(parents=True, exist_ok=True)
            holder.mkdir()
        except FileExistsError:
            # a collision might BE another call, and the only safe answer
            # is to refuse rather than make room
            raise ApplicationRefused("uygulama dizini zaten var") from None
        except OSError:
            raise ApplicationRefused("uygulama dizini yaratilamadi") from None

        payload = {
            "protocol_version": contract.PROTOCOL_VERSION,
            "journal_version": JOURNAL_VERSION,
            "application_id": application_id,
            "repo_id": state_module.repo_identity(repo_path),
            "run_id": run_id, "workspace_id": workspace_id,
            "baseline_sha": baseline_sha, "manifest_digest": manifest_digest,
            "candidate_fingerprint": candidate.fingerprint,
            "state": JournalState.REGISTERED, "created_at": _now(),
            "moves": [{"index": move.index, "from_side": move.from_side,
                       "from_name": move.from_name, "to_side": move.to_side,
                       "to_name": move.to_name, "sha256": move.sha256,
                       "mode": move.mode} for move in moves],
            "directories": list(directories), "applied_index": 0}
        # WRITE-AHEAD OWNERSHIP: the journal exists, and has been read
        # back through a handle, before a single object moves -- so a
        # process that dies during the apply still leaves a record that
        # says whose this is and how far it got.
        _write_journal(holder, payload)
        if _read_journal(holder) != payload:
            raise ApplicationRefused("uygulama gunlugu geri okunamadi")
        (holder / SLOTS_DIRNAME).mkdir()
        slots_tree = _Tree(holder)
        slots = slots_tree.directory((SLOTS_DIRNAME,))

        _stage(slots, implementer, staged, ceiling=ceiling)
        payload = _advance_journal(holder, payload, JournalState.PREPARED)

        for relative in directories:
            parts = _split(relative)
            main.directory(parts, create=True)
        payload = _advance_journal(holder, payload, JournalState.APPLYING)
        for move in moves:
            # THE INTENT IS DURABLE BEFORE THE ACT, every time and not
            # only the first: a journal that stops being updated cannot
            # say where a crash landed, which is the one question
            # recovery has to answer.
            payload = {**payload, "applied_index": move.index + 1}
            _write_journal(holder, payload)
            _execute(move, main, slots, ceiling=ceiling)
        transport.fsync_directory(main.root)

        _verify(candidate, repo_path=repo_path, key=key, policy=policy,
                main_before=main_before, identity=identity,
                task_file=task_file, task_before=task_before)
        payload = _advance_journal(holder, payload, JournalState.APPLIED)
    except BaseException as primary:
        # EVERY exit, interrupts included. A rollback that skipped
        # KeyboardInterrupt would leave half a change set on disk every
        # time an operator changed their mind.
        if payload is not None and slots_tree is not None:
            try:
                _rollback(payload, holder, main, slots, moves, ceiling=ceiling)
                rolled_back = True
            except RollbackFailed:
                # STRONGER THAN WHATEVER FAILED FIRST: a red gate and a
                # tree nobody has described are different events, and
                # only one of them needs a human before anything else
                # runs. The holder is deliberately NOT removed.
                _release(main, reference, implementer, slots_tree)
                raise
        _release(main, reference, implementer, slots_tree)
        if payload is not None and not _remove_holder(holder):
            try:
                primary.add_note(CLEANUP_NOTE)
            except AttributeError:                 # pragma: no cover -- <3.11
                pass
        raise
    _release(main, reference, implementer, slots_tree)
    if not _remove_holder(holder):
        raise ApplicationCleanupFailed("uygulama dizini silinemedi")

    kinds = [change.kind for change in candidate.changes]
    return ApplicationReport(
        run_id=run_id, workspace_id=workspace_id, baseline_sha=baseline_sha,
        manifest_digest=manifest_digest,
        candidate_fingerprint=candidate.fingerprint,
        application_id=application_id,
        applied_files=tuple(change.path for change in candidate.changes),
        added=kinds.count(changes.ADDED),
        modified=kinds.count(changes.MODIFIED),
        deleted=kinds.count(changes.DELETED), rollback_performed=rolled_back,
        event=contract.EventCode.STATE_TRANSITION)


def _release(*trees) -> None:
    """Every handle this call opened, released -- attempting all of them
    whatever any one does."""
    for tree in trees:
        if tree is not None:
            tree.close()


def _verify(candidate, *, repo_path, key, policy, main_before, identity,
            task_file, task_before) -> None:
    """The application is not finished when the last byte lands.

    Four things are asked, and a success is reported only when all four
    answer: the checkout's semantic difference across this call IS the
    candidate and nothing else; the workspace still holds the candidate
    that was accepted; the manifest that chose it is still the same
    bytes; and the checkout is still the same root object -- which
    `main_difference` refuses to diff across."""
    after = changes.main_projection(repo_path, key=key, policy=policy)
    try:
        observed = changes.main_difference(main_before, after)
    except changes.ChangeSetError as refused:
        raise ConcurrentMainChange(str(refused)) from None
    expected = tuple((change.path, change.kind, change.sha256)
                     for change in candidate.changes)
    if tuple((item.path, item.kind, item.sha256)
             for item in observed) != expected:
        raise ConcurrentMainChange("ana agac farki adayla ayni degil")
    try:
        again = changes.derive_candidate_changes(**identity)
    except changes.ChangeSetError as refused:
        raise ConcurrentMainChange(str(refused)) from None
    if again.fingerprint != candidate.fingerprint:
        raise ConcurrentMainChange("calisma alani uygulama sirasinda degisti")
    try:
        moved = preflight.snapshot_manifest(task_file).digest != task_before
    except (OSError, ValueError):
        raise ConcurrentMainChange("gorev dosyasi okunamadi") from None
    if moved:
        raise ConcurrentMainChange("gorev dosyasi degistirildi")


# ---------------------------------------------------------------------
# CRASH RECOVERY
# ---------------------------------------------------------------------

def find_pending_applications(repo) -> tuple:
    """Every application of THIS repository that has not finished.

    By enumeration of the exact sibling root and nothing else: a caller
    cannot name a directory, a holder without a readable journal is not
    ours to have an opinion about, and a journal naming another
    repository is somebody else's problem by construction.

    APPLIED and ROLLED_BACK are finished and are not returned. FAILED is,
    because it is precisely the state a human has to clear."""
    root = apply_root_for(repo)
    try:
        names = sorted(entry.name for entry in os.scandir(root))
    except OSError:
        return ()
    repo_id = state_module.repo_identity(Path(repo))
    pending = []
    for name in names:
        if not name.startswith(HOLDER_PREFIX):
            continue
        application_id = name[len(HOLDER_PREFIX):]
        if not APPLICATION_ID.match(application_id):
            continue
        try:
            payload = _read_journal(root / name)
        except ApplicationError:
            # unreadable, unmarked, or a link where a directory should
            # be: all of them mean "not demonstrably ours", and this
            # function is not allowed to guess
            continue
        if payload["repo_id"] != repo_id or payload["application_id"] != \
                application_id:
            continue
        if payload["state"] in (JournalState.APPLIED,
                                JournalState.ROLLED_BACK):
            continue
        pending.append(application_id)
    return tuple(pending)


def recover_application(repo, *, application_id) -> RecoveryReport:
    """Finish ONE crashed application, in the ROLLBACK direction.

    Never forwards. A process that died mid-apply cannot say whether the
    candidate is still the right thing to install, and the only state
    this package can restore without asking anybody is the one the
    checkout was in before it started.

    A caller supplies an id, never a path; the holder is derived, its
    journal has to name this repository and this id, and a terminal
    record is left exactly as it is."""
    repo_path = Path(_exact_text(repo, "depo yolu"))
    application_id = _exact_match(application_id, APPLICATION_ID,
                                  "uygulama kimligi")
    holder = holder_for(repo_path, application_id)
    if _comparable(holder.parent) != _comparable(apply_root_for(repo_path)):
        raise ApplicationContainment("uygulama dizini kokun icinde degil")
    if not _present(holder):
        raise ApplicationRefused("uygulama kaydi yok")
    payload = _read_journal(holder)
    if payload["application_id"] != application_id or \
            payload["repo_id"] != state_module.repo_identity(repo_path):
        raise ApplicationContainment("uygulama kaydi bu depoya ait degil")
    if payload["state"] in (JournalState.APPLIED, JournalState.ROLLED_BACK):
        # a finished record is not re-applied and not re-undone: doing
        # either would be inventing a second opinion about a decision
        # that has already been made
        return RecoveryReport(application_id=application_id,
                              state=payload["state"], rolled_back=False,
                              event=contract.EventCode.STATE_TRANSITION)

    ceiling = fs_evidence.Limits().max_content_file_bytes
    moves = tuple(_Move(**move) for move in payload["moves"])
    main = _Tree(repo_path)
    slots_tree = _Tree(holder)
    try:
        slots = slots_tree.directory((SLOTS_DIRNAME,))
        if payload["state"] == JournalState.FAILED:
            # a FAILED record is one whose rollback could not be proven,
            # so the walk starts again from the same table rather than
            # from a state the transition table has closed
            payload = {**payload, "state": JournalState.APPLYING}
        payload = _rollback(payload, holder, main, slots, moves,
                            ceiling=ceiling)
    finally:
        _release(main, slots_tree)
    if not _remove_holder(holder):
        raise ApplicationCleanupFailed("uygulama dizini silinemedi")
    return RecoveryReport(application_id=application_id,
                          state=JournalState.ROLLED_BACK, rolled_back=True,
                          event=contract.EventCode.STATE_TRANSITION)
