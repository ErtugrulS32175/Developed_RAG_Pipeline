"""Persistent run state. PACKAGE B1.

Two documents, on purpose.

`state.json` is the frozen Phase A `STATE_SCHEMA` and nothing else. That
schema sets `additionalProperties: false` and has no field for the
repository, the manifest digest or the disposable worktree -- so binding
those INTO it would mean editing a frozen contract to make an
implementation convenient, which is the one move this whole process
exists to prevent.

`binding.json` therefore holds the identity a run is pinned to, under a
schema this package owns. Phase A already expects the state directory to
carry more than one artefact (`events.jsonl` is named in the contract),
so a second document is a use of the design rather than a change to it.

EVERY WRITE IS ATOMIC and every read is validated. A truncated state is
not a state to recover from by guessing -- it stops the run. Nothing
here silently resets and starts over: a corrupt file is evidence that
something went wrong, and throwing it away destroys the evidence while
leaving the cause in place.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import contract, schemas

STATE_FILENAME = "state.json"
BINDING_FILENAME = "binding.json"

# Fields that describe WHICH run this is. A legal state transition may
# move the run forward; it may not rewrite whose run it is. `advance`
# used to merge every keyword it was handed straight over the current
# document, so a caller could change `run_id` and `baseline_sha` in the
# middle of an otherwise valid move -- and the resulting file still
# validated, because the schema constrains shapes, not continuity.
IMMUTABLE_STATE_FIELDS = ("protocol_version", "run_id", "started_at",
                          "baseline_sha")

# What a run is pinned to. Owned by B1; the frozen STATE_SCHEMA is
# untouched.
BINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    # `worktree_id` is REQUIRED. It was optional, and the value
    # `worktree.create` produced could not satisfy the pattern anyway,
    # so the binding to a worktree was never established and nothing
    # reported that. Optional plus unsatisfiable is a field that exists
    # only in the documentation.
    "required": ["protocol_version", "run_id", "repo_id", "baseline_sha",
                 "manifest_digest", "worktree_id"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "run_id": {"type": "string", "pattern": contract.IDENTIFIER_PATTERN},
        # a digest of the repository's own identity, never its path: an
        # absolute path in a state file is a private path in a report
        "repo_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "manifest_digest": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
        "worktree_id": {"type": "string", "pattern": r"^[0-9a-f]{32}$"},
    },
}


class StateError(RuntimeError):
    """Base for every refusal that protects a run's recorded identity."""


class CorruptState(StateError):
    """The state document cannot be read or does not validate. Never
    repaired, never reset: it is evidence."""


class IncompatibleState(StateError):
    """Valid state, wrong run -- another repository, another baseline or
    another task manifest."""


class IllegalTransition(StateError):
    """A move the frozen transition table does not allow."""


def repo_identity(repo) -> str:
    """A stable, path-free id for a repository.

    Derived from the absolute path but never REVEALING it: reports and
    state files may carry identity, not location. Case folding follows
    `os.name` -- Windows's STANDARD case-insensitive assumption, not a
    per-directory measurement, so a case-sensitive NTFS directory on
    Windows is a known, accepted limit. Folding everywhere gave two
    distinct case-twin repositories on Linux the SAME identity, and an
    identity two repositories share authorises either one against
    records the other wrote."""
    resolved = str(Path(repo).resolve())
    if os.name == "nt":
        resolved = resolved.casefold()
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:32]


def _validate(payload, schema, what):
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as invalid:
        # the failing PATH, never the failing value: a value here could
        # be anything the run was carrying
        where = "/".join(str(part) for part in invalid.absolute_path) or "kok"
        raise CorruptState(f"{what} sema disi (alan: {where})") from None
    _assert_finite(payload, what)
    return payload


def _assert_finite(payload, what):
    """No NaN, no infinity, anywhere in the document.

    JSON Schema does not catch these: `{"type": "number", "minimum": 0}`
    accepts NaN, because every comparison against NaN is false and the
    `minimum` check asks whether the value is BELOW the bound. Python's
    json module then writes the literal `NaN` -- which is not JSON --
    and reads it straight back, so a budget could be spent-unknown and
    survive a full round trip looking valid."""
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise CorruptState(f"{what} sonlu olmayan sayi tasiyor")


def _assert_state_invariants(payload, what):
    """The relations JSON Schema cannot express.

    `contract.BUDGET_INVARIANT` is recorded in the frozen schemas as a
    RUNNER invariant precisely because it relates two fields. Recorded
    and unenforced is the weaker half of that: it is checked here, on
    the way in and on the way out, so a state that has already overrun
    cannot be written and cannot be resumed from."""
    # finiteness FIRST: `inf > max_usd` is true, so a non-finite budget
    # would otherwise be reported as an overrun, which names the wrong
    # defect and sends the operator looking at spend
    _assert_finite(payload, what)
    budget = payload.get("budget")
    if isinstance(budget, dict):
        spent, ceiling = budget.get("spent_usd"), budget.get("max_usd")
        if isinstance(spent, (int, float)) and isinstance(ceiling, (int, float)) \
                and spent > ceiling:
            raise IncompatibleState(
                f"{what}: butce degismezi ihlal edildi "
                f"({contract.BUDGET_INVARIANT})")


def write_json_atomically(path, payload, schema, what):
    """Temp file in the SAME directory, flush, fsync, then replace.

    A partial write must never become accepted state, and `os.replace`
    is the only step that makes a file visible under the real name. The
    temp file lives beside the target because replace across filesystems
    is not atomic."""
    _validate(payload, schema, what)
    target = Path(path)
    ensure_directory(target.parent)
    handle, temporary = tempfile.mkstemp(dir=str(target.parent),
                                         prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            # `allow_nan=False` is the belt to `_assert_finite`'s braces:
            # if a non-finite value ever reaches here it raises instead
            # of emitting a bare `NaN` token that is not valid JSON
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        fsync_directory(target.parent)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return target


def ensure_directory(directory) -> Path:
    """Create a directory chain and make each new link durable.

    `mkdir(parents=True)` was enough to make the path exist and not
    enough to make it survive: the FIRST record in a state directory
    creates `worktrees/` on the way in, and flushing only the directory
    the file lands in leaves that new directory's own entry -- in ITS
    parent -- unflushed. After a power cut the record could be gone
    together with the directory holding it, while the worktree it
    describes was already on disk.

    Each level is flushed in the order it was created: a parent's entry
    is only meaningful once the parent itself is on disk."""
    target = Path(directory)
    missing = []
    probe = target
    while not probe.exists():
        missing.append(probe)
        if probe.parent == probe:
            break
        probe = probe.parent
    target.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):             # outermost first
        fsync_directory(created.parent)
    return target


def fsync_directory(directory) -> bool:
    """Make the RENAME durable, not only the bytes. Returns whether the
    platform actually did it -- the return value is the honest part.

    `os.fsync` on the file guarantees its CONTENTS survive; it says
    nothing about the directory entry `os.replace` created. On POSIX,
    without this, a power loss can leave the entry missing while the
    thing the record describes already exists on disk -- which is the
    write-ahead ordering inverted, and exactly the failure the ordering
    was introduced to prevent."""
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(handle)
        return True
    except OSError:
        return False
    finally:
        os.close(handle)


def durability_of(directory) -> str:
    """What an atomic write to this directory actually guarantees.

    Two different claims, kept apart on purpose:

    `power-loss` -- file fsync, atomic replace, and a successful fsync
    of the parent directory.

    `process-crash` -- file fsync and atomic replace only. A killed
    process cannot leave a torn or half-visible record, and recovery by
    enumeration works. Survival of a power cut is NOT demonstrated.

    Windows lands on the second. A directory handle cannot be opened for
    this from Python, and the operations Microsoft documents for
    directory handles do not include a buffer flush, so there is no
    supported way to make the claim -- and swallowing the error while
    still calling the result durable would be the claim without the
    evidence."""
    return "power-loss" if fsync_directory(directory) else "process-crash"


def read_json_checked(path, schema, what):
    target = Path(path)
    if not target.exists():
        raise CorruptState(f"{what} yok")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"),
                             parse_constant=_reject_constant)
    except (ValueError, OSError):
        # truncated, half-written or unreadable -- all the same verdict
        raise CorruptState(f"{what} okunamadi ya da bozuk") from None
    return _validate(payload, schema, what)


def _reject_constant(literal):
    """`NaN`, `Infinity` and `-Infinity` are Python extensions, not JSON.
    A file carrying one was hand-edited or written by something that
    ignored the same rule."""
    raise ValueError(f"JSON disi sabit: {literal}")


def write_state(state_dir, payload):
    _assert_state_invariants(payload, "durum")
    return write_json_atomically(Path(state_dir) / STATE_FILENAME, payload,
                                 schemas.STATE_SCHEMA, "durum")


def read_state(state_dir):
    payload = read_json_checked(Path(state_dir) / STATE_FILENAME,
                                schemas.STATE_SCHEMA, "durum")
    _assert_state_invariants(payload, "durum")
    return payload


def write_binding(state_dir, payload):
    return write_json_atomically(Path(state_dir) / BINDING_FILENAME, payload,
                                 BINDING_SCHEMA, "baglama")


def read_binding(state_dir):
    return read_json_checked(Path(state_dir) / BINDING_FILENAME,
                             BINDING_SCHEMA, "baglama")


def assert_binding(state_dir, *, repo_id, baseline_sha, manifest_digest,
                   worktree_id=None):
    """Refuse state that belongs to a different run.

    Four separate ways a state directory can be the wrong one, and each
    is worth telling apart: a copied checkout, a moved baseline, an
    edited task, a worktree that is no longer the one this run built."""
    binding = read_binding(state_dir)
    checks = [("repo_id", repo_id), ("baseline_sha", baseline_sha),
              ("manifest_digest", manifest_digest)]
    if worktree_id is not None:
        checks.append(("worktree_id", worktree_id))
    for field, expected in checks:
        if binding.get(field) != expected:
            raise IncompatibleState(
                f"durum baska bir kosuya ait: {field} uyusmuyor")
    return binding


def assert_state_directory(state_dir, *, repo_id, baseline_sha,
                           manifest_digest, allow_resume=False):
    """What a state directory is allowed to be: empty, or a complete and
    matching pair. There is no third shape.

    A `state.json` with no `binding.json` used to pass unexamined --
    the caller only validated the binding IF the binding was there, so
    deleting one file of the two turned a foreign run's state into an
    acceptable one. The two documents describe the same run and are
    only meaningful together, which also means their `run_id` values
    have to agree; nothing compared them before.

    A previous run that never reached a terminal state is refused too.
    That refusal used to be the lock's job, via a stale-lock rule that
    guessed from a pid. Liveness and resumability are different
    questions and only this one has evidence on disk."""
    directory = Path(state_dir)
    has_state = (directory / STATE_FILENAME).exists()
    has_binding = (directory / BINDING_FILENAME).exists()
    if not has_state and not has_binding:
        return None                                   # a fresh run
    if has_state != has_binding:
        raise CorruptState(
            "yarim durum: durum ve baglama yalniz birlikte gecerli")
    binding = assert_binding(state_dir, repo_id=repo_id,
                             baseline_sha=baseline_sha,
                             manifest_digest=manifest_digest)
    current = read_state(state_dir)
    if current["run_id"] != binding["run_id"]:
        raise IncompatibleState("durum ve baglama farkli kosulari gosteriyor")
    if current["state"] not in contract.TERMINAL_STATES and not allow_resume:
        raise IncompatibleState(
            "onceki kosu bitmemis; devam etmek acikca istenmeli")
    return {"state": current, "binding": binding}


def assert_transition(current, target):
    """The frozen transition table decides, not the caller's intent."""
    allowed = contract.ALLOWED_TRANSITIONS.get(current)
    if allowed is None:
        raise IllegalTransition(f"bilinmeyen durum: {current!r}")
    if current in contract.TERMINAL_STATES:
        raise IllegalTransition(
            f"terminal durumdan cikis yok: {current} -> {target}")
    if target not in allowed:
        raise IllegalTransition(f"izinsiz gecis: {current} -> {target}")
    return target


def advance(state_dir, target, **changes):
    """Move the run one legal step and persist it atomically.

    A legal transition is not a licence to rewrite the run's identity:
    the fields that say WHICH run this is are refused here even when
    the move itself is allowed and the result would still validate."""
    current = read_state(state_dir)
    assert_transition(current["state"], target)
    for field_name in IMMUTABLE_STATE_FIELDS:
        if field_name in changes and changes[field_name] != current.get(
                field_name):
            raise IncompatibleState(
                f"degismez alan gecis sirasinda degistirilemez: {field_name}")
    updated = {**current, **changes, "state": target}
    # a terminal state carries exactly one reason; a running one carries
    # none -- the frozen schema enforces it, this keeps callers honest
    if target not in contract.TERMINAL_STATES:
        updated.pop("stop_reason", None)
    write_state(state_dir, updated)
    return updated
