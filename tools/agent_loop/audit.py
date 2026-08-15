"""The evaluator subprocess adapter. PACKAGE B3.

ONE primitive: run the READ-ONLY evaluator binary the caller supplied
against the candidate's implementer tree, and bring back a reply that has
been validated against the frozen schema the runner chose. It does not
decide what a verdict means, run acceptance, repair anything or advance
the run -- those live in `runner`, and keeping them out is the only
reason this file can be read in one sitting.

WHY IT IS NOT `execution.py`. The two adapters look alike and are not the
same mechanism. The implementer WRITES and answers on stdout under an
inline schema; the evaluator writes NOTHING and answers into a FILE,
because `codex exec` takes `--output-schema` and `--output-last-message`
as paths rather than inline JSON. That difference is the whole reason
this module exists: a file the child writes is a file whose SIZE nobody
bounds unless somebody watches it, and stdout ceilings do not cover it.

THE LAST MESSAGE IS BOUNDED WHILE THE CHILD RUNS. Measuring it after the
process exits is measuring a disk that has already been filled -- the
bytes were written, the space was spent, and "refuse afterwards" is a
verdict about a machine that already lost. A watcher polls the file and
trips the same event an overflowing pipe trips, which stops the process
tree instead of the parse.

WHERE THE FILE LIVES, AND WHY NOT IN THE OBVIOUS PLACES. Not in the state
directory: that is the run's own record, and a model-written file inside
it is model output inside the audit trail. Not inside the candidate:
the evaluator must not change the tree it is judging, and a file it
wrote there would be indistinguishable from one it edited. It lives in a
holder this module mints and removes on every path.

THE CWD IS DERIVED, NEVER RECEIVED. The caller passes identities and
`flat_workspace.assert_binding` turns them into the one directory the
evaluator may run in. There is no parameter through which a caller can
inject a path -- the defect a distant predecessor of this shape had, when
`is_dir()` was the whole check and the main checkout passed it.

WHAT MAY LEAVE. Closed event and reason codes from the frozen contract,
the validated reply, and numbers. Never an argv, never a path, never
captured stderr, never an OS message: `OSError` text names absolute
paths, and this package's refusals travel into reports.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from jsonschema import ValidationError

from tools.agent_loop import cli, contract, flat_workspace, schemas
from tools.agent_loop.process import (
    BoundedStream, ContainmentError, PromptWriter, READ_COMPLETED,
    REAP_SECONDS, join_within, launch_contained, stop)

EVALUATOR = "evaluator"
_POLL_SECONDS = 0.02
# How often the last-message watcher asks the filesystem. Fast enough
# that a runaway write is stopped in the same order of magnitude as a
# runaway pipe, slow enough not to be a spin loop.
_WATCH_SECONDS = 0.05

HOLDER_PREFIX = "agent-loop-audit-"
SCHEMA_NAME = "audit-schema.json"
LAST_MESSAGE_NAME = "audit-last-message.json"

# Read from the frozen task schema rather than invented here, so this
# adapter cannot accept a call the contract would have refused.
_OUTPUT_BYTES = schemas.TASK_SCHEMA["properties"]["max_output_bytes"]
MIN_OUTPUT_BYTES = _OUTPUT_BYTES["minimum"]
MAX_OUTPUT_BYTES = _OUTPUT_BYTES["maximum"]
_TIMEOUT = schemas.TASK_SCHEMA["properties"]["model_call_timeout_seconds"]
MIN_TIMEOUT_SECONDS = _TIMEOUT["minimum"]
MAX_TIMEOUT_SECONDS = _TIMEOUT["maximum"]
MAX_PROMPT_BYTES = MAX_OUTPUT_BYTES

_RUN_ID_PATTERN = re.compile(contract.IDENTIFIER_PATTERN)
_BASELINE_PATTERN = re.compile(
    flat_workspace.RECORD_SCHEMA["properties"]["baseline_sha"]["pattern"])
_OPAQUE_PATTERN = re.compile(contract.OPAQUE_ID_PATTERN)


class AuditError(RuntimeError):
    """A refused or failed evaluator call.

    Carries measurements and closed codes only. The message is a fixed
    sentence chosen HERE -- never captured output, never a path, never an
    OS message."""

    def __init__(self, message, *, event, reason, exit_code=None,
                 duration_ms=0, stdout_bytes=0, stderr_bytes=0,
                 cleanup_complete=True):
        super().__init__(message)
        self.role = EVALUATOR
        self.event = event
        self.reason = reason
        self.exit_code = exit_code
        self.duration_ms = duration_ms
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes
        self.cleanup_complete = cleanup_complete


class AuditInputRefused(AuditError):
    """An argument, bound or identity refused before a process existed."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


class WorkspaceNotBound(AuditInputRefused):
    """The identities do not name a recorded, READY flat workspace, so no
    process was started."""


class ContainmentFailed(AuditError):
    """The container could not be established, so nothing was run.

    An uncontained call is one nobody can stop, and it is billable."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED)


class ProcessFailed(AuditError):
    """The evaluator process ended non-zero."""

    def __init__(self, message, **measurements):
        super().__init__(message,
                         event=contract.EventCode.MODEL_CALL_FINISHED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED,
                         **measurements)


# ---------------------------------------------------------------------
# THE STDERR SUBFAMILY (B4-R8)
#
# WHY IT EXISTS. The first run that reached `auditing` ended with a
# non-zero evaluator exit, zero bytes of stdout and a couple of kilobytes
# of stderr -- and nothing anywhere could say WHICH refusal that was. The
# bytes existed, in a bounded buffer, at the moment the failure was
# raised; they were dropped one line later. Every one of these classes
# is still a `ProcessFailed`, carrying the same measurements: what is
# added is the name of the mechanism, never the text that proved it.
#
# WHAT MAY CREATE ONE. An EXACT line in `STDERR_FAILURE_MARKERS` below.
# There is no substring search: `"if"` matched inside `verify` once on
# this road already, and a classifier that guesses is worse than the
# generic code it replaces, because an operator acts on it.

class StartupRefused(ProcessFailed):
    """The CLI refused before it began the audit."""


class AuthFailed(ProcessFailed):
    """The CLI has no usable session. Distinct from the plan-only gate,
    which refuses BEFORE the call: this is the CLI's own answer."""


class RepositoryRefused(ProcessFailed):
    """The CLI refused the working directory it was given.

    THE ONE CLASS WITH A PROVEN MARKER TODAY. The flat workspace is a
    copy rather than a clone and holds no `.git`, which is exactly the
    condition this refusal names."""


class InvalidArgv(ProcessFailed):
    """The CLI rejected the command line itself."""


class SchemaRefused(ProcessFailed):
    """The CLI refused the `--output-schema` file.

    Distinct from `SchemaViolation`, which is THIS package refusing a
    reply: here no reply was ever produced."""


class ProviderFailed(ProcessFailed):
    """The CLI reported a provider-side failure."""


# EXACT stderr LINES, each proven locally, mapped to the class it names.
#
# THE MATCH IS WHOLE-LINE AND EXACT. The real refusal arrives with a
# preamble line above it (`Reading prompt from stdin...`), so the whole
# buffer never equals a marker -- but one of its lines does. Nothing is
# matched by substring, prefix or case-folding.
#
# WHAT IS DELIBERATELY EMPTY. The other five classes above have no
# marker yet. Their codes exist so the road is spelled out, and a code
# without a marker is unreachable ON PURPOSE: inventing a sentence this
# machine has never seen would produce a confident wrong answer, which
# is the failure mode this whole subfamily was written to end.
STDERR_FAILURE_MARKERS = {
    # codex-cli, measured on 0.147.0-alpha.6.6 and reproduced by this
    # package's own stub: exit 1 before any work, this exact sentence on
    # stderr, nothing on stdout.
    "Not inside a trusted directory and --skip-git-repo-check was not "
    "specified.": "RepositoryRefused",
}


class OutputLimitExceeded(AuditError):
    """A stream or the last-message file crossed its ceiling and the tree
    was stopped.

    `stream` says WHICH, because "stdout" and "last_message" are two
    different defects: one is a chatty model, the other is a model
    filling a disk through a file nobody was watching."""

    def __init__(self, message, *, stream, **measurements):
        super().__init__(message, event=contract.EventCode.OUTPUT_TRUNCATED,
                         reason=contract.StopReason.SCHEMA_VIOLATION,
                         **measurements)
        self.stream = stream


class Timeout(AuditError):
    """The wall clock ran out; the tree was stopped."""

    def __init__(self, message, *, timeout_seconds, **measurements):
        super().__init__(message, event=contract.EventCode.INTERRUPTED,
                         reason=contract.StopReason.TIMEOUT, **measurements)
        self.timeout_seconds = timeout_seconds


class TransportFailed(AuditError):
    """A reader ended in a state that is not a complete read.

    A pipe that broke mid-read used to be indistinguishable from a short
    answer, so a truncated buffer arrived beside a clean exit code and
    was treated as the whole output. EXIT 0 IS NOT THE QUESTION."""

    def __init__(self, message, **measurements):
        super().__init__(message, event=contract.EventCode.INTERRUPTED,
                         reason=contract.StopReason.INTERRUPTED,
                         **measurements)


class SchemaViolation(AuditError):
    """The reply is not something the frozen schema accepts. Never
    repaired: a guessed field is a field nobody agreed to."""

    def __init__(self, message, **measurements):
        super().__init__(message, event=contract.EventCode.SCHEMA_VIOLATION,
                         reason=contract.StopReason.SCHEMA_VIOLATION,
                         **measurements)


class ProcessTreeSurvived(AuditError):
    """The call ended but its container could not be emptied.

    OUTRANKS THE VERDICT. Something the evaluator started is still
    running against the candidate, so whatever it said about that
    candidate describes a tree that is still moving. This is raised in
    preference to the audit answer, not logged beside it."""

    def __init__(self, message, **measurements):
        super().__init__(message,
                         event=contract.EventCode.MODEL_CALL_FINISHED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED,
                         **measurements)


class HolderNotReleased(AuditError):
    """The audit holder could not be removed.

    Also outranks the verdict: a holder left behind carries the model's
    own output, and a run that cannot prove it cleaned up cannot claim
    the boundary this module exists to keep."""

    def __init__(self, message, **measurements):
        super().__init__(message,
                         event=contract.EventCode.MODEL_CALL_FINISHED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED,
                         **measurements)


@dataclass(frozen=True, slots=True)
class AuditRun:
    """A validated evaluator reply plus what the call cost.

    `schema_sha256` names the EXACT canonical bytes of the AUTHORITY
    that validated this reply. `transport_sha256` names the bytes that
    were in the file on the argv. They were one field until B4-R11,
    which is precisely the conflation that sent the acceptance schema to
    the provider -- so they are two now, and a report that wants to know
    which document did which job can ask."""

    reply: dict
    audit_kind: str
    exit_code: int
    duration_ms: int
    stdout_bytes: int
    stderr_bytes: int
    last_message_bytes: int
    event: str
    schema_sha256: str
    transport_sha256: str


def _now():
    """The clock, behind one name -- a test seam, and deliberately the
    only one: the contract's minimum per-call timeout is 30 seconds, so
    proving timeout behaviour would otherwise mean half-minute tests or a
    production range loosened to suit them."""
    return time.monotonic()


def _canonical_limits(timeout_seconds, max_output_bytes):
    """Bounds are checked BEFORE a process exists, and RETURNED.

    Exact `int` excludes `bool` by type and excludes the subclass whose
    `__le__` and `__ge__` agree with every range while carrying a value
    far outside it -- a bound whose comparisons the value itself defines
    is not a bound."""
    if type(timeout_seconds) is not int:
        raise AuditInputRefused("sure siniri tam sayi degil")
    if not MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise AuditInputRefused("sure siniri sozlesme araligi disinda")
    if type(max_output_bytes) is not int:
        raise AuditInputRefused("cikti siniri tam sayi degil")
    if not MIN_OUTPUT_BYTES <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise AuditInputRefused("cikti siniri sozlesme araligi disinda")
    return timeout_seconds, max_output_bytes


def _exact_identity(value, pattern, what):
    """An exact `str` its frozen pattern accepts. A `str` subclass whose
    `__eq__` returns True for everything satisfies every comparison a
    binding makes, so the binding would agree with a record it had never
    been shown."""
    if type(value) is not str or not pattern.fullmatch(value):
        raise AuditInputRefused(f"{what} sozlesme desenine uymuyor")
    return value


def _exact_path(value, what):
    """One conversion, one concrete `Path`, and nothing afterwards asks
    the original object anything."""
    try:
        text = cli.exact_text(value, what=what)
    except cli.UnsafeInvocation:
        raise AuditInputRefused(f"{what} bir yol degil") from None
    if not text:
        raise AuditInputRefused(f"{what} bos")
    return Path(text)


def _prompt_bytes(prompt):
    """Encoded and measured before launch. EXACTLY `str`: a subclass
    checked as one thing returned entirely different bytes from
    `encode()`, so the instruction validated and the instruction received
    were two different questions."""
    if type(prompt) is not str:
        raise AuditInputRefused("istem bir metin degil")
    try:
        payload = prompt.encode("utf-8")
    except UnicodeEncodeError:
        raise AuditInputRefused("istem UTF-8'e kodlanamiyor") from None
    if not payload:
        raise AuditInputRefused("istem bos")
    if len(payload) > MAX_PROMPT_BYTES:
        raise AuditInputRefused("istem sozlesme tavanini asiyor")
    return payload


def _usable_binary(binary):
    """An existing regular file, as ONE absolute canonical path.

    `Popen` searches for a bare name, which is how a call that meant to
    reach a stub reaches the real, billable CLI. There is ONE conversion
    and its result is what gets launched: an object whose `__fspath__`
    named binary A and whose `__str__` named binary B used to be verified
    as A and launched as B."""
    try:
        text = cli.exact_text(binary, what="ikili dosya")
    except cli.UnsafeInvocation:
        raise AuditInputRefused("ikili dosya bir yol degil") from None
    if not text:
        raise AuditInputRefused("ikili dosya verilmedi")
    path = Path(text)
    if path.parent == Path(""):
        raise AuditInputRefused("ikili dosya bir yol degil ya da mevcut degil")
    resolved = path.resolve()
    if not resolved.is_file():
        raise AuditInputRefused("ikili dosya bir yol degil ya da mevcut degil")
    return resolved


def _opaque_ids(values, what):
    """The ids the RUNNER minted, as an exact tuple.

    Model-chosen slugs are free text on a short leash. These are checked
    against the frozen opaque pattern here so a caller cannot hand the
    schema builder something that would widen the enum it becomes."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise AuditInputRefused(f"{what} bir kimlik listesi degil")
    items = tuple(values)
    for item in items:
        _exact_identity(item, _OPAQUE_PATTERN, what)
    if len(set(items)) != len(items):
        raise AuditInputRefused(f"{what} yinelemeli")
    return items


def _schema_for(audit_kind, *, issued_run_id, issued_finding_ids,
                issued_mechanism_ids):
    """The TWO schemas this call needs, as `(authoritative, transport)`.

    THEY ARE NOT INTERCHANGEABLE, and B4-R11 exists because one object
    used to play both parts. The authoritative schema decides whether a
    reply is acceptable and never leaves this process. The transport
    schema is written to the file on the argv and constrains generation
    only -- Codex refuses `allOf` at the root, which is exactly where
    the conditional rules live, so every evaluator call was rejected by
    the provider before the model saw anything.

    The transport copy is DERIVED FROM the authoritative one, per call,
    so the two can never describe different documents: a locked audit's
    transport carries the same issued-id bindings its authority does.

    A LOCKED audit gets the per-call bound schema: the reply's `run_id`
    pinned with `const` and each id field an `enum` of exactly what was
    issued, so an id the runner never minted is a schema violation rather
    than a new case to act on. The static pattern alone proves shape only
    -- any 32 hex characters satisfy it.

    THE LOCKED REPLY'S `run_id` IS NOT THE LOOP'S `run_id`, and conflating
    them makes a locked audit impossible to run. The loop's is a readable
    identifier (`kurgu-run-1`) and it is what binds the workspace; the
    locked envelope is TEXTLESS, so its `run_id` is opaque -- a readable
    slug there is free text on a short leash, and `gizli-belge-adi` is a
    valid one. They are two identifiers with two grammars, and the runner
    mints the second."""
    if audit_kind == contract.AuditKind.CODE:
        if issued_run_id is not None or issued_finding_ids \
                or issued_mechanism_ids:
            # a value that does nothing is a value somebody will believe
            # in: a caller handing out ids here would think they bound
            # something, and nothing would be bound
            raise AuditInputRefused(
                "kod denetimi verilmis kimlik almaz")
        # STATIC on this road, so the transport copy is the one derived
        # and hash-pinned at import rather than rebuilt here
        return (schemas.CODE_AUDIT_RESULT_SCHEMA,
                schemas.CODE_AUDIT_TRANSPORT_SCHEMA)
    if audit_kind == contract.AuditKind.LOCKED:
        if issued_run_id is None or not issued_finding_ids \
                or not issued_mechanism_ids:
            # an empty enum accepts nothing and would refuse every reply,
            # including an honest approval -- a gate that cannot pass is
            # a gate nobody can tell from a broken one
            raise AuditInputRefused(
                "kilitli denetim icin kimlik verilmedi")
        # BOUND FIRST, then reduced: the transport copy is derived from
        # this call's bound authority, so the issued ids travel with it.
        # A static locked transport would describe a document accepting
        # ids the runner never minted.
        authoritative = schemas.locked_audit_schema(
            run_id=issued_run_id, issued_finding_ids=issued_finding_ids,
            issued_mechanism_ids=issued_mechanism_ids)
        return authoritative, schemas.codex_transport_schema(authoritative)
    raise AuditInputRefused("denetim turu sozlesmede yok")


@dataclass(frozen=True, slots=True)
class _CanonicalAuditCall:
    """Every input, validated once and converted once.

    THE POINT OF THIS CLASS is that after it exists the caller's objects
    are irrelevant. A validator that only inspects a value and leaves the
    original alive is not a boundary: an object is free to answer
    differently every time it is asked."""

    binary: Path
    repo: Path
    state_dir: Path
    run_id: str
    workspace_id: str
    baseline_sha: str
    audit_kind: str
    prompt_bytes: bytes
    timeout_seconds: int
    max_output_bytes: int
    model: object
    # THE ACCEPTANCE AUTHORITY. Never written to a file, never sent.
    schema: dict
    # The generation constraint, and the only schema that reaches an
    # argv. Weaker on purpose; it decides nothing.
    transport: dict


def _canonical_call(binary, *, repo, state_dir, run_id, workspace_id,
                    baseline_sha, audit_kind, prompt, timeout_seconds,
                    max_output_bytes, issued_run_id, issued_finding_ids,
                    issued_mechanism_ids, model):
    """Validate and convert, cheapest refusals first, so a caller who got
    a bound wrong hears about it without a filesystem question."""
    canonical_timeout, canonical_output = _canonical_limits(
        timeout_seconds, max_output_bytes)
    canonical_run = _exact_identity(run_id, _RUN_ID_PATTERN, "kosu kimligi")
    findings = _opaque_ids(issued_finding_ids, "bulgu kimlikleri")
    mechanisms = _opaque_ids(issued_mechanism_ids, "mekanizma kimlikleri")
    if issued_run_id is not None:
        issued_run_id = _exact_identity(issued_run_id, _OPAQUE_PATTERN,
                                        "verilen kosu kimligi")
    if type(audit_kind) is not str or audit_kind not in contract.ALL_AUDIT_KINDS:
        raise AuditInputRefused("denetim turu sozlesmede yok")
    schema, transport = _schema_for(audit_kind, issued_run_id=issued_run_id,
                                    issued_finding_ids=findings,
                                    issued_mechanism_ids=mechanisms)
    try:
        return _CanonicalAuditCall(
            binary=_usable_binary(binary),
            prompt_bytes=_prompt_bytes(prompt),
            timeout_seconds=canonical_timeout,
            max_output_bytes=canonical_output,
            model=cli.exact_model(model),
            repo=_exact_path(repo, "depo yolu"),
            state_dir=_exact_path(state_dir, "durum dizini"),
            run_id=canonical_run,
            workspace_id=_exact_identity(workspace_id,
                                         flat_workspace.WORKSPACE_ID,
                                         "calisma alani kimligi"),
            baseline_sha=_exact_identity(baseline_sha, _BASELINE_PATTERN,
                                         "taban surum"),
            audit_kind=audit_kind, schema=schema, transport=transport)
    except cli.UnsafeInvocation as refused:
        raise AuditInputRefused(str(refused)) from None


class _FileCeiling(threading.Thread):
    """Watches the last-message file and trips the shared event.

    THE WHOLE REASON THIS THREAD EXISTS. `codex exec` answers into a
    file, so the pipe ceilings that bound stdout say nothing about it. A
    child can write gigabytes there while stdout stays empty and the exit
    code stays 0, and a size checked after the process ended is a size
    checked after the disk was already filled.

    It never READS the file -- only `stat` -- so nothing the child wrote
    enters this process before the size has been judged."""

    def __init__(self, path, limit, tripped, finished):
        super().__init__(daemon=True)
        self.path = path
        self.exceeded = False
        self.peak = 0
        self._limit = limit
        self._tripped = tripped
        self._finished = finished

    def run(self):
        while not self._finished.is_set():
            try:
                size = os.path.getsize(self.path)
            except OSError:
                # not created yet, or already gone: neither is an
                # overflow, and neither is this thread's to report
                size = 0
            if size > self.peak:
                self.peak = size
            if size > self._limit:
                self.exceeded = True
                self._tripped.set()
                return
            self._finished.wait(_WATCH_SECONDS)


def _read_bounded(path, limit):
    """The last message, or a closed refusal. Never more than `limit`.

    Read through a handle with an explicit ceiling rather than
    `read_bytes()`: the watcher above bounds what the child WROTE, and
    this bounds what this process takes in even if the file grew between
    the last poll and the process exiting."""
    try:
        with open(path, "rb") as handle:
            payload = handle.read(limit + 1)
    except OSError:
        # the name of the file that refused is not text this module may
        # repeat, and the failure leaves the handler before it flies
        return None
    return payload


def _stderr_failure(stream, measurements, exit_code):
    """A more precise `ProcessFailed`, or `None` to keep the generic one.

    EVERY CONDITION MUST HOLD, and any one of them failing is silence
    rather than a guess: the stream must have been read to completion
    (an overflowed or unreadable buffer is a partial document, and the
    existing overflow and read-failure authorities outrank this one),
    the bytes must decode as strict UTF-8, and one whole LINE must equal
    a marker this package has evidence for.

    NORMALISATION IS EXACTLY THREE THINGS: strict UTF-8, CRLF to LF, and
    stripping the ends. No case-folding, no whitespace collapsing, no
    substring search -- each of those turns "looks similar" into
    "classified", and the wrong closed code is acted on as confidently
    as the right one.

    NOTHING IS READ OUT OF THE OUTPUT. The matched line is a lookup KEY
    and is dropped; the unmatched lines, the vendor's wording, any path
    it printed and the byte count's contents never enter the exception,
    whose sentence is a fixed one chosen here."""
    if stream.outcome != READ_COMPLETED or stream.overflowed:
        return None
    try:
        text = bytes(stream.buffer).decode("utf-8")
    except UnicodeDecodeError:
        return None
    for line in text.replace("\r\n", "\n").strip().split("\n"):
        name = STDERR_FAILURE_MARKERS.get(line)
        if name is not None:
            return globals()[name](
                "denetci sureci bildirilen bir nedenle durdu",
                exit_code=exit_code, **measurements)
    return None


def _release(holder):
    """Remove the audit holder, and report whether it is really gone.

    A holder that could not be removed carries model output on the disk
    after a run that claims to bound it, so the answer is returned rather
    than swallowed."""
    if holder is None:
        return True
    shutil.rmtree(holder, ignore_errors=True)
    return not os.path.exists(holder)


def run_evaluator(binary, *, repo, state_dir, run_id, workspace_id,
                  baseline_sha, audit_kind, prompt, timeout_seconds,
                  max_output_bytes, issued_run_id=None,
                  issued_finding_ids=(), issued_mechanism_ids=(),
                  model=None) -> AuditRun:
    """Run the evaluator once, READ-ONLY, and return its validated reply.

    The working directory is derived from the identities by
    `flat_workspace.assert_binding` -- the candidate's implementer tree,
    the same one the model under audit worked in. There is no way to pass
    a path in. Raises a typed `AuditError` for every refusal. Nothing
    here retries, repairs or decides what the verdict means.

    THE CLEANUP ENVELOPE IS THE CONTRACT. Once `launch_contained`
    returns, every path runs stop, drain, join and reap, and every one of
    those is attempted even when an earlier one failed. A cleanup that
    cannot be proven outranks whatever the evaluator said."""
    call = _canonical_call(
        binary, repo=repo, state_dir=state_dir, run_id=run_id,
        workspace_id=workspace_id, baseline_sha=baseline_sha,
        audit_kind=audit_kind, prompt=prompt,
        timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes,
        issued_run_id=issued_run_id, issued_finding_ids=issued_finding_ids,
        issued_mechanism_ids=issued_mechanism_ids, model=model)
    # THE RAW ARGUMENTS ARE GONE, and that is the enforcement rather than
    # a comment asking future readers to be careful: reaching for one now
    # is a NameError the positive-control tests hit on the first run.
    del binary, repo, state_dir, run_id, workspace_id, baseline_sha
    del audit_kind, prompt, timeout_seconds, max_output_bytes
    del issued_run_id, issued_finding_ids, issued_mechanism_ids, model

    try:
        cwd = flat_workspace.assert_binding(
            call.repo, state_dir=call.state_dir, run_id=call.run_id,
            workspace_id=call.workspace_id,
            baseline_sha=call.baseline_sha).implementer_root
    except flat_workspace.FlatWorkspaceError as refused:
        raise WorkspaceNotBound(str(refused)) from None

    # TWO BINDINGS, pinned independently (B4-R11). `binding` is the
    # acceptance authority and is the only thing a reply is judged by;
    # `transport_binding` is the document the provider is asked to
    # generate against, and its bytes are the only ones that reach disk.
    binding = schemas.SchemaBinding(call.schema)
    transport_binding = schemas.SchemaBinding(call.transport)
    holder = Path(tempfile.mkdtemp(prefix=HOLDER_PREFIX))
    try:
        return _measure(call, cwd=cwd, holder=holder, binding=binding,
                        transport_binding=transport_binding)
    finally:
        # the holder's own removal is checked inside `_measure` on the
        # success path so it can outrank a verdict; this is the net for
        # every path that left before that point
        shutil.rmtree(holder, ignore_errors=True)


def _measure(call, *, cwd, holder, binding, transport_binding):
    """Everything from the first file to the last reap, with exactly one
    way out of each phase.

    THE FILE ON THE ARGV CARRIES THE TRANSPORT BYTES. Writing the
    authority here is what made every evaluator call fail with a
    provider 400, and it is also what would send the acceptance rules
    to a vendor. The authority stays in `binding`, which is what
    `_parse_reply` judges the answer with."""
    schema_file = holder / SCHEMA_NAME
    last_message = holder / LAST_MESSAGE_NAME
    schema_file.write_bytes(transport_binding.canonical_bytes)

    argv = cli.build_evaluator_argv(
        call.binary, repo=cwd, schema_path=schema_file,
        last_message_path=last_message, model=call.model)

    started = _now()
    try:
        process, container = launch_contained(argv, cwd=cwd)
    except ContainmentError as refused:
        raise ContainmentFailed(str(refused)) from None

    drained = False
    grace = None
    started_readers = []
    finished = threading.Event()
    try:
        tripped = threading.Event()
        streams = [BoundedStream("stdout", process.stdout,
                                 call.max_output_bytes, tripped),
                   BoundedStream("stderr", process.stderr,
                                 call.max_output_bytes, tripped)]
        # ONLY the readers that really started are joined later. A thread
        # whose `start()` raised was never alive, and joining it raises
        # again -- inside the cleanup, where it would replace the failure
        # that got us there.
        for stream in streams:
            stream.start()
            started_readers.append(stream)
        watcher = _FileCeiling(last_message, call.max_output_bytes, tripped,
                               finished)
        watcher.start()
        started_readers.append(watcher)
        writer = PromptWriter(process.stdin, call.prompt_bytes)
        writer.start()
        started_readers.append(writer)

        deadline = started + call.timeout_seconds
        timed_out = False
        while True:
            if tripped.is_set():
                break
            if process.poll() is not None:
                break
            if _now() >= deadline:
                timed_out = True
                break
            time.sleep(_POLL_SECONDS)

        finished.set()
        # ONE grace budget covers the kill and every join, so a call that
        # is already failing cannot spend three separate timeouts.
        grace = time.monotonic() + REAP_SECONDS
        cleanup_complete = True
        if tripped.is_set() or timed_out:
            cleanup_complete = stop(process, grace)
        # DRAIN BEFORE JOINING. A grandchild holding the pipes open keeps
        # the readers alive, so joining first waits out the whole grace
        # period and only then kills anything.
        drained = container.drain(grace)
        if not drained:
            cleanup_complete = False
        # `started_readers` ALREADY holds the writer. Spelling it again
        # here joined one thread twice and, worse, made the list look
        # like it was assembled at the call site -- where a future reader
        # would be tempted to add a thread that never started.
        if not join_within(started_readers, grace):
            cleanup_complete = False

        duration_ms = int((_now() - started) * 1000)
        measurements = {"duration_ms": duration_ms,
                        "stdout_bytes": streams[0].total,
                        "stderr_bytes": streams[1].total,
                        "cleanup_complete": cleanup_complete}

        if timed_out:
            raise Timeout("denetci cagrisi sure sinirini asti",
                          timeout_seconds=call.timeout_seconds,
                          exit_code=process.returncode, **measurements)
        if watcher.exceeded:
            raise OutputLimitExceeded(
                "son mesaj dosyasi sinirini asti; surec agaci durduruldu",
                stream="last_message", exit_code=process.returncode,
                **measurements)
        overflowed = [stream for stream in streams if stream.overflowed]
        if overflowed:
            raise OutputLimitExceeded(
                "cikti sinirini asti; surec agaci durduruldu",
                stream=overflowed[0].label, exit_code=process.returncode,
                **measurements)

        exit_code = process.wait()
        # A READER THAT DID NOT COMPLETE IS NOT A SHORT ANSWER, and exit
        # 0 does not answer this question: a pipe that broke mid-read
        # produced a truncated buffer beside a clean exit code, and the
        # truncation was then treated as the whole output.
        broken = [stream for stream in streams
                  if stream.outcome != READ_COMPLETED]
        if broken:
            raise TransportFailed("denetci ciktisi eksiksiz okunamadi",
                                  exit_code=exit_code, **measurements)
        if exit_code != 0:
            # THE STDERR IS STILL IN HAND (B4-R8). The bounded buffer
            # holds it right now -- the readers were joined above and
            # the buffer is dropped moments later -- so this is the only
            # moment at which the refusal can be named without keeping a
            # byte of it. `_stderr_failure` returns a more precise class
            # or `None`; it never turns a failure into a success, and
            # the broken-reader and overflow checks above already ran,
            # so a partial buffer is never classified.
            raise _stderr_failure(
                streams[1], measurements, exit_code) or ProcessFailed(
                    "denetci sureci sifirdan farkli koda dondu",
                    exit_code=exit_code, **measurements)
        # A child can answer WITHOUT reading: it writes a valid reply,
        # exits, and the asynchronous write dies on a closed pipe. The
        # reply then judges a candidate against instructions the
        # evaluator was never given.
        if not writer.completed:
            raise TransportFailed("istem denetciye tamamen teslim edilemedi",
                                  exit_code=exit_code, **measurements)
        if not drained:
            raise ProcessTreeSurvived(
                "denetci sureci kapsayicisi bosaltilamadi",
                exit_code=exit_code, **measurements)
        measurements.pop("cleanup_complete")

        raw = _read_bounded(last_message, call.max_output_bytes)
        if raw is None:
            raise TransportFailed("son mesaj dosyasi okunamadi",
                                  exit_code=exit_code, **measurements)
        if len(raw) > call.max_output_bytes:
            raise OutputLimitExceeded(
                "son mesaj dosyasi sinirini asti", stream="last_message",
                exit_code=exit_code, **measurements)
        reply = _parse_reply(raw, dict(measurements, exit_code=exit_code),
                             binding, call.audit_kind)
        # THE HOLDER'S RELEASE OUTRANKS THE VERDICT. A holder left on
        # disk carries the model's own output past the boundary this
        # module exists to keep, so an approval whose cleanup cannot be
        # proven is not returned.
        if not _release(holder):
            raise HolderNotReleased("denetim tutucusu temizlenemedi",
                                    exit_code=exit_code, **measurements)
        return AuditRun(reply=reply, audit_kind=call.audit_kind,
                        exit_code=exit_code,
                        last_message_bytes=len(raw),
                        event=contract.EventCode.MODEL_CALL_FINISHED,
                        schema_sha256=binding.sha256,
                        transport_sha256=transport_binding.sha256,
                        **measurements)
    finally:
        # EVERY exit, including an exception raised between the launch
        # and the drain above. An injected thread-start failure used to
        # leave the model process running with its container built and
        # never emptied.
        finished.set()
        if not drained:
            container.drain(grace if grace is not None
                            else time.monotonic() + REAP_SECONDS)


def _parse_reply(raw, measurements, binding, audit_kind):
    """Decode, parse and validate -- refusing at every step.

    THE DISCRIMINATOR IS CHECKED FIRST. A reply that claims the other
    audit kind must not be judged by this call's schema: the two kinds
    are a TYPE boundary, and a locked reply validated against the code
    schema would be a locked finding carrying free text.

    Strict UTF-8: replacement characters would turn undecodable bytes
    into a string that might then parse into something plausible."""
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        raise SchemaViolation("cikti gecerli UTF-8 degil",
                              **measurements) from None
    try:
        payload = json.loads(text)
    except ValueError:
        raise SchemaViolation("cikti JSON degil", **measurements) from None
    if not isinstance(payload, dict):
        raise SchemaViolation("cikti bir JSON nesnesi degil", **measurements)
    if payload.get("audit_kind") != audit_kind:
        raise SchemaViolation("yanit baska bir denetim turunu adliyor",
                              **measurements)
    try:
        binding.validate(payload)
    except ValidationError as invalid:
        # the failing FIELD PATH, never the failing value: the value is
        # model output and this text travels into reports
        where = "/".join(str(part) for part in invalid.absolute_path) or "kok"
        raise SchemaViolation(f"yanit sema disi (alan: {where})",
                              **measurements) from None
    return payload
