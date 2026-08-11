"""The implementer subprocess adapter. PACKAGE B2A.

ONE primitive: launch the Claude implementer binary the caller supplied,
inside the disposable worktree B1 RECORDED, and bring back a reply that
has been validated against the frozen schema. It does not create
worktrees, read diffs, run acceptance commands, call the evaluator,
apply patches or advance the run. Those live in B2B and B3, and keeping
them out is the only reason this file can be read in one sitting.

THE WORKING DIRECTORY IS DERIVED, NEVER RECEIVED. The first version
took a `worktree` path and checked `is_dir()`, which made the main
checkout -- or any directory at all -- an acceptable place to run the
model, with B1's ownership record sitting unread beside it. Now the
caller passes identities (repository, state directory, run, worktree
id, baseline) and `worktree.assert_execution_binding` turns them into
the one directory they may name, refusing everything else before a
process exists. There is no parameter through which a caller can
inject a path.

NOTHING IS DISCOVERED. The binary, the identities, the budget, the
timeout and the output ceiling are all mandatory arguments. A caller
who forgets one gets a `TypeError`; a caller who passes a bare name
like `claude` gets a refusal rather than whatever is on PATH. The argv
itself is built only by `cli.build_implementer_argv`, so the flag rules
stay in the module that already proves them. The SCHEMA is not an
argument at all: it is the frozen canonical binding, inline on the
argv, hash-checked before launch, and the validator is parsed from the
same bytes.

THE PROMPT GOES OVER STDIN. Never argv, never the environment: a prompt
on a command line is a prompt in every process listing, and an
environment variable is inherited by every child.

BOUNDS ARE ENFORCED WHILE READING. Reading everything and measuring
afterwards means the memory was already spent, and a truncated reply is
exactly the kind of thing a lenient parser will accept as valid. Each
stream is read in chunks against its own ceiling, and crossing it stops
the process tree instead of the parse.

WHAT MAY LEAVE. Closed event and reason codes from the frozen contract,
plus numbers: exit code, duration, byte counts. Never model output.
Anything else would put the text this package exists to bound into
whatever the caller logs.

WHAT IS NOT CLAIMED. `IMPLEMENTER_RESULT_SCHEMA` has no cost field, so
this adapter reports no spend. A zero would be a number a caller could
subtract from a budget, and it would be an invention.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import cli, contract, schemas, worktree
from tools.agent_loop.process import (
    BoundedStream, Container, ContainmentError, PromptWriter,
    REAP_SECONDS, READ_CHUNK_BYTES, join_within,
    launch_contained, stop, terminate_tree)

_POLL_SECONDS = 0.02
IMPLEMENTER = "implementer"

# Taken from the frozen task schema rather than invented here, so the
# adapter cannot accept a call the contract would have refused.
_OUTPUT_BYTES = schemas.TASK_SCHEMA["properties"]["max_output_bytes"]
MIN_OUTPUT_BYTES = _OUTPUT_BYTES["minimum"]
MAX_OUTPUT_BYTES = _OUTPUT_BYTES["maximum"]
# The PER-CALL field, not the run's total wall clock. Binding to
# `max_wall_clock_minutes` accepted 0.5 and 10000 alike: one is a
# timeout too short to be a timeout, the other longer than the contract
# allows a single call to take. They are different budgets and only one
# of them describes this function.
_TIMEOUT = schemas.TASK_SCHEMA["properties"]["model_call_timeout_seconds"]
MIN_TIMEOUT_SECONDS = _TIMEOUT["minimum"]
MAX_TIMEOUT_SECONDS = _TIMEOUT["maximum"]
# A prompt is an instruction, not a document. The ceiling is the
# contract's own output ceiling, so neither direction of the call can
# carry more than the other.
MAX_PROMPT_BYTES = MAX_OUTPUT_BYTES
# The task schema caps a whole run; a single call may not authorise
# more than that. Only "greater than zero" was enforced, so one call
# could be funded for any amount at all. Re-exported from `cli`, which
# owns the budget rule for both roads into it -- not recomputed here.
MAX_BUDGET_USD = cli.MAX_BUDGET_USD

# Identity grammars, read from the modules that already own them.
_RUN_ID_PATTERN = re.compile(contract.IDENTIFIER_PATTERN)
_BASELINE_PATTERN = re.compile(
    worktree.RECORD_SCHEMA["properties"]["baseline_sha"]["pattern"])


class AdapterError(RuntimeError):
    """A refused or failed model call.

    Carries measurements and closed codes only. The message is a fixed
    sentence chosen here -- never captured output, never a path."""

    def __init__(self, message, *, event, reason, exit_code=None,
                 duration_ms=0, stdout_bytes=0, stderr_bytes=0,
                 cleanup_complete=True):
        super().__init__(message)
        self.role = IMPLEMENTER
        self.event = event
        self.reason = reason
        self.exit_code = exit_code
        self.duration_ms = duration_ms
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes
        # A cleanup that did not finish must not overwrite the failure
        # that triggered it -- and must not be invisible either. A
        # swallowed kill error is a surviving process nobody hears
        # about; this is the flag that makes it reportable.
        self.cleanup_complete = cleanup_complete


class BinaryNotUsable(AdapterError):
    """The path given is not a file this adapter may execute."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


class BudgetRefused(AdapterError):
    """Refused BEFORE a process existed."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.BUDGET_CHECK,
                         reason=contract.StopReason.BUDGET_EXHAUSTED)


class ProcessFailed(AdapterError):
    """The model process ended non-zero.

    Reporting `preflight_failed` here once named a gate that had
    already succeeded, and `None` left a terminal result without the
    closed reason Phase A requires. The contract owner added
    `model_process_failed`; this is that code."""

    def __init__(self, message, **measurements):
        super().__init__(message,
                         event=contract.EventCode.MODEL_CALL_FINISHED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED,
                         **measurements)


class OutputLimitExceeded(AdapterError):
    """A stream crossed its ceiling and the tree was stopped."""

    def __init__(self, message, *, stream, **measurements):
        super().__init__(message, event=contract.EventCode.OUTPUT_TRUNCATED,
                         reason=contract.StopReason.SCHEMA_VIOLATION,
                         **measurements)
        self.stream = stream


class Timeout(AdapterError):
    """The wall clock ran out; the tree was stopped."""

    def __init__(self, message, *, timeout_seconds, **measurements):
        super().__init__(message, event=contract.EventCode.INTERRUPTED,
                         reason=contract.StopReason.TIMEOUT, **measurements)
        self.timeout_seconds = timeout_seconds


class SchemaViolation(AdapterError):
    """The reply is not something the frozen schema accepts. Never
    repaired: a guessed field is a field nobody agreed to."""

    def __init__(self, message, **measurements):
        super().__init__(message, event=contract.EventCode.SCHEMA_VIOLATION,
                         reason=contract.StopReason.SCHEMA_VIOLATION,
                         **measurements)


@dataclass
class ImplementerRun:
    """A validated reply plus what the call cost in time and bytes.

    `schema_sha256` names the EXACT canonical schema bytes that were on
    the argv and that validated this reply -- the one 64-hex code a
    report may carry about the schema, and the only thing it needs."""

    reply: dict
    exit_code: int
    duration_ms: int
    stdout_bytes: int
    stderr_bytes: int
    event: str
    schema_sha256: str


def _canonical_budget(budget_usd):
    """Refused here, before anything is launched -- and RETURNED, so
    the number that was bounded is the number that gets spelled.

    Two defects met in this function. A `float` SUBCLASS passed every
    check as 1.0 and then wrote `999999` onto the command line, because
    `isinstance` accepts subclasses and argv took `__str__` afterwards.
    And the only ceiling was zero, so a single call could be authorised
    for any amount while the task schema capped the whole run at
    `MAX_BUDGET_USD`.

    ONE authority, in `cli`, because the builder is a public callable
    too: the rule used to live only here, so calling the builder
    directly spelled `101`, `0` and `inf` onto a command line. Copying
    the rule into both modules would leave two places to forget."""
    try:
        return cli.exact_budget(budget_usd)
    except cli.UnsafeInvocation as refused:
        # the same fixed sentences, re-typed for this package's callers
        raise BudgetRefused(str(refused)) from None


class LimitRefused(AdapterError):
    """A bound that could not bound anything."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


class PromptNotDelivered(AdapterError):
    """The child answered without ever taking the prompt.

    A reply produced from an instruction the model never received is a
    reply to a different question. The write is asynchronous, which is
    what stops a deaf child from blocking the deadline -- and that same
    asynchrony means completion has to be CHECKED rather than assumed.

    `interrupted` is the closest closed code the frozen vocabulary has:
    the transport was cut short. Flagged for the contract owner, who may
    prefer a dedicated one."""

    def __init__(self, message, **measurements):
        super().__init__(message, event=contract.EventCode.INTERRUPTED,
                         reason=contract.StopReason.INTERRUPTED,
                         **measurements)


def _now():
    """The clock, behind one name.

    A test seam, and deliberately the ONLY one: the contract's minimum
    per-call timeout is 30 seconds, so proving timeout behaviour would
    otherwise mean either half-minute tests or a production range
    loosened to suit them. The range stays; the clock moves."""
    return time.monotonic()


def _canonical_limits(timeout_seconds, max_output_bytes):
    """The bounds are checked BEFORE a process exists, and returned.

    `timeout_seconds=NaN` made every deadline comparison false, so a
    hung child ran until something outside killed it. Then an `int`
    SUBCLASS whose `__le__` and `__ge__` always agreed passed both
    ranges while carrying a value nine orders of magnitude outside
    them -- `isinstance` accepts subclasses, and a bound whose
    comparisons the value itself defines is not a bound. Exact `int`
    also excludes `bool` by type."""
    if type(timeout_seconds) is not int:
        raise LimitRefused("sure siniri tam sayi degil")
    if not MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise LimitRefused("sure siniri sozlesme araligi disinda")
    if type(max_output_bytes) is not int:
        raise LimitRefused("cikti siniri tam sayi degil")
    if not MIN_OUTPUT_BYTES <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise LimitRefused("cikti siniri sozlesme araligi disinda")
    return timeout_seconds, max_output_bytes


def _prompt_bytes(prompt):
    """Encoded and measured before launch.

    The prompt used to be written synchronously, before the readers and
    the deadline existed: a child that never read its stdin filled the
    pipe buffer, the write blocked, and the timeout that was supposed to
    cover this had not started. The write is asynchronous now AND the
    payload is bounded, because a call that cannot fit down the pipe is
    better refused than raced.

    EXACTLY `str`: a subclass checked as "kurgu" returned entirely
    different bytes from `encode()`, so the instruction that was
    validated and the instruction the model received were two
    different questions."""
    if type(prompt) is not str:
        raise LimitRefused("istem bir metin degil")
    try:
        payload = prompt.encode("utf-8")
    except UnicodeEncodeError:
        raise LimitRefused("istem UTF-8'e kodlanamiyor") from None
    if not payload:
        raise LimitRefused("istem bos")
    if len(payload) > MAX_PROMPT_BYTES:
        raise LimitRefused("istem sozlesme tavanini asiyor")
    return payload


def _usable_binary(binary):
    """An existing regular file, as ONE absolute canonical path.

    `Popen` searches PATH for a bare name, which is how a test that
    meant to run a stub reaches the real, billable CLI.

    Two conversions used to happen and they could disagree: this
    function asked `Path(binary)` while the argv builder asked
    `str(binary)`, so an object whose `__fspath__` named binary A and
    whose `__str__` named binary B was VERIFIED as A and LAUNCHED as B.
    There is one conversion now, and its result is returned.

    RESOLVED before it is returned, because the process is launched
    with the disposable worktree as its working directory: a relative
    path checked against the current directory and launched against
    another names two different programs. What this does NOT close is
    the declared filesystem race -- a same-user process can still
    replace the file after this check."""
    try:
        text = cli.exact_text(binary, what="ikili dosya")
    except cli.UnsafeInvocation:
        raise BinaryNotUsable("ikili dosya bir yol degil") from None
    if not text:
        raise BinaryNotUsable("ikili dosya verilmedi")
    path = Path(text)
    if path.parent == Path(""):
        raise BinaryNotUsable("ikili dosya bir yol degil ya da mevcut degil")
    resolved = path.resolve()
    if not resolved.is_file():
        raise BinaryNotUsable("ikili dosya bir yol degil ya da mevcut degil")
    return resolved


class SchemaNotBound(AdapterError):
    """The argv's inline schema is not the frozen canonical binding, so
    no process was started.

    The schema used to travel as a caller-chosen FILE PATH -- to a flag
    that takes inline JSON -- while the validator used a separate live
    dictionary; nothing tied the two, and rewriting the file to garbage
    between build and launch changed nothing anybody checked. Now the
    bytes actually on the argv are hashed and compared to the binding
    immediately before launch; anything else refuses here."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


def _argv_schema(argv):
    """The inline schema ACTUALLY on the argv, or a refusal.

    Verified, not trusted: exactly one `--json-schema`, its value's
    exact UTF-8 bytes hashing to the frozen binding's SHA-256. The
    validator returned is parsed from THOSE bytes, so the schema the
    model receives and the schema that judges its reply cannot be two
    things. Never launches anything; never puts schema text in an
    error."""
    positions = [index for index, token in enumerate(argv)
                 if token == "--json-schema"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise SchemaNotBound("argv tam olarak bir inline sema tasimiyor")
    text = argv[positions[0] + 1]
    binding = schemas.IMPLEMENTER_SCHEMA_BINDING
    # EXACTLY `str`, not "a kind of str". `isinstance` accepts
    # SUBCLASSES, and a subclass answers whatever it likes: an audit
    # built one whose real content was `{}` but whose `__eq__` claimed
    # equality with the canonical text and whose `encode()` returned
    # the canonical BYTES. Both guards passed, `json.loads` then read
    # the real `{}`, and a permissive validator accepted an arbitrary
    # reply. Every check below asks the object a question; only an
    # exact `str` cannot lie about the answer.
    #
    # This also subsumes the earlier crash classes -- bytes, numbers,
    # None and unencodable strings -- which used to escape as
    # AttributeError or UnicodeEncodeError: operationally closed,
    # contractually untyped.
    if type(text) is not str:
        raise SchemaNotBound("argv'deki sema degeri metin degil")
    try:
        payload = text.encode("utf-8")
    except UnicodeEncodeError:
        raise SchemaNotBound(
            "argv'deki sema UTF-8'e kodlanamiyor") from None
    # Both the byte-exact text comparison against the frozen canonical
    # form AND a fresh hash of what is actually on the argv: either
    # divergence refuses, and neither trusts the other.
    if text != binding.canonical_json \
            or hashlib.sha256(payload).hexdigest() != binding.sha256:
        raise SchemaNotBound("argv'deki sema kanonik baglamayla eslesmiyor")
    return Draft202012Validator(json.loads(text)), binding.sha256


class CallInputRefused(AdapterError):
    """A call argument the CLI layer refused, re-typed for this
    package's callers.

    `run_implementer` promises a typed `AdapterError` for every
    refusal, and a `cli.UnsafeInvocation` escaping through it broke
    that promise: the runner's closed state machine would have had no
    reason code to record. The message is the CLI's own fixed sentence,
    which carries no caller value -- the builder keeps raising
    `UnsafeInvocation` when it is called directly."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


class IdentityRefused(AdapterError):
    """An identity or path argument that is not exactly what the frozen
    grammars describe, refused before any process exists.

    A `str` subclass whose `__eq__` returned True for everything
    satisfied every comparison the worktree binding makes -- so the
    binding agreed with a record it had never been shown. Identities
    are exact strings matching their existing patterns before the
    binding is consulted at all."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


def _exact_identity(value, pattern, what):
    """An exact `str` its frozen pattern accepts. The pattern comes
    from the module that already owns it; nothing is re-spelled here."""
    if type(value) is not str or not pattern.fullmatch(value):
        raise IdentityRefused(f"{what} sozlesme desenine uymuyor")
    return value


def _canonical_path(value, what):
    """One conversion, one concrete `Path`, and nothing afterwards asks
    the original object anything."""
    try:
        text = cli.exact_text(value, what=what)
    except cli.UnsafeInvocation:
        raise IdentityRefused(f"{what} bir yol degil") from None
    if not text:
        raise IdentityRefused(f"{what} bos")
    return Path(text)


class WorktreeNotBound(AdapterError):
    """The identities do not name a recorded, READY, git-verified
    disposable worktree, so no process was started.

    The predecessor of this class accepted any existing directory --
    `is_dir()` was the whole check -- and the main checkout passed it.
    The refusal text is the binding's own fixed sentence; it carries no
    path, no repository name and no record content."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.PREFLIGHT_FAILED)


class ContainmentFailed(AdapterError):
    """The container could not be established, so nothing was run.

    Refused BEFORE the model starts, deliberately: a call that cannot be
    contained is a call nobody can stop, and discovering that after the
    prompt has been sent means the invoice already exists. The previous
    version launched anyway and noticed at the end."""

    def __init__(self, message):
        super().__init__(message, event=contract.EventCode.PREFLIGHT_FAILED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED)


class ProcessTreeSurvived(AdapterError):
    """The call ended but its container could not be emptied.

    Not a detail to log and move past: something the model started is
    still running against the worktree, so the reply is not returned."""

    def __init__(self, message, **measurements):
        super().__init__(message,
                         event=contract.EventCode.MODEL_CALL_FINISHED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED,
                         **measurements)


def _parse_reply(raw, measurements, validator):
    """Decode, parse and validate -- refusing at every step.

    Strict UTF-8: replacement characters would turn undecodable bytes
    into a string that might then parse into something plausible. The
    validator ARRIVES here, built from the argv's own inline schema
    bytes -- validating against a separate in-process schema is exactly
    the unbound state R2B removed."""
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
    try:
        validator.validate(payload)
    except ValidationError as invalid:
        # the failing FIELD PATH, never the failing value: the value is
        # model output and this text travels into reports
        where = "/".join(str(part) for part in invalid.absolute_path) or "kok"
        raise SchemaViolation(f"yanit sema disi (alan: {where})",
                              **measurements) from None
    return payload


@dataclass(frozen=True, slots=True)
class _CanonicalImplementerCall:
    """Every input, validated once and converted once.

    THE POINT OF THIS CLASS is that after it exists the caller's
    objects are irrelevant. A validator that only inspects a value and
    leaves the original alive is not a boundary: the argv builder, the
    worktree binding, the deadline arithmetic and the stdin writer each
    asked the caller's object again, and an object is free to answer
    differently every time. These fields are exact built-in types and
    concrete paths, and they are the only things anything downstream
    is allowed to see."""

    binary: Path
    repo: Path
    state_dir: Path
    run_id: str
    worktree_id: str
    baseline_sha: str
    prompt_bytes: bytes
    budget_usd: object            # exact int or float, bounded
    timeout_seconds: int
    max_output_bytes: int
    model: object                 # exact str or None


def _canonical_call(binary, *, repo, state_dir, run_id, worktree_id,
                    baseline_sha, prompt, budget_usd, timeout_seconds,
                    max_output_bytes, model):
    """Validate and convert, in the order the refusals are cheapest.

    Budget, bounds, prompt and binary come before the identities so a
    caller who got one of those wrong hears about it without a
    filesystem or git question being asked -- the order existing tests
    already depend on."""
    canonical_budget = _canonical_budget(budget_usd)
    canonical_timeout, canonical_output = _canonical_limits(
        timeout_seconds, max_output_bytes)
    # ONE boundary for the CLI layer's refusals. `cli.UnsafeInvocation`
    # is not an `AdapterError`, and one escaping here broke this
    # function's own promise -- the runner's closed state machine would
    # have had no reason code for it. Wrapping the whole construction
    # rather than a single call keeps that true for every cli helper
    # reached from here, including ones added later.
    try:
        return _CanonicalImplementerCall(
            binary=_usable_binary(binary),
            prompt_bytes=_prompt_bytes(prompt),
            budget_usd=canonical_budget,
            timeout_seconds=canonical_timeout,
            max_output_bytes=canonical_output,
            model=cli.exact_model(model),
            repo=_canonical_path(repo, "depo yolu"),
            state_dir=_canonical_path(state_dir, "durum dizini"),
            run_id=_exact_identity(run_id, _RUN_ID_PATTERN, "kosu kimligi"),
            worktree_id=_exact_identity(worktree_id, worktree.WORKTREE_ID,
                                        "calisma agaci kimligi"),
            baseline_sha=_exact_identity(baseline_sha, _BASELINE_PATTERN,
                                         "taban surum"))
    except cli.UnsafeInvocation as refused:
        raise CallInputRefused(str(refused)) from None


def run_implementer(binary, *, repo, state_dir, run_id, worktree_id,
                    baseline_sha, prompt, budget_usd,
                    timeout_seconds, max_output_bytes, model=None):
    """Run the implementer once and return its validated reply.

    The working directory is derived from the identities by
    `worktree.assert_execution_binding`, and the schema is the frozen
    canonical binding -- there is no way to pass either one in.
    Raises a typed `AdapterError` for every refusal. Nothing here
    retries, repairs or decides what happens next."""
    call = _canonical_call(
        binary, repo=repo, state_dir=state_dir, run_id=run_id,
        worktree_id=worktree_id, baseline_sha=baseline_sha, prompt=prompt,
        budget_usd=budget_usd, timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes, model=model)
    # THE RAW ARGUMENTS ARE GONE, and that is the enforcement rather
    # than a comment asking future readers to be careful: every defect
    # in this family was a second look at a caller's object after a
    # check had already agreed with it. Reaching for one now is a
    # NameError the positive-control tests hit on the first run.
    del binary, repo, state_dir, run_id, worktree_id, baseline_sha
    del prompt, budget_usd, timeout_seconds, max_output_bytes, model

    argv = cli.build_implementer_argv(call.binary, budget_usd=call.budget_usd,
                                      model=call.model)
    # The bytes ACTUALLY on the argv, hashed against the frozen binding
    # before anything runs -- and the validator is parsed from those
    # same bytes, so what the model receives and what judges its reply
    # cannot diverge.
    validator, schema_sha256 = _argv_schema(argv)

    # LAST before the launch, so nothing sits between the proof and the
    # use of it. The filesystem offers no transaction here -- a hostile
    # same-user process can still race this window -- but the window is
    # kept to the launch itself.
    try:
        cwd = worktree.assert_execution_binding(
            call.repo, state_dir=call.state_dir, run_id=call.run_id,
            worktree_id=call.worktree_id, baseline_sha=call.baseline_sha)
    except worktree.WorktreeError as refused:
        raise WorktreeNotBound(str(refused)) from None

    # THE CONTAINER EXISTS BEFORE THE PROGRAM RUNS. Creating the
    # process and containing it afterwards leaves a window in which the
    # child can spawn something that lands outside the container -- and
    # if the container cannot be built at all, the model must not run:
    # an uncontained call is one nobody can stop, and it is billable.
    started = _now()
    try:
        process, container = launch_contained(argv, cwd=cwd)
    except ContainmentError as refused:
        raise ContainmentFailed(str(refused)) from None

    drained = False
    grace = None
    try:
        # READERS FIRST, THEN THE PROMPT. Writing synchronously before
        # this meant a child that never read its stdin -- or one whose
        # output had already filled the pipe -- blocked the adapter in a
        # write no deadline covered, because the deadline loop had not
        # been reached.
        tripped = threading.Event()
        streams = [BoundedStream("stdout", process.stdout,
                                 call.max_output_bytes, tripped),
                   BoundedStream("stderr", process.stderr,
                                 call.max_output_bytes, tripped)]
        for stream in streams:
            stream.start()
        writer = PromptWriter(process.stdin, call.prompt_bytes)
        writer.start()

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

        # ONE grace budget covers the kill and every thread join, so a
        # run that is already failing cannot spend three separate
        # timeouts.
        grace = time.monotonic() + REAP_SECONDS
        cleanup_complete = True
        if tripped.is_set() or timed_out:
            cleanup_complete = stop(process, grace)
        # DRAIN BEFORE JOINING. A grandchild holding the pipes open kept
        # the readers alive, so the adapter waited out the whole grace
        # period and only then killed anything.
        drained = container.drain(grace)
        if not drained:
            cleanup_complete = False
        if not join_within([writer, *streams], grace):
            cleanup_complete = False

        duration_ms = int((_now() - started) * 1000)
        measurements = {"duration_ms": duration_ms,
                        "stdout_bytes": streams[0].total,
                        "stderr_bytes": streams[1].total,
                        "cleanup_complete": cleanup_complete}

        if timed_out:
            raise Timeout("model cagrisi sure sinirini asti",
                          timeout_seconds=call.timeout_seconds,
                          exit_code=process.returncode, **measurements)
        overflowed = [stream for stream in streams if stream.overflowed]
        if overflowed:
            raise OutputLimitExceeded(
                "cikti sinirini asti; surec agaci durduruldu",
                stream=overflowed[0].label, exit_code=process.returncode,
                **measurements)

        exit_code = process.wait()
        if exit_code != 0:
            raise ProcessFailed("model sureci sifirdan farkli koda dondu",
                                exit_code=exit_code, **measurements)
        # A child can answer WITHOUT reading: it writes valid JSON,
        # exits, and the asynchronous write dies on a closed pipe. The
        # reply then answers a question the model was never asked.
        if not writer.completed:
            raise PromptNotDelivered("istem modele tamamen teslim edilemedi",
                                     exit_code=exit_code, **measurements)
        if not drained:
            raise ProcessTreeSurvived(
                "model sureci kapsayicisi bosaltilamadi", exit_code=exit_code,
                **measurements)
        measurements.pop("cleanup_complete")
    finally:
        # EVERY exit, including an exception raised between the launch
        # and the drain above. An injected thread-start failure used to
        # leave the model process running with its container built and
        # never emptied.
        if not drained:
            # an exception before the cleanup window opened gets a fresh
            # budget; otherwise the SAME deadline `stop` and the joins
            # used, so one failing call cannot spend three of them
            container.drain(grace if grace is not None
                            else time.monotonic() + REAP_SECONDS)

    reply = _parse_reply(streams[0].buffer,
                         dict(measurements, exit_code=exit_code), validator)
    return ImplementerRun(reply=reply, exit_code=exit_code,
                          event=contract.EventCode.MODEL_CALL_FINISHED,
                          schema_sha256=schema_sha256, **measurements)
