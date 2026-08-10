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

NOTHING IS DISCOVERED. The binary, the identities, the schema path, the
budget, the timeout and the output ceiling are all mandatory arguments.
A caller who forgets one gets a `TypeError`; a caller who passes a bare
name like `claude` gets a refusal rather than whatever is on PATH. The
argv itself is built only by `cli.build_implementer_argv`, so the flag
rules stay in the module that already proves them.

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

import json
import math
import os
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
    """A validated reply plus what the call cost in time and bytes."""

    reply: dict
    exit_code: int
    duration_ms: int
    stdout_bytes: int
    stderr_bytes: int
    event: str


def _assert_budget(budget_usd):
    """Refused here, before anything is launched.

    A budget that is zero, negative or not a finite number cannot fund a
    call, and finding that out after the process started is finding it
    out too late."""
    if isinstance(budget_usd, bool) or not isinstance(budget_usd,
                                                      (int, float)):
        raise BudgetRefused("butce sayisal degil")
    if not math.isfinite(budget_usd):
        raise BudgetRefused("butce sonlu bir sayi degil")
    if budget_usd <= 0:
        raise BudgetRefused("kalan butce bir cagriyi fonlamiyor")


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


def _assert_limits(timeout_seconds, max_output_bytes):
    """The bounds are checked BEFORE a process exists.

    Only the budget was validated. `timeout_seconds=NaN` made every
    deadline comparison false, so a hung child ran until something
    outside killed it, and `max_output_bytes=NaN` made every ceiling
    comparison false, so a megabyte was read in full and then refused
    for the wrong reason -- a parse error standing in for a limit that
    never applied. A bound that cannot be compared is not a bound."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds,
                                                           int):
        raise LimitRefused("sure siniri tam sayi degil")
    if not MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise LimitRefused("sure siniri sozlesme araligi disinda")
    if isinstance(max_output_bytes, bool) or not isinstance(
            max_output_bytes, int):
        raise LimitRefused("cikti siniri tam sayi degil")
    if not MIN_OUTPUT_BYTES <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise LimitRefused("cikti siniri sozlesme araligi disinda")


def _prompt_bytes(prompt):
    """Encoded and measured before launch.

    The prompt used to be written synchronously, before the readers and
    the deadline existed: a child that never read its stdin filled the
    pipe buffer, the write blocked, and the timeout that was supposed to
    cover this had not started. The write is asynchronous now AND the
    payload is bounded, because a call that cannot fit down the pipe is
    better refused than raced."""
    if not isinstance(prompt, str):
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
    """An existing regular file, and only that.

    `Popen` searches PATH for a bare name, which is how a test that
    meant to run a stub reaches the real, billable CLI."""
    if not binary:
        raise BinaryNotUsable("ikili dosya verilmedi")
    path = Path(binary)
    if path.parent == Path("") or not path.is_file():
        raise BinaryNotUsable("ikili dosya bir yol degil ya da mevcut degil")
    return path


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


def _parse_reply(raw, measurements):
    """Decode, parse and validate -- refusing at every step.

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
    try:
        Draft202012Validator(
            schemas.IMPLEMENTER_RESULT_SCHEMA).validate(payload)
    except ValidationError as invalid:
        # the failing FIELD PATH, never the failing value: the value is
        # model output and this text travels into reports
        where = "/".join(str(part) for part in invalid.absolute_path) or "kok"
        raise SchemaViolation(f"yanit sema disi (alan: {where})",
                              **measurements) from None
    return payload


def run_implementer(binary, *, repo, state_dir, run_id, worktree_id,
                    baseline_sha, prompt, schema_path, budget_usd,
                    timeout_seconds, max_output_bytes, model=None):
    """Run the implementer once and return its validated reply.

    The working directory is derived from the identities by
    `worktree.assert_execution_binding`; there is no way to pass one in.
    Raises a typed `AdapterError` for every refusal. Nothing here
    retries, repairs or decides what happens next."""
    _assert_budget(budget_usd)
    _assert_limits(timeout_seconds, max_output_bytes)
    payload = _prompt_bytes(prompt)
    _usable_binary(binary)
    argv = cli.build_implementer_argv(binary, schema_path=schema_path,
                                      budget_usd=budget_usd, model=model)

    # LAST before the launch, so nothing sits between the proof and the
    # use of it. The filesystem offers no transaction here -- a hostile
    # same-user process can still race this window -- but the window is
    # kept to the launch itself.
    try:
        cwd = worktree.assert_execution_binding(
            repo, state_dir=state_dir, run_id=run_id,
            worktree_id=worktree_id, baseline_sha=baseline_sha)
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
        streams = [BoundedStream("stdout", process.stdout, max_output_bytes,
                                 tripped),
                   BoundedStream("stderr", process.stderr, max_output_bytes,
                                 tripped)]
        for stream in streams:
            stream.start()
        writer = PromptWriter(process.stdin, payload)
        writer.start()

        deadline = started + timeout_seconds
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
                          timeout_seconds=timeout_seconds,
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
                         dict(measurements, exit_code=exit_code))
    return ImplementerRun(reply=reply, exit_code=exit_code,
                          event=contract.EventCode.MODEL_CALL_FINISHED,
                          **measurements)
