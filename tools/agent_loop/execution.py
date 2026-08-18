"""The implementer subprocess adapter. PACKAGE B2A.

ONE primitive: launch the Claude implementer binary the caller supplied,
inside the IMPLEMENTER tree of the D3A flat workspace the ledger
RECORDED, and bring back a reply that has been validated against the
frozen schema. It does not create workspaces, read diffs, run acceptance
commands, call the evaluator, apply patches or advance the run. Those
live in B2B and B3, and keeping them out is the only reason this file
can be read in one sitting.

THE WORKING DIRECTORY IS DERIVED, NEVER RECEIVED. The first version
took a path and checked `is_dir()`, which made the main checkout -- or
any directory at all -- an acceptable place to run the model, with the
ownership record sitting unread beside it. Now the caller passes
identities (repository, state directory, run, workspace id, baseline)
and `flat_workspace.assert_binding` turns them into the one directory
they may name, refusing everything else before a process exists. There
is no parameter through which a caller can inject a path.

ONE EXECUTION IDENTITY. B2B-B1 accepted `worktree_id` beside
`workspace_id` while the flat workspace replaced the disposable git
worktree, and B2B-B2C removed the older of the two along with the module
behind it. A call still naming a worktree is refused by Python itself.

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

from tools.agent_loop import cli, contract, flat_workspace, schemas
from tools.agent_loop.process import (
    BoundedStream, Container, ContainmentError, PromptWriter,
    REAP_SECONDS, READ_CHUNK_BYTES, READ_COMPLETED, join_within,
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
    flat_workspace.RECORD_SCHEMA["properties"]["baseline_sha"]["pattern"])


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


# The CLI's own terminal error envelopes (B4-R6).
#
# MEASURED, NOT GUESSED. Every value below was read out of the installed
# binary's string table and appears within 400 bytes of the literal
# `subtype`, which is where an emitting site puts it;
# `error_max_budget_usd` was additionally observed in a real envelope. A
# fifth candidate, `error_envelope_no_type`, occurs in the binary and
# NEVER near `subtype`, so it is not a subtype value and is deliberately
# absent -- a name whose meaning nobody has proven must not become a
# code the loop reports.
#
# WHY THESE ARE SEPARATE CLASSES. They are four different operator
# actions: raise a ceiling, allow more turns, look at the schema, look
# at the provider. `model_process_failed` cannot tell them apart, and
# the previous real run cost a scripted probe and a session-record hunt
# to answer a question the process itself had already answered.
ENVELOPE_ERROR_SUBTYPES = {
    "error_max_budget_usd": "MaxBudgetReached",
    "error_max_turns": "MaxTurnsReached",
    "error_max_structured_output_retries": "StructuredOutputRetriesExhausted",
    "error_during_execution": "ProviderExecutionFailed",
}


class MaxBudgetReached(ProcessFailed):
    """The CLI stopped itself at `--max-budget-usd`.

    A `ProcessFailed` still: the process ended non-zero and every
    measurement it carries is unchanged. What is added is WHICH ceiling
    ended it."""


class MaxTurnsReached(ProcessFailed):
    """The CLI stopped itself at its turn limit."""


class StructuredOutputRetriesExhausted(ProcessFailed):
    """The CLI could not get a schema-valid answer within its retries.

    Distinct from `SchemaViolation`, which is THIS package refusing a
    reply it received: here the CLI never obtained one to hand over."""


class ProviderExecutionFailed(ProcessFailed):
    """The CLI reported a failure during execution."""


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
    repaired: a guessed field is a field nobody agreed to.

    CARRIES TWO CLOSED WORDS (B5-R3). `schema_issue` says which rule
    refused and `schema_field` says which DECLARED field it refused at,
    both from the contract's own tuples. They exist because the sentence
    below cannot travel: a message is free text, the runner will not put
    free text in the journal, and a real run therefore recorded nothing
    but `implementer_schema_violation`.

    Anything not proven is left out rather than guessed, and neither
    field is ever derived from a value the model chose."""

    def __init__(self, message, *, schema_issue=None, schema_field=None,
                 **measurements):
        super().__init__(message, event=contract.EventCode.SCHEMA_VIOLATION,
                         reason=contract.StopReason.SCHEMA_VIOLATION,
                         **measurements)
        # membership-checked HERE, so a caller cannot attach a word the
        # contract does not own even by accident
        self.schema_issue = (schema_issue
                             if schema_issue in contract.ALL_SCHEMA_ISSUES
                             else None)
        self.schema_field = (schema_field
                             if schema_field in contract.ALL_SCHEMA_FIELDS
                             else None)


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


# B7-R1. Built FROM the derived table rather than typed out again: a
# second hard-coded copy is exactly what goes stale when a rule changes,
# and the model would then be told something the authority contradicts.
# The statuses come FROM the derived table, so the matrix cannot go
# stale about which ones exist. The ACTIONS are deliberately not named:
# the model must not write the field at all, and printing the value it
# would have written is an invitation to try. The evaluator road makes
# the same choice.
PROTOCOL_MATRIX = (
    "SOZLESME (kapali):",
    "  - next_action YAZMA; adapter onu status'tan turetir",
    "  - status su kapali kumeden biri olmali: "
    + ", ".join(sorted(schemas.IMPLEMENTER_NEXT_ACTION)),
    "  - status=blocked veya failed -> stop_reason bos olamaz",
)


def _with_protocol(prompt):
    """The caller's prompt plus the closed protocol matrix.

    Appended rather than merged: the runner owns what the task IS, this
    module owns what a valid ANSWER is, and the two must not be able to
    contradict each other inside one shared string.

    THE CALLER'S HALF IS JUDGED FIRST, and that ordering is load-bearing
    rather than tidy: appending the matrix makes any prompt non-empty, so
    an empty one would sail through the emptiness check below carrying
    nothing but this module's own text -- an instruction the implementer
    was never given, in a call somebody paid for."""
    if type(prompt) is not str:
        raise LimitRefused("istem bir metin degil")
    if not prompt:
        raise LimitRefused("istem bos")
    # EMPTINESS IS CHECKED WITHOUT ENCODING, and that is not a shortcut:
    # a prompt carrying a lone surrogate cannot be encoded at all, and
    # encoding here to measure it would escape as `UnicodeEncodeError`
    # instead of this module's typed refusal. `_prompt_bytes` owns the
    # encoding question and already answers it with `LimitRefused`.
    return "\n".join([prompt, *PROTOCOL_MATRIX])


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
    with the workspace's implementer tree as its working directory: a
    relative path checked against the current directory and launched
    against another names two different programs. What this does NOT close is
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
    exact UTF-8 bytes hashing to the TRANSPORT binding's SHA-256.

    TWO AUTHORITIES SINCE B4-R2, and the split is deliberate. The API
    refuses the acceptance schema outright -- `pattern`, `if`/`then`
    and every length and numeric bound are outside the published
    structured-output subset, and two authorized diagnostics measured
    the resulting 4xx. So the argv carries the reduced TRANSPORT schema,
    which constrains what the model generates, while the validator
    returned here is built from the AUTHORITATIVE binding's own bytes,
    which is what decides whether a reply is acceptable.

    The weaker schema therefore never judges anything: a reply that
    satisfies the transport copy and fails the authority is refused by
    `_parse_reply` exactly as before. Never launches anything; never
    puts schema text in an error."""
    positions = [index for index, token in enumerate(argv)
                 if token == "--json-schema"]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise SchemaNotBound("argv tam olarak bir inline sema tasimiyor")
    text = argv[positions[0] + 1]
    binding = schemas.IMPLEMENTER_TRANSPORT_BINDING
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
    # THE VALIDATOR IS THE AUTHORITY'S, parsed from the authoritative
    # binding's own canonical bytes -- never from the argv text, which
    # is the weaker transport copy. Building it from `text` would make
    # the reduced schema the acceptance gate, which is exactly the
    # confusion this split exists to prevent.
    authority = schemas.IMPLEMENTER_SCHEMA_BINDING
    return Draft202012Validator(
        json.loads(authority.canonical_json)), authority.sha256


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
    satisfied every comparison the execution binding makes -- so the
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


class WorkspaceNotBound(AdapterError):
    """The identities do not name a recorded, READY flat workspace, so
    no process was started.

    A distant predecessor of this class accepted any existing directory
    -- `is_dir()` was the whole check -- and the main checkout passed it.
    The refusal text is the binding's own fixed sentence; it carries no
    path, no repository name and no record content.

    ONE class, and no alias. B2B-B1 made `WorkspaceNotBound` a second
    name for the worktree refusal so a caller could not handle one and
    miss the other while both execution surfaces existed. There is one
    surface now, so there is one name."""

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
    still running against the workspace, so the reply is not
    returned."""

    def __init__(self, message, **measurements):
        super().__init__(message,
                         event=contract.EventCode.MODEL_CALL_FINISHED,
                         reason=contract.StopReason.MODEL_PROCESS_FAILED,
                         **measurements)


# The measured success envelope, as three exact values. `claude
# --print --output-format json` answers with a RESULT ENVELOPE; the
# implementer payload is one field inside it.
ENVELOPE_TYPE = "result"
ENVELOPE_SUBTYPE = "success"
# THE CANONICAL PAYLOAD FIELD. `structured_output` exists because of
# `--json-schema` and arrives already parsed; `result` is the generic
# text rendering every json-format reply carries, payload or not. Two
# fields that can both hold a payload is two authorities, so one is
# named here and the other is only ever used to detect disagreement.
PAYLOAD_FIELD = "structured_output"
TEXT_FIELD = "result"


def _envelope_payload(payload, measurements):
    """The implementer payload carried by a SUCCESSFUL result envelope.

    MEASURED against `claude 2.1.220` rather than assumed: exit 0,
    `type="result"`, `subtype="success"`, `is_error=False`, the payload
    an exact object under `structured_output`, and the same payload
    rendered as text under `result`.

    Everything is exact. `is_error` is compared to `False` by identity
    of value rather than truthiness, because `0`, `""` and a missing key
    are all falsy and none of them is the CLI saying the run succeeded.

    WHAT DOES NOT CROSS: `session_id`, `uuid`, `usage`, `modelUsage`,
    `total_cost_usd`, the timings and the model's own prose. They stay
    in this function's local `payload` and are never read out of it --
    the adapter reports measurements it made itself, and a cost figure
    from the envelope would be a number a caller could subtract from a
    budget on the strength of the model's own bookkeeping."""
    # EVERY refusal in this function is the ENVELOPE stage (B5-R3): the
    # transport document was wrong, so the authority never saw a payload
    # and no declared field can be blamed.
    envelope = {"schema_issue": contract.SchemaIssue.INVALID_ENVELOPE,
                "schema_field": contract.SchemaField.ENVELOPE}
    if payload.get("type") != ENVELOPE_TYPE:
        raise SchemaViolation("cikti bir sonuc zarfi degil",
                              **envelope, **measurements)
    if payload.get("subtype") != ENVELOPE_SUBTYPE:
        raise SchemaViolation("zarf basarili bir sonuc bildirmiyor",
                              **envelope, **measurements)
    if payload.get("is_error") is not False:
        raise SchemaViolation("zarf hata bayragi tasiyor",
                              **envelope, **measurements)
    inner = payload.get(PAYLOAD_FIELD)
    # EXACTLY `dict`, so a subclass cannot answer the validator's
    # questions differently from the mapping that gets returned
    if type(inner) is not dict:
        raise SchemaViolation("zarf yapisal sonucu tasimiyor",
                              **envelope, **measurements)
    # A SECOND payload that DISAGREES is refused. Agreement is the
    # normal case -- the CLI renders the same object into both fields --
    # and prose in `result` is not a competing payload at all. Silently
    # preferring one of two disagreeing objects is how the field nobody
    # watches becomes the field that decides.
    text = payload.get(TEXT_FIELD)
    if type(text) is str:
        try:
            rendered = json.loads(text)
        except ValueError:
            rendered = None
        if isinstance(rendered, dict) and rendered != inner:
            raise SchemaViolation("zarf celisen iki sonuc tasiyor",
                                  **envelope, **measurements)
    return inner


def _envelope_failure(stream, measurements, exit_code):
    """A more precise `ProcessFailed`, or `None` to keep the generic one.

    EVERY CONDITION MUST HOLD, and any one of them failing is silence
    rather than a guess: the stream must have been read to completion
    (an overflowed or unreadable buffer is a partial document, and the
    existing overflow and read-failure authorities outrank this one),
    the bytes must be one JSON object, the envelope must call itself a
    result, `is_error` must be exactly `True`, and the `subtype` must be
    an exact string this package has EVIDENCE for.

    NOTHING IS READ OUT OF THE ENVELOPE. The subtype is used as a lookup
    key and dropped; the vendor's message, the `result` text, the
    session id and the cost never enter the exception, whose sentence is
    a fixed one chosen here. An unknown subtype is not reported as
    itself -- it falls back to the generic failure, because a code this
    package cannot define is a code nobody can act on.

    THIS FUNCTION NEVER SUCCEEDS. A valid-looking SUCCESS envelope beside
    a non-zero exit is still a failure: `is_error is not True` returns
    `None`, and the caller raises the generic `ProcessFailed`."""
    if stream.outcome != READ_COMPLETED or stream.overflowed:
        return None
    try:
        payload = json.loads(bytes(stream.buffer).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if type(payload) is not dict:
        return None
    if payload.get("type") != ENVELOPE_TYPE:
        return None
    if payload.get("is_error") is not True:
        return None
    subtype = payload.get("subtype")
    if type(subtype) is not str:
        return None
    name = ENVELOPE_ERROR_SUBTYPES.get(subtype)
    if name is None:
        return None
    return globals()[name]("model sureci bildirilen bir sinirda durdu",
                           exit_code=exit_code, **measurements)


def _declared_field(path):
    """The first element of a failure path, IF the contract declares it.

    Only the top level is read. A nested name is either a declared
    field's own sub-structure -- in which case the top-level name is the
    honest answer -- or a key the model invented, which never travels."""
    for part in path:
        if isinstance(part, str):
            return (part if part in contract.ALL_SCHEMA_FIELDS
                    else contract.SchemaField.UNKNOWN)
        return contract.SchemaField.UNKNOWN
    return contract.SchemaField.ROOT


def _missing_required_field(error, payload):
    """WHICH required field is absent, computed from the SCHEMA and the
    payload's key set -- never parsed out of the exception message.

    The message embeds the value; the schema's own `required` list does
    not exist. One missing declared field names it, several are
    `multiple`, and anything unprovable stays `unknown`."""
    declared = error.validator_value
    if not isinstance(declared, list) or not isinstance(error.instance, dict):
        return contract.SchemaField.UNKNOWN
    missing = [name for name in declared if name not in error.instance]
    known = [name for name in missing
             if name in contract.ALL_SCHEMA_FIELDS]
    if len(missing) == 1 and len(known) == 1:
        return known[0]
    if len(missing) > 1:
        return contract.SchemaField.MULTIPLE
    return contract.SchemaField.UNKNOWN


def _schema_diagnosis(error, payload):
    """The two closed words for one validation error.

    NOTHING HERE READS A VALUE. The issue comes from a table keyed by
    jsonschema's keyword; the field comes from the failure PATH, which
    is made of names the schema declared -- with the single exception of
    an undeclared key, which is reported as `unknown` rather than
    spelled out."""
    issue = contract.SCHEMA_ISSUE_FOR_VALIDATOR.get(
        str(error.validator), contract.SchemaIssue.UNKNOWN)
    path = list(error.absolute_path)
    # A missing field is only nameable AT THE ROOT: deeper down the
    # absent name belongs to a nested structure -- a test result, a
    # finding -- and the honest answer is the declared field that
    # contains it, not a name from a sub-schema nobody asked about.
    if issue == contract.SchemaIssue.REQUIRED and not path:
        return issue, _missing_required_field(error, payload)
    return issue, _declared_field(path)


def _parse_reply(raw, measurements, validator):
    """Decode, parse, UNWRAP and validate -- refusing at every step.

    Strict UTF-8: replacement characters would turn undecodable bytes
    into a string that might then parse into something plausible.

    THE VALIDATOR IS THE AUTHORITY'S (B4-R2), not the argv's: the schema
    that travels is the reduced transport copy, and validating with it
    would silently stop enforcing every constraint the API cannot
    compile. THE PAYLOAD IS UNWRAPPED FIRST (B4-R3): the whole of stdout
    used to be validated against the implementer schema, so every real
    reply -- including a successful one -- was a schema violation."""
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        raise SchemaViolation("cikti gecerli UTF-8 degil",
                              schema_issue=contract.SchemaIssue.INVALID_UTF8,
                              schema_field=contract.SchemaField.ROOT,
                              **measurements) from None
    try:
        payload = json.loads(text)
    except ValueError:
        raise SchemaViolation("cikti JSON degil",
                              schema_issue=contract.SchemaIssue.INVALID_JSON,
                              schema_field=contract.SchemaField.ROOT,
                              **measurements) from None
    if not isinstance(payload, dict):
        raise SchemaViolation(
            "cikti bir JSON nesnesi degil",
            schema_issue=contract.SchemaIssue.WRONG_ROOT_TYPE,
            schema_field=contract.SchemaField.ROOT, **measurements)
    payload = _envelope_payload(payload, measurements)
    # THE DERIVATION, BEFORE THE AUTHORITY (B7-R1). The transport no
    # longer asks the model for `next_action` -- the conditional `const`
    # rules cannot survive the provider subset, so demanding the field
    # meant demanding a value whose rule the model had never been shown.
    # It is filled in here from `status`, out of the table derived from
    # the authority itself, and the authority then judges the completed
    # document exactly as before. This adds a VALUE, never a verdict.
    try:
        payload = schemas.project_implementer_fields(payload)
    except schemas.ProjectionError as refused_projection:
        # the closed words come from the projection, which is the one
        # thing that knows why it refused; the reply's own values never
        # travel into a refusal
        raise SchemaViolation(
            "yanit turetilebilir bir durum tasimiyor",
            schema_issue=refused_projection.schema_issue,
            schema_field=refused_projection.schema_field,
            **measurements) from None
    refusal = None
    try:
        validator.validate(payload)
    except ValidationError as invalid:
        # the failing FIELD PATH, never the failing value: the value is
        # model output and this text travels into reports. The two
        # closed words are what actually reaches the journal (B5-R3);
        # this sentence stays for a human reading a traceback.
        where = "/".join(str(part) for part in invalid.absolute_path) or "kok"
        issue, field = _schema_diagnosis(invalid, payload)
        refusal = (f"yanit sema disi (alan: {where})", issue, field)
    # RAISED OUTSIDE THE HANDLER, and that is not style. `from None`
    # suppresses the DISPLAY of the cause but leaves `__context__`
    # pointing at the `ValidationError`, whose message embeds the value
    # that broke the rule -- an invented key, a token, a path. Leaving
    # the handler first means the exception this package hands upwards
    # has no chain to walk. Measured: a sentinel test caught the leak
    # through `__context__` while `str`, `repr` and `__dict__` were all
    # already clean.
    if refusal is not None:
        message, issue, field = refusal
        raise SchemaViolation(message, schema_issue=issue, schema_field=field,
                              **measurements)
    return payload


@dataclass(frozen=True, slots=True)
class _CanonicalImplementerCall:
    """Every input, validated once and converted once.

    THE POINT OF THIS CLASS is that after it exists the caller's
    objects are irrelevant. A validator that only inspects a value and
    leaves the original alive is not a boundary: the argv builder, the
    workspace binding, the deadline arithmetic and the stdin writer each
    asked the caller's object again, and an object is free to answer
    differently every time. These fields are exact built-in types and
    concrete paths, and they are the only things anything downstream
    is allowed to see."""

    binary: Path
    repo: Path
    state_dir: Path
    run_id: str
    # THE execution identity. There is one, so nothing downstream has to
    # ask which of two fields decided the working directory -- that was a
    # second question about an already-checked value, and this package's
    # whole family of defects was exactly that.
    workspace_id: str
    baseline_sha: str
    prompt_bytes: bytes
    budget_usd: object            # exact int or float, bounded
    timeout_seconds: int
    max_output_bytes: int
    model: object                 # exact str or None


def _canonical_call(binary, *, repo, state_dir, run_id, workspace_id,
                    baseline_sha, prompt, budget_usd,
                    timeout_seconds, max_output_bytes, model):
    """Validate and convert, in the order the refusals are cheapest.

    Budget, bounds, prompt and binary come before the identities so a
    caller who got one of those wrong hears about it without a
    filesystem question being asked -- the order existing tests already
    depend on."""
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
            # the matrix is appended HERE, so both roads that reach this
            # constructor -- the initial implementation and the verified
            # repair -- carry it without either having to remember
            prompt_bytes=_prompt_bytes(_with_protocol(prompt)),
            budget_usd=canonical_budget,
            timeout_seconds=canonical_timeout,
            max_output_bytes=canonical_output,
            model=cli.exact_model(model),
            repo=_canonical_path(repo, "depo yolu"),
            state_dir=_canonical_path(state_dir, "durum dizini"),
            run_id=_exact_identity(run_id, _RUN_ID_PATTERN, "kosu kimligi"),
            workspace_id=_exact_identity(workspace_id,
                                         flat_workspace.WORKSPACE_ID,
                                         "calisma alani kimligi"),
            baseline_sha=_exact_identity(baseline_sha, _BASELINE_PATTERN,
                                         "taban surum"))
    except cli.UnsafeInvocation as refused:
        raise CallInputRefused(str(refused)) from None


def run_implementer(binary, *, repo, state_dir, run_id, workspace_id,
                    baseline_sha, prompt, budget_usd,
                    timeout_seconds, max_output_bytes, model=None):
    """Run the implementer once and return its validated reply.

    The working directory is derived from the identities by
    `flat_workspace.assert_binding`, and the schema is the frozen
    canonical binding. There is no way to pass either one in. Raises a
    typed `AdapterError` for every refusal. Nothing here retries,
    repairs or decides what happens next.

    `workspace_id` is required and keyword-only. B2B-B1 accepted a
    second identity beside it while the flat workspace replaced the
    disposable git worktree; B2B-B2C removed it, so a caller still
    passing `worktree_id=` fails at the call itself rather than being
    quietly translated into something else."""
    call = _canonical_call(
        binary, repo=repo, state_dir=state_dir, run_id=run_id,
        workspace_id=workspace_id,
        baseline_sha=baseline_sha, prompt=prompt,
        budget_usd=budget_usd, timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes, model=model)
    # THE RAW ARGUMENTS ARE GONE, and that is the enforcement rather
    # than a comment asking future readers to be careful: every defect
    # in this family was a second look at a caller's object after a
    # check had already agreed with it. Reaching for one now is a
    # NameError the positive-control tests hit on the first run.
    del binary, repo, state_dir, run_id, workspace_id
    del baseline_sha
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
        # the IMPLEMENTER root, never the reference tree: the model must
        # have no path to the copy it is going to be compared against
        cwd = flat_workspace.assert_binding(
            call.repo, state_dir=call.state_dir, run_id=call.run_id,
            workspace_id=call.workspace_id,
            baseline_sha=call.baseline_sha).implementer_root
    except flat_workspace.FlatWorkspaceError as refused:
        raise WorkspaceNotBound(str(refused)) from None

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
            # THE ENVELOPE IS STILL IN HAND (B4-R6). The bounded buffer
            # holds it right now and is dropped moments later, so the
            # only chance to learn WHICH ceiling ended the call is here.
            # `_envelope_failure` returns a more precise class or None;
            # it never turns a failure into a success and never persists
            # a byte of what it read.
            raise _envelope_failure(
                streams[0], measurements, exit_code) or ProcessFailed(
                    "model sureci sifirdan farkli koda dondu",
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
