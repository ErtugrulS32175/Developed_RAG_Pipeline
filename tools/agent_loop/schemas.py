"""JSON Schemas for every message the loop reads or writes. PHASE A.

DATA ONLY. These are the boundary between a language model and a state
machine, so they are strict on purpose.

WHAT THE FIRST DRAFT GOT WRONG, because the shape of the mistake is
worth keeping:

  * It opened EVERY status to BOTH roles and then checked the role/status
    split in a Python constant. The constant was right and the schema
    was not, so `{"role": "implementer", "status": "approved"}` -- a
    model grading its own work -- validated cleanly. A rule that lives
    only next to the schema is not enforced by the schema.
  * Its path pattern rejected `../` and drive letters but accepted
    `tools\\..\\contracts\\x.md`, because it never looked at
    backslashes. Repo-relative paths use forward slashes; a backslash is
    now a rejection, not a separator to interpret.
  * `implementer` and `evaluator` were free-form objects, so a task
    could smuggle arbitrary CLI configuration -- including the very
    flags the contract forbids -- past validation.
  * Nothing tied `status` to `next_action`, to `findings` or to
    `stop_reason`. "changes_requested with no findings" and "blocked
    with no reason" both validated.

There is no `$schema` key. The dialect is chosen explicitly in code by
instantiating `Draft202012Validator`, so the URL carried no information
-- and it carried a four-digit year, which is a filename-fragment class
in this corpus and made the leak scanner fail closed. Removing a
redundant key beats teaching the scanner an exception.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import MappingProxyType

from jsonschema import Draft202012Validator

from tools.agent_loop.contract import (
    ALL_AUDIT_KINDS,
    ALL_EVENT_CODES,
    ALL_FAILURE_CODES,
    ALL_LOCKED_FINDING_CLASSES,
    ALL_ROLES,
    ALL_SCHEMA_FIELDS,
    ALL_SCHEMA_ISSUES,
    ALL_STOP_REASONS,
    ALL_SUMMARY_CODES,
    COMMAND_REGISTRY,
    CONTROL_PLANE_BLOCKED_PATHS,
    CONTROL_PLANE_GLOBS,
    CONTROL_PLANE_PATHS,
    DEFAULTS,
    IDENTIFIER_PATTERN,
    OPAQUE_ID_PATTERN,
    PROTOCOL_VERSION,
    AuditKind,
    Role,
    ROLE_STATUSES,
    State,
)

_ALL_STATES = sorted(
    value for name, value in vars(State).items()
    if not name.startswith("_") and isinstance(value, str))

# A path the RUNNING loop may never be given permission to edit. The
# task cannot widen this: `allowed_paths: ["tools/agent_loop/"]` would
# otherwise let the implementer rewrite the registry, the forbidden-flag
# list, the schemas and its own tests, then pass against the rules it
# had just written.
# Protected prefixes AND every directory above one. `tools/` was
# accepted because only the exact prefix was refused -- and `tools/`
# contains `tools/agent_loop/`. Permission over a parent is permission
# over the child.
#
# THREE RELATIONS, AND THEY ARE NOT THE SAME ONE (B5-R1). Joining every
# blocked entry into a single alternation anchored only at the start
# made the ANCESTOR `tests` match everything beneath it, so a manifest
# naming `tests/test_db_lifecycle.py` -- an ordinary test file, no part
# of this loop -- was refused with `path_not_allowed`. The runtime gate
# in `preflight` never had that defect: it compares ancestors by
# EQUALITY. Two layers answering the same question differently is a
# policy split nobody can see, so these rules are now spelled the way
# that gate spells them:
#
#   1. a protected prefix, itself and everything under it;
#   2. a broad ancestor, EXACTLY -- `tests`, not `tests/anything`;
#   3. the frozen glob family, so a test file invented tomorrow is
#      protected today.
#
# Derived from the contract's own constants, never a hand-kept list of
# today's exceptions: an entry added to `CONTROL_PLANE_PATHS` or
# `CONTROL_PLANE_GLOBS` closes here without anybody editing this file.


def _glob_to_regex(pattern):
    """The frozen glob family as a regex, FAIL-CLOSED.

    `*` becomes `.*` -- deliberately the same span `fnmatch` gives it,
    including across `/`, so the schema can never be more permissive
    than the runtime gate that uses `fnmatch`. Any other wildcard is
    unsupported and raises at import rather than being silently escaped
    into a literal, which would leave the family open."""
    out = []
    for character in pattern:
        if character == "*":
            out.append(".*")
        elif character in "?[]":
            raise ValueError(
                "kontrol duzlemi glob'u desteklenmeyen joker tasiyor")
        else:
            out.append(re.escape(character))
    return "".join(out)


def _control_plane_pattern():
    """One pattern, three relations, each anchored on its own terms."""
    prefixes = tuple(CONTROL_PLANE_PATHS)
    exact = {prefix.rstrip("/") for prefix in prefixes}
    ancestors = sorted(
        {entry.rstrip("/") for entry in CONTROL_PLANE_BLOCKED_PATHS
         if entry.rstrip("/")} - exact)
    parts = []
    for prefix in sorted(prefixes):
        # everything under it, and the prefix itself without its slash
        parts.append("^" + re.escape(prefix) + ".*$")
        parts.append("^" + re.escape(prefix.rstrip("/")) + "$")
    if ancestors:
        parts.append("^(" + "|".join(re.escape(entry) for entry in ancestors)
                     + ")/?$")
    if CONTROL_PLANE_GLOBS:
        parts.append("^(" + "|".join(_glob_to_regex(glob)
                                     for glob in CONTROL_PLANE_GLOBS) + ")$")
    return "|".join(parts)


_CONTROL_PLANE_RE = _control_plane_pattern()

_OPAQUE_ID = {"type": "string", "pattern": OPAQUE_ID_PATTERN}

# A repo-relative path. No absolute root, no drive letter, no parent
# escape, and NO BACKSLASH in any position -- `a\..\b` is the escape the
# first version let through.
_RELATIVE_PATH = {
    "type": "string",
    "minLength": 1,
    "maxLength": 400,
    "pattern": (r"^(?![/\\])(?![A-Za-z]:)(?!.*(^|/)\.\.(/|$))"
                r"[^\x00-\x1f\\]+$"),
}

_IDENTIFIER = {"type": "string", "pattern": IDENTIFIER_PATTERN}

_COMMAND_ID = {"enum": sorted(COMMAND_REGISTRY)}

_TEST_RESULT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["command_id", "exit_code", "passed", "failed", "skipped",
                 "duration_ms"],
    "properties": {
        "command_id": _COMMAND_ID,
        "exit_code": {"type": "integer", "minimum": -255, "maximum": 255},
        "passed": {"type": "integer", "minimum": 0},
        "failed": {"type": "integer", "minimum": 0},
        "skipped": {"type": "integer", "minimum": 0},
        "duration_ms": {"type": "integer", "minimum": 0},
    },
}

# A finding about TRACKED SOURCE. Free text is allowed here precisely
# because the implementer can already open the file it names.
_CODE_FINDING = {
    "type": "object",
    "additionalProperties": False,
    "required": ["finding_id", "mechanism_id", "severity", "claim",
                 "reproduction_result", "required_action"],
    "properties": {
        "finding_id": _IDENTIFIER,
        "mechanism_id": _IDENTIFIER,   # the second-patch rule keys on this
        "severity": {"enum": ["critical", "high", "medium", "low"]},
        "file": _RELATIVE_PATH,
        "line": {"type": "integer", "minimum": 1},
        "claim": {"type": "string", "minLength": 1, "maxLength": 500},
        "reproduction_result": {
            "enum": ["reproduced", "not_reproduced", "not_attempted"]},
        "required_action": {"type": "string", "minLength": 1,
                            "maxLength": 500},
    },
}

# A finding from the LOCKED holdout. No text field exists, so there is
# nowhere for a question, an answer or a passage to travel. Counters and
# a closed vocabulary only.
_LOCKED_FINDING = {
    "type": "object",
    "additionalProperties": False,
    "required": ["finding_id", "mechanism_id", "severity", "error_class",
                 "case_count"],
    "properties": {
        # opaque and fixed-length, minted by the runner before the call
        # and validated against the allowlist it handed out
        "finding_id": _OPAQUE_ID,
        "mechanism_id": _OPAQUE_ID,
        "severity": {"enum": ["critical", "high", "medium", "low"]},
        "error_class": {"enum": list(ALL_LOCKED_FINDING_CLASSES)},
        "case_count": {"type": "integer", "minimum": 1, "maximum": 100000},
    },
}


def _conditional_rules(statuses):
    """Tie status to next_action, findings and stop_reason.

    Without these, "changes_requested carrying no findings" and "blocked
    with no reason" both validate -- a refusal nobody can act on and a
    stop nobody can explain."""
    rules = []
    if "changes_requested" in statuses:
        rules.append({
            "if": {"properties": {"status": {"const": "changes_requested"}},
                   "required": ["status"]},
            "then": {"required": ["findings", "next_action"],
                     "properties": {
                         "findings": {"minItems": 1},
                         "next_action": {"const": "await_repair"}}},
        })
    if "approved" in statuses:
        rules.append({
            "if": {"properties": {"status": {"const": "approved"}},
                   "required": ["status"]},
            "then": {"properties": {"findings": {"maxItems": 0},
                                    "next_action": {"const": "stop"}}},
        })
    if "implemented" in statuses:
        rules.append({
            "if": {"properties": {"status": {"const": "implemented"}},
                   "required": ["status"]},
            "then": {"properties": {
                "next_action": {"const": "await_acceptance"}}},
        })
    for terminal in ("blocked", "failed"):
        rules.append({
            "if": {"properties": {"status": {"const": terminal}},
                   "required": ["status"]},
            "then": {"required": ["stop_reason"],
                     "properties": {"next_action": {"const": "stop"}}},
        })
    return rules


def _result_schema(role, finding_schema, *, audit_kind=None,
                   textless=False):
    """`textless` closes the ENVELOPE, not just the findings.

    The locked findings were made textless and the envelope was not: a
    required 2000-character `summary` sat right beside them, and a
    passage fits in it comfortably. `stop_reason` was free text too, and
    `run_id`/`finding_id` were model-chosen slugs -- a slug is free text
    on a short leash, and "gizli-belge-adi" is a valid one. A textless
    result carries a CODE where the prose was and opaque fixed-length
    ids the runner minted itself."""
    statuses = list(ROLE_STATUSES[role])
    identifier = _OPAQUE_ID if textless else _IDENTIFIER
    properties = {
        "protocol_version": {"const": PROTOCOL_VERSION},
        "run_id": identifier,
        # `const`, not `enum`: the role is asserted by the schema the
        # runner chose, so a reply cannot claim to be the other actor
        "role": {"const": role},
        # per-role statuses, IN THE SCHEMA. The Python constant is the
        # documentation; this is the enforcement.
        "status": {"enum": statuses},
        "changed_files": {"type": "array", "maxItems": 200,
                          "items": _RELATIVE_PATH},
        "tests": {"type": "array", "maxItems": 50, "items": _TEST_RESULT},
        "findings": {"type": "array", "maxItems": 50,
                     "items": finding_schema},
        "next_action": {
            "enum": ["await_acceptance", "await_audit", "await_repair",
                     "await_final_audit", "stop"]},
        # An ENUM, never a sentence: a stop reason the runner cannot
        # match against its own vocabulary is a stop nobody can act on,
        # and a 120-character string is 120 characters of document.
        "stop_reason": {"enum": list(ALL_STOP_REASONS)},
    }
    if textless:
        properties["summary_code"] = {"enum": list(ALL_SUMMARY_CODES)}
        summary_field = "summary_code"
    else:
        properties["summary"] = {"type": "string", "minLength": 1,
                                 "maxLength": 2000}
        summary_field = "summary"
    required = ["protocol_version", "run_id", "role", "status",
                summary_field, "next_action"]
    if audit_kind is not None:
        properties["audit_kind"] = {"const": audit_kind}
        required.append("audit_kind")
    if role == Role.EVALUATOR:
        # an evaluator that reports edited files has edited files
        properties.pop("changed_files")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
        "allOf": _conditional_rules(statuses),
    }


IMPLEMENTER_RESULT_SCHEMA = _result_schema(Role.IMPLEMENTER, _CODE_FINDING)
CODE_AUDIT_RESULT_SCHEMA = _result_schema(
    Role.EVALUATOR, _CODE_FINDING, audit_kind=AuditKind.CODE)
LOCKED_AUDIT_RESULT_SCHEMA = _result_schema(
    Role.EVALUATOR, _LOCKED_FINDING, audit_kind=AuditKind.LOCKED,
    textless=True)

EVALUATOR_RESULT_SCHEMAS = {
    AuditKind.CODE: CODE_AUDIT_RESULT_SCHEMA,
    AuditKind.LOCKED: LOCKED_AUDIT_RESULT_SCHEMA,
}

# A model configuration a task may express. Deliberately tiny: anything
# richer becomes a way to smuggle CLI flags past validation, which is
# what a free-form object was.
_MODEL_CONFIG = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"model": {"type": "string", "maxLength": 60,
                             "pattern": r"^[a-z0-9][a-z0-9._-]{1,59}$"}},
}

_COMMAND_REFERENCE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["command_id"],
    "properties": {
        # NAMED, never spelled. The argv lives in the code registry.
        "command_id": _COMMAND_ID,
        "paths": {"type": "array", "maxItems": 20, "items": _RELATIVE_PATH},
    },
}

TASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "protocol_version", "objective", "baseline_sha", "allowed_paths",
        "forbidden_paths", "acceptance_commands", "acceptance_criteria",
        "max_implementation_rounds", "max_repair_rounds",
        "max_wall_clock_minutes", "max_budget_usd", "max_output_bytes",
        "leak_policy", "dirty_tree_allowlist",
    ],
    "properties": {
        "protocol_version": {"const": PROTOCOL_VERSION},
        "objective": {"type": "string", "minLength": 1, "maxLength": 2000},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        # A task may NOT grant edit rights over the loop's own code.
        # `allowed_paths: ["tools/agent_loop/"]` would let the
        # implementer rewrite the registry, the forbidden-flag list, the
        # schemas and its own tests, then pass against the rules it had
        # just written. Refused here AND re-checked by hashing the
        # control plane around every implementer call.
        "allowed_paths": {
            "type": "array", "minItems": 1, "maxItems": 100,
            "items": {**_RELATIVE_PATH,
                      "not": {"pattern": _CONTROL_PLANE_RE}}},
        "forbidden_paths": {"type": "array", "maxItems": 100,
                            "items": _RELATIVE_PATH},
        "acceptance_commands": {"type": "array", "minItems": 1,
                                "maxItems": 20, "items": _COMMAND_REFERENCE},
        "acceptance_criteria": {"type": "array", "minItems": 1, "maxItems": 50,
                                "items": {"type": "string", "maxLength": 500}},
        "max_implementation_rounds": {"type": "integer", "minimum": 1,
                                      "maximum": 1},
        "max_repair_rounds": {"type": "integer", "minimum": 0, "maximum": 1},
        "max_wall_clock_minutes": {"type": "integer", "minimum": 1,
                                   "maximum": 480},
        "max_budget_usd": {"type": "number", "minimum": 0, "maximum": 100},
        "max_output_bytes": {"type": "integer", "minimum": 1024,
                             "maximum": 4194304},
        "model_call_timeout_seconds": {"type": "integer", "minimum": 30,
                                       "maximum": 7200},
        # NOTE: there is no `user_gates` property. The human gates are a
        # frozen contract constant, not configuration -- a task that
        # could set `user_gates: []` could authorise its own commit.
        "leak_policy": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command_id", "max_hard_findings"],
            "properties": {
                "command_id": _COMMAND_ID,
                "max_hard_findings": {"const": 0},
            },
        },
        "dirty_tree_allowlist": {"type": "array", "maxItems": 50,
                                 "items": _RELATIVE_PATH},
        "implementer": _MODEL_CONFIG,
        "evaluator": _MODEL_CONFIG,
    },
}

STATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "run_id", "state", "started_at",
                 "updated_at", "rounds", "budget"],
    "properties": {
        "protocol_version": {"const": PROTOCOL_VERSION},
        "run_id": _IDENTIFIER,
        # an ENUM: `state: "uydurma_basarili"` validated cleanly before,
        # which is a run that can invent its own successful ending
        "state": {"enum": _ALL_STATES},
        "started_at": {"type": "string", "maxLength": 40},
        "updated_at": {"type": "string", "maxLength": 40},
        "heartbeat_at": {"type": "string", "maxLength": 40},
        "baseline_sha": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
        "rounds": {
            "type": "object",
            "additionalProperties": False,
            "required": ["implementation", "repair", "evaluator"],
            # capped at the contract's own limits: 999 rounds validated
            "properties": {
                "implementation": {
                    "type": "integer", "minimum": 0,
                    "maximum": DEFAULTS["max_implementation_rounds"]},
                "repair": {"type": "integer", "minimum": 0,
                           "maximum": DEFAULTS["max_repair_rounds"]},
                "evaluator": {"type": "integer", "minimum": 0,
                              "maximum": DEFAULTS["max_evaluator_rounds"]},
            },
        },
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "required": ["max_usd", "spent_usd"],
            "properties": {
                "max_usd": {"type": "number", "minimum": 0},
                "spent_usd": {"type": "number", "minimum": 0},
            },
        },
        "mechanisms_seen": {
            "type": "array", "maxItems": 100,
            "items": {"type": "array", "maxItems": 50, "items": _IDENTIFIER}},
        "stop_reason": {"enum": list(ALL_STOP_REASONS)},
    },
    # A terminal state carries exactly one reason, and a state that is
    # still running carries none. `approved` with no stop_reason
    # validated before, which contradicts the contract it is supposed to
    # record.
    "allOf": [
        {"if": {"properties": {"state": {"const": State.APPROVED}},
                "required": ["state"]},
         "then": {"required": ["stop_reason"],
                  "properties": {"stop_reason": {"const": "completed"}}}},
        {"if": {"properties": {"state": {"enum": [State.BLOCKED,
                                                  State.FAILED]}},
                "required": ["state"]},
         "then": {"required": ["stop_reason"]}},
        {"if": {"properties": {
            "state": {"enum": [s for s in _ALL_STATES
                               if s not in (State.APPROVED, State.BLOCKED,
                                            State.FAILED)]}},
            "required": ["state"]},
         "then": {"not": {"required": ["stop_reason"]}}},
    ],
}

# `spent_usd <= max_usd` relates two fields, which JSON Schema cannot
# express here, so it is a RUNNER invariant (contract.BUDGET_INVARIANT):
# checked after every model call and asserted before every new one.
# Recorded next to the schema so the gap is visible rather than assumed.

EVENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ts", "run_id", "event"],
    "properties": {
        "ts": {"type": "string", "maxLength": 40},
        "run_id": _IDENTIFIER,
        # a CLOSED vocabulary plus numbers. `detail` was a 500-character
        # free string in a file written on every single step -- a
        # document-sized hole in the most frequently touched artefact.
        "event": {"enum": list(ALL_EVENT_CODES)},
        "state": {"enum": _ALL_STATES},
        "command_id": _COMMAND_ID,
        "exit_code": {"type": "integer"},
        "duration_ms": {"type": "integer", "minimum": 0},
        "bytes_truncated": {"type": "integer", "minimum": 0},
        # B4-R4: WHICH mechanism failed, beside the fact that one did.
        # All closed: a code from the frozen vocabulary, a role from the
        # frozen pair, and numbers the lower layer measured itself.
        # `additionalProperties: false` above still bounds the record, so
        # nothing outside this list can ever reach the journal.
        "failure_code": {"enum": list(ALL_FAILURE_CODES)},
        "role": {"enum": list(ALL_ROLES)},
        "stdout_bytes": {"type": "integer", "minimum": 0},
        "stderr_bytes": {"type": "integer", "minimum": 0},
        "cleanup_complete": {"type": "boolean"},
        # B5-R3: which schema rule refused a reply, and at which
        # DECLARED field. Closed enums, and optional -- an event that is
        # not a schema violation carries neither, which is why they are
        # absent from `required` rather than nullable.
        "schema_issue": {"enum": list(ALL_SCHEMA_ISSUES)},
        "schema_field": {"enum": list(ALL_SCHEMA_FIELDS)},
    },
}

ALL_SCHEMAS = {
    "task": TASK_SCHEMA,
    "state": STATE_SCHEMA,
    "event": EVENT_SCHEMA,
    "implementer_result": IMPLEMENTER_RESULT_SCHEMA,
    "code_audit_result": CODE_AUDIT_RESULT_SCHEMA,
    "locked_audit_result": LOCKED_AUDIT_RESULT_SCHEMA,
}

ALL_AUDIT_KINDS = ALL_AUDIT_KINDS  # re-exported for the tests' convenience


def canonical_json(document) -> str:
    """ONE spelling for a JSON document, so equal documents have equal
    bytes and equal bytes have equal hashes.

    The exact rules are the contract: `sort_keys=True` (insertion order
    must not change the hash), `separators=(",", ":")` (compact, no
    whitespace to disagree about), `ensure_ascii=True` (no encoding
    ambiguity), `allow_nan=False` (NaN/Infinity are not JSON and would
    otherwise serialize into something no parser owes us a meaning
    for)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


class SchemaBinding:
    """A schema frozen into canonical bytes, their SHA-256, and a
    validator parsed back from THOSE SAME BYTES.

    R2B exists because the schema handed to the Claude CLI and the
    schema the adapter validated with were two unrelated things: the
    CLI got a file path (the flag takes INLINE JSON, so the path was
    wrong twice over), the validator got a live module dictionary, and
    rewriting the file to garbage between building the argv and running
    it changed nothing anybody checked.

    The validator here is constructed from `json.loads` of the
    canonical text -- a fresh object -- so the mutable source
    dictionary stops being a live authority the moment the binding is
    built. Construction is fail-closed: the canonical text must
    round-trip to the source document exactly, and the document must
    satisfy `Draft202012Validator.check_schema`.

    FROZEN THROUGH THE NORMAL API. The first version's fields were
    ordinary writable attributes, and an evaluator probe rewrote the
    canonical text and the hash TOGETHER: the substitute entered the
    argv, the hash comparison agreed with itself, and a permissive
    validator accepted an arbitrary reply. Attribute assignment and
    deletion now refuse. What this does NOT claim: `object.__setattr__`
    still exists in the language -- a hostile actor with code execution
    in this process is outside the threat model, and the refusal below
    is for the normal API, exactly as stated."""

    __slots__ = ("canonical_json", "canonical_bytes", "sha256", "_validator")

    def __init__(self, schema):
        text = canonical_json(schema)
        parsed = json.loads(text)
        if parsed != schema:
            raise ValueError("kanonik gosterim kaynak semayla ayni degil")
        Draft202012Validator.check_schema(parsed)
        object.__setattr__(self, "canonical_json", text)
        object.__setattr__(self, "canonical_bytes", text.encode("utf-8"))
        object.__setattr__(self, "sha256",
                           hashlib.sha256(self.canonical_bytes).hexdigest())
        object.__setattr__(self, "_validator", Draft202012Validator(parsed))

    def __setattr__(self, name, value):
        raise AttributeError("baglama donduruldu; alan yeniden yazilamaz")

    def __delattr__(self, name):
        raise AttributeError("baglama donduruldu; alan silinemez")

    def validate(self, payload):
        """Raises `jsonschema.ValidationError` exactly like the raw
        validator would; carries no schema text into the error path."""
        self._validator.validate(payload)


# THE implementer binding: the ACCEPTANCE authority. Built once at
# import; every reply is finally judged by this and by nothing else.
IMPLEMENTER_SCHEMA_BINDING = SchemaBinding(IMPLEMENTER_RESULT_SCHEMA)

# The same object under the name that says what it IS, now that a
# second schema exists and the difference between them is the point.
AUTHORITATIVE_RESULT_SCHEMA = IMPLEMENTER_RESULT_SCHEMA


# ---------------------------------------------------------------------
# B4-R2 -- THE TRANSPORT SCHEMA
# ---------------------------------------------------------------------
#
# WHAT CHANGED AND WHY, because this replaces an invariant rather than
# adding to one. R2B established "the schema on the argv and the schema
# that validates the reply are the same bytes", and that invariant
# quietly assumed the API would accept the same rich schema the
# acceptance gate needs. It does not. Two authorized diagnostics against
# the real CLI both ended in a 4xx client error with
# `terminal_reason: api_error`, and the published structured-output
# contract refuses `pattern`, `if`/`then`, `minLength`/`maxLength`,
# `minimum`/`maximum` and array constraints beyond `minItems` -- of
# which this schema uses 28 occurrences.
#
# The replacement invariant, stated so nobody has to infer it:
#
#     argv schema constrains generation;
#     authoritative local schema decides acceptance;
#     both exact canonical bindings are independently hash-pinned.
#
# THE TRANSPORT SCHEMA IS NOT AN ACCEPTANCE AUTHORITY. It is strictly
# weaker, on purpose, and nothing validates a reply with it. A payload
# that satisfies it and fails `IMPLEMENTER_SCHEMA_BINDING` is refused --
# there is no road in this package on which the weaker schema decides.

class TransportSchemaError(ValueError):
    """A schema this package cannot honestly reduce for transport.

    Raised at IMPORT, deliberately: an unrepresentable schema is a
    defect to learn about when the module loads, not at the first
    billable call."""


# The subset the published structured-output contract accepts, spelled
# as an EXPLICIT ALLOWLIST rather than discovered by trial.
CLAUDE_TRANSPORT_KEYWORDS = frozenset({
    "type", "properties", "items", "required", "additionalProperties",
    "enum", "const", "allOf", "anyOf",
})

# The keywords this schema uses that the subset refuses. Named ONE BY
# ONE. A generic recursive sanitizer would drop whatever it did not
# recognise, so a constraint added tomorrow would vanish in silence and
# the model would be generating against less than its author wrote --
# anything neither supported nor listed here RAISES.
CLAUDE_TRANSPORT_DROPPED = frozenset({
    "pattern", "minLength", "maxLength", "minimum", "maximum",
    "minItems", "maxItems", "if", "then", "else",
})


def _transport_node(node, *, supported, dropped):
    """One schema node, reduced to the supported subset.

    STRUCTURE-AWARE, not a blind key walk: under `properties` the keys
    are property NAMES chosen by this contract, and a blind walk would
    have tried to classify `stop_reason` as a JSON Schema keyword.

    THE POLICY IS A PARAMETER (B4-R11), because there are two providers
    and they do not accept the same subset. It is not a default: a
    caller who forgets to say which provider gets a `TypeError` rather
    than silently the other one's rules.

    Dropped constraints are NOT restated as `description` prose the way
    the vendor SDKs do it. This boundary exists to keep free text out,
    and a description is free text travelling to a model."""
    if not isinstance(node, dict):
        raise TransportSchemaError("sema dugumu bir nesne degil")
    reduced = {}
    for key, value in node.items():
        if key in dropped:
            continue
        if key not in supported:
            # the KEYWORD, which is this package's own vocabulary --
            # never a value, which could be model-facing content
            raise TransportSchemaError(f"siniflandirilmamis anahtar: {key}")
        if key == "properties":
            reduced[key] = {name: _transport_node(sub, supported=supported,
                                                  dropped=dropped)
                            for name, sub in value.items()}
        elif key == "items":
            reduced[key] = _transport_node(value, supported=supported,
                                           dropped=dropped)
        elif key in ("allOf", "anyOf"):
            # A branch whose whole content was unsupported becomes an
            # EMPTY schema, which constrains nothing and would only add
            # depth for the API to compile. Empty branches are removed,
            # and a combinator left with none goes with them.
            branches = [_transport_node(sub, supported=supported,
                                        dropped=dropped) for sub in value]
            branches = [branch for branch in branches if branch]
            if branches:
                reduced[key] = branches
        elif key == "additionalProperties":
            # the subset accepts this keyword ONLY as `false`, and
            # `false` is exactly what closes the object
            if value is not False:
                raise TransportSchemaError(
                    "additionalProperties yalnizca false olabilir")
            reduced[key] = False
        elif key == "enum":
            if any(isinstance(member, (dict, list)) for member in value):
                raise TransportSchemaError("enum karmasik tur tasiyor")
            reduced[key] = list(value)
        elif key == "required":
            reduced[key] = list(value)
        else:                                    # type, const
            reduced[key] = value
    return reduced


def claude_transport_schema(schema):
    """The API-compatible copy of a schema. Deterministic and pure: the
    source document is never mutated, and equal sources give equal
    canonical bytes."""
    return _transport_node(schema, supported=CLAUDE_TRANSPORT_KEYWORDS,
                           dropped=CLAUDE_TRANSPORT_DROPPED)


CLAUDE_TRANSPORT_SCHEMA = claude_transport_schema(AUTHORITATIVE_RESULT_SCHEMA)

# The SECOND binding, hash-pinned independently of the first. It is the
# only schema that travels on an argv.
IMPLEMENTER_TRANSPORT_BINDING = SchemaBinding(CLAUDE_TRANSPORT_SCHEMA)


def locked_audit_schema(*, run_id, issued_finding_ids, issued_mechanism_ids):
    """The locked schema BOUND to the ids this call actually issued.

    The static pattern proves shape only: any 32 hex characters satisfy
    it, including ids the runner never minted -- so "opaque" was a
    promise about a format rather than a binding to a run. Here `run_id`
    is pinned with `const` and each id field becomes an `enum` of
    exactly what was handed out, which turns an unissued id into a
    schema violation instead of a new case to act on."""
    schema = json.loads(json.dumps(LOCKED_AUDIT_RESULT_SCHEMA))
    schema["properties"]["run_id"] = {"const": run_id}
    finding = schema["properties"]["findings"]["items"]["properties"]
    finding["finding_id"] = {"enum": sorted(issued_finding_ids)}
    finding["mechanism_id"] = {"enum": sorted(issued_mechanism_ids)}
    return schema


# ---------------------------------------------------------------------
# B4-R11 -- THE CODEX TRANSPORT SCHEMA
# ---------------------------------------------------------------------
#
# MEASURED, not inferred. The first run that reached `auditing` failed
# every time with the evaluator's own session record naming the cause:
#
#     status 400, invalid_request_error, code `invalid_json_schema`,
#     param `text.format.schema`, message: Invalid schema for
#     response_format 'codex_output_schema': In context=(), 'allOf' is
#     not permitted.
#
# `context=()` is the ROOT, and the root of every audit schema is where
# `_conditional_rules` puts its `allOf`. So the evaluator road had been
# sending the acceptance authority straight to the API since it existed,
# while the implementer road has sent a derived transport schema since
# B4-R2. This closes that asymmetry with the SAME pure helper rather
# than a second copy of it -- one difference, spelled once:
#
#     Claude accepts `allOf`;  Codex does not.
#
# THE INVARIANT IS UNCHANGED, and it is worth restating because this is
# the road where an evaluator's verdict decides whether a candidate is
# applied:
#
#     argv schema constrains generation;
#     authoritative local schema decides acceptance.
#
# Nothing validates a reply with a transport schema. A payload that
# satisfies the transport copy and fails the authoritative binding is
# refused, which is exactly what the conditional rules stripped here are
# for: `approved` carrying findings, `changes_requested` carrying none.
CODEX_TRANSPORT_KEYWORDS = CLAUDE_TRANSPORT_KEYWORDS - {"allOf"}
CODEX_TRANSPORT_DROPPED = CLAUDE_TRANSPORT_DROPPED | {"allOf"}


def _primitive_type(value):
    """The JSON type of a literal, EXACTLY.

    `bool` is checked before `int` because it is a subclass of it in
    Python and nowhere else: `True` typed as `integer` would tell the
    provider to generate a number where the contract wants a boolean.
    A non-finite float is not JSON at all."""
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        if value != value or value in (float("inf"), float("-inf")):
            raise TransportSchemaError("sonlu olmayan sayi JSON degil")
        return "number"
    if type(value) is str:
        return "string"
    raise TransportSchemaError("ilkel olmayan sema degeri")


def _scalar_type(node):
    """The explicit type of a leaf node, declared or INFERRED.

    The provider refuses a property schema with no `type` -- measured:
    400 `invalid_json_schema` at `('properties', 'audit_kind')`, which
    this contract spells as a bare `const`. Inference is closed and
    fail-shut: `const` gives its literal's type, `enum` gives the one
    type all its members share, and anything else raises rather than
    guessing a type the model would then generate against."""
    declared = node.get("type")
    if type(declared) is str:
        return declared
    if declared is not None:
        raise TransportSchemaError("tip gosterimi metin degil")
    if "const" in node:
        return _primitive_type(node["const"])
    if "enum" in node:
        kinds = {_primitive_type(member) for member in node["enum"]}
        kinds.discard("null")
        if len(kinds) != 1:
            raise TransportSchemaError("enum tek bir ilkel tip tasimiyor")
        return kinds.pop()
    raise TransportSchemaError("alt sema icin tip cikarilamiyor")


def _nullable(node, base):
    """An OPTIONAL field, expressed the way the strict subset allows.

    The subset has no optional properties: everything is required, and
    absence is spelled as `null`. So an authoritative optional field
    keeps its shape and gains `null` -- and a `const` becomes a
    two-member `enum`, because a `const` that also accepts `null` is a
    contradiction the provider would be right to refuse."""
    node["type"] = [base, "null"]
    if "const" in node:
        node["enum"] = [node.pop("const"), None]
    elif "enum" in node and None not in node["enum"]:
        node["enum"] = list(node["enum"]) + [None]
    return node


def _strict_node(node, *, optional):
    """One reduced node, made to satisfy the strict subset.

    THREE RULES, applied together because the provider applies them
    together: every node carries an explicit `type`; every object
    requires ALL of its properties and closes itself with
    `additionalProperties: false`; and a property the AUTHORITY treats
    as optional becomes required-but-nullable here.

    The authority's own `required` list is what decides optionality --
    it survived the keyword reduction, so this pass never has to guess
    which fields were optional, and `elide_optional_nulls` reads the
    same list when the answer comes back."""
    if not isinstance(node, dict):
        raise TransportSchemaError("sema dugumu bir nesne degil")
    strict = dict(node)
    if "properties" in strict:
        zorunlu = set(strict.get("required", ()))
        strict["properties"] = {
            name: _strict_node(sub, optional=name not in zorunlu)
            for name, sub in strict["properties"].items()}
        # ALL of them, in the source's own order so the bytes stay
        # deterministic, and the object closed against anything else
        strict["required"] = list(strict["properties"])
        strict["additionalProperties"] = False
        base = "object"
    elif "items" in strict:
        # an array's ITEMS are never optional: absence is expressed by
        # the array being empty or null, not by a hole inside it
        strict["items"] = _strict_node(strict["items"], optional=False)
        base = "array"
    else:
        base = _scalar_type(strict)
    if optional:
        return _nullable(strict, base)
    strict["type"] = base
    return strict


def codex_transport_schema(schema):
    """The Codex-compatible copy of an audit schema.

    Deterministic and pure, exactly like its Claude twin: the source
    document is never mutated and equal sources give equal canonical
    bytes. It is derived from whatever authoritative schema this call
    will be judged by -- for a LOCKED audit that is the schema already
    BOUND to the ids the runner issued, so the run id and the issued
    ids travel with it, now carrying explicit types. Deriving a locked
    transport from the static schema instead would hand the model a
    document that accepts ids nobody minted.

    TWO PASSES, in this order: drop what the subset cannot express,
    then make what remains satisfy the rest of the contract. Reversing
    them would infer types for keywords about to be discarded."""
    reduced = _transport_node(schema, supported=CODEX_TRANSPORT_KEYWORDS,
                              dropped=CODEX_TRANSPORT_DROPPED)
    strict = _strict_node(reduced, optional=False)
    # THE ROOT IS NEVER NULLABLE and never a union: the reply is an
    # object or it is nothing, and a root that could be `null` would
    # make "no answer" a valid answer.
    if strict.get("type") != "object":
        raise TransportSchemaError("tasima semasinin koku nesne degil")
    return strict


# ---------------------------------------------------------------------
# B4-R17 -- A FIELD THE MODEL SHOULD NEVER HAVE BEEN ASKED FOR
# ---------------------------------------------------------------------
#
# MEASURED on the first evaluator call that ever reached the model: the
# reply was contract-shaped in every other respect and failed on ONE
# rule -- `status: approved` paired with a `next_action` that was not
# `stop`. That pairing is not a judgement the evaluator makes; it is a
# consequence of the verdict, spelled out four times in the authority's
# own conditional rules.
#
# The strict transport subset cannot express a conditional, so the rule
# was unenforceable at generation time BY CONSTRUCTION. Rather than ask
# the model for a value it cannot be constrained to get right, the
# evaluator road stops asking: `next_action` leaves the transport
# schema and the adapter derives it from `status`.
#
# WHAT THIS IS NOT. It is not a relaxation: the authority still requires
# `next_action` and still refuses every wrong pairing, and the derived
# value is validated by it like any other. A projection that produced
# the wrong action would be caught by the same rule that caught the
# model.
#
# The values are READ FROM the authority, never invented: each entry
# below is the `const` inside one `allOf/*/then/properties/next_action`
# branch, and a test pins the table against those branches.
EVALUATOR_NEXT_ACTION = MappingProxyType({
    "approved": "stop",
    "changes_requested": "await_repair",
    "blocked": "stop",
    "failed": "stop",
})

# The fields the EVALUATOR road derives rather than receives. Named as a
# constant so the projection and the schema reduction cannot drift: one
# removes them from what is asked for, the other puts them back.
DERIVED_EVALUATOR_FIELDS = ("next_action",)


class ProjectionError(ValueError):
    """A reply the derivation cannot honestly complete.

    Raised for an unknown status and for a reply that supplied a
    derived field itself -- never for a value this package can fix by
    guessing, because there is no such value."""


def _without_derived(schema):
    """A COPY of an authoritative schema with the derived fields gone.

    A copy, and through JSON, so nothing here can reach the authority:
    it is a module-level dictionary and one in-place edit would remove
    `next_action` from the acceptance gate for the rest of the
    process."""
    document = json.loads(json.dumps(schema))
    for name in DERIVED_EVALUATOR_FIELDS:
        document.get("properties", {}).pop(name, None)
        if name in document.get("required", ()):
            document["required"] = [entry for entry in document["required"]
                                    if entry != name]
    return document


def evaluator_transport_schema(authoritative):
    """The transport copy for the EVALUATOR road.

    The strict subset, MINUS the fields the adapter derives. This is a
    deliberate second step rather than a change to
    `codex_transport_schema`: that helper serves every caller and must
    keep meaning "the same document, expressed in what the provider
    accepts". Dropping a property is a different claim, and it belongs
    to the road that knows how to put the property back."""
    return codex_transport_schema(_without_derived(authoritative))


def project_derived_fields(payload):
    """The reply, with the derived fields filled in from `status`.

    PURE: the argument is not mutated and a new document is returned.
    The order this sits in matters and is fixed by the caller -- derive,
    then elide transport nulls, then let the AUTHORITY judge the
    result. Nothing here decides whether a reply is acceptable.

    A reply that supplies a derived field itself is REFUSED rather than
    overwritten: silently replacing it would hide a model that had been
    told not to send it, and the refusal is the only way anybody learns
    the instruction stopped working."""
    if type(payload) is not dict:
        raise ProjectionError("yanit bir JSON nesnesi degil")
    for name in DERIVED_EVALUATOR_FIELDS:
        if name in payload:
            raise ProjectionError("yanit turetilen bir alani kendisi yazdi")
    status = payload.get("status")
    if type(status) is not str or status not in EVALUATOR_NEXT_ACTION:
        raise ProjectionError("durum kapali degerlendirici kumesinde degil")
    return dict(payload, next_action=EVALUATOR_NEXT_ACTION[status])


def elide_optional_nulls(schema, payload):
    """The reply, with transport `null`s for OPTIONAL fields removed.

    WHY THIS EXISTS. The strict subset has no optional properties, so
    the transport copy asks for every field and lets the absent ones be
    `null`. The authority spells the same thing as ABSENCE. Without
    this seam the two documents would disagree about every optional
    field, and the fix would have been to loosen the authority -- which
    is the one thing this split exists to avoid.

    NARROW ON PURPOSE, and every exclusion is a refusal the authority
    still gets to make: a `null` under a REQUIRED name stays (the
    authority rejects it), a `null` under a name the schema does not
    declare stays (`additionalProperties: false` rejects it), and no
    non-null value is ever touched -- `"null"`, `0`, `False`, `[]` and
    `{}` are values somebody meant. Pure: neither argument is mutated
    and equal inputs give equal output."""
    if isinstance(payload, dict) and isinstance(schema, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        zorunlu = set(schema.get("required", ()))
        temiz = {}
        for key, value in payload.items():
            if key in properties and key not in zorunlu and value is None:
                continue
            alt = properties.get(key)
            temiz[key] = (elide_optional_nulls(alt, value)
                          if isinstance(alt, dict) else value)
        return temiz
    if isinstance(payload, list) and isinstance(schema, dict):
        items = schema.get("items")
        if isinstance(items, dict):
            return [elide_optional_nulls(items, entry) for entry in payload]
        return list(payload)
    return payload


# The CODE audit's transport copy: static, so it is derived once and
# hash-pinned here rather than rebuilt per call. Built through the
# EVALUATOR projection (B4-R17), so `next_action` is absent from what
# the model is asked for and present in what the authority demands.
CODE_AUDIT_TRANSPORT_SCHEMA = evaluator_transport_schema(
    CODE_AUDIT_RESULT_SCHEMA)

# The two bindings of the CODE road, pinned INDEPENDENTLY. The first is
# the acceptance authority and never leaves this process; the second is
# the only one whose bytes reach a file on an argv.
CODE_AUDIT_SCHEMA_BINDING = SchemaBinding(CODE_AUDIT_RESULT_SCHEMA)
CODE_AUDIT_TRANSPORT_BINDING = SchemaBinding(CODE_AUDIT_TRANSPORT_SCHEMA)
