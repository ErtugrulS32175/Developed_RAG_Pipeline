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

from jsonschema import Draft202012Validator

from tools.agent_loop.contract import (
    ALL_AUDIT_KINDS,
    ALL_EVENT_CODES,
    ALL_LOCKED_FINDING_CLASSES,
    ALL_STOP_REASONS,
    ALL_SUMMARY_CODES,
    COMMAND_REGISTRY,
    CONTROL_PLANE_BLOCKED_PATHS,
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
_CONTROL_PLANE_RE = "^(" + "|".join(
    re.escape(prefix) for prefix in
    sorted(CONTROL_PLANE_BLOCKED_PATHS | set(CONTROL_PLANE_PATHS))
    if prefix) + ")"

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


# THE implementer binding. Built once at import, used by the argv
# builder and the adapter alike, so there is exactly one schema in
# flight and one hash that names it.
IMPLEMENTER_SCHEMA_BINDING = SchemaBinding(IMPLEMENTER_RESULT_SCHEMA)


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
