"""The run's own record: the event journal and the findings artefact.

TWO DOCUMENTS THE STATE MECHANISM CANNOT CARRY, which is the only reason
this module exists rather than a third authority growing next to
`state.py`.

`state.json` is the frozen Phase A `STATE_SCHEMA`: one document, one
state, `additionalProperties: false`. It has no room for a SEQUENCE of
things that happened -- and the contract names `events.jsonl` in the
state directory, so the journal is a use of the design rather than a
change to it. It has no room for FINDINGS either: `mechanisms_seen`
carries mechanism ids and nothing else, so the class and the count a
locked audit returns would have nowhere to go but a field that means
something different.

WHAT MAY BE WRITTEN HERE, and the list is the enforcement rather than a
promise: a timestamp, a state, a closed event code, a closed stop
reason, closed severity and error-class vocabularies, counts, opaque or
identifier-shaped ids, and repo-relative tracked source paths. Both
schemas set `additionalProperties: false`, so a field nobody agreed to
is a refusal rather than a new column.

WHAT MAY NOT, and why the omission is deliberate rather than an
oversight: `claim` and `required_action` are the free-text fields a CODE
finding is allowed to carry TO THE IMPLEMENTER, and they are model
output. They travel in the repair PROMPT, which lives in memory for the
length of one call; they do not travel into a file that survives the
run. The contract's state-directory rule is that the loop writes about
the work and never the material, and a 500-character model-authored
sentence in the most frequently written artefact is exactly the hole
that rule names.

A LOCKED FINDING HAS NO TEXT FIELD AT ALL, so the boundary between the
two audit kinds is a TYPE here as well: the record schema below accepts
`error_class` and `case_count` from one kind and `file` and `line` from
the other, and nothing that could carry a passage from either.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from tools.agent_loop import contract, schemas, state as state_module

EVENTS_FILENAME = "events.jsonl"
FINDINGS_FILENAME = "findings.json"

# One line's ceiling. An event is counts and closed codes, so this is
# generous by two orders of magnitude -- it exists so a defect upstream
# cannot turn an append-only journal into a disk-filling loop.
MAX_EVENT_BYTES = 4096

_EVENT_VALIDATOR = Draft202012Validator(schemas.EVENT_SCHEMA)

_IDENTIFIER = {"type": "string", "pattern": contract.IDENTIFIER_PATTERN}

# A finding as the RUN records it. Note what is absent: every free-text
# field the model authored. See the module docstring for why.
_FINDING_RECORD = {
    "type": "object",
    "additionalProperties": False,
    "required": ["finding_id", "mechanism_id", "severity"],
    "properties": {
        "finding_id": _IDENTIFIER,
        "mechanism_id": _IDENTIFIER,
        "severity": {"enum": ["critical", "high", "medium", "low"]},
        # LOCKED audits only: a closed vocabulary and a counter.
        "error_class": {"enum": list(contract.ALL_LOCKED_FINDING_CLASSES)},
        "case_count": {"type": "integer", "minimum": 1, "maximum": 100000},
        # CODE audits only: tracked source the implementer can already
        # open. Borrowed from the frozen schema rather than re-spelled,
        # so one path grammar serves the whole loop.
        "file": schemas.CODE_AUDIT_RESULT_SCHEMA["properties"]["findings"][
            "items"]["properties"]["file"],
        "line": {"type": "integer", "minimum": 1},
    },
}

FINDINGS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["protocol_version", "run_id", "rounds"],
    "properties": {
        "protocol_version": {"const": contract.PROTOCOL_VERSION},
        "run_id": _IDENTIFIER,
        "rounds": {
            "type": "array",
            "maxItems": contract.DEFAULTS["max_evaluator_rounds"],
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["round", "state", "audit_kind", "status",
                             "findings"],
                "properties": {
                    "round": {"type": "integer", "minimum": 0, "maximum": 2},
                    "state": {"enum": [contract.State.AUDITING,
                                       contract.State.FINAL_AUDITING]},
                    "audit_kind": {"enum": list(contract.ALL_AUDIT_KINDS)},
                    "status": {"enum": list(contract.ALL_STATUSES)},
                    "summary_code": {
                        "enum": list(contract.ALL_SUMMARY_CODES)},
                    "findings": {"type": "array", "maxItems": 50,
                                 "items": _FINDING_RECORD},
                },
            },
        },
    },
}

# The fields a finding may be COPIED with, per audit kind. A record is
# built by naming these rather than by filtering the reply, because a
# filter answers "what did I remember to remove" and an allowlist
# answers "what did I agree to keep" -- and only the second one is still
# right when the reply grows a field.
CODE_RECORD_FIELDS = ("finding_id", "mechanism_id", "severity", "file",
                      "line")
LOCKED_RECORD_FIELDS = ("finding_id", "mechanism_id", "severity",
                        "error_class", "case_count")

RECORD_FIELDS = {
    contract.AuditKind.CODE: CODE_RECORD_FIELDS,
    contract.AuditKind.LOCKED: LOCKED_RECORD_FIELDS,
}


class JournalError(RuntimeError):
    """A record that may not be written, or could not be.

    Carries a fixed sentence chosen here. Never a path, never an OS
    message, never the payload that was refused."""


def events_path(state_dir) -> Path:
    """Derived, never supplied: a caller-chosen journal is a journal
    somewhere nobody else looks."""
    return Path(state_dir) / EVENTS_FILENAME


def findings_path(state_dir) -> Path:
    return Path(state_dir) / FINDINGS_FILENAME


def record_finding(finding, audit_kind) -> dict:
    """One reply finding, reduced to the fields this run may keep.

    The allowlist is chosen by the audit KIND, not by what the object
    happens to carry: a locked reply that somehow arrived with a `claim`
    would lose it here, and a code reply's `error_class` is not a field
    a code finding has."""
    allowed = RECORD_FIELDS.get(audit_kind)
    if allowed is None:
        raise JournalError("denetim turu sozlesmede yok")
    if not isinstance(finding, dict):
        raise JournalError("bulgu bir nesne degil")
    return {name: finding[name] for name in allowed if name in finding}


def _validate(payload, schema, what):
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as invalid:
        # the failing FIELD PATH, never the failing value: the value can
        # be model output and this text travels into reports
        where = "/".join(str(part) for part in invalid.absolute_path) or "kok"
        raise JournalError(f"{what} sema disi (alan: {where})") from None
    return payload


def append_event(state_dir, payload) -> Path:
    """Add one validated line to the journal.

    APPEND-ONLY and validated BEFORE the handle is opened: a record that
    the frozen event schema refuses must not exist half-written in a file
    whose whole value is that every line in it is readable.

    The line is canonical JSON, so two runs recording the same event
    produce the same bytes and a reader never has to guess at key
    order."""
    _validate(payload, schemas.EVENT_SCHEMA, "olay")
    line = schemas.canonical_json(payload)
    data = (line + "\n").encode("utf-8")
    if len(data) > MAX_EVENT_BYTES:
        raise JournalError("olay kaydi tavani asiyor")
    target = events_path(state_dir)
    state_module.ensure_directory(target.parent)
    try:
        with open(target, "ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        # the name of the file that refused is not text this module may
        # repeat, and the failure leaves the handler before it flies
        raise JournalError("olay gunlugune yazilamadi") from None
    return target


def write_findings(state_dir, payload) -> Path:
    """Replace the findings artefact, atomically and validated.

    Rewritten rather than appended because it is a SUMMARY of the rounds
    that have happened, and a half-written summary that still parses is
    worse than none: it would describe a run nobody had."""
    _validate(payload, FINDINGS_SCHEMA, "bulgu kaydi")
    return state_module.write_json_atomically(
        findings_path(state_dir), payload, FINDINGS_SCHEMA, "bulgu kaydi")


def read_findings(state_dir):
    """The artefact, or `None` when this run has not audited yet."""
    target = findings_path(state_dir)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise JournalError("bulgu kaydi okunamadi ya da bozuk") from None
    return _validate(payload, FINDINGS_SCHEMA, "bulgu kaydi")


def backup_path(state_dir) -> Path:
    """Where the last KNOWN-GOOD state document is kept.

    Beside the state file and under the same directory, so restoring it
    is a rename on one filesystem rather than a copy across two."""
    return Path(state_dir) / f"{state_module.STATE_FILENAME}.backup"


def save_state_backup(state_dir, payload) -> Path:
    """Keep the document that is about to be replaced.

    Written through the state module's own atomic writer and against the
    frozen state schema: a backup that does not validate is not a backup,
    it is a second corrupt file to choose between."""
    return state_module.write_json_atomically(
        backup_path(state_dir), payload, schemas.STATE_SCHEMA, "durum yedegi")


def restore_state_backup(state_dir):
    """Put the backup back as the live state, and return it.

    Read and VALIDATED first: restoring an unreadable backup over a
    corrupt state leaves the run with two broken documents and no way to
    tell which failure came first."""
    payload = state_module.read_json_checked(
        backup_path(state_dir), schemas.STATE_SCHEMA, "durum yedegi")
    state_module.write_state(state_dir, payload)
    return payload
