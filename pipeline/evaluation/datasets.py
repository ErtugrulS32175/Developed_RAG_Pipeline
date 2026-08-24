"""Pure, fail-closed authority for versioned evaluation datasets.

The existing evaluation runners consume five fields: ``q``, ``key``,
``answer``, ``pages`` and ``type``.  A governed dataset adds one random UUIDv4
``case_key`` to that exact shape.  This module validates that document before
storage, emits a deterministic JSON representation, and can project the
validated case back to the legacy runner shape without weakening either
contract.

No exception contains an offered value.  Dataset material may contain user or
document text, so diagnostics are deliberately limited to closed error codes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import uuid


CASE_TYPES = frozenset({"metin", "sayisal", "tablo"})
CASE_KEYS = frozenset({"case_key", "q", "key", "answer", "pages", "type"})
VERSION_KEYS = frozenset({"version", "cases"})

MIN_CASES = 1
MAX_CASES = 500
MAX_VERSION = 2_147_483_647
MAX_JSON_BYTES = 16 * 1024 * 1024
TEXT_BYTE_LIMITS = MappingProxyType({
    "q": 4096,
    "key": 16_384,
    "answer": 16_384,
})


class EvalDatasetError(ValueError):
    """An untrusted dataset failed a closed contract gate."""


def _refuse(code: str):
    raise EvalDatasetError(code)


def _text(value, field: str) -> str:
    if type(value) is not str:
        _refuse("eval_text_invalid")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        _refuse("eval_text_invalid")
    if (not encoded or len(encoded) > TEXT_BYTE_LIMITS[field]
            or value != value.strip()):
        _refuse("eval_text_invalid")
    # JSON permits C0 controls after escaping, but accepting them gives logs
    # and exported fixtures two different visual interpretations.
    if any(ord(char) < 0x20 for char in value):
        _refuse("eval_text_invalid")
    return value


def _case_key(value) -> str:
    if type(value) is not str:
        _refuse("eval_case_key_invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        _refuse("eval_case_key_invalid")
    if parsed.version != 4 or value != str(parsed):
        _refuse("eval_case_key_invalid")
    return value


def _pages(value) -> tuple[int, ...]:
    if type(value) is not list or not value:
        _refuse("eval_pages_invalid")
    result = []
    previous = 0
    for page in value:
        if type(page) is not int or page <= 0 or page > MAX_VERSION:
            _refuse("eval_pages_invalid")
        if page <= previous:
            _refuse("eval_order_invalid")
        result.append(page)
        previous = page
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_key: str
    q: str
    key: str
    answer: str
    pages: tuple[int, ...]
    type: str

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_case_bytes(self)

    @property
    def sha256(self) -> str:
        return case_sha256(self)


@dataclass(frozen=True, slots=True)
class EvalDatasetVersion:
    version: int
    cases: tuple[EvalCase, ...]

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_version_bytes(self)

    @property
    def sha256(self) -> str:
        return version_sha256(self)


def validate_case(value) -> EvalCase:
    """Validate one closed case object and return an immutable copy."""
    if type(value) is not dict or set(value) != CASE_KEYS:
        _refuse("eval_case_invalid")
    case_type = value["type"]
    if type(case_type) is not str or case_type not in CASE_TYPES:
        _refuse("eval_type_invalid")
    return EvalCase(
        case_key=_case_key(value["case_key"]),
        q=_text(value["q"], "q"),
        key=_text(value["key"], "key"),
        answer=_text(value["answer"], "answer"),
        pages=_pages(value["pages"]),
        type=case_type,
    )


def validate_version(value) -> EvalDatasetVersion:
    """Validate a complete version in its one canonical case order."""
    if type(value) is not dict or set(value) != VERSION_KEYS:
        _refuse("eval_version_invalid")
    version = value["version"]
    if (type(version) is not int or version <= 0
            or version > MAX_VERSION):
        _refuse("eval_version_invalid")
    offered = value["cases"]
    if (type(offered) is not list
            or not MIN_CASES <= len(offered) <= MAX_CASES):
        _refuse("eval_case_count_invalid")
    cases = tuple(validate_case(case) for case in offered)
    keys = tuple(case.case_key for case in cases)
    if len(set(keys)) != len(keys):
        _refuse("eval_case_key_duplicate")
    if keys != tuple(sorted(keys)):
        _refuse("eval_order_invalid")
    return EvalDatasetVersion(version=version, cases=cases)


def _as_case(value) -> EvalCase:
    if type(value) is EvalCase:
        return validate_case(_case_object(value))
    if type(value) is dict and type(value.get("pages")) is tuple:
        copied = dict(value)
        copied["pages"] = list(value["pages"])
        return validate_case(copied)
    return validate_case(value)


def _case_sequence(value) -> tuple[EvalCase, ...]:
    if type(value) not in (list, tuple):
        _refuse("eval_case_count_invalid")
    if not MIN_CASES <= len(value) <= MAX_CASES:
        _refuse("eval_case_count_invalid")
    cases = tuple(_as_case(case) for case in value)
    keys = tuple(case.case_key for case in cases)
    if len(set(keys)) != len(keys):
        _refuse("eval_case_key_duplicate")
    if keys != tuple(sorted(keys)):
        _refuse("eval_order_invalid")
    return cases


def normalize_cases(value) -> tuple[dict, ...]:
    """Return fresh normalized dictionaries for DB/API persistence seams.

    The tuple and integer-page tuples make the returned collection's order
    explicit.  The dictionaries are fresh copies so no caller-owned list or
    mapping is retained.
    """
    return tuple({
        "case_key": case.case_key,
        "q": case.q,
        "key": case.key,
        "answer": case.answer,
        "pages": tuple(case.pages),
        "type": case.type,
    } for case in _case_sequence(value))


def new_case_key() -> str:
    """Issue a canonical random UUIDv4 suitable for a new case."""
    return str(uuid.uuid4())


def _case_object(case: EvalCase) -> dict:
    if type(case) is not EvalCase:
        _refuse("eval_case_invalid")
    return {
        "answer": case.answer,
        "case_key": case.case_key,
        "key": case.key,
        "pages": list(case.pages),
        "q": case.q,
        "type": case.type,
    }


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_case_bytes(case: EvalCase) -> bytes:
    checked = validate_case(_case_object(case))
    return _canonical(_case_object(checked))


def case_sha256(case: EvalCase) -> str:
    return hashlib.sha256(canonical_case_bytes(case)).hexdigest()


def case_digest(case) -> str:
    """Return the lowercase SHA-256 hex digest of one accepted case."""
    return case_sha256(_as_case(case))


def canonical_version_bytes(version: EvalDatasetVersion) -> bytes:
    if type(version) is not EvalDatasetVersion:
        _refuse("eval_version_invalid")
    checked = validate_version({
        "cases": [_case_object(case) for case in version.cases],
        "version": version.version,
    })
    return _canonical({
        "cases": [_case_object(case) for case in checked.cases],
        "version": checked.version,
    })


def version_sha256(version: EvalDatasetVersion) -> str:
    return hashlib.sha256(canonical_version_bytes(version)).hexdigest()


def version_digest(cases) -> str:
    """Digest normalized version content, independent of its DB version id."""
    checked = _case_sequence(cases)
    body = _canonical({"cases": [_case_object(case) for case in checked]})
    return hashlib.sha256(body).hexdigest()


def project_legacy_case(case: EvalCase) -> dict:
    """Return a fresh legacy runner case, excluding the governance key."""
    checked = validate_case(_case_object(case))
    return {
        "q": checked.q,
        "key": checked.key,
        "answer": checked.answer,
        "pages": list(checked.pages),
        "type": checked.type,
    }


def project_legacy(cases) -> tuple[dict, ...]:
    """Project a validated collection to exact legacy runner dictionaries."""
    return tuple(project_legacy_case(case) for case in _case_sequence(cases))


def _closed_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _refuse("eval_json_duplicate_key")
        result[key] = value
    return result


def _bad_constant(_value):
    _refuse("eval_json_invalid")


def load_version_json(raw: bytes) -> EvalDatasetVersion:
    """Decode strict UTF-8 JSON while retaining duplicate-key detection."""
    if type(raw) is not bytes or not raw or len(raw) > MAX_JSON_BYTES:
        _refuse("eval_json_invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
        if text.startswith("\ufeff"):
            _refuse("eval_json_invalid")
        value = json.loads(
            text,
            object_pairs_hook=_closed_object,
            parse_constant=_bad_constant,
        )
    except EvalDatasetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        _refuse("eval_json_invalid")
    return validate_version(value)
