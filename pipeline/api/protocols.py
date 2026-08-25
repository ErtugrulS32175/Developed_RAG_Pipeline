"""Shared API protocol rules that do not belong to one domain router."""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType


PROTOCOL_VERSION = 1
PAGE_LIMIT_MIN = 1
PAGE_LIMIT_MAX = 100
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENCY_KEY_MIN = 1
IDEMPOTENCY_KEY_MAX = 200


class PaginationRefused(ValueError):
    """A cursor is incomplete or incompatible with the requested page."""


@dataclass(frozen=True, slots=True)
class PageWindow:
    items: tuple
    has_more: bool
    next_cursor: dict | int | None


def paired_cursor(first, second, *, first_name, second_name,
                  error_message=None):
    """Return an all-or-nothing cursor pair without guessing missing fields."""
    if (first is None) != (second is None):
        raise PaginationRefused(error_message or (
            f"{first_name} ve {second_name} birlikte verilmeli"))
    return None if first is None else (first, second)


def page_window(rows, limit, cursor):
    """Project a limit-plus-one probe to one page and its continuation."""
    if (type(limit) is not int or not PAGE_LIMIT_MIN <= limit <= PAGE_LIMIT_MAX):
        raise PaginationRefused("sayfa limiti gecersiz")
    materialized = tuple(rows)
    page = materialized[:limit]
    has_more = len(materialized) > limit
    next_cursor = cursor(page[-1]) if has_more and page else None
    if (next_cursor is not None and type(next_cursor) not in (dict, int)):
        raise PaginationRefused("sayfa imleci nesne veya tamsayi olmali")
    return PageWindow(page, has_more, next_cursor)


PAGINATION_OPERATIONS = MappingProxyType({
    ("get", "/documents"): {
        "mode": "cursor_or_offset",
        "cursor_fields": ["before_uploaded_at", "before_id"],
        "legacy_offset_field": "offset",
        "response_cursor_field": "next_cursor",
    },
    ("get", "/documents/{document_id}/versions"): {
        "mode": "cursor",
        "cursor_fields": ["before_version_number"],
        "response_cursor_field": "next_before_version_number",
    },
    ("get", "/v1/org/admin/audit-events"): {
        "mode": "cursor",
        "cursor_fields": ["before_created_at", "before_id"],
        "response_cursor_field": "next_cursor",
    },
    ("get", "/v1/org/admin/retention-documents"): {
        "mode": "cursor",
        "cursor_fields": ["before_uploaded_at", "before_id"],
        "response_cursor_field": "next_cursor",
    },
    ("get", "/v1/reviews/queue"): {
        "mode": "cursor",
        "cursor_fields": ["before_created_at", "before_id"],
        "response_cursor_field": "next_cursor",
    },
})

IDEMPOTENCY_OPERATIONS = MappingProxyType({
    ("post", "/documents/{document_id}/ingest-jobs"): {
        "header": IDEMPOTENCY_KEY_HEADER,
        "scope": "document_lifetime",
        "replay": "same_job",
        "storage": "sha256_digest_only",
        "conflict_status": 409,
    },
})

PROTOCOL_CONTRACT = {
    "version": PROTOCOL_VERSION,
    "pagination": {
        "limit_minimum": PAGE_LIMIT_MIN,
        "limit_maximum": PAGE_LIMIT_MAX,
        "ordering": "stable_descending",
        "continuation": "exclusive_before_cursor",
        "has_more": "limit_plus_one_probe",
        "incomplete_cursor_status": 422,
    },
    "idempotency": {
        "header": IDEMPOTENCY_KEY_HEADER,
        "key_min_length": IDEMPOTENCY_KEY_MIN,
        "key_max_length": IDEMPOTENCY_KEY_MAX,
        "raw_key_persisted": False,
    },
    "conflict": {
        "status": 409,
        "error_code": "conflict",
        "automatic_mutation_retry": False,
        "recovery": "refresh_authoritative_state",
    },
    "deprecation": {
        "openapi_flag": "deprecated",
        "required_extension": "x-ragtest-deprecation",
        "required_fields": ["replacement", "sunset"],
        "required_headers": ["Deprecation", "Sunset", "Link"],
    },
}


def _parameter(operation, name, location):
    for parameter in operation.get("parameters", []):
        if parameter.get("name") == name and parameter.get("in") == location:
            return parameter
    return None


def annotate_openapi(document):
    document["x-ragtest-protocols"] = deepcopy(PROTOCOL_CONTRACT)
    for (method, path), contract in PAGINATION_OPERATIONS.items():
        operation = document["paths"][path][method]
        if _parameter(operation, "limit", "query") is None:
            raise RuntimeError(f"pagination limit is absent: {method} {path}")
        for field in contract["cursor_fields"]:
            if _parameter(operation, field, "query") is None:
                raise RuntimeError(
                    f"pagination cursor field is absent: {method} {path} {field}")
        operation["x-ragtest-pagination"] = {
            "version": PROTOCOL_VERSION,
            **deepcopy(contract),
        }
    for (method, path), contract in IDEMPOTENCY_OPERATIONS.items():
        operation = document["paths"][path][method]
        parameter = _parameter(operation, contract["header"], "header")
        if parameter is None or parameter.get("required") is not True:
            raise RuntimeError(f"idempotency header is absent: {method} {path}")
        schema = parameter.get("schema", {})
        if (schema.get("minLength") != IDEMPOTENCY_KEY_MIN
                or schema.get("maxLength") != IDEMPOTENCY_KEY_MAX):
            raise RuntimeError(f"idempotency bounds drifted: {method} {path}")
        operation["x-ragtest-idempotency"] = {
            "version": PROTOCOL_VERSION,
            **deepcopy(contract),
        }
    for path, item in document.get("paths", {}).items():
        for method, operation in item.items():
            if not isinstance(operation, dict) or not operation.get("deprecated"):
                continue
            policy = operation.get("x-ragtest-deprecation")
            required = PROTOCOL_CONTRACT["deprecation"]["required_fields"]
            if (not isinstance(policy, dict)
                    or any(not policy.get(field) for field in required)):
                raise RuntimeError(
                    f"deprecated operation lacks policy: {method} {path}")
    return document


def install(app):
    """Publish and validate the shared protocols in the generated OpenAPI."""
    base_openapi = app.openapi

    def protocol_openapi():
        document = base_openapi()
        if "x-ragtest-protocols" not in document:
            annotate_openapi(document)
        return document

    app.openapi = protocol_openapi
