"""Shared pagination, idempotency, conflict, and deprecation contracts."""
from copy import deepcopy

import pytest

from pipeline.api import app as api
from pipeline.api import errors
from pipeline.api import protocols


def test_a_cursor_pair_is_all_or_nothing_and_preserves_values():
    assert protocols.paired_cursor(
        "instant", "identifier",
        first_name="before_time", second_name="before_id",
    ) == ("instant", "identifier")
    assert protocols.paired_cursor(
        None, None, first_name="before_time", second_name="before_id",
    ) is None
    with pytest.raises(protocols.PaginationRefused, match="together"):
        protocols.paired_cursor(
            "instant", None, first_name="before_time",
            second_name="before_id", error_message="together",
        )


def test_limit_plus_one_has_one_authoritative_projection():
    rows = ({"id": 1}, {"id": 2}, {"id": 3})
    window = protocols.page_window(
        rows, 2, lambda row: {"before_id": row["id"]})
    assert window.items == rows[:2]
    assert window.has_more is True
    assert window.next_cursor == {"before_id": 2}
    last = protocols.page_window(rows[:2], 2, lambda _row: {"unused": 1})
    assert last.items == rows[:2]
    assert last.has_more is False
    assert last.next_cursor is None


@pytest.mark.parametrize("value", [True, 0, 101, "2", None])
def test_page_limits_fail_closed_before_cursor_projection(value):
    called = []
    with pytest.raises(protocols.PaginationRefused):
        protocols.page_window([{"id": 1}], value,
                              lambda row: called.append(row) or {})
    assert called == []


def test_openapi_publishes_the_versioned_shared_protocol_vocabulary():
    contract = api.app.openapi()["x-ragtest-protocols"]
    assert contract == protocols.PROTOCOL_CONTRACT
    assert contract["pagination"] == {
        "limit_minimum": 1,
        "limit_maximum": 100,
        "ordering": "stable_descending",
        "continuation": "exclusive_before_cursor",
        "has_more": "limit_plus_one_probe",
        "incomplete_cursor_status": 422,
    }
    assert contract["conflict"] == {
        "status": 409,
        "error_code": "conflict",
        "automatic_mutation_retry": False,
        "recovery": "refresh_authoritative_state",
    }


def test_every_registered_page_operation_carries_its_exact_protocol():
    document = api.app.openapi()
    for (method, path), contract in protocols.PAGINATION_OPERATIONS.items():
        operation = document["paths"][path][method]
        assert operation["x-ragtest-pagination"] == {
            "version": protocols.PROTOCOL_VERSION,
            **contract,
        }
        parameters = {(item["name"], item["in"])
                      for item in operation["parameters"]}
        assert ("limit", "query") in parameters
        assert all((field, "query") in parameters
                   for field in contract["cursor_fields"])


def test_the_ingest_idempotency_contract_is_bounded_and_content_free():
    document = api.app.openapi()
    operation = document["paths"][
        "/documents/{document_id}/ingest-jobs"]["post"]
    policy = operation["x-ragtest-idempotency"]
    assert policy == {
        "version": 1,
        "header": "Idempotency-Key",
        "scope": "document_lifetime",
        "replay": "same_job",
        "storage": "sha256_digest_only",
        "conflict_status": 409,
    }
    header = next(item for item in operation["parameters"]
                  if item["name"] == "Idempotency-Key")
    assert header["required"] is True
    assert header["schema"]["minLength"] == 1
    assert header["schema"]["maxLength"] == 200
    assert "request-alpha" not in str(document)


def test_conflicts_use_the_shared_versioned_error_code():
    assert errors.ERROR_CODE_BY_STATUS[409] == "conflict"
    response = api.app.openapi()["paths"][
        "/documents/{document_id}/ingest-jobs"]["post"]["responses"]["409"]
    schema = response["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/ErrorResponse"}


def test_a_future_deprecated_operation_requires_replacement_and_sunset():
    document = deepcopy(api.app.openapi())
    operation = document["paths"]["/health"]["get"]
    operation["deprecated"] = True
    document.pop("x-ragtest-protocols")
    with pytest.raises(RuntimeError, match="lacks policy"):
        protocols.annotate_openapi(document)
    operation["x-ragtest-deprecation"] = {
        "replacement": "/ready",
        "sunset": "Wed, 31 Dec 2026 23:59:59 GMT",
    }
    protocols.annotate_openapi(document)
    assert document["x-ragtest-protocols"]["deprecation"][
        "required_headers"] == ["Deprecation", "Sunset", "Link"]


def test_no_current_operation_is_silently_deprecated():
    document = api.app.openapi()
    deprecated = [
        (method, path)
        for path, item in document["paths"].items()
        for method, operation in item.items()
        if isinstance(operation, dict) and operation.get("deprecated")
    ]
    assert deprecated == []
