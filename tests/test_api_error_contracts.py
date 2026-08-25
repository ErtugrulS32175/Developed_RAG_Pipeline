"""Versioned, content-free API errors and the bounded detail transition."""
from contextlib import contextmanager

from fastapi.testclient import TestClient

from pipeline.api import app as api
from pipeline.api import contracts
from pipeline.api import errors


def _assert_bound_error(response, code):
    body = response.json()
    assert body["error"] == {
        "version": 1,
        "code": code,
        "request_id": response.headers["X-Request-ID"],
    }
    contracts.ErrorResponse.model_validate(body)
    return body


def test_the_error_vocabulary_is_closed_and_status_owned():
    assert dict(errors.ERROR_CODE_BY_STATUS) == {
        400: "invalid_request",
        401: "authentication_required",
        403: "permission_denied",
        404: "resource_not_found",
        405: "method_not_allowed",
        409: "conflict",
        413: "payload_too_large",
        422: "validation_failed",
        429: "rate_limited",
        500: "internal_error",
        502: "upstream_failed",
        503: "service_unavailable",
    }
    schema = contracts.ErrorResponse.model_json_schema()
    assert schema["additionalProperties"] is False
    for nested in schema["$defs"].values():
        if nested.get("type") == "object":
            assert nested["additionalProperties"] is False


def test_http_errors_add_the_new_envelope_without_removing_legacy_detail():
    response = TestClient(api.app).get("/route-that-does-not-exist")
    assert response.status_code == 404
    body = _assert_bound_error(response, "resource_not_found")
    assert body["detail"] == "Not Found"


def test_request_validation_drops_the_reflected_input_value():
    sentinel = "PRIVATE_REQUEST_VALUE_MUST_NOT_RETURN"
    response = TestClient(api.app).get(
        "/documents", params={"limit": sentinel})
    assert response.status_code == 422
    body = _assert_bound_error(response, "validation_failed")
    assert sentinel not in response.text
    assert body["detail"]
    assert all(set(issue) == {"type", "loc", "msg"}
               for issue in body["detail"])


def test_unhandled_failures_use_only_the_fixed_internal_error(monkeypatch):
    private = "PRIVATE_DATABASE_FAILURE_MUST_NOT_RETURN"

    def fail(*_args, **_kwargs):
        raise RuntimeError(private)

    @contextmanager
    def connection():
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    monkeypatch.setattr(api.db, "get_document", fail)
    response = TestClient(api.app, raise_server_exceptions=False).get(
        "/documents/fixed-document-id")
    assert response.status_code == 500
    body = _assert_bound_error(response, "internal_error")
    assert body["detail"] == "internal server error"
    assert private not in response.text


def test_openapi_declares_the_same_error_model_on_every_operation():
    document = api.app.openapi()
    for path in document["paths"].values():
        for operation in path.values():
            if not isinstance(operation, dict) or "responses" not in operation:
                continue
            for status in errors.ERROR_CODE_BY_STATUS:
                response = operation["responses"][str(status)]
                assert response["content"]["application/json"]["schema"] == {
                    "$ref": "#/components/schemas/ErrorResponse"
                }
