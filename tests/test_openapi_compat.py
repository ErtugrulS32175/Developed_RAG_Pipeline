"""Permanent regression tests for the checked OpenAPI compatibility gate."""
from copy import deepcopy

from scripts.check_openapi_compat import incompatibilities


def _document():
    return {
        "openapi": "3.1.0",
        "components": {
            "schemas": {
                "Reply": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
            "securitySchemes": {"key": {"type": "apiKey", "in": "header"}},
        },
        "paths": {
            "/items": {
                "get": {
                    "parameters": [],
                    "responses": {"200": {"description": "ok"}},
                    "security": [{"key": []}],
                },
            },
        },
    }


def test_additive_paths_and_components_are_compatible():
    baseline = _document()
    current = deepcopy(baseline)
    current["components"]["schemas"]["NewReply"] = {"type": "string"}
    current["paths"]["/new"] = {"get": {"responses": {}}}
    assert incompatibilities(baseline, current) == ()


def test_removing_or_changing_a_published_contract_is_refused():
    baseline = _document()
    current = deepcopy(baseline)
    del current["paths"]["/items"]["get"]
    current["components"]["schemas"]["Reply"]["required"] = []
    assert incompatibilities(baseline, current) == (
        "component changed: schemas/Reply",
        "operation removed: GET /items",
    )


def test_request_response_parameter_and_security_drift_are_each_refused():
    baseline = _document()
    for key in ("parameters", "requestBody", "responses", "security"):
        current = deepcopy(baseline)
        current["paths"]["/items"]["get"][key] = {"changed": True}
        assert incompatibilities(baseline, current) == (
            f"operation {key} changed: GET /items",
        )
