"""The first closed response contracts for the enterprise API surface."""
from datetime import datetime, timezone
import json

from fastapi import Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from pipeline.api import app as api
from pipeline.api import contracts
from pipeline.api import owui_chat


def _route(path, method="GET"):
    matches = [
        route for route in api.app.routes
        if (isinstance(route, APIRoute) and route.path == path
            and method in route.methods)
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("path,method,model", [
    ("/health", "GET", contracts.HealthResponse),
    ("/ready", "GET", contracts.ReadinessResponse),
    ("/v1/models", "GET", contracts.ModelListResponse),
    ("/v1/org/me", "GET", contracts.OrganizationMeResponse),
    ("/v1/org/visible-members", "GET",
     contracts.VisibleOrgMembersResponse),
    ("/v1/org/admin/topology", "GET", contracts.OrgTopologyResponse),
    ("/v1/org/admin/topology", "PUT", contracts.OrgVersionResponse),
    ("/v1/org/admin/members/{identity_id}", "PUT",
     contracts.OrgMembershipResponse),
    ("/v1/org/admin/audit-events", "GET",
     contracts.OrgAuditEventListResponse),
    ("/v1/org/admin/retention-policy", "GET",
     contracts.RetentionPolicyResponse),
    ("/v1/org/admin/retention-policy", "PUT",
     contracts.RetentionPolicyResponse),
    ("/v1/org/admin/retention-documents", "GET",
     contracts.RetentionDocumentListResponse),
    ("/documents/{document_id}/legal-holds", "GET",
     contracts.LegalHoldListResponse),
    ("/documents/{document_id}/legal-holds", "POST",
     contracts.LegalHoldResponse),
    ("/documents/{document_id}/legal-holds/{hold_id}/release", "POST",
     contracts.LegalHoldResponse),
    ("/documents/{document_id}/purge-jobs", "GET",
     contracts.PurgeJobListResponse),
    ("/documents/{document_id}/purge-jobs", "POST",
     contracts.PurgeJobResponse),
    ("/documents", "GET", contracts.DocumentListResponse),
    ("/documents/{document_id}", "GET",
     contracts.DocumentDetailResponse),
    ("/documents/{document_id}/archive", "POST",
     contracts.DocumentLifecycleResponse),
    ("/documents/{document_id}/restore", "POST",
     contracts.DocumentLifecycleResponse),
    ("/documents/{document_id}/versions", "GET",
     contracts.DocumentVersionListResponse),
    ("/documents/{document_id}/versions/{version_id}/activate", "POST",
     contracts.DocumentVersionActivationResponse),
    ("/documents/upload", "POST", contracts.DocumentUploadResponse),
    ("/documents/{document_id}/process", "POST",
     contracts.DocumentProcessResponse),
    ("/documents/{document_id}/ingest-jobs", "POST",
     contracts.IngestJobResponse),
    ("/ingest-jobs/{job_id}", "GET", contracts.IngestJobResponse),
    ("/ingest-jobs/{job_id}", "DELETE", contracts.IngestJobResponse),
    ("/v1/evidence/tickets", "POST",
     contracts.ShortLivedTicketResponse),
    ("/v1/evidence/preview", "POST",
     contracts.EvidencePreviewResponse),
    ("/v1/exports/tickets", "POST",
     contracts.ShortLivedTicketResponse),
    ("/v1/reviews/feedback", "POST",
     contracts.ReviewFeedbackResponse),
    ("/v1/reviews/queue", "GET", contracts.ReviewQueueResponse),
    ("/v1/reviews/{case_id}/decision", "POST",
     contracts.ReviewDecisionResponse),
    ("/v1/eval/datasets", "GET", contracts.EvalDatasetListResponse),
    ("/v1/eval/datasets", "POST", contracts.EvalDatasetResponse),
    ("/v1/eval/datasets/{dataset_id}/versions", "GET",
     contracts.EvalVersionListResponse),
    ("/v1/eval/datasets/{dataset_id}/drafts", "POST",
     contracts.EvalVersionResponse),
    ("/v1/eval/datasets/{dataset_id}/versions/{version_id}/cases/import",
     "POST", contracts.EvalVersionResponse),
    ("/v1/eval/datasets/{dataset_id}/versions/{version_id}/cases", "GET",
     contracts.EvalCaseListResponse),
    ("/v1/eval/datasets/{dataset_id}/versions/{version_id}/publish", "POST",
     contracts.EvalVersionResponse),
    ("/v1/eval/datasets/{dataset_id}/retire", "POST",
     contracts.EvalDatasetResponse),
    ("/v1/chat/completions", "POST", contracts.ChatCompletionResponse),
])
def test_enterprise_routes_enforce_named_response_models(path, method, model):
    assert _route(path, method).response_model is model


def test_export_download_declares_only_its_binary_media_type():
    response = api.app.openapi()["paths"]["/v1/exports/download"]["post"][
        "responses"]["200"]
    assert set(response["content"]) == {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }


def test_the_response_models_are_recursively_closed_in_openapi():
    schemas = api.app.openapi()["components"]["schemas"]
    for name in (
            "HealthResponse", "ReadinessChecks", "ReadinessResponse",
            "ModelDescriptor", "ModelListResponse", "OrgSelfMembership",
            "OrganizationMeResponse", "VisibleOrgMember",
            "VisibleOrgMembersResponse", "OrgPositionResponse",
            "OrgTopologyMember", "OrgTopologyResponse", "OrgVersionResponse",
            "OrgMembershipResponse", "DocumentSummary", "DocumentCursor",
            "OrgAuditEvent", "OrgAuditCursor", "OrgAuditEventListResponse",
            "RetentionPolicyResponse", "RetentionDocument",
            "RetentionDocumentCursor", "RetentionDocumentListResponse",
            "LegalHoldResponse", "LegalHoldListResponse",
            "PurgeJobResponse", "PurgeJobListResponse",
            "DocumentListResponse", "DocumentDetailResponse",
            "DocumentLifecycleResponse", "DocumentVersion",
            "DocumentVersionListResponse",
            "DocumentVersionActivationResponse", "DocumentUploadResponse",
            "DocumentProcessResponse", "IngestJobResponse",
            "ShortLivedTicketResponse", "EvidencePreviewResponse",
            "ReviewFeedbackResponse", "ReviewCase", "ReviewCursor",
            "ReviewQueueResponse", "ReviewDecisionResponse",
            "EvalVersionMetadata", "EvalDatasetMetadata",
            "EvalDatasetListResponse", "EvalDatasetResponse",
            "EvalVersionListResponse", "EvalVersionResponse", "EvalCase",
            "EvalCaseListResponse", "RagCitationResponse",
            "RetrievalTraceStagesResponse", "RetrievalTraceResponse",
            "ChatMessageResponse", "ChatChoiceResponse",
            "ChatCompletionResponse"):
        assert schemas[name]["additionalProperties"] is False


def _trace_contract():
    return {
        "trace_version": 2,
        "trace_id": "a" * 32,
        "backend": "native",
        "planner_policy_version": 1,
        "query_class": "factual",
        "retrieval_mode": "hybrid_balanced",
        "fallback": "none",
        "scope_kind": "all_visible",
        "policy_epoch": 1,
        "top_k": 15,
        "candidate_limit": 60,
        "scope_document_count": None,
        "retrieved_count": 1,
        "reranked_count": 1,
        "context_passage_count": 1,
        "context_utf8_bytes": 10,
        "stages_ms": {
            "plan": 1,
            "retrieve": 2,
            "rerank": 3,
            "context": 4,
            "generate": 5,
            "validate": 6,
        },
    }


def test_chat_openapi_distinguishes_json_from_event_stream():
    content = api.app.openapi()["paths"]["/v1/chat/completions"]["post"][
        "responses"]["200"]["content"]
    assert set(content) == {"application/json", "text/event-stream"}
    assert content["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ChatCompletionResponse"
    }
    assert content["text/event-stream"]["schema"]["type"] == "string"


def test_every_stream_event_has_a_closed_chunk_contract():
    chunks = tuple(owui_chat.stream_text(
        "checked answer",
        "ragtest-rag",
        rag_status="answered",
        rag_citations=[{"page": 1, "source": "model"}],
        rag_trace=_trace_contract(),
    ))
    assert chunks[-1] == "data: [DONE]\n\n"
    for chunk in chunks[:-1]:
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        payload = json.loads(chunk[len("data: "):-2])
        contracts.ChatCompletionChunkResponse.model_validate(payload)


def test_stream_chunk_contract_is_recursively_closed():
    schema = contracts.ChatCompletionChunkResponse.model_json_schema()
    assert schema["additionalProperties"] is False
    for nested in schema["$defs"].values():
        if nested.get("type") == "object":
            assert nested["additionalProperties"] is False


def test_chat_contract_refuses_unknown_fields_in_nested_rag_evidence():
    payload = {
        "id": "chatcmpl-fixed",
        "object": "chat.completion",
        "created": 0,
        "model": "ragtest-rag",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "answer"},
            "finish_reason": "stop",
        }],
        "rag_status": "answered",
        "rag_citations": [{
            "page": 1,
            "source": "model",
            "private_chunk_text": "must-not-pass",
        }],
    }
    with pytest.raises(ValidationError):
        contracts.ChatCompletionResponse.model_validate(payload)


def test_timestamp_contract_preserves_legacy_wire_spelling():
    legacy = "0999-01-01T00:00:00+00:00"
    common = {
        "document_id": "doc", "filename": "doc.pdf", "file_type": "pdf",
        "status": "done", "status_note": None, "active_generation": 1,
        "archived_at": None,
    }
    from_text = contracts.DocumentSummary(
        **common, uploaded_at=legacy).model_dump(mode="json")
    from_database = contracts.DocumentSummary(
        **common,
        uploaded_at=datetime(999, 1, 1, tzinfo=timezone.utc),
    ).model_dump(mode="json")
    assert from_text["uploaded_at"] == legacy
    assert from_database["uploaded_at"] == legacy


def test_document_detail_projects_before_the_closed_contract():
    row = {
        "id": "doc", "filename": "doc.pdf", "file_type": "pdf",
        "status": "done", "active_generation": 1,
        "private_future_column": "must-not-pass",
    }
    projected = api._document_detail(row)
    assert "private_future_column" not in projected
    contracts.DocumentDetailResponse.model_validate(projected)


def test_liveness_and_model_discovery_keep_their_existing_wire_shape():
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}
    assert api.list_models() == {
        "object": "list",
        "data": [
            {"id": "ragtest-rag", "object": "model", "owned_by": "ragtest"},
            {"id": "ragtest-table", "object": "model", "owned_by": "ragtest"},
            {"id": "ragtest-rag-llamaindex", "object": "model",
             "owned_by": "ragtest"},
        ],
    }
    contracts.ModelListResponse.model_validate(api.list_models())


def test_readiness_preserves_both_closed_outcomes(monkeypatch):
    monkeypatch.setattr(
        api, "_probe", lambda name, fn: (name, name != "embedding"))
    response = Response()
    payload = api.ready(response)
    assert response.status_code == 503
    assert payload == {
        "status": "degraded",
        "kontroller": {
            "veritabani": True,
            "sema": True,
            "embedding": False,
        },
    }
    contracts.ReadinessResponse.model_validate(payload)


@pytest.mark.parametrize("model,payload", [
    (contracts.HealthResponse, {"status": "ok", "secret": "must-not-pass"}),
    (contracts.ReadinessResponse, {
        "status": "ready",
        "kontroller": {
            "veritabani": True, "sema": True, "embedding": True,
            "database_host": "must-not-pass",
        },
    }),
    (contracts.ModelListResponse, {
        "object": "list",
        "data": [{
            "id": "ragtest-rag", "object": "model", "owned_by": "ragtest",
            "vendor_payload": "must-not-pass",
        }],
    }),
    (contracts.OrganizationMeResponse, {
        "tenant_id": "tenant", "identity_id": "identity",
        "architecture_admin": False,
        "membership": {
            "identity_id": "identity", "position_id": "position",
            "title": "Member", "kind": "member", "level": 2,
            "can_monitor_descendants": False,
            "protected_from_monitoring": False,
            "architecture_version": 1, "policy_epoch": 1,
            "display_label": "Member", "app_role": "reader",
            "ancestor_identity": "must-not-pass",
        },
    }),
    (contracts.DocumentListResponse, {
        "documents": [{
            "document_id": "doc", "filename": "doc.pdf",
            "file_type": "pdf", "uploaded_at": None, "status": "done",
            "status_note": None, "active_generation": 1,
            "archived_at": None, "candidate_id": "must-not-pass",
        }],
        "limit": 20, "offset": 0, "has_more": False,
        "next_cursor": None,
    }),
])
def test_an_undeclared_response_field_is_refused_at_every_depth(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)
