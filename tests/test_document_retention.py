"""Closed retention, legal-hold and purge API contracts."""
from contextlib import contextmanager
from datetime import datetime, timezone
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pipeline.api import app as api
from pipeline.api import auth
from pipeline.index import db


TENANT = uuid.UUID("43000000-0000-4000-8000-000000000001")
ARCHITECT = uuid.UUID("43000000-0000-4000-8000-000000000002")
DOCUMENT = uuid.UUID("43000000-0000-4000-8000-000000000003")
HOLD = uuid.UUID("43000000-0000-4000-8000-000000000004")


@pytest.fixture
def retention_api(monkeypatch):
    current = {"principal": auth.Principal(
        TENANT, "org_architect", ARCHITECT, "openwebui",
        org_architect=True)}
    monkeypatch.setattr(
        api, "_request_principal", lambda _request: current["principal"])

    @contextmanager
    def connection():
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    return TestClient(api.app), current


def test_retention_models_are_strict_and_closed():
    policy = api.RetentionPolicyUpdateRequest.model_validate({
        "expected_revision": 1,
        "expected_policy_epoch": 2,
        "archive_retention_days": 90,
    })
    assert policy.archive_retention_days == 90

    for model, payload in (
        (api.RetentionPolicyUpdateRequest, {
            "expected_revision": True, "expected_policy_epoch": 2,
            "archive_retention_days": 90}),
        (api.LegalHoldCreateRequest, {
            "expected_document_revision": 1, "expected_policy_epoch": 2,
            "reason_code": "free_text"}),
        (api.PurgeScheduleRequest, {
            "expected_document_revision": 1, "expected_policy_epoch": 2,
            "extra": "not-allowed"}),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_schema_carries_closed_retention_evidence_and_rls():
    schema = db.Path(db.__file__).with_name("schema.sql").read_text(
        encoding="utf-8")
    assert db.SCHEMA_VERSION == 13
    for table in (
            "tenant_retention_policies", "document_legal_holds",
            "document_purge_jobs", "document_retention_events"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in schema
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in schema
    assert "CREATE TRIGGER document_retention_events_immutable" in schema
    assert "CREATE TRIGGER documents_guard_purge_state" in schema
    assert "WHERE state IN ('pending', 'running')" in schema
    assert "purge_execute" in schema
    assert "retention_inventory_view" in schema


def test_retention_inventory_is_opaque_keyset_paginated_and_audited(
        retention_api, monkeypatch):
    client, _current = retention_api
    uploaded = datetime(2026, 8, 25, 7, 0, tzinfo=timezone.utc)
    later = uuid.UUID("43000000-0000-4000-8000-000000000005")
    captured = []
    audited = []
    rows = [
        {
            "document_id": DOCUMENT, "status": "archived", "revision": 4,
            "uploaded_at": uploaded, "archived_at": uploaded,
            "purged_at": None, "active_hold_count": 1,
            "latest_purge_job_id": HOLD,
            "latest_purge_state": "cancelled",
        },
        {
            "document_id": later, "status": "purged", "revision": 5,
            "uploaded_at": uploaded, "archived_at": uploaded,
            "purged_at": uploaded, "active_hold_count": 0,
            "latest_purge_job_id": None, "latest_purge_state": None,
        },
    ]
    monkeypatch.setattr(
        api.db, "list_retention_documents",
        lambda _conn, **fields: captured.append(fields) or rows)
    monkeypatch.setattr(
        api.db, "record_org_decision",
        lambda _conn, **fields: audited.append(fields))

    response = client.get("/v1/org/admin/retention-documents?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is True
    assert body["documents"] == [{
        "document_id": str(DOCUMENT), "status": "archived", "revision": 4,
        "uploaded_at": uploaded.isoformat(),
        "archived_at": uploaded.isoformat(), "purged_at": None,
        "active_hold_count": 1, "latest_purge_job_id": str(HOLD),
        "latest_purge_state": "cancelled",
    }]
    assert set(body["documents"][0]).isdisjoint({
        "filename", "status_note", "content_sha256", "candidate_id",
    })
    assert body["next_cursor"] == {
        "before_uploaded_at": uploaded.isoformat(),
        "before_id": str(DOCUMENT),
    }
    assert captured == [{"actor_id": ARCHITECT, "limit": 1,
                         "before": None}]
    assert audited[0]["action"] == "retention_inventory_view"
    assert audited[0]["allowed"] is True


def test_retention_inventory_requires_a_complete_cursor(retention_api):
    client, _current = retention_api
    response = client.get(
        "/v1/org/admin/retention-documents?before_id=" + str(DOCUMENT))
    assert response.status_code == 422


def test_policy_endpoints_preserve_cas_and_request_identity(
        retention_api, monkeypatch):
    client, _current = retention_api
    captured = []
    monkeypatch.setattr(
        api.db, "get_tenant_retention_policy",
        lambda _conn, **fields: {
            "tenant_id": TENANT, "archive_retention_days": 365,
            "revision": 1, "policy_epoch": 7, "updated_at": None})
    monkeypatch.setattr(
        api.db, "update_tenant_retention_policy",
        lambda _conn, **fields: captured.append(fields) or {
            "tenant_id": TENANT, "archive_retention_days": 90,
            "revision": 2, "policy_epoch": 8, "updated_at": None})

    current = client.get("/v1/org/admin/retention-policy")
    updated = client.put("/v1/org/admin/retention-policy", json={
        "expected_revision": 1,
        "expected_policy_epoch": 7,
        "archive_retention_days": 90,
    })

    assert current.status_code == 200
    assert current.json()["archive_retention_days"] == 365
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert captured[0] == {
        "actor_id": ARCHITECT,
        "archive_retention_days": 90,
        "expected_revision": 1,
        "expected_policy_epoch": 7,
        "request_id": updated.headers["X-Request-ID"],
    }


def test_hold_and_purge_routes_keep_closed_authority_fields(
        retention_api, monkeypatch):
    client, _current = retention_api
    calls = []
    monkeypatch.setattr(
        api.db, "create_document_legal_hold",
        lambda _conn, **fields: calls.append(("hold", fields)) or {
            "id": HOLD, "document_id": DOCUMENT, "reason_code": "litigation",
            "state": "active", "revision": 1, "policy_epoch": 4,
            "created_at": None, "released_at": None})
    monkeypatch.setattr(
        api.db, "release_document_legal_hold",
        lambda _conn, **fields: calls.append(("release", fields)) or {
            "id": HOLD, "document_id": DOCUMENT, "reason_code": "litigation",
            "state": "released", "revision": 2, "policy_epoch": 5,
            "created_at": None, "released_at": None})
    monkeypatch.setattr(
        api.db, "schedule_document_purge",
        lambda _conn, **fields: calls.append(("purge", fields)) or {
            "id": HOLD, "document_id": DOCUMENT, "state": "pending",
            "eligible_at": None, "attempt_count": 0, "created_at": None})

    created = client.post(f"/documents/{DOCUMENT}/legal-holds", json={
        "expected_document_revision": 3,
        "expected_policy_epoch": 3,
        "reason_code": "litigation",
    })
    released = client.post(
        f"/documents/{DOCUMENT}/legal-holds/{HOLD}/release", json={
            "expected_revision": 1,
            "expected_policy_epoch": 4,
        })
    scheduled = client.post(f"/documents/{DOCUMENT}/purge-jobs", json={
        "expected_document_revision": 3,
        "expected_policy_epoch": 5,
    })

    assert (created.status_code, released.status_code,
            scheduled.status_code) == (201, 200, 202)
    assert [kind for kind, _fields in calls] == ["hold", "release", "purge"]
    assert calls[0][1]["document_id"] == DOCUMENT
    assert calls[0][1]["reason_code"] == "litigation"
    assert calls[1][1]["hold_id"] == HOLD
    assert calls[2][1]["expected_revision"] == 3
    assert all(fields["actor_id"] == ARCHITECT for _kind, fields in calls)
    assert all(len(fields["request_id"]) == 8 for _kind, fields in calls)


def test_document_admin_without_architect_capability_is_refused_and_audited(
        retention_api, monkeypatch):
    client, current = retention_api
    current["principal"] = auth.Principal(
        TENANT, "admin", ARCHITECT, "openwebui", org_architect=False)
    recorded = []
    monkeypatch.setattr(
        api.db, "record_org_decision",
        lambda _conn, **fields: recorded.append(fields))

    response = client.get("/v1/org/admin/retention-policy")

    assert response.status_code == 403
    assert recorded == [{
        "actor_id": ARCHITECT,
        "subject_id": None,
        "action": "retention_policy_change",
        "reason_code": "system_operation",
        "allowed": False,
        "request_id": response.headers["X-Request-ID"],
    }]

    recorded.clear()
    response = client.get("/v1/org/admin/retention-documents")
    assert response.status_code == 403
    assert recorded[0]["action"] == "retention_inventory_view"
    assert recorded[0]["allowed"] is False
