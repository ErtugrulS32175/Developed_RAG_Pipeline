"""Tenant credentials, role boundaries, and tenant-aware storage/scopes."""
import json
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from pipeline.api import auth
from pipeline.api import app as api
from pipeline.index import db, publication
from pipeline.retrieval import query


TENANT_A = UUID("10000000-0000-0000-0000-000000000001")
TENANT_B = UUID("20000000-0000-0000-0000-000000000002")


def _registry():
    return auth.load_registry("", json.dumps([
        {"key": "reader-token", "tenant_id": str(TENANT_A),
         "role": "reader"},
        {"key": "editor-token", "tenant_id": str(TENANT_A),
         "role": "editor"},
        {"key": "admin-token", "tenant_id": str(TENANT_B),
         "role": "admin"},
    ]))


def test_registry_is_closed_and_never_retains_raw_tokens():
    registry = _registry()
    assert registry.configured and registry.multi_tenant
    assert "reader-token" not in repr(registry)
    assert auth.authenticate(registry, "Bearer reader-token") == auth.Principal(
        TENANT_A, "reader")
    assert auth.authenticate(registry, "bearer admin-token") == auth.Principal(
        TENANT_B, "admin")
    assert auth.authenticate(registry, "Bearer wrong") is None


@pytest.mark.parametrize("payload", [
    "{}", "[]", '[{"key":"x"}]',
    '[{"key":"x","tenant_id":"bad","role":"reader"}]',
    '[{"key":"x","tenant_id":"10000000-0000-0000-0000-000000000001",'
    '"role":"owner"}]',
])
def test_malformed_multi_key_configuration_fails_closed(payload):
    with pytest.raises(auth.AuthConfigurationError):
        auth.load_registry("", payload)


def test_duplicate_key_across_tenants_is_refused():
    rows = [
        {"key": "same", "tenant_id": str(TENANT_A), "role": "reader"},
        {"key": "same", "tenant_id": str(TENANT_B), "role": "admin"},
    ]
    with pytest.raises(auth.AuthConfigurationError):
        auth.load_registry("", json.dumps(rows))


def test_roles_are_monotonic_and_unknown_roles_never_pass():
    reader = auth.Principal(TENANT_A, "reader")
    editor = auth.Principal(TENANT_A, "editor")
    admin = auth.Principal(TENANT_A, "admin")
    assert auth.permits(reader, "reader")
    assert not auth.permits(reader, "editor")
    assert auth.permits(editor, "reader") and auth.permits(editor, "editor")
    assert not auth.permits(editor, "admin")
    assert all(auth.permits(admin, role) for role in auth.ROLES)
    assert not auth.permits(admin, "owner")


def test_reader_editor_and_admin_dependencies_are_wired_to_route_classes():
    expected = {
        ("/v1/models", "GET"): api.require_api_key,
        ("/documents", "GET"): api.require_api_key,
        ("/documents/upload", "POST"): api.require_editor,
        ("/documents/{document_id}/process", "POST"): api.require_editor,
        ("/collections", "POST"): api.require_editor,
        ("/documents/{document_id}/archive", "POST"): api.require_admin,
        ("/collections/{collection_id}", "DELETE"): api.require_admin,
    }
    actual = {}
    for route in api.app.routes:
        for method in getattr(route, "methods", ()):
            key = (getattr(route, "path", ""), method)
            if key in expected:
                actual[key] = route.dependencies[0].dependency
    assert actual == expected


def test_role_refusal_happens_before_request_body_or_database(monkeypatch):
    monkeypatch.setattr(api, "AUTH_REGISTRY", _registry())
    borrowed = []

    @contextmanager
    def borrowing():
        borrowed.append(True)
        yield object()

    monkeypatch.setattr(api, "db_conn", borrowing)
    client = TestClient(api.app)
    reader = {"Authorization": "Bearer reader-token"}
    editor = {"Authorization": "Bearer editor-token"}
    assert client.post("/documents/upload", headers=reader).status_code == 403
    assert client.post("/documents/x/archive", headers=editor).status_code == 403
    assert borrowed == []


def test_request_connection_binds_and_clears_exact_tenant(monkeypatch):
    monkeypatch.setattr(api, "AUTH_REGISTRY", _registry())
    seen = []
    conn = SimpleNamespace(rollback=lambda: seen.append("rollback"))

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(api.db, "get_pool",
                        lambda: SimpleNamespace(connection=connection))
    monkeypatch.setattr(api.db, "init_schema", lambda _conn: None)
    monkeypatch.setattr(api, "_schema_ready", True)
    monkeypatch.setattr(api.db, "set_tenant_context",
                        lambda _conn, tenant, *, service=False,
                        actor_id=None: seen.append(
                            (tenant, actor_id, service)))
    monkeypatch.setattr(api.db, "clear_tenant_context",
                        lambda _conn: seen.append("clear"))
    monkeypatch.setattr(api.db, "get_document", lambda *_args: None)
    response = TestClient(api.app).get(
        "/documents/missing",
        headers={"Authorization": "Bearer reader-token"})
    assert response.status_code == 404
    assert seen == [(TENANT_A, None, False), "rollback", "clear"]


def test_non_default_tenant_storage_is_namespaced(tmp_path):
    assert publication.tenant_upload_root(
        tmp_path, db.DEFAULT_TENANT_ID) == tmp_path.resolve()
    first = publication.tenant_upload_root(tmp_path, TENANT_A)
    second = publication.tenant_upload_root(tmp_path, TENANT_B)
    assert first != second
    assert first.parent == second.parent == tmp_path.resolve() / "tenants"
    assert publication.source_path(tmp_path, "same.pdf", TENANT_A).parent == first
    assert publication.source_path(tmp_path, "same.pdf", TENANT_B).parent == second


def test_multitenant_unscoped_rag_is_resolved_by_the_checked_backend(
        monkeypatch):
    monkeypatch.setattr(api, "AUTH_REGISTRY", _registry())
    captured = []

    def answer(_question, **kwargs):
        captured.append((kwargs, db._EXECUTION_TENANT.get(),
                         db.current_execution_actor()))
        return SimpleNamespace(status="abstained", answer="bilmiyorum",
                               citations=(), trace=None)

    monkeypatch.setattr(api.rag_backends, "answer_checked", answer)
    monkeypatch.setattr(api, "_publish_checked",
                        lambda _result: ("abstained", "bilmiyorum", (), None))
    response = TestClient(api.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer reader-token"},
        json={"model": api.RAG_MODEL_ID,
              "messages": [{"role": "user", "content": "soru"}]})
    assert response.status_code == 200
    assert captured[0][0] == {"backend": "native"}
    assert captured[0][1] == (TENANT_A, False)
    assert captured[0][2] is None
    assert db._EXECUTION_TENANT.get() == (db.DEFAULT_TENANT_ID, False)
    assert db.current_execution_actor() is None


def test_native_retrieval_binds_and_clears_the_execution_tenant(monkeypatch):
    seen = []
    conn = SimpleNamespace(rollback=lambda: seen.append("rollback"))

    @contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(
        query.db, "get_pool",
        lambda: SimpleNamespace(connection=connection))
    monkeypatch.setattr(query, "embed_dense", lambda _value: [0.0])
    monkeypatch.setattr(query, "embed_sparse", lambda _value: ([1], [1.0]))
    monkeypatch.setattr(query.db, "begin_retrieval_snapshot",
                        lambda actual: seen.append(("snapshot", actual)))
    monkeypatch.setattr(query.db, "retrieval_policy_epoch", lambda _conn: 1)
    monkeypatch.setattr(query.db, "active_document_ids",
                        lambda _conn: ("visible-document",))
    monkeypatch.setattr(query.db, "current_execution_actor", lambda: None)
    monkeypatch.setattr(
        query.db, "set_tenant_context",
        lambda actual, tenant, service=False, actor_id=None:
        seen.append((actual, tenant, service, actor_id)))
    monkeypatch.setattr(
        query.db, "clear_tenant_context",
        lambda actual: seen.append(("clear", actual)))
    monkeypatch.setattr(
        query.db, "hybrid_search",
        lambda actual, *_args, **_kwargs:
        seen.append(("search", actual)) or [{"id": "visible"}])

    token = db.bind_execution_tenant(TENANT_A)
    try:
        assert query.retrieve("kurgu") == [{"id": "visible"}]
    finally:
        db.reset_execution_tenant(token)

    assert seen == [
        (conn, TENANT_A, False, None),
        ("snapshot", conn),
        ("search", conn),
        "rollback",
        ("clear", conn),
    ]


def test_execution_actor_is_bound_and_reset_beside_the_legacy_tenant_seam():
    actor = UUID("10000000-0000-0000-0000-000000000099")
    token = db.bind_execution_tenant(TENANT_A, actor_id=actor)
    try:
        assert db.current_execution_tenant() == (TENANT_A, False)
        assert db.current_execution_actor() == actor
        # Existing callers that inspect the historical context retain its
        # exact two-item shape; actor identity is a separate authority.
        assert db._EXECUTION_TENANT.get() == (TENANT_A, False)
    finally:
        db.reset_execution_tenant(token)
    assert db.current_execution_tenant() == (db.DEFAULT_TENANT_ID, False)
    assert db.current_execution_actor() is None


def test_checked_chat_binds_the_verified_openwebui_actor(monkeypatch):
    actor = UUID("10000000-0000-0000-0000-000000000099")
    principal = auth.Principal(
        TENANT_A, "reader", subject_id=actor, source="openwebui")
    seen = []
    monkeypatch.setattr(
        api.db, "bind_execution_tenant",
        lambda tenant_id, *, service=False, actor_id=None:
        seen.append((tenant_id, service, actor_id)) or "token")
    monkeypatch.setattr(api.db, "reset_execution_tenant",
                        lambda token: seen.append(("reset", token)))
    monkeypatch.setattr(
        api.rag_backends, "answer_checked",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="abstained", answer="bilmiyorum", citations=(),
            trace=None))
    monkeypatch.setattr(api, "_publish_checked",
                        lambda _result: ("abstained", "bilmiyorum", (), None))
    monkeypatch.setattr(api, "_persist_review_interaction",
                        lambda _result: None)
    token = auth.bind(principal)
    try:
        response = api.chat_completions(api.ChatRequest(
            model=api.RAG_MODEL_ID,
            messages=[api.ChatMessage(role="user", content="soru")]))
    finally:
        auth.reset(token)
    assert response["choices"][0]["message"]["content"] == "bilmiyorum"
    assert seen == [(TENANT_A, False, actor), ("reset", "token")]


def test_schema_declares_forced_row_level_isolation_for_every_tenant_table():
    sql = (publication.Path(__file__).parents[1] /
           "pipeline" / "index" / "schema.sql").read_text(encoding="utf-8")
    for table in ("documents", "collections", "tags",
                  "collection_documents", "document_tags", "ingest_jobs",
                  "chunks", "attempts"):
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in sql
        assert f"CREATE POLICY tenant_isolation ON {table}" in sql
