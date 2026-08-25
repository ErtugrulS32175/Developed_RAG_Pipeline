"""Machine identities stay scope-only from Bearer parsing to tenant context."""
from contextlib import contextmanager
from types import SimpleNamespace
import uuid

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api, auth
from pipeline.control import db as control_db, service_accounts


TENANT_ID = uuid.UUID("10000000-0000-0000-0000-000000000001")
ACCOUNT_ID = uuid.UUID("30000000-0000-0000-0000-000000000003")
TOKEN = service_accounts.PREFIX + "reserved"


def _principal(*scopes):
    return auth.ServicePrincipal(TENANT_ID, ACCOUNT_ID, frozenset(scopes))


def _machine_client(monkeypatch, principal):
    monkeypatch.setattr(api, "SERVICE_ACCOUNT_AUTH_ENABLED", True)
    monkeypatch.setattr(api, "OIDC_CLIENT", None)
    monkeypatch.setattr(api, "OPENWEBUI_IDENTITY", None)
    monkeypatch.setattr(api, "AUTH_REGISTRY", auth.Registry((), None))
    monkeypatch.setattr(api, "_resolve_service_account",
                        lambda _authorization: principal)
    return TestClient(api.app)


def test_machine_principal_has_no_human_authority_surface():
    principal = _principal("rag.query")
    assert not hasattr(principal, "role")
    assert not hasattr(principal, "subject_id")
    assert not hasattr(principal, "position_id")
    assert not hasattr(principal, "org_architect")
    assert not auth.permits(principal, "reader")
    assert auth.permits_service(principal, "rag.query")
    assert not auth.permits_service(principal, "documents.read")


def test_service_scope_grants_only_the_exact_registered_route(monkeypatch):
    client = _machine_client(monkeypatch, _principal("rag.query"))
    headers = {"Authorization": "Bearer " + TOKEN}
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.get("/documents", headers=headers).status_code == 403


def test_missing_scope_refuses_before_the_data_plane(monkeypatch):
    client = _machine_client(monkeypatch, _principal("documents.read"))
    borrowed = []

    @contextmanager
    def connection():
        borrowed.append(True)
        yield object()

    monkeypatch.setattr(api, "db_conn", connection)
    response = client.post(
        "/documents/upload",
        headers={"Authorization": "Bearer " + TOKEN},
    )
    assert response.status_code == 403
    assert borrowed == []


def test_reserved_token_never_falls_back_to_legacy_registry(monkeypatch):
    registry = auth.load_registry("", "[{\"key\":\"" + TOKEN +
                                  "\",\"tenant_id\":\"" + str(TENANT_ID) +
                                  "\",\"role\":\"admin\"}]")
    monkeypatch.setattr(api, "SERVICE_ACCOUNT_AUTH_ENABLED", True)
    monkeypatch.setattr(api, "OIDC_CLIENT", None)
    monkeypatch.setattr(api, "AUTH_REGISTRY", registry)
    monkeypatch.setattr(api, "_resolve_service_account", lambda _value: None)
    response = TestClient(api.app).get(
        "/v1/models", headers={"Authorization": "Bearer " + TOKEN})
    assert response.status_code == 401

    upper = TOKEN.upper()
    upper_registry = auth.load_registry(
        "", "[{\"key\":\"" + upper + "\",\"tenant_id\":\"" +
        str(TENANT_ID) + "\",\"role\":\"admin\"}]")
    monkeypatch.setattr(api, "AUTH_REGISTRY", upper_registry)
    response = TestClient(api.app).get(
        "/v1/models", headers={"Authorization": "Bearer " + upper})
    assert response.status_code == 401


def test_resolver_binds_the_credential_to_the_control_route(monkeypatch):
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    token, proof = service_accounts.issue_credential(ACCOUNT_ID, 4)
    route = SimpleNamespace(
        service_account_id=ACCOUNT_ID,
        facts=SimpleNamespace(tenant_id=TENANT_ID),
        scopes=("documents.read", "rag.query"),
    )
    connection = object()

    @contextmanager
    def checkout():
        yield connection

    monkeypatch.setattr(api, "SERVICE_ACCOUNT_AUTH_ENABLED", True)
    monkeypatch.setattr(
        api.control_db, "get_pool",
        lambda: SimpleNamespace(connection=checkout))
    seen = []
    monkeypatch.setattr(
        api.control_db, "resolve_service_account",
        lambda conn, account, version, digest: (
            seen.append((conn, account, version, digest)) or route))

    principal = api._resolve_service_account("Bearer " + token)
    assert principal == _principal("documents.read", "rag.query")
    assert seen == [(connection, ACCOUNT_ID, 4, proof.digest)]


def test_malformed_service_token_never_checks_out_control_or_data(monkeypatch):
    monkeypatch.setattr(api, "SERVICE_ACCOUNT_AUTH_ENABLED", True)
    checked_out = []
    monkeypatch.setattr(
        api.control_db, "get_pool",
        lambda: checked_out.append(True))
    assert api._resolve_service_account("Bearer " + TOKEN) is None
    assert checked_out == []


def test_service_identity_cannot_enter_human_organization_routes(monkeypatch):
    client = _machine_client(monkeypatch, _principal(*control_db.SERVICE_ACCOUNT_SCOPES))
    response = client.get(
        "/v1/org/me", headers={"Authorization": "Bearer " + TOKEN})
    assert response.status_code == 403


def test_browser_session_and_machine_header_are_ambiguous(monkeypatch):
    session = SimpleNamespace(
        identity=object(), csrf_token="c" * 43, expires_at=4_000_000_000)
    store = SimpleNamespace(authenticate=lambda _cookie: session)
    monkeypatch.setattr(api, "OIDC_CLIENT", SimpleNamespace(store=store))
    monkeypatch.setattr(api, "SERVICE_ACCOUNT_AUTH_ENABLED", True)
    monkeypatch.setattr(api, "AUTH_REGISTRY", auth.Registry((), None))
    client = TestClient(api.app)
    client.cookies.set("rag_oidc_session", "opaque")
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer " + TOKEN})
    assert response.status_code == 401


def test_machine_context_is_never_an_internal_service_bypass(monkeypatch):
    seen = []
    connection = SimpleNamespace(rollback=lambda: seen.append("rollback"))

    @contextmanager
    def checkout():
        yield connection

    monkeypatch.setattr(
        api.db, "get_pool", lambda: SimpleNamespace(connection=checkout))
    monkeypatch.setattr(api, "_schema_ready", True)
    monkeypatch.setattr(
        api.db, "set_tenant_context",
        lambda _conn, tenant, *, service=False, actor_id=None: seen.append(
            (tenant, actor_id, service)))
    monkeypatch.setattr(
        api.db, "clear_tenant_context",
        lambda _conn: seen.append("clear"))

    with api._principal_db_conn(_principal("documents.read")):
        pass
    assert seen == [(TENANT_ID, None, False), "rollback", "clear"]


def test_table_model_needs_both_query_and_table_scopes(monkeypatch):
    client = _machine_client(monkeypatch, _principal("rag.query"))
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + TOKEN},
        json={"model": api.TABLE_MODEL_ID,
              "messages": [{"role": "user", "content": "extract"}]},
    )
    assert response.status_code == 403


def test_service_route_table_is_closed_real_and_machine_only():
    route_keys = {
        (method, route.path)
        for route in api.app.routes
        for method in getattr(route, "methods", ())
    }
    assert set(api.SERVICE_ROUTE_SCOPES).issubset(route_keys)
    assert set(api.SERVICE_ROUTE_SCOPES.values()).issubset(
        control_db.SERVICE_ACCOUNT_SCOPES)
    assert all(not path.startswith("/v1/org/")
               for _method, path in api.SERVICE_ROUTE_SCOPES)
    assert all(not path.startswith("/v1/eval/")
               for _method, path in api.SERVICE_ROUTE_SCOPES)
    with pytest.raises(TypeError):
        api.SERVICE_ROUTE_SCOPES[("GET", "/v1/org/me")] = "rag.query"
