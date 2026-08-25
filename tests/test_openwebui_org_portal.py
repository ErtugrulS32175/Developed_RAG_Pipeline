"""Static and cryptographic contract for the trusted OpenWebUI Event Function."""
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline.api import identity


ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "openwebui" / "functions" / "ragtest_org_portal.py"


def _module(monkeypatch):
    monkeypatch.setenv(
        "OPENWEBUI_GATEWAY_KEY", "portal-gateway-key-with-more-than-32-bytes")
    monkeypatch.setenv(
        "FORWARD_USER_INFO_HEADER_JWT_SECRET",
        "portal-identity-key-with-more-than-32-bytes")
    spec = importlib.util.spec_from_file_location("ragtest_org_portal", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registered(monkeypatch):
    module = _module(monkeypatch)
    user = SimpleNamespace(id="owui-subject", email="display@example.invalid",
                           name="Display", role="user")
    root = types.ModuleType("open_webui")
    utils = types.ModuleType("open_webui.utils")
    auth_module = types.ModuleType("open_webui.utils.auth")
    auth_module.get_verified_user = lambda: user
    monkeypatch.setitem(sys.modules, "open_webui", root)
    monkeypatch.setitem(sys.modules, "open_webui.utils", utils)
    monkeypatch.setitem(sys.modules, "open_webui.utils.auth", auth_module)
    app = FastAPI()
    module.Event()._register(app)
    return module, TestClient(app), user


def test_portal_assertion_is_accepted_by_the_backend_verifier(monkeypatch):
    module = _module(monkeypatch)
    user = SimpleNamespace(id="owui-subject", email="display@example.invalid",
                           name="Display", role="admin")
    token = module.signed_identity(user, now=1_000)
    verifier = identity.Verifier.configured(
        "portal-identity-key-with-more-than-32-bytes")
    resolved = verifier.verify(token, now=1_030)
    assert resolved.subject == "owui-subject"
    assert not hasattr(resolved, "role")


def test_portal_keeps_secrets_server_side_and_renders_values_as_text(monkeypatch):
    module = _module(monkeypatch)
    html = module.PORTAL_HTML
    assert "OPENWEBUI_GATEWAY_KEY" not in html
    assert "FORWARD_USER_INFO_HEADER_JWT_SECRET" not in html
    assert "innerHTML" not in html
    assert "textContent" in html
    assert "eval(" not in html
    assert "<textarea" not in html
    assert "item.filename" not in html
    assert "item.content_sha256" not in html


def test_portal_csp_pins_exact_static_assets_and_refuses_ambient_sources(
        monkeypatch):
    module = _module(monkeypatch)
    assert "default-src 'none'" in module.PORTAL_CSP
    assert "connect-src 'self'" in module.PORTAL_CSP
    assert "frame-ancestors 'self'" in module.PORTAL_CSP
    assert ("sha256-" + module._source_hash(module.PORTAL_STYLE)
            in module.PORTAL_CSP)
    assert ("sha256-" + module._source_hash(module.PORTAL_SCRIPT)
            in module.PORTAL_CSP)
    assert "unsafe-inline" not in module.PORTAL_CSP
    assert "https:" not in module.PORTAL_CSP


def test_registered_page_is_session_bound_uncached_and_csp_protected(
        monkeypatch):
    module, client, _user = _registered(monkeypatch)
    response = client.get(module.PORTAL_PATH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-security-policy"] == module.PORTAL_CSP
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_portal_registers_only_session_authenticated_routes():
    source = PLUGIN.read_text(encoding="utf-8")
    assert source.count("Depends(get_verified_user)") == 18
    assert source.count("app.add_api_route(") == 18
    assert "Authorization\": f\"Bearer {GATEWAY_KEY}" in source
    assert "X-OpenWebUI-User-Jwt" in source


def test_every_mutation_requires_a_non_simple_same_origin_header(monkeypatch):
    module = _module(monkeypatch)
    assert not module._mutation_allowed(SimpleNamespace(headers={}))
    assert not module._mutation_allowed(SimpleNamespace(headers={
        "X-RAGTest-Portal": "same-origin",
        "content-type": "text/plain",
    }))
    assert module._mutation_allowed(SimpleNamespace(headers={
        "X-RAGTest-Portal": "same-origin",
        "content-type": "application/json; charset=utf-8",
    }))
    source = PLUGIN.read_text(encoding="utf-8")
    assert source.count("rejection = mutation_refused(request)") == 7
    assert "headers['X-RAGTest-Portal']='same-origin'" in source


def test_proxy_payloads_are_closed_before_the_backend_sees_them(monkeypatch):
    module = _module(monkeypatch)
    assert module._closed_payload(
        {"expected_revision": 1, "state": "active"},
        required={"expected_revision"}, optional={"state"})
    assert not module._closed_payload(
        {"expected_revision": 1, "secret": "x"},
        required={"expected_revision"}, optional={"state"})
    assert not module._closed_payload(
        {"state": "active"}, required={"expected_revision"},
        optional={"state"})


def test_registered_mutation_refuses_csrf_then_proxies_one_closed_body(
        monkeypatch):
    module, client, user = _registered(monkeypatch)
    calls = []

    async def proxy(actor, method, path, payload=None):
        calls.append((actor, method, path, payload))
        return 200, {"architecture_version": 2, "policy_epoch": 2}

    monkeypatch.setattr(module, "_proxy", proxy)
    payload = {
        "expected_version": 1, "name": "Example",
        "positions": [], "members": [],
    }
    refused = client.put(module.PORTAL_PATH + "/api/topology", json=payload)
    accepted = client.put(
        module.PORTAL_PATH + "/api/topology", json=payload,
        headers={"X-RAGTest-Portal": "same-origin"})
    assert refused.status_code == 403 and calls == [
        (user, "PUT", "/v1/org/admin/topology", payload)]
    assert accepted.status_code == 200


def test_portal_review_queue_is_content_free_and_decisions_are_closed(
        monkeypatch):
    module = _module(monkeypatch)
    html = module.PORTAL_HTML
    assert "/ragtest-org/api/reviews?reason_code=management_duty" in html
    assert "expected_revision:item.revision" in html
    assert "expected_policy_epoch:item.policy_epoch" in html
    assert "item.question" not in html and "item.answer" not in html
    assert "innerHTML" not in html
    source = PLUGIN.read_text(encoding="utf-8")
    assert 'set(payload) != allowed' in source
    assert '{"resolved", "dismissed"}' in source
    assert '{"detail": "gecersiz inceleme karari"}' in source


def test_portal_exposes_audit_events_endpoint():
    source = PLUGIN.read_text(encoding="utf-8")
    assert "/ragtest-org/api/events" in source
    assert "before_created_at" in source and "before_id" in source
    assert "EVENT_ACTIONS" in source


def test_audit_events_are_inside_the_architect_only_panel(monkeypatch):
    html = _module(monkeypatch).PORTAL_HTML
    assert 'data-tab="events" hidden' in html
    assert 'id="admin-nav"' in html
    assert "by('admin-nav').hidden=!me.architecture_admin" in html


def test_production_panel_has_structured_hierarchy_membership_and_retention(
        monkeypatch):
    html = _module(monkeypatch).PORTAL_HTML
    for control in (
            'id="position-form"', 'id="member-form"',
            'id="topology-save"', 'id="event-action"',
            'id="retention-days"', 'id="document-rows"',
            'id="hold-create"', 'id="purge-schedule"'):
        assert control in html
    assert "JSON.parse(document.getElementById('topology')" not in html
    assert "expected_architecture_version" in html
    assert "expected_policy_epoch" in html
    assert "/api/retention-documents" in html
    assert "/holds/" in html and "/purges" in html


def test_content_blind_document_panel_uses_only_closed_lifecycle_fields(
        monkeypatch):
    script = _module(monkeypatch).PORTAL_SCRIPT
    for field in (
            "document_id", "status", "revision", "active_hold_count",
            "latest_purge_state"):
        assert "item." + field in script
    for field in (
            "filename", "status_note", "content_sha256", "candidate_id"):
        assert "item." + field not in script
