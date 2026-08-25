"""Static and cryptographic contract for the trusted OpenWebUI Event Function."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_portal_registers_only_session_authenticated_routes():
    source = PLUGIN.read_text(encoding="utf-8")
    assert source.count("Depends(get_verified_user)") == 9
    assert source.count("app.add_api_route(") == 9
    assert "Authorization\": f\"Bearer {GATEWAY_KEY}" in source
    assert "X-OpenWebUI-User-Jwt" in source


def test_portal_review_queue_is_content_free_and_decisions_are_closed(
        monkeypatch):
    module = _module(monkeypatch)
    html = module.PORTAL_HTML
    assert "/ragtest-org/api/reviews?reason_code=management_duty" in html
    assert "expected_revision:c.revision" in html
    assert "expected_policy_epoch:c.policy_epoch" in html
    assert "c.question" not in html and "c.answer" not in html
    assert "innerHTML" not in html
    source = PLUGIN.read_text(encoding="utf-8")
    assert 'set(payload) != allowed' in source
    assert '{"resolved", "dismissed"}' in source
    assert '{"detail": "gecersiz inceleme karari"}' in source


def test_portal_exposes_audit_events_endpoint():
    source = PLUGIN.read_text(encoding="utf-8")
    assert "/ragtest-org/api/events" in source


def test_audit_events_are_inside_the_architect_only_panel(monkeypatch):
    html = _module(monkeypatch).PORTAL_HTML
    admin_start = html.index('<section id="admin" hidden>')
    events_start = html.index("<h2>Kurumsal olaylar</h2>")
    admin_end = html.index("</section>", admin_start)
    assert admin_start < events_start < admin_end
    assert "if(me.architecture_admin)" in html
