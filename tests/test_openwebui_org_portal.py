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
    assert source.count("Depends(get_verified_user)") == 6
    assert source.count("app.add_api_route(") == 6
    assert "Authorization\": f\"Bearer {GATEWAY_KEY}" in source
    assert "X-OpenWebUI-User-Jwt" in source
