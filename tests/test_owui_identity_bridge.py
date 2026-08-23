import base64
import hashlib
import hmac
import json
import importlib
import time
import uuid

import pytest

from pipeline.api import identity


SECRET = "identity-test-secret-with-more-than-32-bytes"
GATEWAY_KEY = "gateway-only-test-key-with-more-than-32-bytes"


def _part(value):
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _token(*, secret=SECRET, header=None, **overrides):
    claims = {
        "sub": "owui-user-123",
        "email": "display@example.invalid",
        "name": "Display Only",
        "role": "admin",
        "iss": "open-webui",
        "iat": 1_000,
        "exp": 1_060,
    }
    claims.update(overrides)
    head = _part(header or {"alg": "HS256", "typ": "JWT"})
    body = _part(claims)
    signature = hmac.new(
        secret.encode("utf-8"), f"{head}.{body}".encode("ascii"),
        hashlib.sha256).digest()
    tail = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{head}.{body}.{tail}"


def test_a_valid_short_lived_assertion_returns_only_identity_authority():
    verifier = identity.Verifier.configured(SECRET)

    resolved = verifier.verify(_token(), now=1_030)

    assert resolved == identity.ForwardedIdentity(
        "open-webui", "owui-user-123", 1_000, 1_060)
    assert not hasattr(resolved, "role")
    assert not hasattr(resolved, "email")
    assert not hasattr(resolved, "name")


@pytest.mark.parametrize("token", [
    _token(secret="different-secret-with-more-than-32-bytes"),
    _token(iss="somewhere-else"),
    _token(exp=999),
    _token(iat=1_040, exp=1_060),
    _token(iat=1_000, exp=1_061),
    _token(sub=""),
    _token(sub=" leading-space"),
    _token(header={"alg": "none", "typ": "JWT"}),
])
def test_invalid_assertions_fail_closed(token):
    verifier = identity.Verifier.configured(SECRET)
    with pytest.raises(identity.IdentityRefused):
        verifier.verify(token, now=1_030)


def test_claims_are_closed_not_a_client_capability_channel():
    verifier = identity.Verifier.configured(SECRET)
    token = _token(level=99)
    with pytest.raises(identity.IdentityRefused, match="alanlari"):
        verifier.verify(token, now=1_030)


@pytest.mark.parametrize("secret", ["", "short", " leading"])
def test_weak_or_ambiguous_secrets_are_configuration_errors(secret):
    with pytest.raises(identity.IdentityConfigurationError):
        identity.Verifier.configured(secret)


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("API_KEYS_JSON", "")
    monkeypatch.setenv("OPENWEBUI_GATEWAY_KEY", GATEWAY_KEY)
    monkeypatch.setenv("OPENWEBUI_USER_JWT_SECRET", SECRET)
    import pipeline.api.app as api
    importlib.reload(api)
    resolved = api.auth.Principal(
        uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        "reader",
        uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        "openwebui",
    )
    monkeypatch.setattr(api, "_resolve_forwarded_principal", lambda _a: resolved)
    yield api
    monkeypatch.delenv("OPENWEBUI_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("OPENWEBUI_USER_JWT_SECRET", raising=False)
    importlib.reload(api)


def _live_token():
    now = int(time.time())
    return _token(iat=now, exp=now + 60)


def test_gateway_requires_both_provider_key_and_signed_human(gateway):
    from fastapi.testclient import TestClient

    client = TestClient(gateway.app)
    headers = {
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-OpenWebUI-User-Jwt": _live_token(),
    }
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
    ).status_code == 401
    assert client.get(
        "/v1/models",
        headers={"X-OpenWebUI-User-Jwt": _live_token()},
    ).status_code == 401


def test_plain_spoof_headers_are_refused_even_with_the_gateway_key(gateway):
    from fastapi.testclient import TestClient

    response = TestClient(gateway.app).get("/v1/models", headers={
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-OpenWebUI-User-Id": "forged-user",
        "X-OpenWebUI-User-Role": "admin",
    })
    assert response.status_code == 401


def test_openwebui_admin_claim_does_not_grant_an_editor_action(gateway):
    from fastapi.testclient import TestClient

    response = TestClient(gateway.app).post("/documents/upload", headers={
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-OpenWebUI-User-Jwt": _live_token(),
    })
    assert response.status_code == 403


def test_gateway_key_cannot_equal_a_legacy_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", GATEWAY_KEY)
    monkeypatch.setenv("OPENWEBUI_GATEWAY_KEY", GATEWAY_KEY)
    monkeypatch.setenv("OPENWEBUI_USER_JWT_SECRET", SECRET)
    import pipeline.api.app as api
    with pytest.raises(identity.IdentityConfigurationError, match="ayni"):
        importlib.reload(api)
    monkeypatch.delenv("OPENWEBUI_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("OPENWEBUI_USER_JWT_SECRET", raising=False)
    monkeypatch.setenv("API_KEY", "")
    importlib.reload(api)


def test_local_open_mode_still_binds_its_explicit_default_principal(
        monkeypatch):
    from starlette.requests import Request

    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("API_KEYS_JSON", "")
    monkeypatch.delenv("OPENWEBUI_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("OPENWEBUI_USER_JWT_SECRET", raising=False)
    import pipeline.api.app as api
    importlib.reload(api)
    request = Request({"type": "http", "headers": []})

    assert api._request_principal(request) == api.AUTH_REGISTRY.open_principal
    assert api.AUTH_REGISTRY.open_principal is not None
