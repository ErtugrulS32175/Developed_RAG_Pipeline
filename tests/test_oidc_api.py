"""The browser OIDC road reaches the same tenant authority without tokens."""
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from pipeline.api import app as api
from pipeline.api import auth, oidc, oidc_session


TENANT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SUBJECT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
POSITION_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
IDENTITY = oidc.OIDCIdentity(
    "https://identity.example/realms/pilot",
    "opaque-subject",
    100,
    4_000_000_000,
)
PRINCIPAL = auth.Principal(
    TENANT_ID,
    "reader",
    SUBJECT_ID,
    "oidc",
    POSITION_ID,
    False,
)


class _Store:
    def __init__(self):
        self.cookie = "c" * 43
        self.csrf = "x" * 43
        self.revocations = []

    def authenticate(self, cookie):
        if cookie != self.cookie:
            return None
        return oidc_session.SessionIdentity(IDENTITY, self.csrf, 4_000_000_000)

    def revoke(self, cookie, csrf):
        self.revocations.append((cookie, csrf))
        return cookie == self.cookie and csrf == self.csrf


class _Client:
    def __init__(self):
        self.store = _Store()
        self.callbacks = []

    def begin(self):
        return oidc_session.LoginStart(
            "https://identity.example/authorize?closed=value",
            int(time.time()) + 60,
        )

    def callback(self, *, state, code):
        self.callbacks.append((state, code))
        session = oidc_session.SessionIdentity(
            IDENTITY, self.store.csrf, int(time.time()) + 60)
        return self.store.cookie, session


def _browser(monkeypatch):
    client = _Client()
    monkeypatch.setattr(api, "OIDC_CLIENT", client)
    # OIDC configuration disables the local-development open principal at
    # import. The module used by this unit process starts in local mode, so the
    # fixture must reproduce that production boundary explicitly.
    monkeypatch.setattr(api, "AUTH_REGISTRY", auth.Registry((), None))
    monkeypatch.setattr(api, "OIDC_REDIRECT_URI",
                        "https://rag.example/auth/callback")
    monkeypatch.setattr(api, "_resolve_oidc_principal", lambda _item: PRINCIPAL)
    return TestClient(api.app), client


def test_login_redirects_only_to_the_provider_authorization_endpoint(
        monkeypatch):
    browser, _client = _browser(monkeypatch)

    response = browser.get("/auth/login", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == (
        "https://identity.example/authorize?closed=value")
    assert "client_secret" not in response.headers["location"]


def test_callback_sets_one_secure_opaque_server_session(monkeypatch):
    browser, client = _browser(monkeypatch)

    response = browser.get(
        "/auth/callback?state=" + "s" * 43 + "&code=one-time-code",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert client.callbacks == [("s" * 43, "one-time-code")]
    cookie = response.headers["set-cookie"]
    assert cookie.startswith(oidc_session.COOKIE_NAME + "=" + "c" * 43)
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=lax" in cookie
    assert "opaque-subject" not in cookie


def test_session_publishes_only_closed_resolved_authority(monkeypatch):
    browser, client = _browser(monkeypatch)
    browser.cookies.set(oidc_session.COOKIE_NAME, client.store.cookie)

    response = browser.get("/auth/session")

    assert response.status_code == 200
    assert response.json() == {
        "authenticated": True,
        "tenant_id": str(TENANT_ID),
        "role": "reader",
        "source": "oidc",
        "position_id": str(POSITION_ID),
        "org_architect": False,
        "csrf_token": client.store.csrf,
        "expires_at": 4_000_000_000,
    }
    assert "opaque-subject" not in response.text
    assert "issuer" not in response.text


def test_logout_requires_csrf_and_revokes_the_server_session(monkeypatch):
    browser, client = _browser(monkeypatch)
    browser.cookies.set(oidc_session.COOKIE_NAME, client.store.cookie)

    refused = browser.post("/auth/logout")
    accepted = browser.post(
        "/auth/logout", headers={"X-RAGTest-CSRF": client.store.csrf})

    assert refused.status_code == 403
    assert accepted.status_code == 204
    assert client.store.revocations == [
        (client.store.cookie, ""),
        (client.store.cookie, client.store.csrf),
    ]
    assert (oidc_session.COOKIE_NAME + '=""' in
            accepted.headers["set-cookie"])


def test_every_unsafe_cookie_request_requires_the_session_csrf(monkeypatch):
    browser, client = _browser(monkeypatch)
    browser.cookies.set(oidc_session.COOKIE_NAME, client.store.cookie)

    path = "/documents/11111111-1111-1111-1111-111111111111/archive"
    absent = browser.post(path)
    accepted_by_identity = browser.post(
        path,
        headers={"X-RAGTest-CSRF": client.store.csrf},
    )

    assert absent.status_code == 401
    # The bound identity is only a reader, so the route's independent editor
    # gate still refuses it. Reaching that gate proves CSRF enabled auth; it
    # grants no role and touches neither a request body nor the database.
    assert accepted_by_identity.status_code == 403


def test_an_invalid_cookie_never_falls_back_to_an_open_principal(monkeypatch):
    browser, _client = _browser(monkeypatch)
    monkeypatch.setattr(
        api, "AUTH_REGISTRY", auth.Registry((), PRINCIPAL))
    browser.cookies.set(oidc_session.COOKIE_NAME, "not-a-session")

    response = browser.get("/auth/session")

    assert response.status_code == 401


class _ControlConnection:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _ControlPool:
    def connection(self):
        return _ControlConnection()


def test_control_routing_precedes_and_binds_the_data_plane(monkeypatch):
    order = []
    monkeypatch.setattr(api.control_db, "identity_digest",
                        lambda issuer, subject: b"d" * 32)
    monkeypatch.setattr(api.control_db, "get_pool", lambda: _ControlPool())

    def route(_connection, version, digest):
        order.append(("control", version, digest))
        return SimpleNamespace(facts=SimpleNamespace(tenant_id=TENANT_ID))

    def data(assertion, source):
        order.append(("data", assertion, source))
        return PRINCIPAL

    monkeypatch.setattr(api.control_db, "resolve_identity", route)
    monkeypatch.setattr(api, "_resolve_external_principal", data)
    monkeypatch.setattr(api, "OIDC_IDENTITY_KEY_VERSION", 7)

    assert api._resolve_oidc_principal(IDENTITY) is PRINCIPAL
    assert order == [
        ("control", 7, b"d" * 32),
        ("data", IDENTITY, "oidc"),
    ]


def test_an_unrouted_identity_never_touches_the_data_plane(monkeypatch):
    monkeypatch.setattr(api.control_db, "identity_digest",
                        lambda issuer, subject: b"d" * 32)
    monkeypatch.setattr(api.control_db, "get_pool", lambda: _ControlPool())
    monkeypatch.setattr(api.control_db, "resolve_identity",
                        lambda _connection, _version, _digest: None)

    def forbidden(*_args):
        raise AssertionError("data plane was touched")

    monkeypatch.setattr(api, "_resolve_external_principal", forbidden)

    assert api._resolve_oidc_principal(IDENTITY) is None


def test_oidc_humans_share_content_authority_but_api_keys_do_not():
    for source, expected in (
            ("openwebui", True), ("oidc", True), ("api_key", False)):
        principal = auth.Principal(
            TENANT_ID, "reader", SUBJECT_ID, source, POSITION_ID, False)
        token = auth.bind(principal)
        try:
            assert api._browser_evidence_enabled() is expected
        finally:
            auth.reset(token)


def _startup(tmp_path, values):
    environment = os.environ.copy()
    for name in (
            "API_KEY", "API_KEYS_JSON", "OPENWEBUI_GATEWAY_KEY",
            "OPENWEBUI_USER_JWT_SECRET", "EVIDENCE_HMAC_SECRET",
            "OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET",
            "OIDC_REDIRECT_URI", "OIDC_SESSION_SECRET",
            "CONTROL_IDENTITY_HMAC_SECRET", "PG_CONTROL_DSN",
            "ALLOW_INSECURE_LOCAL", "API_BIND_HOST"):
        environment.pop(name, None)
    environment.update(values)
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    environment["UPLOAD_DIR"] = str(tmp_path / "uploads")
    return subprocess.run(
        [sys.executable, "-c", (
            "from pipeline.api import app; "
            "assert app.OIDC_CLIENT is not None; "
            "assert app.AUTH_REGISTRY.open_principal is None; "
            "assert app._DOCS_OPEN is False")],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_partial_oidc_configuration_fails_startup_closed(tmp_path):
    result = _startup(tmp_path, {
        "OIDC_ISSUER": "https://identity.example/realms/pilot",
    })

    assert result.returncode != 0
    assert "OIDCConfigurationError" in result.stderr


def test_complete_oidc_configuration_disables_local_open_mode(tmp_path):
    result = _startup(tmp_path, {
        "OIDC_ISSUER": "https://identity.example/realms/pilot",
        "OIDC_CLIENT_ID": "ragtest-bff",
        "OIDC_CLIENT_SECRET": "c" * 32,
        "OIDC_REDIRECT_URI": "https://rag.example/auth/callback",
        "OIDC_SESSION_SECRET": "s" * 32,
        "CONTROL_IDENTITY_HMAC_SECRET": "i" * 32,
        "PG_CONTROL_DSN": "postgresql://not-contacted.invalid/control",
        "EVIDENCE_HMAC_SECRET": "e" * 32,
    })

    assert result.returncode == 0, result.stderr


def test_oidc_configuration_requires_an_independent_evidence_key(tmp_path):
    result = _startup(tmp_path, {
        "OIDC_ISSUER": "https://identity.example/realms/pilot",
        "OIDC_CLIENT_ID": "ragtest-bff",
        "OIDC_CLIENT_SECRET": "c" * 32,
        "OIDC_REDIRECT_URI": "https://rag.example/auth/callback",
        "OIDC_SESSION_SECRET": "s" * 32,
        "CONTROL_IDENTITY_HMAC_SECRET": "i" * 32,
        "PG_CONTROL_DSN": "postgresql://not-contacted.invalid/control",
    })

    assert result.returncode != 0
    assert "IdentityConfigurationError" in result.stderr
    assert "evidence HMAC" in result.stderr
