"""Authorization-code/PKCE and opaque browser-session security tests."""
import base64
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256

from pipeline.api import oidc
from pipeline.api import oidc_session


ISSUER = "https://identity.example/realms/pilot"
SESSION_SECRET = "s" * 32


def _b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _config():
    return oidc.OIDCConfig.configured(
        issuer=ISSUER,
        client_id="ragtest-bff",
        redirect_uri="https://rag.example/auth/callback",
        clock_skew_seconds=5,
        jwks_cache_seconds=30,
        jwks_overlap_seconds=60,
    )


def _provider_key():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_numbers()
    return private, {
        "kty": "RSA", "kid": "provider-key", "use": "sig", "alg": "RS256",
        "n": _b64(public.n.to_bytes((public.n.bit_length() + 7) // 8, "big")),
        "e": _b64(public.e.to_bytes((public.e.bit_length() + 7) // 8, "big")),
    }


def _token(private, nonce):
    header = {"alg": "RS256", "kid": "provider-key", "typ": "JWT"}
    claims = {
        "iss": ISSUER, "sub": "opaque-subject", "aud": "ragtest-bff",
        "iat": 100, "exp": 500, "nonce": nonce,
    }
    head = _b64(json.dumps(header, separators=(",", ":")).encode())
    body = _b64(json.dumps(claims, separators=(",", ":")).encode())
    signed = f"{head}.{body}".encode("ascii")
    signature = private.sign(signed, padding.PKCS1v15(), SHA256())
    return f"{head}.{body}.{_b64(signature)}"


class _Response:
    def __init__(self, value, *, status=200, chunks=None):
        self.value = value
        self.status = status
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError("provider error with sensitive prose")

    def iter_content(self, chunk_size):
        assert chunk_size == 4096
        if self.chunks is not None:
            yield from self.chunks
        else:
            yield json.dumps(self.value, separators=(",", ":")).encode()

    def close(self):
        self.closed = True


class _Transport:
    def __init__(self, jwk):
        self.jwk = jwk
        self.token = None
        self.posts = []
        self.responses = []

    def get(self, url, **kwargs):
        assert kwargs == {
            "timeout": 5, "stream": True,
            "headers": {"Accept": "application/json"},
        }
        if url.endswith("/.well-known/openid-configuration"):
            value = {
                "issuer": ISSUER,
                "authorization_endpoint": ISSUER + "/auth",
                "token_endpoint": ISSUER + "/token",
                "jwks_uri": ISSUER + "/certs",
                "id_token_signing_alg_values_supported": ["RS256"],
                "code_challenge_methods_supported": ["S256"],
            }
        elif url.endswith("/certs"):
            value = {"keys": [self.jwk]}
        else:
            raise AssertionError("unexpected provider URL")
        response = _Response(value)
        self.responses.append(response)
        return response

    def post(self, url, **kwargs):
        assert url == ISSUER + "/token"
        self.posts.append(kwargs)
        response = _Response({"id_token": self.token, "token_type": "Bearer"})
        self.responses.append(response)
        return response


def test_login_url_carries_one_use_state_nonce_and_s256_challenge():
    store = oidc_session.SessionStore(
        SESSION_SECRET, login_seconds=60, session_seconds=300)
    start = store.begin(ISSUER + "/auth", _config(), now=100)
    query = parse_qs(urlsplit(start.authorization_url).query)
    assert set(query) == {
        "client_id", "redirect_uri", "response_type", "scope", "state",
        "nonce", "code_challenge", "code_challenge_method",
    }
    assert query["code_challenge_method"] == ["S256"]
    assert query["response_type"] == ["code"]
    assert query["scope"] == ["openid"]
    pending = store.consume_login(query["state"][0], now=101)
    expected = _b64(hashlib.sha256(pending.verifier.encode("ascii")).digest())
    assert query["code_challenge"] == [expected]
    assert query["nonce"] == [pending.nonce]
    with pytest.raises(oidc_session.OIDCSessionRefused, match="replayed"):
        store.consume_login(query["state"][0], now=102)


def test_expired_login_state_is_burned_and_cannot_be_replayed():
    store = oidc_session.SessionStore(
        SESSION_SECRET, login_seconds=30, session_seconds=300)
    start = store.begin(ISSUER + "/auth", _config(), now=100)
    state = parse_qs(urlsplit(start.authorization_url).query)["state"][0]
    with pytest.raises(oidc_session.OIDCSessionRefused, match="expired"):
        store.consume_login(state, now=131)
    with pytest.raises(oidc_session.OIDCSessionRefused, match="replayed"):
        store.consume_login(state, now=110)


def test_callback_exchanges_verifier_once_and_creates_an_opaque_session():
    private, jwk = _provider_key()
    transport = _Transport(jwk)
    store = oidc_session.SessionStore(
        SESSION_SECRET, login_seconds=60, session_seconds=300)
    client = oidc_session.OIDCClient(
        _config(), "c" * 32, store, transport=transport)
    start = client.begin(now=100)
    query = parse_qs(urlsplit(start.authorization_url).query)
    transport.token = _token(private, query["nonce"][0])
    cookie, session = client.callback(
        state=query["state"][0], code="one-use-code", now=110)

    assert session.identity.subject == "opaque-subject"
    assert session.expires_at == 410
    assert cookie not in store._sessions
    assert store._digest(cookie) in store._sessions
    post = transport.posts[0]
    assert post["timeout"] == 5 and post["stream"] is True
    assert post["data"]["code_verifier"]
    assert post["data"]["client_secret"] == "c" * 32
    assert all(response.closed for response in transport.responses)
    with pytest.raises(oidc_session.OIDCSessionRefused, match="replayed"):
        client.callback(state=query["state"][0], code="again", now=111)
    assert len(transport.posts) == 1


def test_cookie_authentication_expiry_logout_and_csrf_are_closed():
    identity = oidc.OIDCIdentity(ISSUER, "subject", 100, 500)
    store = oidc_session.SessionStore(
        SESSION_SECRET, login_seconds=60, session_seconds=300)
    cookie, session = store.create(identity, now=110)
    assert store.authenticate(cookie, now=120) == session
    assert not store.revoke(cookie, "wrong-csrf-token-value-that-is-long-enough",
                            now=120)
    assert store.authenticate(cookie, now=120) is not None
    assert store.revoke(cookie, session.csrf_token, now=120)
    assert store.authenticate(cookie, now=120) is None

    short_cookie, _short_session = store.create(identity, now=490)
    assert store.authenticate(short_cookie, now=499) is not None
    assert store.authenticate(short_cookie, now=500) is None


def test_invalid_cookie_and_csrf_never_raise_or_delete_another_session():
    identity = oidc.OIDCIdentity(ISSUER, "subject", 100, 500)
    store = oidc_session.SessionStore(
        SESSION_SECRET, login_seconds=60, session_seconds=300)
    cookie, session = store.create(identity, now=110)
    for hostile in (None, 7, "", "short", "x y", "x" * 129):
        assert store.authenticate(hostile, now=120) is None
        assert not store.revoke(cookie, hostile, now=120)
    assert store.authenticate(cookie, now=120) == session


def test_provider_response_is_bounded_closed_and_always_closed():
    response = _Response({}, chunks=[b"x" * 10, b"y" * 10])
    with pytest.raises(oidc_session.OIDCSessionRefused, match="too large"):
        oidc_session._bounded_json(response, 15)
    assert response.closed

    response = _Response({}, status=500)
    with pytest.raises(RuntimeError):
        oidc_session._bounded_json(response, 100)
    assert response.closed


def test_provider_failure_and_bad_token_response_hide_vendor_values():
    _private, jwk = _provider_key()

    class FailedTransport(_Transport):
        def get(self, _url, **_kwargs):
            raise RuntimeError("provider host client secret and path")

    client = oidc_session.OIDCClient(
        _config(), "c" * 32,
        oidc_session.SessionStore(
            SESSION_SECRET, login_seconds=60, session_seconds=300),
        transport=FailedTransport(jwk))
    with pytest.raises(oidc_session.OIDCSessionRefused) as caught:
        client.begin(now=100)
    assert "provider host" not in str(caught.value)


def test_callback_burns_state_before_invalid_code_or_provider_work():
    _private, jwk = _provider_key()
    transport = _Transport(jwk)
    client = oidc_session.OIDCClient(
        _config(), "c" * 32,
        oidc_session.SessionStore(
            SESSION_SECRET, login_seconds=60, session_seconds=300),
        transport=transport)
    start = client.begin(now=100)
    state = parse_qs(urlsplit(start.authorization_url).query)["state"][0]
    with pytest.raises(oidc_session.OIDCSessionRefused, match="code"):
        client.callback(state=state, code="", now=110)
    with pytest.raises(oidc_session.OIDCSessionRefused, match="replayed"):
        client.callback(state=state, code="valid", now=111)
    assert transport.posts == []
