"""Server-held OIDC authorization-code/PKCE browser sessions."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

from pipeline.api.oidc import (
    MAX_DOCUMENT_BYTES,
    Discovery,
    JWKSVerifier,
    OIDCConfig,
    OIDCIdentity,
    OIDCRefused,
    parse_discovery,
)


MAX_CODE_LENGTH = 4096
MAX_PENDING_LOGINS = 1024
MAX_SESSIONS = 4096
COOKIE_NAME = "ragtest_session"
CSRF_HEADER = "x-ragtest-csrf"


class OIDCSessionRefused(ValueError):
    """A browser login, callback, session or CSRF value failed closed."""


@dataclass(frozen=True, slots=True)
class LoginStart:
    authorization_url: str
    expires_at: int


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    identity: OIDCIdentity
    csrf_token: str
    expires_at: int


@dataclass(frozen=True, slots=True)
class _PendingLogin:
    verifier: str
    nonce: str
    expires_at: int


@dataclass(frozen=True, slots=True)
class _Session:
    identity: OIDCIdentity
    csrf_token: str
    expires_at: int


class SessionStore:
    """Process-local bounded state; cookie material is stored only as a digest."""

    def __init__(self, secret, *, login_seconds=300, session_seconds=28800):
        if (type(secret) is not str
                or len(secret.encode("utf-8")) < 32):
            raise OIDCSessionRefused("OIDC session secret is invalid")
        if type(login_seconds) is not int or not 30 <= login_seconds <= 600:
            raise OIDCSessionRefused("OIDC login lifetime is invalid")
        if (type(session_seconds) is not int
                or not 300 <= session_seconds <= 86400):
            raise OIDCSessionRefused("OIDC session lifetime is invalid")
        self.login_seconds = login_seconds
        self.session_seconds = session_seconds
        self._secret = secret.encode("utf-8")
        self._pending = {}
        self._sessions = {}
        self._lock = threading.Lock()

    def begin(self, authorization_endpoint: str, config: OIDCConfig,
              *, now=None) -> LoginStart:
        current = _now(now)
        state = _random()
        nonce = _random()
        verifier = _random(48)
        challenge = _b64(hashlib.sha256(verifier.encode("ascii")).digest())
        with self._lock:
            self._purge(current)
            if len(self._pending) >= MAX_PENDING_LOGINS:
                raise OIDCSessionRefused("OIDC login capacity is exhausted")
            self._pending[self._digest(state)] = _PendingLogin(
                verifier, nonce, current + self.login_seconds)
        query = urlencode({
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "response_type": "code",
            "scope": "openid",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
        return LoginStart(
            authorization_endpoint + "?" + query,
            current + self.login_seconds,
        )

    def consume_login(self, state, *, now=None) -> _PendingLogin:
        current = _now(now)
        if not _opaque(state):
            raise OIDCSessionRefused("OIDC state is invalid")
        with self._lock:
            pending = self._pending.pop(self._digest(state), None)
        if pending is None or pending.expires_at <= current:
            raise OIDCSessionRefused("OIDC state is expired or replayed")
        return pending

    def create(self, identity: OIDCIdentity, *, now=None):
        current = _now(now)
        if type(identity) is not OIDCIdentity or identity.expires_at <= current:
            raise OIDCSessionRefused("OIDC identity is not active")
        cookie = _random(48)
        csrf = _random()
        expires = min(
            identity.expires_at,
            current + self.session_seconds,
        )
        with self._lock:
            self._purge(current)
            if len(self._sessions) >= MAX_SESSIONS:
                raise OIDCSessionRefused("OIDC session capacity is exhausted")
            self._sessions[self._digest(cookie)] = _Session(
                identity, csrf, expires)
        return cookie, SessionIdentity(identity, csrf, expires)

    def authenticate(self, cookie, *, now=None) -> SessionIdentity | None:
        current = _now(now)
        if not _opaque(cookie):
            return None
        digest = self._digest(cookie)
        with self._lock:
            session = self._sessions.get(digest)
            if session is not None and session.expires_at <= current:
                self._sessions.pop(digest, None)
                session = None
        if session is None:
            return None
        return SessionIdentity(
            session.identity, session.csrf_token, session.expires_at)

    def revoke(self, cookie, csrf_token, *, now=None) -> bool:
        session = self.authenticate(cookie, now=now)
        if session is None or not _opaque(csrf_token):
            return False
        if not secrets.compare_digest(session.csrf_token, csrf_token):
            return False
        with self._lock:
            return self._sessions.pop(self._digest(cookie), None) is not None

    def _digest(self, value):
        return hmac.new(
            self._secret, value.encode("ascii"), hashlib.sha256).digest()

    def _purge(self, now):
        self._pending = {
            key: value for key, value in self._pending.items()
            if value.expires_at > now
        }
        self._sessions = {
            key: value for key, value in self._sessions.items()
            if value.expires_at > now
        }


class OIDCClient:
    """Bounded network adapter; token values never leave the callback seam."""

    def __init__(self, config: OIDCConfig, client_secret: str,
                 store: SessionStore, *, transport=requests):
        if (type(client_secret) is not str
                or len(client_secret.encode("utf-8")) < 32):
            raise OIDCSessionRefused("OIDC client secret is invalid")
        self.config = config
        self._secret = client_secret
        self.store = store
        self._transport = transport
        self._discovery: Discovery | None = None
        self._verifier: JWKSVerifier | None = None

    def discover(self) -> Discovery:
        document = self._get_json(self.config.discovery_url)
        try:
            discovery = parse_discovery(document, self.config)
        except OIDCRefused as exc:
            raise OIDCSessionRefused(
                "OIDC discovery document was refused") from exc
        self._discovery = discovery
        self._verifier = JWKSVerifier(
            self.config, discovery, self._get_json)
        return discovery

    def begin(self, *, now=None) -> LoginStart:
        discovery = self._discovery or self.discover()
        return self.store.begin(
            discovery.authorization_endpoint, self.config, now=now)

    def callback(self, *, state, code, now=None):
        current = _now(now)
        pending = self.store.consume_login(state, now=current)
        if (type(code) is not str or not code or code != code.strip()
                or len(code) > MAX_CODE_LENGTH):
            raise OIDCSessionRefused("OIDC authorization code is invalid")
        discovery = self._discovery or self.discover()
        verifier = self._verifier
        if verifier is None:
            raise OIDCSessionRefused("OIDC verifier is unavailable")
        token_document = self._post_token(discovery.token_endpoint, {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "client_secret": self._secret,
            "code_verifier": pending.verifier,
        })
        if (type(token_document) is not dict
                or type(token_document.get("id_token")) is not str
                or token_document.get("token_type") != "Bearer"):
            raise OIDCSessionRefused("OIDC token response is invalid")
        try:
            identity = verifier.verify(
                token_document["id_token"],
                expected_nonce=pending.nonce,
                now=current,
            )
        except OIDCRefused as exc:
            raise OIDCSessionRefused("OIDC identity was refused") from exc
        return self.store.create(identity, now=current)

    def _get_json(self, url, *, max_bytes=MAX_DOCUMENT_BYTES,
                  timeout_seconds=5):
        try:
            response = self._transport.get(
                url, timeout=timeout_seconds, stream=True,
                headers={"Accept": "application/json"})
            return _bounded_json(response, max_bytes)
        except OIDCSessionRefused:
            raise
        except Exception as exc:
            raise OIDCSessionRefused("OIDC provider is unavailable") from exc

    def _post_token(self, url, data):
        try:
            response = self._transport.post(
                url, data=data, timeout=5, stream=True,
                headers={"Accept": "application/json"})
            return _bounded_json(response, MAX_DOCUMENT_BYTES)
        except OIDCSessionRefused:
            raise
        except Exception as exc:
            raise OIDCSessionRefused("OIDC provider is unavailable") from exc


def _bounded_json(response, maximum):
    chunks = []
    size = 0
    try:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if type(chunk) is not bytes:
                raise OIDCSessionRefused("OIDC provider response is invalid")
            size += len(chunk)
            if size > maximum:
                raise OIDCSessionRefused("OIDC provider response is too large")
            chunks.append(chunk)
    finally:
        response.close()
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OIDCSessionRefused("OIDC provider response is invalid") from exc
    if type(value) is not dict:
        raise OIDCSessionRefused("OIDC provider response is invalid")
    return value


def _random(size=32):
    return secrets.token_urlsafe(size)


def _opaque(value):
    return (type(value) is str and 32 <= len(value) <= 128
            and all(char.isascii() and (char.isalnum() or char in "-_")
                    for char in value))


def _b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _now(value):
    result = int(time.time()) if value is None else value
    if type(result) is not int:
        raise TypeError("now must be an integer")
    return result
