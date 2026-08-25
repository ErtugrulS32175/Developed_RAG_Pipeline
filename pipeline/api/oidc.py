"""Closed OIDC discovery, rotating JWKS and ID-token verification."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from types import MappingProxyType
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256


MAX_DOCUMENT_BYTES = 65536
MAX_TOKEN_BYTES = 16384
MAX_KEYS = 8
MAX_SUBJECT_LENGTH = 512


class OIDCConfigurationError(RuntimeError):
    """Deployment OIDC settings are absent or unsafe."""


class OIDCRefused(ValueError):
    """Provider metadata, key material or an identity token failed closed."""


@dataclass(frozen=True, slots=True)
class OIDCConfig:
    issuer: str
    client_id: str
    redirect_uri: str
    clock_skew_seconds: int
    jwks_cache_seconds: int
    jwks_overlap_seconds: int

    @classmethod
    def configured(cls, *, issuer, client_id, redirect_uri,
                   clock_skew_seconds=30, jwks_cache_seconds=300,
                   jwks_overlap_seconds=900, allow_loopback_http=False):
        issuer_value = _absolute_url(
            issuer, "issuer", allow_loopback_http=allow_loopback_http,
            query=False)
        redirect_value = _absolute_url(
            redirect_uri, "redirect_uri",
            allow_loopback_http=allow_loopback_http, query=False)
        if issuer_value.endswith("/"):
            raise OIDCConfigurationError("OIDC issuer must not end with slash")
        if not _closed_text(client_id, 200):
            raise OIDCConfigurationError("OIDC client_id is invalid")
        for value, low, high, name in (
                (clock_skew_seconds, 0, 120, "clock skew"),
                (jwks_cache_seconds, 30, 3600, "JWKS cache"),
                (jwks_overlap_seconds, 60, 86400, "JWKS overlap")):
            if type(value) is not int or not low <= value <= high:
                raise OIDCConfigurationError(f"OIDC {name} is invalid")
        return cls(issuer_value, client_id, redirect_value,
                   clock_skew_seconds, jwks_cache_seconds,
                   jwks_overlap_seconds)

    @property
    def discovery_url(self) -> str:
        return self.issuer + "/.well-known/openid-configuration"


@dataclass(frozen=True, slots=True)
class Discovery:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str


@dataclass(frozen=True, slots=True)
class OIDCIdentity:
    issuer: str
    subject: str
    issued_at: int
    expires_at: int


@dataclass(frozen=True, slots=True)
class _CachedKey:
    key: rsa.RSAPublicKey
    retired_at: int | None


def parse_discovery(document, config: OIDCConfig) -> Discovery:
    if type(document) is not dict:
        raise OIDCRefused("OIDC discovery is invalid")
    required = {
        "issuer", "authorization_endpoint", "token_endpoint", "jwks_uri",
        "id_token_signing_alg_values_supported",
        "code_challenge_methods_supported",
    }
    if not required <= document.keys():
        raise OIDCRefused("OIDC discovery is incomplete")
    if document["issuer"] != config.issuer:
        raise OIDCRefused("OIDC discovery issuer mismatch")
    algorithms = document["id_token_signing_alg_values_supported"]
    challenges = document["code_challenge_methods_supported"]
    if (type(algorithms) is not list or "RS256" not in algorithms
            or type(challenges) is not list or "S256" not in challenges):
        raise OIDCRefused("OIDC discovery capabilities are insufficient")
    endpoints = {}
    for name in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        endpoint = _provider_endpoint(document[name], config)
        endpoints[name] = endpoint
    return Discovery(config.issuer, endpoints["authorization_endpoint"],
                     endpoints["token_endpoint"], endpoints["jwks_uri"])


class JWKSVerifier:
    """Bounded key cache with one refresh for an unknown key id."""

    def __init__(self, config: OIDCConfig, discovery: Discovery, fetch_json):
        self.config = config
        self.discovery = discovery
        self._fetch_json = fetch_json
        self._keys = MappingProxyType({})
        self._refreshed_at = 0

    @property
    def key_ids(self):
        return frozenset(self._keys)

    def refresh(self, *, now=None) -> None:
        current = _now(now)
        try:
            document = self._fetch_json(
                self.discovery.jwks_uri,
                max_bytes=MAX_DOCUMENT_BYTES,
                timeout_seconds=5,
            )
        except OIDCRefused:
            raise
        except Exception as exc:
            raise OIDCRefused("OIDC JWKS is unavailable") from exc
        fresh = _parse_jwks(document)
        combined = {}
        for kid, cached in self._keys.items():
            if kid in fresh:
                continue
            retired_at = cached.retired_at
            if retired_at is None:
                retired_at = current
            if current - retired_at <= self.config.jwks_overlap_seconds:
                combined[kid] = _CachedKey(cached.key, retired_at)
        for kid, key in fresh.items():
            combined[kid] = _CachedKey(key, None)
        if len(combined) > MAX_KEYS * 2:
            raise OIDCRefused("OIDC JWKS overlap is too large")
        self._keys = MappingProxyType(combined)
        self._refreshed_at = current

    def verify(self, token, *, expected_nonce, now=None) -> OIDCIdentity:
        current = _now(now)
        header, claims, signed, signature = _token_parts(token)
        kid = header["kid"]
        cached = self._keys.get(kid)
        stale = current - self._refreshed_at >= self.config.jwks_cache_seconds
        if cached is None or stale:
            self.refresh(now=current)
            cached = self._keys.get(kid)
        if cached is None:
            raise OIDCRefused("OIDC signing key is unknown")
        if (cached.retired_at is not None
                and current - cached.retired_at
                > self.config.jwks_overlap_seconds):
            raise OIDCRefused("OIDC signing key is stale")
        try:
            cached.key.verify(signature, signed, padding.PKCS1v15(), SHA256())
        except (InvalidSignature, ValueError) as exc:
            raise OIDCRefused("OIDC token signature is invalid") from exc
        return _validate_claims(
            claims, self.config, expected_nonce=expected_nonce, now=current)


def _parse_jwks(document):
    if type(document) is not dict or set(document) != {"keys"}:
        raise OIDCRefused("OIDC JWKS is invalid")
    keys = document["keys"]
    if type(keys) is not list or not 1 <= len(keys) <= MAX_KEYS:
        raise OIDCRefused("OIDC JWKS key count is invalid")
    parsed = {}
    for item in keys:
        if (type(item) is not dict
                or not {"kty", "kid", "use", "alg", "n", "e"}
                <= item.keys()
                or item["kty"] != "RSA" or item["use"] != "sig"
                or item["alg"] != "RS256"
                or not _closed_text(item["kid"], 128)):
            raise OIDCRefused("OIDC JWK is invalid")
        kid = item["kid"]
        if kid in parsed:
            raise OIDCRefused("OIDC JWK kid is duplicated")
        modulus = int.from_bytes(_decode(item["n"]), "big")
        exponent = int.from_bytes(_decode(item["e"]), "big")
        if modulus.bit_length() < 2048 or exponent not in (3, 65537):
            raise OIDCRefused("OIDC RSA key is invalid")
        try:
            parsed[kid] = rsa.RSAPublicNumbers(exponent, modulus).public_key()
        except ValueError as exc:
            raise OIDCRefused("OIDC RSA key is invalid") from exc
    return parsed


def _token_parts(token):
    if (type(token) is not str or not token or token != token.strip()
            or len(token.encode("utf-8")) > MAX_TOKEN_BYTES):
        raise OIDCRefused("OIDC token is invalid")
    parts = token.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise OIDCRefused("OIDC token is invalid")
    header = _json_part(parts[0])
    claims = _json_part(parts[1])
    if (type(header) is not dict
            or header != {
                "alg": "RS256", "kid": header.get("kid"), "typ": "JWT"}):
        raise OIDCRefused("OIDC token header is invalid")
    if not _closed_text(header["kid"], 128) or type(claims) is not dict:
        raise OIDCRefused("OIDC token is invalid")
    try:
        signed = f"{parts[0]}.{parts[1]}".encode("ascii")
    except UnicodeEncodeError as exc:
        raise OIDCRefused("OIDC token is invalid") from exc
    return header, claims, signed, _decode(parts[2])


def _validate_claims(claims, config, *, expected_nonce, now):
    if not _closed_text(expected_nonce, 256):
        raise OIDCRefused("OIDC nonce authority is invalid")
    required = {"iss", "sub", "aud", "exp", "iat", "nonce"}
    if not required <= claims.keys():
        raise OIDCRefused("OIDC token claims are incomplete")
    if claims["iss"] != config.issuer:
        raise OIDCRefused("OIDC token issuer mismatch")
    subject = claims["sub"]
    if not _closed_text(subject, MAX_SUBJECT_LENGTH):
        raise OIDCRefused("OIDC token subject is invalid")
    audience = claims["aud"]
    if type(audience) is str:
        audiences = [audience]
    elif (type(audience) is list and audience
          and all(_closed_text(value, 200) for value in audience)
          and len(set(audience)) == len(audience)):
        audiences = audience
    else:
        raise OIDCRefused("OIDC token audience is invalid")
    if config.client_id not in audiences:
        raise OIDCRefused("OIDC token audience mismatch")
    if len(audiences) > 1 and claims.get("azp") != config.client_id:
        raise OIDCRefused("OIDC token authorized party mismatch")
    issued_at = claims["iat"]
    expires_at = claims["exp"]
    if (type(issued_at) is not int or type(expires_at) is not int
            or expires_at <= issued_at):
        raise OIDCRefused("OIDC token time is invalid")
    if issued_at > now + config.clock_skew_seconds:
        raise OIDCRefused("OIDC token is not active")
    if expires_at <= now - config.clock_skew_seconds:
        raise OIDCRefused("OIDC token is expired")
    if claims["nonce"] != expected_nonce:
        raise OIDCRefused("OIDC token nonce mismatch")
    return OIDCIdentity(config.issuer, subject, issued_at, expires_at)


def _provider_endpoint(value, config):
    issuer_scheme = urlsplit(config.issuer).scheme
    endpoint = _absolute_url(value, "provider endpoint",
                             allow_loopback_http=issuer_scheme == "http",
                             query=False,
                             refusal=OIDCRefused)
    issuer = urlsplit(config.issuer)
    parsed = urlsplit(endpoint)
    prefix = issuer.path.rstrip("/") + "/"
    if (parsed.scheme != issuer.scheme or parsed.netloc != issuer.netloc
            or not parsed.path.startswith(prefix)):
        raise OIDCRefused("OIDC provider endpoint escaped issuer")
    return endpoint


def _absolute_url(value, name, *, allow_loopback_http, query, 
                  refusal=OIDCConfigurationError):
    if type(value) is not str or not value or value != value.strip():
        raise refusal(f"OIDC {name} is invalid")
    parsed = urlsplit(value)
    if (parsed.fragment or (parsed.query and not query) or parsed.username
            or parsed.password or not parsed.hostname or not parsed.path
            or "%" in parsed.path
            or any(part in {".", ".."} for part in parsed.path.split("/"))):
        raise refusal(f"OIDC {name} is invalid")
    loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (
            allow_loopback_http and loopback and parsed.scheme == "http"):
        raise refusal(f"OIDC {name} must use HTTPS")
    return value


def _closed_text(value, maximum):
    return (type(value) is str and value and value == value.strip()
            and len(value) <= maximum
            and all(32 <= ord(char) < 127 for char in value))


def _decode(value):
    if (type(value) is not str or not value or any(
            char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789-_" for char in value)):
        raise OIDCRefused("OIDC base64 is invalid")
    try:
        return base64.urlsafe_b64decode(
            value + "=" * ((4 - len(value) % 4) % 4))
    except Exception as exc:
        raise OIDCRefused("OIDC base64 is invalid") from exc


def _json_part(value):
    try:
        result = json.loads(_decode(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OIDCRefused("OIDC JSON is invalid") from exc
    return result


def _now(value):
    result = int(time.time()) if value is None else value
    if type(result) is not int:
        raise TypeError("now must be an integer")
    return result
