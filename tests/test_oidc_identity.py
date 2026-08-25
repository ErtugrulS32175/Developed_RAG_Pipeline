"""OIDC discovery, issuer, audience, nonce and JWKS rotation attacks."""
import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256

from pipeline.api import oidc


ISSUER = "https://identity.example/realms/pilot"
CLIENT = "ragtest-bff"
REDIRECT = "https://rag.example/auth/callback"


def _b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _config(**overrides):
    values = {
        "issuer": ISSUER,
        "client_id": CLIENT,
        "redirect_uri": REDIRECT,
        "clock_skew_seconds": 5,
        "jwks_cache_seconds": 30,
        "jwks_overlap_seconds": 60,
    }
    values.update(overrides)
    return oidc.OIDCConfig.configured(**values)


def _discovery(**overrides):
    values = {
        "issuer": ISSUER,
        "authorization_endpoint": ISSUER + "/protocol/openid-connect/auth",
        "token_endpoint": ISSUER + "/protocol/openid-connect/token",
        "jwks_uri": ISSUER + "/protocol/openid-connect/certs",
        "id_token_signing_alg_values_supported": ["RS256"],
        "code_challenge_methods_supported": ["S256"],
    }
    values.update(overrides)
    return values


def _key(kid):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
        "n": _b64(public.n.to_bytes((public.n.bit_length() + 7) // 8, "big")),
        "e": _b64(public.e.to_bytes((public.e.bit_length() + 7) // 8, "big")),
    }
    return private, jwk


def _token(private, kid="key-one", **claim_overrides):
    header = {"alg": "RS256", "kid": kid, "typ": "JWT"}
    claims = {
        "iss": ISSUER,
        "sub": "opaque-subject",
        "aud": CLIENT,
        "iat": 100,
        "exp": 200,
        "nonce": "one-use-nonce",
    }
    claims.update(claim_overrides)
    head = _b64(json.dumps(header, separators=(",", ":")).encode())
    body = _b64(json.dumps(claims, separators=(",", ":")).encode())
    signed = f"{head}.{body}".encode("ascii")
    signature = private.sign(signed, padding.PKCS1v15(), SHA256())
    return f"{head}.{body}.{_b64(signature)}"


class _Documents:
    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = []

    def __call__(self, url, **limits):
        self.calls.append((url, limits))
        if not self.documents:
            raise AssertionError("unexpected network refresh")
        return self.documents.pop(0)


@pytest.mark.parametrize("value", [
    "http://identity.example/realms/pilot",
    "https://user@identity.example/realms/pilot",
    "https://identity.example/realms/pilot?tenant=x",
    "https://identity.example/realms/pilot#fragment",
    " https://identity.example/realms/pilot",
])
def test_issuer_configuration_refuses_unsafe_coordinates(value):
    with pytest.raises(oidc.OIDCConfigurationError):
        _config(issuer=value)


def test_literal_loopback_is_the_only_http_configuration_exception():
    config = _config(
        issuer="http://127.0.0.1:8080/realms/pilot",
        redirect_uri="http://localhost:8000/auth/callback",
        allow_loopback_http=True,
    )
    assert config.issuer.startswith("http://127.0.0.1:")
    with pytest.raises(oidc.OIDCConfigurationError):
        _config(
            issuer="http://identity.local/realms/pilot",
            redirect_uri="http://localhost:8000/auth/callback",
            allow_loopback_http=True,
        )


@pytest.mark.parametrize("change", [
    {"issuer": "https://forged.example/realms/pilot"},
    {"jwks_uri": "https://forged.example/keys"},
    {"jwks_uri": ISSUER + "/../other/keys"},
    {"id_token_signing_alg_values_supported": ["HS256"]},
    {"code_challenge_methods_supported": ["plain"]},
])
def test_discovery_cannot_move_authority_or_weaken_algorithms(change):
    with pytest.raises(oidc.OIDCRefused):
        oidc.parse_discovery(_discovery(**change), _config())


def test_valid_discovery_and_token_use_bounded_fetch_and_closed_identity():
    private, jwk = _key("key-one")
    config = _config()
    discovery = oidc.parse_discovery(_discovery(extra_provider_field=True),
                                     config)
    documents = _Documents([{"keys": [jwk]}])
    verifier = oidc.JWKSVerifier(config, discovery, documents)
    identity = verifier.verify(
        _token(private), expected_nonce="one-use-nonce", now=150)
    assert identity == oidc.OIDCIdentity(
        ISSUER, "opaque-subject", 100, 200)
    assert documents.calls == [(discovery.jwks_uri, {
        "max_bytes": oidc.MAX_DOCUMENT_BYTES,
        "timeout_seconds": 5,
    })]


@pytest.mark.parametrize("claims", [
    {"iss": "https://forged.example/realms/pilot"},
    {"aud": "other-client"},
    {"aud": [CLIENT, "second"], "azp": "other-client"},
    {"exp": 145},
    {"iat": 160},
    {"nonce": "replayed-nonce"},
    {"sub": ""},
])
def test_forged_or_stale_claims_fail_closed(claims):
    private, jwk = _key("key-one")
    documents = _Documents([{"keys": [jwk]}])
    verifier = oidc.JWKSVerifier(
        _config(), oidc.parse_discovery(_discovery(), _config()), documents)
    with pytest.raises(oidc.OIDCRefused):
        verifier.verify(_token(private, **claims),
                        expected_nonce="one-use-nonce", now=150)


def test_multiple_audiences_require_the_exact_authorized_party():
    private, jwk = _key("key-one")
    verifier = oidc.JWKSVerifier(
        _config(), oidc.parse_discovery(_discovery(), _config()),
        _Documents([{"keys": [jwk]}]))
    identity = verifier.verify(
        _token(private, aud=["other", CLIENT], azp=CLIENT),
        expected_nonce="one-use-nonce", now=150)
    assert identity.subject == "opaque-subject"


@pytest.mark.parametrize("header_change", [
    {"alg": "none"},
    {"alg": "HS256"},
    {"jku": "https://forged.example/keys"},
    {"typ": "at+jwt"},
])
def test_token_header_cannot_select_an_algorithm_or_key_source(header_change):
    private, jwk = _key("key-one")
    token = _token(private)
    head, body, _signature = token.split(".")
    header = json.loads(base64.urlsafe_b64decode(head + "=="))
    header.update(header_change)
    forged_head = _b64(json.dumps(header, separators=(",", ":")).encode())
    signed = f"{forged_head}.{body}".encode("ascii")
    signature = private.sign(signed, padding.PKCS1v15(), SHA256())
    forged = f"{forged_head}.{body}.{_b64(signature)}"
    verifier = oidc.JWKSVerifier(
        _config(), oidc.parse_discovery(_discovery(), _config()),
        _Documents([{"keys": [jwk]}]))
    with pytest.raises(oidc.OIDCRefused):
        verifier.verify(forged, expected_nonce="one-use-nonce", now=150)


def test_unknown_kid_gets_one_refresh_and_never_uses_token_urls():
    private, _jwk = _key("unknown")
    documents = _Documents([{"keys": [_key("known")[1]]}])
    discovery = oidc.parse_discovery(_discovery(), _config())
    verifier = oidc.JWKSVerifier(_config(), discovery, documents)
    with pytest.raises(oidc.OIDCRefused, match="unknown"):
        verifier.verify(_token(private, kid="unknown"),
                        expected_nonce="one-use-nonce", now=150)
    assert [call[0] for call in documents.calls] == [discovery.jwks_uri]


def test_matching_kid_with_a_forged_signature_fails_closed():
    trusted_private, trusted_jwk = _key("shared-kid")
    forged_private, _forged_jwk = _key("shared-kid")
    verifier = oidc.JWKSVerifier(
        _config(), oidc.parse_discovery(_discovery(), _config()),
        _Documents([{"keys": [trusted_jwk]}]))
    with pytest.raises(oidc.OIDCRefused, match="signature"):
        verifier.verify(
            _token(forged_private, kid="shared-kid"),
            expected_nonce="one-use-nonce", now=150)
    assert trusted_private is not None


def test_jwks_transport_failure_does_not_escape_vendor_prose():
    class TransportFailure:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("vendor endpoint and credential prose")

    verifier = oidc.JWKSVerifier(
        _config(), oidc.parse_discovery(_discovery(), _config()),
        TransportFailure())
    with pytest.raises(oidc.OIDCRefused) as caught:
        verifier.verify("not-a-token", expected_nonce="one-use-nonce", now=150)
    assert "vendor" not in str(caught.value)

    with pytest.raises(oidc.OIDCRefused) as caught:
        verifier.refresh(now=150)
    assert str(caught.value) == "OIDC JWKS is unavailable"


def test_rotation_overlaps_then_rejects_a_removed_key():
    first_private, first_jwk = _key("first")
    second_private, second_jwk = _key("second")
    documents = _Documents([
        {"keys": [first_jwk]},
        {"keys": [second_jwk]},
        {"keys": [second_jwk]},
    ])
    config = _config(jwks_cache_seconds=30, jwks_overlap_seconds=60)
    verifier = oidc.JWKSVerifier(
        config, oidc.parse_discovery(_discovery(), config), documents)
    assert verifier.verify(
        _token(first_private, kid="first", exp=400),
        expected_nonce="one-use-nonce", now=150).subject == "opaque-subject"
    assert verifier.verify(
        _token(second_private, kid="second", exp=400),
        expected_nonce="one-use-nonce", now=181).subject == "opaque-subject"
    assert verifier.verify(
        _token(first_private, kid="first", exp=400),
        expected_nonce="one-use-nonce", now=210).subject == "opaque-subject"
    with pytest.raises(oidc.OIDCRefused):
        verifier.verify(
            _token(first_private, kid="first", exp=400),
            expected_nonce="one-use-nonce", now=243)


def test_jwks_refuses_duplicate_weak_or_mixed_keys():
    _private, jwk = _key("same")
    for keys in (
            [jwk, jwk],
            [{**jwk, "alg": "HS256"}],
            [{**jwk, "use": "enc"}],
            [{**jwk, "e": _b64((17).to_bytes(1, "big"))}],
            [{**jwk, "n": _b64((257).to_bytes(2, "big"))}]):
        with pytest.raises(oidc.OIDCRefused):
            oidc._parse_jwks({"keys": keys})
