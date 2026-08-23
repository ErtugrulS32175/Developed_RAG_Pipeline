"""Verified Open WebUI identity assertions.

The provider credential identifies the trusted Open WebUI instance.  This
module identifies the human behind that request.  Neither is sufficient on
its own, and display claims never become authorization input.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass


HEADER_NAME = "x-openwebui-user-jwt"
ISSUER = "open-webui"
MAX_ASSERTION_BYTES = 4096
MAX_SUBJECT_LENGTH = 200
REQUIRED_CLAIMS = frozenset({
    "sub", "email", "name", "role", "iss", "iat", "exp",
})


class IdentityConfigurationError(RuntimeError):
    """The trusted identity bridge is not configured safely."""


class IdentityRefused(ValueError):
    """A caller-supplied assertion failed a closed verification rule."""


@dataclass(frozen=True, slots=True)
class ForwardedIdentity:
    issuer: str
    subject: str
    issued_at: int
    expires_at: int


@dataclass(frozen=True, slots=True)
class Verifier:
    secret: bytes
    max_lifetime_seconds: int = 60
    clock_skew_seconds: int = 5

    @classmethod
    def configured(cls, secret, *, max_lifetime_seconds=60,
                   clock_skew_seconds=5):
        if (type(secret) is not str or secret != secret.strip()
                or len(secret.encode("utf-8")) < 32):
            raise IdentityConfigurationError(
                "OpenWebUI JWT anahtari en az 32 bayt olmali")
        if (type(max_lifetime_seconds) is not int
                or not 1 <= max_lifetime_seconds <= 300):
            raise IdentityConfigurationError("JWT omru 1-300 saniye olmali")
        if (type(clock_skew_seconds) is not int
                or not 0 <= clock_skew_seconds <= 30):
            raise IdentityConfigurationError("JWT saat toleransi gecersiz")
        return cls(secret.encode("utf-8"), max_lifetime_seconds,
                   clock_skew_seconds)

    def verify(self, token, *, now=None):
        if (type(token) is not str or not token
                or len(token.encode("utf-8")) > MAX_ASSERTION_BYTES
                or token != token.strip()):
            raise IdentityRefused("kimlik iddiasi gecersiz")
        parts = token.split(".")
        if len(parts) != 3 or any(not part for part in parts):
            raise IdentityRefused("kimlik iddiasi gecersiz")
        header = _json_part(parts[0])
        claims = _json_part(parts[1])
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise IdentityRefused("kimlik algoritmasi gecersiz")
        if type(claims) is not dict or set(claims) != REQUIRED_CLAIMS:
            raise IdentityRefused("kimlik alanlari gecersiz")

        signed = f"{parts[0]}.{parts[1]}".encode("ascii")
        expected = hmac.new(self.secret, signed, hashlib.sha256).digest()
        supplied = _decode(parts[2])
        if not hmac.compare_digest(expected, supplied):
            raise IdentityRefused("kimlik imzasi gecersiz")

        subject = claims["sub"]
        issuer = claims["iss"]
        issued_at = claims["iat"]
        expires_at = claims["exp"]
        if issuer != ISSUER:
            raise IdentityRefused("kimlik issuer gecersiz")
        if not _closed_text(subject, MAX_SUBJECT_LENGTH):
            raise IdentityRefused("kimlik subject gecersiz")
        if (type(issued_at) is not int or type(expires_at) is not int
                or expires_at <= issued_at
                or expires_at - issued_at > self.max_lifetime_seconds):
            raise IdentityRefused("kimlik zamani gecersiz")
        current = int(time.time()) if now is None else now
        if type(current) is not int:
            raise TypeError("now tamsayi olmali")
        if issued_at > current + self.clock_skew_seconds:
            raise IdentityRefused("kimlik henuz gecerli degil")
        if expires_at <= current - self.clock_skew_seconds:
            raise IdentityRefused("kimlik suresi doldu")
        return ForwardedIdentity(issuer, subject, issued_at, expires_at)


def _closed_text(value, maximum):
    return (type(value) is str and value and value == value.strip()
            and len(value) <= maximum
            and all(32 <= ord(char) < 127 for char in value))


def _decode(value):
    if type(value) is not str or any(
            char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789-_" for char in value):
        raise IdentityRefused("kimlik kodlamasi gecersiz")
    try:
        return base64.urlsafe_b64decode(
            value + "=" * ((4 - len(value) % 4) % 4))
    except Exception as exc:
        raise IdentityRefused("kimlik kodlamasi gecersiz") from exc


def _json_part(value):
    try:
        decoded = _decode(value).decode("utf-8")
        result = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityRefused("kimlik JSON gecersiz") from exc
    return result
