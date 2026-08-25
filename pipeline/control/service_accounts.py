"""Opaque service-account credential format and keyed digest authority."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from dataclasses import dataclass, field


PREFIX = "ragsa.v1."


class ServiceAccountRefused(ValueError):
    """A service-account credential or secret configuration failed closed."""


@dataclass(frozen=True, slots=True)
class CredentialProof:
    service_account_id: uuid.UUID
    credential_version: int
    digest: bytes = field(repr=False)


def _key() -> bytes:
    value = os.getenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "")
    try:
        encoded = value.encode("utf-8")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ServiceAccountRefused(
            "service account HMAC key is invalid") from exc
    configured_peers = {
        os.getenv("CONTROL_IDENTITY_HMAC_SECRET", ""),
        os.getenv("OIDC_SESSION_SECRET", ""),
    }
    if (value != value.strip() or len(encoded) < 32
            or any(ord(char) < 33 or ord(char) == 127 for char in value)
            or value in configured_peers - {""}):
        raise ServiceAccountRefused(
            "service account HMAC key is invalid")
    return encoded


def _digest(account_id: uuid.UUID, version: int, secret: str) -> bytes:
    material = (
        b"ragtest-service-account-v1\x00" + account_id.bytes
        + version.to_bytes(4, "big") + secret.encode("ascii")
    )
    return hmac.new(_key(), material, hashlib.sha256).digest()


def issue_credential(account_id, credential_version: int):
    try:
        account = uuid.UUID(str(account_id))
    except (AttributeError, TypeError, ValueError):
        raise ServiceAccountRefused("service account id is invalid") from None
    if (type(credential_version) is not int
            or not 1 <= credential_version <= 2147483647):
        raise ServiceAccountRefused("credential version is invalid")
    secret = secrets.token_urlsafe(32)
    token = f"{PREFIX}{account.hex}.{credential_version}.{secret}"
    return token, CredentialProof(
        account, credential_version, _digest(account, credential_version, secret))


def parse_credential(token) -> CredentialProof:
    if (type(token) is not str or not token.startswith(PREFIX)
            or token != token.strip() or len(token) > 160
            or not token.isascii()):
        raise ServiceAccountRefused("service account token is invalid")
    parts = token[len(PREFIX):].split(".")
    if len(parts) != 3:
        raise ServiceAccountRefused("service account token is invalid")
    account_text, version_text, secret = parts
    try:
        account = uuid.UUID(hex=account_text)
        version = int(version_text)
    except (ValueError, AttributeError):
        raise ServiceAccountRefused("service account token is invalid") from None
    if (account.hex != account_text or str(version) != version_text
            or not 1 <= version <= 2147483647 or len(secret) != 43
            or any(not (char.isalnum() or char in "-_") for char in secret)):
        raise ServiceAccountRefused("service account token is invalid")
    return CredentialProof(account, version, _digest(account, version, secret))
