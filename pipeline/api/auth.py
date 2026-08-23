"""Closed API principal registry and request-local tenant binding."""
from __future__ import annotations

import contextvars
import hashlib
import json
import secrets
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID


DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
ROLES = MappingProxyType({"reader": 1, "editor": 2, "admin": 3})


class AuthConfigurationError(RuntimeError):
    """Authentication environment is malformed; startup must fail closed."""


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: UUID
    role: str
    subject_id: UUID | None = None
    source: str = "api_key"
    position_id: UUID | None = None
    org_architect: bool = False


@dataclass(frozen=True, slots=True)
class Credential:
    digest: bytes
    principal: Principal


@dataclass(frozen=True, slots=True)
class Registry:
    credentials: tuple[Credential, ...]
    open_principal: Principal | None

    @property
    def configured(self):
        return bool(self.credentials)

    @property
    def multi_tenant(self):
        return any(item.principal.tenant_id != DEFAULT_TENANT_ID
                   for item in self.credentials)


_CURRENT = contextvars.ContextVar("rag_principal", default=None)


def _digest(token):
    return hashlib.sha256(token.encode("utf-8")).digest()


def _token(value, label):
    if (type(value) is not str or not value or value != value.strip()
            or len(value) > 500
            or any(ord(char) < 33 or ord(char) == 127 for char in value)):
        raise AuthConfigurationError(f"{label} gecersiz")
    return value


def load_registry(api_key, keys_json):
    """Build one immutable registry from legacy and multi-key settings."""
    credentials = []
    if api_key:
        token = _token(api_key, "API_KEY")
        credentials.append(Credential(
            _digest(token), Principal(DEFAULT_TENANT_ID, "admin")))
    if keys_json:
        try:
            rows = json.loads(keys_json)
        except json.JSONDecodeError as exc:
            raise AuthConfigurationError("API_KEYS_JSON gecersiz JSON") from exc
        if type(rows) is not list or not rows:
            raise AuthConfigurationError("API_KEYS_JSON bos olmayan liste olmali")
        for index, row in enumerate(rows):
            if type(row) is not dict or set(row) != {"key", "tenant_id", "role"}:
                raise AuthConfigurationError(
                    f"API_KEYS_JSON[{index}] alanlari kapali olmali")
            token = _token(row["key"], f"API_KEYS_JSON[{index}].key")
            try:
                tenant_id = UUID(row["tenant_id"])
            except (AttributeError, TypeError, ValueError) as exc:
                raise AuthConfigurationError(
                    f"API_KEYS_JSON[{index}].tenant_id gecersiz") from exc
            role = row["role"]
            if role not in ROLES:
                raise AuthConfigurationError(
                    f"API_KEYS_JSON[{index}].role gecersiz")
            credentials.append(Credential(
                _digest(token), Principal(tenant_id, role)))
    digests = [item.digest for item in credentials]
    if len(set(digests)) != len(digests):
        raise AuthConfigurationError("ayni API anahtari birden cok kez tanimli")
    open_principal = None if credentials else Principal(DEFAULT_TENANT_ID, "admin")
    return Registry(tuple(credentials), open_principal)


def authenticate(registry, authorization):
    """Resolve a Bearer header using fixed-size digest comparisons."""
    if registry.open_principal is not None:
        return registry.open_principal
    if type(authorization) is not str:
        return None
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token:
        return None
    offered = _digest(token)
    matched = None
    for credential in registry.credentials:
        if secrets.compare_digest(offered, credential.digest):
            matched = credential.principal
    return matched


def permits(principal, minimum_role):
    if principal is None or minimum_role not in ROLES:
        return False
    return ROLES.get(principal.role, 0) >= ROLES[minimum_role]


def bind(principal):
    return _CURRENT.set(principal)


def reset(token):
    _CURRENT.reset(token)


def current_principal():
    principal = _CURRENT.get()
    if principal is None:
        raise RuntimeError("istek tenant baglami kurulmamisti")
    return principal


def bound_principal():
    """Return the request binding, including ``None`` for an unauthenticated call."""
    return _CURRENT.get()
