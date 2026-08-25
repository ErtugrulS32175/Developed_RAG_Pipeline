"""Bounded database seam for content-free tenant routing facts."""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import psycopg
from psycopg.rows import dict_row


CONTROL_SCHEMA_VERSION = 1
_SCHEMA_LOCK_NAME = "ragtest-control-schema-migration"
_pool = None


class ControlPlaneRefused(RuntimeError):
    """A control-plane input or result was not inside the closed contract."""


@dataclass(frozen=True, slots=True)
class TenantFacts:
    tenant_id: uuid.UUID
    deployment_profile: str
    region_code: str
    route_kind: str
    configuration_revision: int
    policy_revision: int
    features: Mapping[str, bool]
    quotas: Mapping[str, int]
    quota_enforcement: str


@dataclass(frozen=True, slots=True)
class TenantRoute:
    facts: TenantFacts
    connection_ref: str


def _required_dsn(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ControlPlaneRefused(f"{name} is required")
    return value


def get_conn() -> psycopg.Connection:
    """Open only the control database; a data-plane fallback is forbidden."""
    return psycopg.connect(_required_dsn("PG_CONTROL_DSN"))


def get_migration_conn() -> psycopg.Connection:
    return psycopg.connect(_required_dsn("PG_CONTROL_MIGRATION_DSN"))


def get_pool():
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(
            _required_dsn("PG_CONTROL_DSN"),
            min_size=0,
            max_size=int(os.getenv("PG_CONTROL_POOL_MAX", "4")),
            check=ConnectionPool.check_connection,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def _identity_secret() -> bytes:
    value = os.getenv("CONTROL_IDENTITY_HMAC_SECRET", "")
    encoded = value.encode("utf-8")
    if len(encoded) < 32:
        raise ControlPlaneRefused(
            "CONTROL_IDENTITY_HMAC_SECRET must be at least 32 bytes")
    return encoded


def _coordinate(value: str, name: str) -> bytes:
    if type(value) is not str or not value or len(value) > 512:
        raise ControlPlaneRefused(f"{name} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ControlPlaneRefused(f"{name} is invalid") from exc
    return len(encoded).to_bytes(4, "big") + encoded


def identity_digest(issuer: str, subject: str) -> bytes:
    """Return a non-reversible, delimiter-safe external identity coordinate."""
    material = _coordinate(issuer, "issuer") + _coordinate(subject, "subject")
    return hmac.new(_identity_secret(), material, hashlib.sha256).digest()


def init_schema(conn: psycopg.Connection) -> None:
    path = Path(__file__).with_name("schema.sql")
    schema_bytes = path.read_bytes()
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    schema_sql = schema_bytes.decode("utf-8")
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            (_SCHEMA_LOCK_NAME,),
        )
        cursor.execute("CREATE SCHEMA IF NOT EXISTS rag_control")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS rag_control.control_schema_state ("
            "singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), "
            "schema_version integer NOT NULL CHECK (schema_version > 0), "
            "schema_sha256 text NOT NULL CHECK (length(schema_sha256) = 64), "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cursor.execute(
            "SELECT schema_version, schema_sha256 "
            "FROM rag_control.control_schema_state "
            "WHERE singleton FOR UPDATE"
        )
        previous = cursor.fetchone()
        if previous is not None:
            version, digest = previous
            if version > CONTROL_SCHEMA_VERSION:
                raise ControlPlaneRefused("control schema downgrade refused")
            if version == CONTROL_SCHEMA_VERSION and digest != schema_sha256:
                raise ControlPlaneRefused("control schema digest mismatch")
        cursor.execute(schema_sql)
        cursor.execute(
            "INSERT INTO rag_control.control_schema_history "
            "(schema_version, schema_sha256) VALUES (%s, %s) "
            "ON CONFLICT (schema_version) DO NOTHING",
            (CONTROL_SCHEMA_VERSION, schema_sha256),
        )
        cursor.execute(
            "INSERT INTO rag_control.control_schema_state "
            "(singleton, schema_version, schema_sha256) VALUES "
            "(true, %s, %s) ON CONFLICT (singleton) DO UPDATE SET "
            "schema_version = EXCLUDED.schema_version, "
            "schema_sha256 = EXCLUDED.schema_sha256, applied_at = now()",
            (CONTROL_SCHEMA_VERSION, schema_sha256),
        )
    conn.commit()


def _closed_mapping(value, name: str, value_type) -> Mapping:
    if type(value) is not dict:
        raise ControlPlaneRefused(f"{name} result is invalid")
    closed = {}
    for key, item in value.items():
        if type(key) is not str or type(item) is not value_type:
            raise ControlPlaneRefused(f"{name} result is invalid")
        closed[key] = item
    return MappingProxyType(closed)


def _facts(row, *, route: bool = False) -> TenantFacts:
    expected = {
        "tenant_id", "deployment_profile", "region_code", "route_kind",
        "configuration_revision", "policy_revision", "features", "quotas",
        "quota_enforcement",
    }
    if route:
        expected.add("connection_ref")
    if type(row) is not dict or set(row) != expected:
        raise ControlPlaneRefused("tenant facts result is invalid")
    try:
        tenant_id = uuid.UUID(str(row["tenant_id"]))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControlPlaneRefused("tenant facts result is invalid") from exc
    for field in ("deployment_profile", "region_code", "route_kind",
                  "quota_enforcement"):
        if type(row[field]) is not str:
            raise ControlPlaneRefused("tenant facts result is invalid")
    for field in ("configuration_revision", "policy_revision"):
        if type(row[field]) is not int or row[field] < 0:
            raise ControlPlaneRefused("tenant facts result is invalid")
    return TenantFacts(
        tenant_id=tenant_id,
        deployment_profile=row["deployment_profile"],
        region_code=row["region_code"],
        route_kind=row["route_kind"],
        configuration_revision=row["configuration_revision"],
        policy_revision=row["policy_revision"],
        features=_closed_mapping(row["features"], "features", bool),
        quotas=_closed_mapping(row["quotas"], "quotas", int),
        quota_enforcement=row["quota_enforcement"],
    )


def resolve_identity(conn, key_version: int, digest: bytes) -> TenantRoute | None:
    if (type(key_version) is not int or key_version < 1
            or type(digest) is not bytes or len(digest) != 32):
        raise ControlPlaneRefused("identity digest is invalid")
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT * FROM rag_control.control_resolve_identity(%s, %s)",
            (key_version, digest))
        rows = cursor.fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ControlPlaneRefused("identity route is ambiguous")
    row = rows[0]
    facts = _facts(row, route=True)
    return TenantRoute(facts=facts, connection_ref=row["connection_ref"])


def tenant_facts(conn, tenant_id) -> TenantFacts | None:
    try:
        tenant = uuid.UUID(str(tenant_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControlPlaneRefused("tenant_id is invalid") from exc
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT * FROM rag_control.control_tenant_facts(%s)", (tenant,))
        rows = cursor.fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ControlPlaneRefused("tenant route is ambiguous")
    return _facts(rows[0])
