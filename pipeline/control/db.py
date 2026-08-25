"""Bounded database seam for content-free tenant routing facts."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import psycopg
from psycopg.rows import dict_row


CONTROL_SCHEMA_VERSION = 4
_SCHEMA_LOCK_NAME = "ragtest-control-schema-migration"
_SCHEMA_MONOTONIC_GUARD_DDL = """
CREATE OR REPLACE FUNCTION rag_control.control_guard_schema_monotonic()
RETURNS trigger LANGUAGE plpgsql
SET search_path = pg_catalog, rag_control
AS $control_schema_guard$
BEGIN
    IF NEW.schema_version < OLD.schema_version THEN
        RAISE EXCEPTION 'control schema downgrade refused';
    END IF;
    RETURN NEW;
END
$control_schema_guard$;
DROP TRIGGER IF EXISTS control_schema_state_monotonic
    ON rag_control.control_schema_state;
CREATE TRIGGER control_schema_state_monotonic
BEFORE UPDATE OF schema_version ON rag_control.control_schema_state
FOR EACH ROW EXECUTE FUNCTION rag_control.control_guard_schema_monotonic();
REVOKE ALL ON FUNCTION rag_control.control_guard_schema_monotonic()
    FROM PUBLIC;
"""
_pool = None
_admin_pool = None
_ROLE_FACTS_SQL = """
SELECT current_user AS role_name,
       session_user AS session_role,
       role.rolcanlogin AS can_login,
       (role.rolsuper OR role.rolbypassrls OR role.rolcreatedb
        OR role.rolcreaterole OR role.rolinherit OR role.rolreplication)
           AS elevated,
       EXISTS (
           SELECT 1 FROM pg_catalog.pg_auth_members AS membership
           WHERE membership.member = role.oid
       ) AS role_membership_power,
       has_schema_privilege(current_user, 'rag_control', 'USAGE')
           AS schema_usage,
       has_schema_privilege(current_user, 'rag_control', 'CREATE')
           AS schema_create,
       has_database_privilege(current_user, current_database(), 'CREATE')
           AS database_create,
       EXISTS (
           SELECT 1
           FROM pg_catalog.pg_class AS object
           JOIN pg_catalog.pg_namespace AS namespace
             ON namespace.oid = object.relnamespace
           WHERE namespace.nspname = 'rag_control'
             AND object.relkind IN ('r', 'p', 'v', 'm')
             AND (
                 (object.relname <> 'control_schema_state'
                  AND has_table_privilege(
                      current_user, object.oid,
                      'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'))
                 OR (object.relname = 'control_schema_state'
                     AND has_table_privilege(
                         current_user, object.oid,
                         'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'))
             )
       ) AS table_power,
       EXISTS (
           SELECT 1
           FROM pg_catalog.pg_class AS object
           JOIN pg_catalog.pg_namespace AS namespace
             ON namespace.oid = object.relnamespace
           WHERE namespace.nspname = 'rag_control'
             AND object.relkind = 'S'
             AND has_sequence_privilege(
                 current_user, object.oid, 'USAGE,SELECT,UPDATE')
       ) AS sequence_power,
       has_table_privilege(
           current_user, 'rag_control.control_schema_state', 'SELECT')
           AS schema_state_select,
       has_function_privilege(current_user,
           'rag_control.control_tenant_facts(uuid)', 'EXECUTE')
           AS tenant_facts_execute,
       has_function_privilege(current_user,
           'rag_control.control_resolve_identity(integer,bytea)', 'EXECUTE')
           AS identity_execute,
       has_function_privilege(current_user,
           'rag_control.control_resolve_platform_operator(integer,bytea)',
           'EXECUTE') AS operator_execute,
       has_function_privilege(current_user,
           'rag_control.control_resolve_service_account(uuid,integer,bytea)',
           'EXECUTE') AS service_execute,
       has_function_privilege(current_user,
           'rag_control.control_issue_service_account(integer,bytea,uuid,uuid,'
           'bytea,text[],timestamptz,timestamptz,text,bytea,bytea)', 'EXECUTE')
           AS issue_execute,
       has_function_privilege(current_user,
           'rag_control.control_rotate_service_account(integer,bytea,uuid,uuid,'
           'bigint,bytea,timestamptz,text,bytea,bytea)', 'EXECUTE')
           AS rotate_execute,
       has_function_privilege(current_user,
           'rag_control.control_revoke_service_account(integer,bytea,uuid,uuid,'
           'bigint,text,bytea,bytea)', 'EXECUTE') AS revoke_execute,
       has_function_privilege(current_user,
           'rag_control.control_approve_service_account_issue('
           'integer,bytea,uuid,uuid,uuid,text[],timestamptz,timestamptz,'
           'bigint,text,bytea,bytea)', 'EXECUTE') AS approve_issue_execute,
       has_function_privilege(current_user,
           'rag_control.control_approve_service_account_rotation('
           'integer,bytea,uuid,uuid,uuid,bigint,timestamptz,bigint,text,'
           'bytea,bytea)', 'EXECUTE') AS approve_rotate_execute,
       has_function_privilege(current_user,
           'rag_control.control_list_redeemable_service_account_approvals('
           'uuid,integer)', 'EXECUTE') AS list_approvals_execute,
       has_function_privilege(current_user,
           'rag_control.control_cancel_service_account_approval('
           'integer,bytea,uuid,uuid,uuid,bigint,text)', 'EXECUTE')
           AS cancel_approval_execute,
       has_function_privilege(current_user,
           'rag_control.control_redeem_service_account_issue('
           'uuid,uuid,uuid,bigint,bytea,bigint,bytea,bytea,bytea)', 'EXECUTE')
           AS redeem_issue_execute,
       has_function_privilege(current_user,
           'rag_control.control_redeem_service_account_rotation('
           'uuid,uuid,uuid,bigint,bytea,bigint,bytea,bytea,bytea)', 'EXECUTE')
           AS redeem_rotate_execute,
       EXISTS (
           SELECT 1
           FROM pg_catalog.pg_proc AS procedure
           JOIN pg_catalog.pg_namespace AS namespace
             ON namespace.oid = procedure.pronamespace
           WHERE namespace.nspname = 'rag_control'
             AND has_function_privilege(
                 current_user, procedure.oid, 'EXECUTE')
             AND NOT (
                 (current_user = 'rag_control_runtime'
                  AND procedure.oid IN (
                      'rag_control.control_tenant_facts(uuid)'::regprocedure,
                      'rag_control.control_resolve_identity('
                      'integer,bytea)'::regprocedure,
                      'rag_control.control_resolve_platform_operator('
                      'integer,bytea)'::regprocedure,
                      'rag_control.control_resolve_service_account('
                      'uuid,integer,bytea)'::regprocedure))
                  OR (current_user = 'rag_control_admin'
                      AND procedure.oid IN (
                          'rag_control.control_revoke_service_account('
                         'integer,bytea,uuid,uuid,bigint,text,bytea,'
                         'bytea)'::regprocedure,
                         'rag_control.control_approve_service_account_issue('
                         'integer,bytea,uuid,uuid,uuid,text[],timestamptz,'
                         'timestamptz,bigint,text,bytea,bytea)'::regprocedure,
                         'rag_control.control_approve_service_account_rotation('
                         'integer,bytea,uuid,uuid,uuid,bigint,timestamptz,'
                          'bigint,text,bytea,bytea)'::regprocedure,
                          'rag_control.control_cancel_service_account_approval('
                          'integer,bytea,uuid,uuid,uuid,bigint,text)'
                          ::regprocedure))
             )
       ) AS unexpected_function_execute
FROM pg_catalog.pg_roles AS role
WHERE role.rolname = current_user
"""
SERVICE_ACCOUNT_SCOPES = (
    "collections.manage",
    "documents.lifecycle",
    "documents.read",
    "documents.write",
    "rag.query",
    "tables.extract",
)
SERVICE_ACCOUNT_ISSUE_REASONS = (
    "incident_response", "security_provisioning",
)
SERVICE_ACCOUNT_ROTATE_REASONS = (
    "scheduled_rotation", "suspected_compromise",
)
SERVICE_ACCOUNT_REVOKE_REASONS = (
    "access_removed", "security_response", "suspected_compromise",
    "tenant_suspension",
)


class ControlPlaneRefused(RuntimeError):
    """A control-plane input or result was not inside the closed contract."""


class ControlPlaneDenied(ControlPlaneRefused):
    """A platform mutation lacked the exact control-plane authority."""


class ControlPlaneConflict(ControlPlaneRefused):
    """A platform mutation lost its compare-and-swap revision."""


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


@dataclass(frozen=True, slots=True)
class ServiceAccountRoute:
    service_account_id: uuid.UUID
    facts: TenantFacts
    connection_ref: str
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlatformOperator:
    operator_id: uuid.UUID
    role: str
    revision: int


@dataclass(frozen=True, slots=True)
class ServiceAccountApproval:
    approval_id: uuid.UUID
    tenant_id: uuid.UUID
    service_account_id: uuid.UUID
    action: str
    state: str
    approval_revision: int
    reason_code: str
    scopes: tuple[str, ...] | None
    account_expires_at: datetime | None
    credential_expires_at: datetime
    expected_account_revision: int | None
    control_policy_revision: int
    expires_at: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ServiceAccountRedemption:
    service_account_id: uuid.UUID
    account_revision: int
    credential_version: int
    account_expires_at: datetime
    credential_expires_at: datetime


def _required_dsn(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ControlPlaneRefused(f"{name} is required")
    return value


def get_conn() -> psycopg.Connection:
    """Open only the control database; a data-plane fallback is forbidden."""
    connection = psycopg.connect(_required_dsn("PG_CONTROL_DSN"))
    try:
        _configure_runtime_connection(connection)
    except Exception:
        connection.close()
        raise
    return connection


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
            configure=_configure_runtime_connection,
        )
    return _pool


def get_admin_pool():
    """Return the lifecycle-writer pool; runtime lookup credentials never apply."""
    global _admin_pool
    if _admin_pool is None:
        from psycopg_pool import ConnectionPool
        _admin_pool = ConnectionPool(
            _required_dsn("PG_CONTROL_ADMIN_DSN"),
            min_size=0,
            max_size=int(os.getenv("PG_CONTROL_ADMIN_POOL_MAX", "2")),
            check=ConnectionPool.check_connection,
            configure=_configure_admin_connection,
        )
    return _admin_pool


def close_pool() -> None:
    global _pool, _admin_pool
    if _pool is not None:
        _pool.close()
        _pool = None
    if _admin_pool is not None:
        _admin_pool.close()
        _admin_pool = None


def _assert_control_role(conn, expected_role: str, expected: dict) -> None:
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(_ROLE_FACTS_SQL)
            rows = cursor.fetchall()
        conn.rollback()
    except psycopg.Error:
        try:
            conn.rollback()
        finally:
            raise ControlPlaneRefused("control database role refused") from None
    if len(rows) != 1 or type(rows[0]) is not dict:
        raise ControlPlaneRefused("control database role refused")
    facts = rows[0]
    if (set(facts) != {"role_name", "session_role", *expected}
            or facts["role_name"] != expected_role
            or facts["session_role"] != expected_role
            or any(type(facts[name]) is not bool or facts[name] is not value
                   for name, value in expected.items())):
        raise ControlPlaneRefused("control database role refused")


def _configure_runtime_connection(conn) -> None:
    _assert_control_role(conn, "rag_control_runtime", {
        "can_login": True, "elevated": False,
        "schema_usage": True, "schema_create": False,
        "database_create": False,
        "role_membership_power": False,
        "table_power": False, "sequence_power": False,
        "schema_state_select": True,
        "tenant_facts_execute": True, "identity_execute": True,
        "operator_execute": True, "service_execute": True,
        "issue_execute": False, "rotate_execute": False,
        "revoke_execute": False, "unexpected_function_execute": False,
        "approve_issue_execute": False, "approve_rotate_execute": False,
        "list_approvals_execute": False,
        "cancel_approval_execute": False,
        "redeem_issue_execute": False, "redeem_rotate_execute": False,
    })


def _configure_admin_connection(conn) -> None:
    _assert_control_role(conn, "rag_control_admin", {
        "can_login": True, "elevated": False,
        "schema_usage": True, "schema_create": False,
        "database_create": False,
        "role_membership_power": False,
        "table_power": False, "sequence_power": False,
        "schema_state_select": True,
        "tenant_facts_execute": False, "identity_execute": False,
        "operator_execute": False, "service_execute": False,
        "issue_execute": False, "rotate_execute": False,
        "revoke_execute": True, "unexpected_function_execute": False,
        "approve_issue_execute": True, "approve_rotate_execute": True,
        "list_approvals_execute": False, "cancel_approval_execute": True,
        "redeem_issue_execute": False, "redeem_rotate_execute": False,
    })


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


def _audit_secret() -> bytes:
    value = os.getenv("CONTROL_AUDIT_HMAC_SECRET", "")
    try:
        encoded = value.encode("utf-8")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise ControlPlaneRefused("control audit HMAC key is invalid") from exc
    peers = {
        os.getenv("CONTROL_IDENTITY_HMAC_SECRET", ""),
        os.getenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", ""),
        os.getenv("OIDC_SESSION_SECRET", ""),
    }
    if (value != value.strip() or len(encoded) < 32
            or any(ord(char) < 33 or ord(char) == 127 for char in value)
            or value in peers - {""}):
        raise ControlPlaneRefused("control audit HMAC key is invalid")
    return encoded


def _audit_digest(kind: str, fields: dict) -> bytes:
    if (type(kind) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", kind)
            or type(fields) is not dict
            or any(type(key) is not str for key in fields)):
        raise ControlPlaneRefused("control audit facts are invalid")
    try:
        document = json.dumps(
            {"kind": kind, "facts": fields}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ControlPlaneRefused("control audit facts are invalid") from exc
    return hmac.new(
        _audit_secret(), b"ragtest-control-audit-v1\x00" + document,
        hashlib.sha256,
    ).digest()


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
            "CREATE TABLE IF NOT EXISTS rag_control.control_schema_history ("
            "schema_version integer PRIMARY KEY CHECK (schema_version > 0), "
            "schema_sha256 text NOT NULL CHECK (length(schema_sha256) = 64), "
            "applied_at timestamptz NOT NULL DEFAULT now())"
        )
        cursor.execute(_SCHEMA_MONOTONIC_GUARD_DDL)
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
        cursor.execute(
            "SELECT schema_sha256 FROM rag_control.control_schema_history "
            "WHERE schema_version = %s FOR UPDATE",
            (CONTROL_SCHEMA_VERSION,),
        )
        receipt = cursor.fetchone()
        if receipt is not None and receipt[0] != schema_sha256:
            raise ControlPlaneRefused("control schema history mismatch")
        cursor.execute(schema_sql)
        if receipt is None:
            cursor.execute(
                "INSERT INTO rag_control.control_schema_history "
                "(schema_version, schema_sha256) VALUES (%s, %s)",
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


def resolve_platform_operator(
        conn, key_version: int, digest: bytes) -> PlatformOperator | None:
    if (type(key_version) is not int or key_version < 1
            or type(digest) is not bytes or len(digest) != 32):
        raise ControlPlaneRefused("platform operator digest is invalid")
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT * FROM rag_control.control_resolve_platform_operator("
            "%s, %s)", (key_version, digest))
        rows = cursor.fetchall()
    if not rows:
        return None
    if len(rows) != 1 or type(rows[0]) is not dict or set(rows[0]) != {
            "operator_id", "role", "revision"}:
        raise ControlPlaneRefused("platform operator result is invalid")
    row = rows[0]
    try:
        operator_id = uuid.UUID(str(row["operator_id"]))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControlPlaneRefused(
            "platform operator result is invalid") from exc
    if (row["role"] not in {
            "platform_reader", "platform_operator", "platform_security"}
            or type(row["revision"]) is not int or row["revision"] < 1):
        raise ControlPlaneRefused("platform operator result is invalid")
    return PlatformOperator(operator_id, row["role"], row["revision"])


def resolve_service_account(
        conn, account_id, credential_version: int,
        digest: bytes) -> ServiceAccountRoute | None:
    try:
        account = uuid.UUID(str(account_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControlPlaneRefused("service account id is invalid") from exc
    if (type(credential_version) is not int or credential_version < 1
            or type(digest) is not bytes or len(digest) != 32):
        raise ControlPlaneRefused("service account credential is invalid")
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            "SELECT * FROM rag_control.control_resolve_service_account("
            "%s, %s, %s)",
            (account, credential_version, digest))
        rows = cursor.fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ControlPlaneRefused("service account route is ambiguous")
    row = rows[0]
    extra = {"service_account_id", "scopes"}
    facts_row = {key: value for key, value in row.items() if key not in extra}
    facts = _facts(facts_row, route=True)
    try:
        service_account_id = uuid.UUID(str(row["service_account_id"]))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControlPlaneRefused(
            "service account result is invalid") from exc
    scopes = row["scopes"]
    if (type(scopes) is not list or not scopes
            or any(type(scope) is not str for scope in scopes)
            or tuple(scopes) != tuple(sorted(set(scopes)))
            or not set(scopes).issubset(SERVICE_ACCOUNT_SCOPES)):
        raise ControlPlaneRefused("service account result is invalid")
    return ServiceAccountRoute(
        service_account_id=service_account_id,
        facts=facts,
        connection_ref=row["connection_ref"],
        scopes=tuple(scopes),
    )


def _uuid_value(value, name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ControlPlaneRefused(f"{name} is invalid") from exc


def _expiry(value, name: str) -> datetime:
    if (type(value) is not datetime or value.tzinfo is None
            or value.utcoffset() is None):
        raise ControlPlaneRefused(f"{name} is invalid")
    return value.astimezone(timezone.utc)


def _reason(value, allowed: tuple[str, ...]) -> str:
    if type(value) is not str or value not in allowed:
        raise ControlPlaneRefused("reason_code is invalid")
    return value


def _credential_digest(value) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ControlPlaneRefused("service account credential is invalid")
    return value


def _operator_proof(key_version, digest) -> tuple[int, bytes]:
    if (type(key_version) is not int or not 1 <= key_version <= 2147483647
            or type(digest) is not bytes or len(digest) != 32):
        raise ControlPlaneRefused("platform operator proof is invalid")
    return key_version, digest


def _scope_tuple(value) -> tuple[str, ...]:
    if (type(value) is not tuple or not value
            or any(type(item) is not str for item in value)
            or value != tuple(sorted(set(value)))
            or not set(value).issubset(SERVICE_ACCOUNT_SCOPES)):
        raise ControlPlaneRefused("service account scopes are invalid")
    return value


def _mutation_revision(conn, query: str, params: tuple) -> int:
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
    except psycopg.Error as exc:
        if exc.sqlstate == "42501":
            raise ControlPlaneDenied("service account lifecycle denied") from None
        if exc.sqlstate == "40001":
            raise ControlPlaneConflict(
                "service account lifecycle conflict") from None
        raise ControlPlaneRefused("service account lifecycle refused") from None
    if (type(row) not in {tuple, list} or len(row) != 1
            or type(row[0]) is not int or row[0] < 1):
        raise ControlPlaneRefused("service account lifecycle result is invalid")
    return row[0]


def _positive_revision(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ControlPlaneRefused(f"{name} is invalid")
    return value


def _approval_created(conn, query: str, params: tuple) -> dict:
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    except psycopg.Error as exc:
        if exc.sqlstate == "42501":
            raise ControlPlaneDenied("service account approval denied") from None
        if exc.sqlstate in {"40001", "23505"}:
            raise ControlPlaneConflict(
                "service account approval conflict") from None
        raise ControlPlaneRefused("service account approval refused") from None
    if (len(rows) != 1 or type(rows[0]) is not dict
            or set(rows[0]) != {
                "approval_revision", "control_policy_revision",
                "created_at", "expires_at"}):
        raise ControlPlaneRefused("service account approval result is invalid")
    row = rows[0]
    _positive_revision(row["approval_revision"], "approval_revision")
    _positive_revision(
        row["control_policy_revision"], "control_policy_revision")
    created = _expiry(row["created_at"], "created_at")
    expiry = _expiry(row["expires_at"], "expires_at")
    if (row["approval_revision"] != 1 or expiry <= created
            or expiry - created > timedelta(minutes=15)):
        raise ControlPlaneRefused("service account approval result is invalid")
    return row


def approve_service_account_issue(
        conn, *, operator_key_version: int, operator_digest: bytes,
        approval_id, tenant_id, account_id, scopes: tuple[str, ...],
        account_expires_at: datetime, credential_expires_at: datetime,
        expected_policy_revision: int, reason_code: str,
        ) -> ServiceAccountApproval:
    key_version, operator_proof = _operator_proof(
        operator_key_version, operator_digest)
    approval = _uuid_value(approval_id, "approval_id")
    tenant = _uuid_value(tenant_id, "tenant_id")
    account = _uuid_value(account_id, "service_account_id")
    canonical_scopes = _scope_tuple(scopes)
    account_expiry = _expiry(account_expires_at, "account_expires_at")
    credential_expiry = _expiry(
        credential_expires_at, "credential_expires_at")
    policy_revision = _positive_revision(
        expected_policy_revision, "expected_policy_revision")
    reason = _reason(reason_code, SERVICE_ACCOUNT_ISSUE_REASONS)
    facts = {
        "account_expires_at": account_expiry.isoformat(),
        "approval_id": str(approval), "approval_revision": 1,
        "control_policy_revision": policy_revision,
        "credential_expires_at": credential_expiry.isoformat(),
        "reason_code": reason, "scopes": list(canonical_scopes),
        "service_account_id": str(account), "state": "approved",
        "tenant_id": str(tenant),
    }
    row = _approval_created(
        conn,
        "SELECT * FROM rag_control.control_approve_service_account_issue("
        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (key_version, operator_proof, approval, tenant, account,
         list(canonical_scopes), account_expiry, credential_expiry,
         policy_revision, reason,
         _audit_digest("service_account_issue_approval_request", facts),
         _audit_digest("service_account_issue_approval_result", facts)),
    )
    if row["control_policy_revision"] != policy_revision:
        raise ControlPlaneRefused("service account approval result is invalid")
    return ServiceAccountApproval(
        approval, tenant, account, "issue", "approved", 1, reason,
        canonical_scopes, account_expiry, credential_expiry, None,
        policy_revision, _expiry(row["expires_at"], "expires_at"),
        _expiry(row["created_at"], "created_at"),
    )


def approve_service_account_rotation(
        conn, *, operator_key_version: int, operator_digest: bytes,
        approval_id, tenant_id, account_id, expected_account_revision: int,
        credential_expires_at: datetime, expected_policy_revision: int,
        reason_code: str) -> ServiceAccountApproval:
    key_version, operator_proof = _operator_proof(
        operator_key_version, operator_digest)
    approval = _uuid_value(approval_id, "approval_id")
    tenant = _uuid_value(tenant_id, "tenant_id")
    account = _uuid_value(account_id, "service_account_id")
    account_revision = _positive_revision(
        expected_account_revision, "expected_account_revision")
    credential_expiry = _expiry(
        credential_expires_at, "credential_expires_at")
    policy_revision = _positive_revision(
        expected_policy_revision, "expected_policy_revision")
    reason = _reason(reason_code, SERVICE_ACCOUNT_ROTATE_REASONS)
    facts = {
        "approval_id": str(approval), "approval_revision": 1,
        "control_policy_revision": policy_revision,
        "credential_expires_at": credential_expiry.isoformat(),
        "expected_account_revision": account_revision,
        "reason_code": reason, "service_account_id": str(account),
        "state": "approved", "tenant_id": str(tenant),
    }
    row = _approval_created(
        conn,
        "SELECT * FROM rag_control.control_approve_service_account_rotation("
        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (key_version, operator_proof, approval, tenant, account,
         account_revision, credential_expiry, policy_revision, reason,
         _audit_digest("service_account_rotate_approval_request", facts),
         _audit_digest("service_account_rotate_approval_result", facts)),
    )
    if row["control_policy_revision"] != policy_revision:
        raise ControlPlaneRefused("service account approval result is invalid")
    return ServiceAccountApproval(
        approval, tenant, account, "rotate", "approved", 1, reason,
        None, None, credential_expiry, account_revision, policy_revision,
        _expiry(row["expires_at"], "expires_at"),
        _expiry(row["created_at"], "created_at"),
    )


def _approval_from_row(row) -> ServiceAccountApproval:
    expected = {
        "approval_id", "tenant_id", "service_account_id", "action", "state",
        "approval_revision", "reason_code", "scopes", "account_expires_at",
        "credential_expires_at", "expected_account_revision",
        "control_policy_revision", "expires_at", "created_at",
    }
    if type(row) is not dict or set(row) != expected:
        raise ControlPlaneRefused("service account approval result is invalid")
    action = row["action"]
    if action not in {"issue", "rotate"} or row["state"] != "approved":
        raise ControlPlaneRefused("service account approval result is invalid")
    approval = _uuid_value(row["approval_id"], "approval_id")
    tenant = _uuid_value(row["tenant_id"], "tenant_id")
    account = _uuid_value(row["service_account_id"], "service_account_id")
    revision = _positive_revision(row["approval_revision"], "approval_revision")
    policy = _positive_revision(
        row["control_policy_revision"], "control_policy_revision")
    reason_allowed = (SERVICE_ACCOUNT_ISSUE_REASONS if action == "issue"
                      else SERVICE_ACCOUNT_ROTATE_REASONS)
    reason = _reason(row["reason_code"], reason_allowed)
    scopes_value = row["scopes"]
    if action == "issue":
        if type(scopes_value) is not list:
            raise ControlPlaneRefused(
                "service account approval result is invalid")
        scopes = _scope_tuple(tuple(scopes_value))
        account_expiry = _expiry(
            row["account_expires_at"], "account_expires_at")
        if row["expected_account_revision"] is not None:
            raise ControlPlaneRefused(
                "service account approval result is invalid")
        account_revision = None
    else:
        if scopes_value is not None or row["account_expires_at"] is not None:
            raise ControlPlaneRefused(
                "service account approval result is invalid")
        scopes = None
        account_expiry = None
        account_revision = _positive_revision(
            row["expected_account_revision"], "expected_account_revision")
    credential_expiry = _expiry(
        row["credential_expires_at"], "credential_expires_at")
    created = _expiry(row["created_at"], "created_at")
    expires = _expiry(row["expires_at"], "expires_at")
    measured_at = datetime.now(timezone.utc)
    if (revision != 1 or expires <= created
            or expires - created > timedelta(minutes=15)
            or expires <= measured_at
            or created > measured_at + timedelta(seconds=5)
            or credential_expiry <= created
            or (account_expiry is not None
                and account_expiry < credential_expiry)):
        raise ControlPlaneRefused("service account approval result is invalid")
    return ServiceAccountApproval(
        approval, tenant, account, action, "approved", revision, reason,
        scopes, account_expiry, credential_expiry, account_revision, policy,
        expires, created,
    )


def list_redeemable_service_account_approvals(
        conn, tenant_id, *, limit: int = 100,
        ) -> tuple[ServiceAccountApproval, ...]:
    """Read the offline queue; no deployable role owns this grant yet."""
    tenant = _uuid_value(tenant_id, "tenant_id")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ControlPlaneRefused("limit is invalid")
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM rag_control."
                "control_list_redeemable_service_account_approvals(%s, %s)",
                (tenant, limit))
            rows = cursor.fetchall()
    except psycopg.Error:
        raise ControlPlaneRefused("service account approvals refused") from None
    if len(rows) > limit:
        raise ControlPlaneRefused("service account approvals result is invalid")
    approvals = tuple(_approval_from_row(row) for row in rows)
    if (any(item.tenant_id != tenant for item in approvals)
            or len({item.approval_id for item in approvals}) != len(approvals)
            or list(approvals) != sorted(
                approvals, key=lambda item: (
                    item.created_at, item.approval_id.int))):
        raise ControlPlaneRefused("service account approvals result is invalid")
    return approvals


def cancel_service_account_approval(
        conn, *, operator_key_version: int, operator_digest: bytes,
        approval_id, tenant_id, account_id,
        expected_approval_revision: int, reason_code: str) -> int:
    key_version, operator_proof = _operator_proof(
        operator_key_version, operator_digest)
    approval = _uuid_value(approval_id, "approval_id")
    tenant = _uuid_value(tenant_id, "tenant_id")
    account = _uuid_value(account_id, "service_account_id")
    revision = _positive_revision(
        expected_approval_revision, "expected_approval_revision")
    reason = _reason(reason_code, (
        "access_removed", "approval_cancelled", "security_response",
        "tenant_suspension"))
    return _mutation_revision(
        conn,
        "SELECT rag_control.control_cancel_service_account_approval("
        "%s, %s, %s, %s, %s, %s, %s)",
        (key_version, operator_proof, approval, tenant, account, revision,
         reason),
    )


def _redemption_result(conn, query: str, params: tuple,
                       account_id: uuid.UUID) -> ServiceAccountRedemption:
    try:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    except psycopg.Error as exc:
        if exc.sqlstate in {"40001", "23505"}:
            raise ControlPlaneConflict(
                "service account redemption conflict") from None
        if exc.sqlstate == "42501":
            raise ControlPlaneDenied(
                "service account redemption denied") from None
        raise ControlPlaneRefused("service account redemption refused") from None
    expected = {
        "account_revision", "credential_version", "account_expires_at",
        "credential_expires_at",
    }
    if len(rows) != 1 or type(rows[0]) is not dict or set(rows[0]) != expected:
        raise ControlPlaneRefused("service account redemption result is invalid")
    row = rows[0]
    account_revision = _positive_revision(
        row["account_revision"], "account_revision")
    credential_version = _positive_revision(
        row["credential_version"], "credential_version")
    account_expiry = _expiry(row["account_expires_at"], "account_expires_at")
    credential_expiry = _expiry(
        row["credential_expires_at"], "credential_expires_at")
    if (credential_version != account_revision
            or credential_expiry > account_expiry):
        raise ControlPlaneRefused("service account redemption result is invalid")
    return ServiceAccountRedemption(
        account_id, account_revision, credential_version,
        account_expiry, credential_expiry)


def redeem_service_account_approval(
        conn, approval: ServiceAccountApproval, *,
        tenant_actor_digest: bytes, org_policy_epoch: int,
        credential_digest: bytes) -> ServiceAccountRedemption:
    """Consume offline authority after a separately proven tenant gate.

    The SQL function is deliberately revoked from every online role until a
    cross-database actor assertion can bind that gate to this transaction.
    """
    if type(approval) is not ServiceAccountApproval:
        raise ControlPlaneRefused("service account approval is invalid")
    if (approval.action not in {"issue", "rotate"}
            or approval.state != "approved"
            or approval.approval_revision != 1):
        raise ControlPlaneRefused("service account approval is invalid")
    actor_digest = _credential_digest(tenant_actor_digest)
    policy_epoch = _positive_revision(org_policy_epoch, "org_policy_epoch")
    credential = _credential_digest(credential_digest)
    next_revision = (1 if approval.action == "issue"
                     else _positive_revision(
                         approval.expected_account_revision,
                         "expected_account_revision") + 1)
    request_facts = {
        "approval_id": str(approval.approval_id),
        "approval_revision": approval.approval_revision,
        "credential_digest": credential.hex(),
        "org_policy_epoch": policy_epoch,
        "service_account_id": str(approval.service_account_id),
        "tenant_id": str(approval.tenant_id),
    }
    result_facts = {
        **request_facts, "account_revision": next_revision,
        "credential_version": next_revision, "state": "active",
    }
    query = {
        "issue": (
            "SELECT * FROM rag_control.control_redeem_service_account_issue("
            "%s, %s, %s, %s, %s, %s, %s, %s, %s)"),
        "rotate": (
            "SELECT * FROM rag_control.control_redeem_service_account_rotation("
            "%s, %s, %s, %s, %s, %s, %s, %s, %s)"),
    }[approval.action]
    return _redemption_result(
        conn,
        query,
        (approval.approval_id, approval.tenant_id,
         approval.service_account_id, approval.approval_revision,
         actor_digest, policy_epoch, credential,
         _audit_digest("service_account_redemption_request", request_facts),
         _audit_digest("service_account_redemption_result", result_facts)),
        approval.service_account_id,
    )


def issue_service_account(
        conn, *, operator_key_version: int, operator_digest: bytes,
        tenant_id, account_id, credential_digest: bytes,
        scopes: tuple[str, ...], account_expires_at: datetime,
        credential_expires_at: datetime, reason_code: str) -> int:
    key_version, operator_proof = _operator_proof(
        operator_key_version, operator_digest)
    tenant = _uuid_value(tenant_id, "tenant_id")
    account = _uuid_value(account_id, "service_account_id")
    digest = _credential_digest(credential_digest)
    canonical_scopes = _scope_tuple(scopes)
    account_expiry = _expiry(account_expires_at, "account_expires_at")
    credential_expiry = _expiry(
        credential_expires_at, "credential_expires_at")
    reason = _reason(reason_code, SERVICE_ACCOUNT_ISSUE_REASONS)
    request_facts = {
        "account_expires_at": account_expiry.isoformat(),
        "credential_digest": digest.hex(),
        "credential_expires_at": credential_expiry.isoformat(),
        "credential_version": 1, "reason_code": reason,
        "scopes": list(canonical_scopes),
        "service_account_id": str(account), "tenant_id": str(tenant),
    }
    resulting_facts = {
        "credential_digest": digest.hex(),
        "credential_expires_at": credential_expiry.isoformat(),
        "credential_version": 1, "revision": 1,
        "scopes": list(canonical_scopes),
        "service_account_id": str(account), "state": "active",
        "tenant_id": str(tenant),
    }
    return _mutation_revision(
        conn,
        "SELECT rag_control.control_issue_service_account("
        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (key_version, operator_proof, tenant, account, digest,
         list(canonical_scopes),
         account_expiry, credential_expiry, reason,
         _audit_digest("service_account_issue_request", request_facts),
         _audit_digest("service_account_issue_result", resulting_facts)),
    )


def rotate_service_account(
        conn, *, operator_key_version: int, operator_digest: bytes,
        tenant_id, account_id, expected_revision: int,
        credential_digest: bytes, credential_expires_at: datetime,
        reason_code: str) -> int:
    key_version, operator_proof = _operator_proof(
        operator_key_version, operator_digest)
    tenant = _uuid_value(tenant_id, "tenant_id")
    account = _uuid_value(account_id, "service_account_id")
    if type(expected_revision) is not int or expected_revision < 1:
        raise ControlPlaneRefused("expected_revision is invalid")
    digest = _credential_digest(credential_digest)
    expiry = _expiry(credential_expires_at, "credential_expires_at")
    reason = _reason(reason_code, SERVICE_ACCOUNT_ROTATE_REASONS)
    next_revision = expected_revision + 1
    common = {
        "reason_code": reason,
        "service_account_id": str(account), "tenant_id": str(tenant),
    }
    request_facts = {
        **common, "credential_digest": digest.hex(),
        "credential_expires_at": expiry.isoformat(),
        "credential_version": next_revision,
        "expected_revision": expected_revision,
    }
    resulting_facts = {
        **common, "credential_digest": digest.hex(),
        "credential_expires_at": expiry.isoformat(),
        "credential_version": next_revision,
        "revision": next_revision, "state": "active",
    }
    return _mutation_revision(
        conn,
        "SELECT rag_control.control_rotate_service_account("
        "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (key_version, operator_proof, tenant, account, expected_revision,
         digest, expiry, reason,
         _audit_digest("service_account_rotate_request", request_facts),
         _audit_digest("service_account_rotate_result", resulting_facts)),
    )


def revoke_service_account(
        conn, *, operator_key_version: int, operator_digest: bytes,
        tenant_id, account_id, expected_revision: int,
        reason_code: str) -> int:
    key_version, operator_proof = _operator_proof(
        operator_key_version, operator_digest)
    tenant = _uuid_value(tenant_id, "tenant_id")
    account = _uuid_value(account_id, "service_account_id")
    if type(expected_revision) is not int or expected_revision < 1:
        raise ControlPlaneRefused("expected_revision is invalid")
    reason = _reason(reason_code, SERVICE_ACCOUNT_REVOKE_REASONS)
    next_revision = expected_revision + 1
    common = {
        "reason_code": reason,
        "service_account_id": str(account), "tenant_id": str(tenant),
    }
    return _mutation_revision(
        conn,
        "SELECT rag_control.control_revoke_service_account("
        "%s, %s, %s, %s, %s, %s, %s, %s)",
        (key_version, operator_proof, tenant, account, expected_revision,
         reason,
         _audit_digest("service_account_revoke_request", {
             **common, "expected_revision": expected_revision}),
         _audit_digest("service_account_revoke_result", {
             **common, "revision": next_revision, "state": "revoked"})),
    )


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
