"""Platform-security lifecycle authority for opaque service credentials."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import psycopg
import pytest

from pipeline.control import db


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = (ROOT / "pipeline" / "control" / "schema.sql").read_text(
    encoding="utf-8")

OPERATOR = uuid.UUID("40000000-0000-0000-0000-000000000004")
TENANT = uuid.UUID("10000000-0000-0000-0000-000000000001")
ACCOUNT = uuid.UUID("30000000-0000-0000-0000-000000000003")


class _Cursor:
    def __init__(self, rows=None, error=None):
        self.rows = [] if rows is None else rows
        self.error = error
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params=None):
        self.query = query
        self.params = params
        if self.error is not None:
            raise self.error

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows=None, error=None):
        self.cursor_instance = _Cursor(rows, error)

    def cursor(self, **_kwargs):
        return self.cursor_instance

    def rollback(self):
        return None


def _configure_audit(monkeypatch):
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "a" * 32)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 32)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 32)


def test_platform_operator_resolution_is_exact_and_closed():
    digest = b"d" * 32
    connection = _Connection([{
        "operator_id": OPERATOR,
        "role": "platform_security",
        "revision": 7,
    }])
    assert db.resolve_platform_operator(connection, 2, digest) == (
        db.PlatformOperator(OPERATOR, "platform_security", 7))
    assert connection.cursor_instance.params == (2, digest)

    for rows in ([{
            "operator_id": OPERATOR, "role": "tenant_admin", "revision": 7,
            }], [{
                "operator_id": OPERATOR, "role": "platform_security",
                "revision": 7, "subject": "forbidden",
            }], [{
                "operator_id": OPERATOR, "role": "platform_security",
                "revision": 7,
            }] * 2):
        with pytest.raises(db.ControlPlaneRefused):
            db.resolve_platform_operator(_Connection(rows), 2, digest)


def test_lifecycle_authority_is_platform_security_only():
    assert "control_identity_capabilities" not in SCHEMA
    assert "control_require_service_manager" not in SCHEMA
    assert "platform_actor.role = 'platform_security'" in SCHEMA
    assert "platform_actor.state = 'active'" in SCHEMA
    assert "requested_actor_identity_id" not in SCHEMA
    assert "requested_operator_id" not in SCHEMA
    assert "requested_operator_digest" in SCHEMA
    assert "control_tenant_facts(\n               requested_tenant_id)" in SCHEMA


def test_lifecycle_events_are_content_free_and_immutable():
    event = SCHEMA.split(
        "CREATE TABLE IF NOT EXISTS "
        "rag_control.control_service_account_events", 1)[1].split(
            "ALTER TABLE rag_control.control_service_account_events", 1)[0]
    for field in (
            "operator_id", "target_tenant_id", "service_account_id", "action",
            "reason_code", "expected_revision", "resulting_revision",
            "request_digest", "resulting_fact_digest"):
        assert field in event
    for forbidden in (
            "token", "secret", "subject", "issuer", "email", "display_name",
            "filename", "prompt", "content"):
        assert forbidden not in event.lower()
    assert ("BEFORE UPDATE OR DELETE ON "
            "rag_control.control_service_account_events") in SCHEMA
    assert "control_service_account_tenant_identity" in SCHEMA
    assert ("ON rag_control.control_service_accounts "
            "(tenant_id, service_account_id)") in SCHEMA
    assert "FOREIGN KEY (target_tenant_id, service_account_id)" in SCHEMA


def test_admin_role_can_approve_cancel_and_revoke_but_never_redeem():
    script = (ROOT / "scripts" / "init_control_admin_role.sh").read_text(
        encoding="utf-8")
    assert ("NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT "
            "NOREPLICATION") in script
    assert "ALTER ROLE rag_control_admin LOGIN PASSWORD :'admin_password'" in script
    assert "\\getenv admin_password CONTROL_ADMIN_PASSWORD" in script
    assert "--set=admin_password" not in script
    assert "REVOKE ALL ON ALL TABLES IN SCHEMA rag_control" in script
    assert "REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control" in script
    assert "REVOKE ALL ON ALL FUNCTIONS IN SCHEMA rag_control" in script
    assert "REVOKE CREATE ON DATABASE" in script
    for name in (
            "control_approve_service_account_issue",
            "control_approve_service_account_rotation",
            "control_cancel_service_account_approval",
            "control_revoke_service_account"):
        assert f"GRANT EXECUTE ON FUNCTION rag_control.{name}" in script
    for name in (
            "control_list_redeemable_service_account_approvals",
            "control_redeem_service_account_issue",
            "control_redeem_service_account_rotation"):
        assert f"GRANT EXECUTE ON FUNCTION rag_control.{name}" not in script
    for forbidden in (
            "control_issue_service_account", "control_rotate_service_account"):
        assert f"GRANT EXECUTE ON FUNCTION rag_control.{forbidden}" not in script
    assert "GRANT INSERT" not in script
    assert "GRANT UPDATE" not in script
    assert "GRANT DELETE" not in script
    assert "control_resolve_identity" not in script
    assert not (ROOT / "scripts" / "init_control_redeemer_role.sh").exists()
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "PG_CONTROL_REDEMPTION_DSN" not in env_example


def test_admin_connection_never_falls_back_to_runtime_or_data(monkeypatch):
    monkeypatch.delenv("PG_CONTROL_ADMIN_DSN", raising=False)
    monkeypatch.setenv("PG_CONTROL_DSN", "must-not-be-used")
    monkeypatch.setenv("PG_DSN", "must-not-be-used")
    db._admin_pool = None
    with pytest.raises(db.ControlPlaneRefused, match="PG_CONTROL_ADMIN_DSN"):
        db.get_admin_pool()


def _role_facts(role_name, *, kind):
    runtime = kind == "runtime"
    admin = kind == "admin"
    return {
        "role_name": role_name,
        "session_role": role_name,
        "can_login": True,
        "elevated": False,
        "role_membership_power": False,
        "schema_usage": True,
        "schema_create": False,
        "database_create": False,
        "table_power": False,
        "sequence_power": False,
        "schema_state_select": True,
        "tenant_facts_execute": runtime,
        "identity_execute": runtime,
        "operator_execute": runtime,
        "service_execute": runtime,
        "issue_execute": False,
        "rotate_execute": False,
        "revoke_execute": admin,
        "approve_issue_execute": admin,
        "approve_rotate_execute": admin,
        "list_approvals_execute": False,
        "cancel_approval_execute": admin,
        "redeem_issue_execute": False,
        "redeem_rotate_execute": False,
        "unexpected_function_execute": False,
    }


def test_each_pool_proves_its_exact_role_and_privilege_shape():
    db._configure_runtime_connection(_Connection([
        _role_facts("rag_control_runtime", kind="runtime")]))
    db._configure_admin_connection(_Connection([
        _role_facts("rag_control_admin", kind="admin")]))
    for configure, row in (
            (db._configure_runtime_connection,
             _role_facts("rag_control_admin", kind="runtime")),
            (db._configure_admin_connection,
             _role_facts("rag_control_runtime", kind="admin")),
            (db._configure_runtime_connection, {
                **_role_facts("rag_control_runtime", kind="runtime"),
                "issue_execute": True,
            }),
            (db._configure_admin_connection, {
                **_role_facts("rag_control_admin", kind="admin"),
                "table_power": True,
            }),
            (db._configure_runtime_connection, {
                **_role_facts("rag_control_runtime", kind="runtime"),
                "unexpected_function_execute": True,
            }),
            (db._configure_admin_connection, {
                **_role_facts("rag_control_admin", kind="admin"),
                "role_membership_power": True,
            }),
            (db._configure_runtime_connection, {
                **_role_facts("rag_control_runtime", kind="runtime"),
                "database_create": True,
            }),
            (db._configure_runtime_connection, {
                **_role_facts("rag_control_runtime", kind="runtime"),
                "session_role": "schema_owner",
            })):
        with pytest.raises(db.ControlPlaneRefused):
            configure(_Connection([row]))


def test_audit_key_is_independent_and_domain_separated(monkeypatch):
    _configure_audit(monkeypatch)
    first = db._audit_digest("service_account_issue_request", {"revision": 1})
    second = db._audit_digest("service_account_issue_result", {"revision": 1})
    assert len(first) == 32
    assert first != second
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "i" * 32)
    with pytest.raises(db.ControlPlaneRefused):
        db._audit_digest("service_account_issue_request", {"revision": 1})


def test_issue_binds_one_way_credential_facts_inside_audit_hmac_only(
        monkeypatch):
    _configure_audit(monkeypatch)
    seen = []

    def capture(kind, fields):
        seen.append((kind, fields))
        return b"h" * 32

    monkeypatch.setattr(db, "_audit_digest", capture)
    now = datetime.now(timezone.utc)
    connection = _Connection([(1,)])
    revision = db.issue_service_account(
        connection,
        operator_key_version=1,
        operator_digest=b"p" * 32,
        tenant_id=TENANT,
        account_id=ACCOUNT,
        credential_digest=b"z" * 32,
        scopes=("documents.read", "rag.query"),
        account_expires_at=now + timedelta(days=30),
        credential_expires_at=now + timedelta(days=7),
        reason_code="security_provisioning",
    )
    assert revision == 1
    assert [kind for kind, _fields in seen] == [
        "service_account_issue_request", "service_account_issue_result"]
    assert all(fields["credential_digest"] == (b"z" * 32).hex()
               for _kind, fields in seen)
    assert all(fields["credential_version"] == 1
               for _kind, fields in seen)
    assert b"z" * 32 not in (connection.cursor_instance.params[-2:])
    assert "control_issue_service_account" in connection.cursor_instance.query


def test_audit_evidence_changes_with_the_credential(monkeypatch):
    _configure_audit(monkeypatch)
    now = datetime.now(timezone.utc)

    def issue(digest):
        connection = _Connection([(1,)])
        db.issue_service_account(
            connection, operator_key_version=1,
            operator_digest=b"p" * 32, tenant_id=TENANT,
            account_id=ACCOUNT, credential_digest=digest,
            scopes=("rag.query",),
            account_expires_at=now + timedelta(days=30),
            credential_expires_at=now + timedelta(days=7),
            reason_code="security_provisioning")
        return connection.cursor_instance.params[-2:]

    assert issue(b"x" * 32) != issue(b"y" * 32)


def test_rotate_and_revoke_are_revision_bound(monkeypatch):
    _configure_audit(monkeypatch)
    now = datetime.now(timezone.utc)
    rotate = _Connection([(5,)])
    assert db.rotate_service_account(
        rotate, operator_key_version=1, operator_digest=b"p" * 32,
        tenant_id=TENANT, account_id=ACCOUNT,
        expected_revision=4, credential_digest=b"n" * 32,
        credential_expires_at=now + timedelta(days=1),
        reason_code="scheduled_rotation") == 5
    assert rotate.cursor_instance.params[4] == 4

    revoke = _Connection([(6,)])
    assert db.revoke_service_account(
        revoke, operator_key_version=1, operator_digest=b"p" * 32,
        tenant_id=TENANT, account_id=ACCOUNT,
        expected_revision=5, reason_code="access_removed") == 6
    assert revoke.cursor_instance.params[4] == 5


@pytest.mark.parametrize("error,exception", [
    (psycopg.errors.SerializationFailure("private database detail"),
     db.ControlPlaneConflict),
    (psycopg.errors.InsufficientPrivilege("private database detail"),
     db.ControlPlaneDenied),
    (psycopg.errors.CheckViolation("private database detail"),
     db.ControlPlaneRefused),
])
def test_database_exception_detail_never_crosses_the_repository(
        monkeypatch, error, exception):
    _configure_audit(monkeypatch)
    with pytest.raises(exception) as caught:
        db.revoke_service_account(
            _Connection(error=error), operator_key_version=1,
            operator_digest=b"p" * 32,
            tenant_id=TENANT, account_id=ACCOUNT, expected_revision=1,
            reason_code="access_removed")
    assert "private database detail" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("change", [
    {"scopes": ("rag.query", "documents.read")},
    {"scopes": ("rag.query", "rag.query")},
    {"reason_code": "UPPER_CASE"},
    {"credential_expires_at": datetime.now()},
    {"credential_digest": bytearray(b"x" * 32)},
])
def test_invalid_issue_request_fails_before_sql(monkeypatch, change):
    _configure_audit(monkeypatch)
    now = datetime.now(timezone.utc)
    values = {
        "operator_key_version": 1, "operator_digest": b"p" * 32,
        "tenant_id": TENANT, "account_id": ACCOUNT,
        "credential_digest": b"x" * 32,
        "scopes": ("documents.read", "rag.query"),
        "account_expires_at": now + timedelta(days=30),
        "credential_expires_at": now + timedelta(days=7),
        "reason_code": "security_provisioning",
    }
    values.update(change)
    connection = _Connection([(1,)])
    with pytest.raises(db.ControlPlaneRefused):
        db.issue_service_account(connection, **values)
    assert connection.cursor_instance.query is None
