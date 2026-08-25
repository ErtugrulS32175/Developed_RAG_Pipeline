"""Real PostgreSQL proof for the content-free routing control plane."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import uuid

import pytest

from pipeline.control import db


DSN = os.getenv("RAGTEST_CONTROL_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_CONTROL_PG_DSN is absent")


@pytest.fixture(scope="module")
def control_database():
    import psycopg

    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute("DROP SCHEMA IF EXISTS rag_control CASCADE")
        connection = psycopg.connect(DSN)
        db.init_schema(connection)
        yield connection
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute("DROP SCHEMA IF EXISTS rag_control CASCADE")
        finally:
            admin.close()


def _seed(connection):
    tenant = uuid.UUID("10000000-0000-0000-0000-000000000001")
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control.control_regions "
            "(region_code, state) VALUES ('eu-central', 'active')")
        cursor.execute(
            "INSERT INTO rag_control.control_tenants "
            "(tenant_id, lifecycle, deployment_profile) "
            "VALUES (%s, 'active', 'enterprise')", (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_routes "
            "(tenant_id, route_kind, region_code, connection_ref, state) "
            "VALUES (%s, 'dedicated_postgres', 'eu-central', %s, 'active')",
            (tenant, "vault:tenant-a/postgres"))
        cursor.execute(
            "INSERT INTO rag_control.control_identity_routes "
            "(tenant_id, state) VALUES (%s, 'active') RETURNING identity_id",
            (tenant,))
        identity = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_identity_route_digests "
            "(identity_id, key_version, digest, state) "
            "VALUES (%s, 1, %s, 'active')", (identity, b"a" * 32))
        cursor.execute(
            "INSERT INTO rag_control.control_feature_catalog "
            "VALUES ('review', 'active', 1)")
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_features "
            "VALUES (%s, 'review', true, 1)", (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_quotas VALUES "
            "(%s, 60, 4, 20, 1000000, 100, 10, 10, 500000, "
            "'declared', 1)", (tenant,))
    connection.commit()
    return tenant, identity


def _install_lifecycle_roles(connection):
    from psycopg import sql

    passwords = {
        "rag_control_runtime": uuid.uuid4().hex + uuid.uuid4().hex,
        "rag_control_admin": uuid.uuid4().hex + uuid.uuid4().hex,
        "rag_control_redeemer": uuid.uuid4().hex + uuid.uuid4().hex,
    }
    with connection.cursor() as cursor:
        cursor.execute("""
            DO $roles$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_roles
                    WHERE rolname = 'rag_control_runtime') THEN
                    CREATE ROLE rag_control_runtime;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_roles
                    WHERE rolname = 'rag_control_admin') THEN
                    CREATE ROLE rag_control_admin;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM pg_roles
                    WHERE rolname = 'rag_control_redeemer') THEN
                    CREATE ROLE rag_control_redeemer;
                END IF;
            END
            $roles$;
            REVOKE ALL ON SCHEMA rag_control
                FROM rag_control_runtime, rag_control_admin,
                     rag_control_redeemer;
            REVOKE ALL ON ALL TABLES IN SCHEMA rag_control
                FROM rag_control_runtime, rag_control_admin,
                     rag_control_redeemer;
            REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control
                FROM rag_control_runtime, rag_control_admin,
                     rag_control_redeemer;
            REVOKE ALL ON ALL FUNCTIONS IN SCHEMA rag_control
                FROM rag_control_runtime, rag_control_admin,
                     rag_control_redeemer;
            GRANT USAGE ON SCHEMA rag_control
                TO rag_control_runtime, rag_control_admin,
                   rag_control_redeemer;
            GRANT SELECT ON rag_control.control_schema_state
                TO rag_control_runtime, rag_control_admin,
                   rag_control_redeemer;
            GRANT EXECUTE ON FUNCTION
                rag_control.control_tenant_facts(uuid),
                rag_control.control_resolve_identity(integer, bytea),
                rag_control.control_resolve_platform_operator(integer, bytea),
                rag_control.control_resolve_service_account(
                    uuid, integer, bytea)
                TO rag_control_runtime;
            GRANT EXECUTE ON FUNCTION
                rag_control.control_approve_service_account_issue(
                    integer, bytea, uuid, uuid, uuid, text[], timestamptz,
                    timestamptz, bigint, text, bytea, bytea),
                rag_control.control_approve_service_account_rotation(
                    integer, bytea, uuid, uuid, uuid, bigint, timestamptz,
                    bigint, text, bytea, bytea),
                rag_control.control_cancel_service_account_approval(
                    integer, bytea, uuid, uuid, uuid, bigint, text),
                rag_control.control_revoke_service_account(
                    integer, bytea, uuid, uuid, bigint, text, bytea, bytea)
                TO rag_control_admin;
        """)
        for role, password in passwords.items():
            cursor.execute(sql.SQL(
                "ALTER ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS "
                "NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION"
            ).format(sql.Identifier(role), sql.Literal(password)))
            cursor.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(connection.info.dbname), sql.Identifier(role)))
            cursor.execute(sql.SQL(
                "REVOKE CREATE ON DATABASE {} FROM {}"
            ).format(
                sql.Identifier(connection.info.dbname), sql.Identifier(role)))
    connection.commit()
    return passwords


def _connect_as_role(role, password):
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    parameters = conninfo_to_dict(DSN)
    parameters.update(user=role, password=password)
    return psycopg.connect(make_conninfo(**parameters))


def _owner_as_role(role):
    import psycopg
    from psycopg import sql

    connection = psycopg.connect(DSN, autocommit=True)
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(role)))
    connection.autocommit = False
    return connection


def test_active_identity_resolves_only_through_an_active_route(control_database):
    tenant, _identity = _seed(control_database)
    route = db.resolve_identity(control_database, 1, b"a" * 32)
    assert route is not None
    assert route.facts.tenant_id == tenant
    assert route.connection_ref == "vault:tenant-a/postgres"
    assert route.facts.features == {"review": True}
    assert route.facts.quotas["document_count"] == 100
    assert route.facts.quota_enforcement == "declared"

    with control_database.cursor() as cursor:
        cursor.execute(
            "UPDATE rag_control.control_tenants SET lifecycle = 'suspended' "
            "WHERE tenant_id = %s", (tenant,))
    control_database.commit()
    assert db.resolve_identity(control_database, 1, b"a" * 32) is None


def test_digest_rotation_and_revocation_stay_bound_to_one_identity(
        control_database):
    tenant = uuid.UUID("10000000-0000-0000-0000-000000000001")
    with control_database.cursor() as cursor:
        cursor.execute(
            "UPDATE rag_control.control_tenants SET lifecycle = 'active' "
            "WHERE tenant_id = %s", (tenant,))
        cursor.execute(
            "SELECT identity_id FROM rag_control.control_identity_routes "
            "WHERE tenant_id = %s", (tenant,))
        identity = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_identity_route_digests "
            "(identity_id, key_version, digest, state) "
            "VALUES (%s, 2, %s, 'active')", (identity, b"b" * 32))
    control_database.commit()
    assert db.resolve_identity(control_database, 1, b"a" * 32) is not None
    assert db.resolve_identity(control_database, 2, b"b" * 32) is not None

    with control_database.cursor() as cursor:
        cursor.execute(
            "UPDATE rag_control.control_identity_route_digests "
            "SET state = 'retired' WHERE identity_id = %s AND key_version = 1",
            (identity,))
    control_database.commit()
    assert db.resolve_identity(control_database, 1, b"a" * 32) is None
    assert db.resolve_identity(control_database, 2, b"b" * 32) is not None


def test_service_account_resolver_uses_database_time_and_one_live_credential(
        control_database):
    tenant = uuid.UUID("10000000-0000-0000-0000-000000000001")
    account = uuid.UUID("30000000-0000-0000-0000-000000000003")
    now = datetime.now(timezone.utc)
    with control_database.cursor() as cursor:
        cursor.execute(
            "UPDATE rag_control.control_tenants SET lifecycle = 'active' "
            "WHERE tenant_id = %s", (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_service_accounts "
            "(service_account_id, tenant_id, state, expires_at) "
            "VALUES (%s, %s, 'active', %s)",
            (account, tenant, now + timedelta(hours=2)))
        cursor.execute(
            "INSERT INTO rag_control.control_service_account_scopes "
            "VALUES (%s, 'rag.query'), (%s, 'documents.read')",
            (account, account))
        cursor.execute(
            "INSERT INTO rag_control.control_service_account_credentials "
            "(service_account_id, credential_version, digest, state, "
            "not_before, expires_at) VALUES (%s, 1, %s, 'active', %s, %s)",
            (account, b"s" * 32, now - timedelta(minutes=1),
             now + timedelta(hours=1)))
    control_database.commit()

    route = db.resolve_service_account(control_database, account, 1, b"s" * 32)
    assert route is not None
    assert route.facts.tenant_id == tenant
    assert route.scopes == ("documents.read", "rag.query")

    with pytest.raises(Exception) as overlap:
        with control_database.cursor() as cursor:
            cursor.execute(
                "INSERT INTO rag_control.control_service_account_credentials "
                "(service_account_id, credential_version, digest, state, "
                "not_before, expires_at) "
                "VALUES (%s, 2, %s, 'active', %s, %s)",
                (account, b"t" * 32, now - timedelta(minutes=1),
                 now + timedelta(hours=1)))
    control_database.rollback()
    assert type(overlap.value).__name__ == "UniqueViolation"

    with control_database.cursor() as cursor:
        cursor.execute(
            "UPDATE rag_control.control_service_account_credentials "
            "SET state = 'revoked' WHERE service_account_id = %s", (account,))
    control_database.commit()
    assert db.resolve_service_account(
        control_database, account, 1, b"s" * 32) is None

def test_digest_is_globally_unique_and_events_are_owner_immutable(
        control_database):
    tenant = uuid.UUID("10000000-0000-0000-0000-000000000001")
    other = uuid.UUID("20000000-0000-0000-0000-000000000002")
    with control_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control.control_tenants "
            "(tenant_id, lifecycle, deployment_profile) "
            "VALUES (%s, 'provisioning', 'team')", (other,))
        cursor.execute(
            "INSERT INTO rag_control.control_identity_routes "
            "(tenant_id, state) VALUES (%s, 'active') RETURNING identity_id",
            (other,))
        other_identity = cursor.fetchone()[0]
    control_database.commit()
    with pytest.raises(Exception) as collision:
        with control_database.cursor() as cursor:
            cursor.execute(
                "INSERT INTO rag_control.control_identity_route_digests "
                "(identity_id, key_version, digest, state) "
                "VALUES (%s, 2, %s, 'active')",
                (other_identity, b"b" * 32))
    control_database.rollback()
    assert type(collision.value).__name__ == "UniqueViolation"

    with control_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operators "
            "(role, state) VALUES ('platform_security', 'active') "
            "RETURNING operator_id")
        operator = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_admin_events "
            "(operator_id, target_tenant_id, action, target_kind, reason_code, "
            "decision, request_digest) VALUES "
            "(%s, %s, 'tenant_suspend', 'tenant', 'policy_change', "
            "'accepted', %s) RETURNING event_id",
            (operator, tenant, b"r" * 32))
        event = cursor.fetchone()[0]
    control_database.commit()
    with pytest.raises(Exception) as immutable:
        with control_database.cursor() as cursor:
            cursor.execute(
                "DELETE FROM rag_control.control_admin_events "
                "WHERE event_id = %s", (event,))
    control_database.rollback()
    assert type(immutable.value).__name__ == "ObjectNotInPrerequisiteState"


@pytest.mark.parametrize("connection_ref", [
    "postgresql://user:secret@host/database",
    "vault:user@host",
    "opaque value",
])
def test_route_reference_cannot_carry_a_dsn_or_inline_credential(
        control_database, connection_ref):
    tenant = uuid.uuid4()
    with pytest.raises(Exception) as caught:
        with control_database.cursor() as cursor:
            cursor.execute(
                "INSERT INTO rag_control.control_tenants "
                "(tenant_id, lifecycle, deployment_profile) "
                "VALUES (%s, 'provisioning', 'enterprise')", (tenant,))
            cursor.execute(
                "INSERT INTO rag_control.control_tenant_routes "
                "(tenant_id, route_kind, region_code, connection_ref, state) "
                "VALUES (%s, 'dedicated_postgres', 'eu-central', %s, 'pending')",
                (tenant, connection_ref))
    control_database.rollback()
    assert type(caught.value).__name__ == "CheckViolation"


def test_platform_security_service_account_lifecycle_is_atomic(
        control_database, monkeypatch):
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "a" * 32)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 32)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 32)
    tenant = uuid.uuid4()
    account = uuid.uuid4()
    refused_account = uuid.uuid4()
    now = datetime.now(timezone.utc)
    passwords = _install_lifecycle_roles(control_database)
    with pytest.raises(db.ControlPlaneRefused):
        db._configure_runtime_connection(control_database)
    with pytest.raises(db.ControlPlaneRefused):
        db._configure_admin_connection(control_database)
    with control_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control.control_regions "
            "(region_code, state) VALUES ('lifecycle-test', 'active') "
            "ON CONFLICT (region_code) DO UPDATE SET state = 'active'")
        cursor.execute(
            "INSERT INTO rag_control.control_tenants "
            "(tenant_id, lifecycle, deployment_profile) "
            "VALUES (%s, 'active', 'enterprise')", (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_routes "
            "(tenant_id, route_kind, region_code, connection_ref, state) "
            "VALUES (%s, 'shared_rls', 'lifecycle-test', %s, 'active')",
            (tenant, f"route:{tenant.hex}"))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_quotas VALUES "
            "(%s, 10, 2, 2, 1000, 10, 1, 1, 1000, 'declared', 1)",
            (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operators "
            "(role, state) VALUES ('platform_security', 'active') "
            "RETURNING operator_id")
        security = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operator_digests "
            "(operator_id, key_version, digest, state) "
            "VALUES (%s, 1, %s, 'active')", (security, b"p" * 32))
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operators "
            "(role, state) VALUES ('platform_operator', 'active') "
            "RETURNING operator_id")
        ordinary = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operator_digests "
            "(operator_id, key_version, digest, state) "
            "VALUES (%s, 1, %s, 'active')", (ordinary, b"q" * 32))
    control_database.commit()

    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        db._configure_runtime_connection(runtime)
        db._configure_admin_connection(admin)
        assert db.resolve_platform_operator(
            runtime, 1, b"p" * 32) == db.PlatformOperator(
                security, "platform_security", 1)
        with pytest.raises(db.ControlPlaneDenied):
            db.issue_service_account(
                admin, operator_key_version=1,
                operator_digest=b"q" * 32, tenant_id=tenant,
                account_id=refused_account, credential_digest=b"r" * 32,
                scopes=("rag.query",),
                account_expires_at=now + timedelta(days=30),
                credential_expires_at=now + timedelta(days=7),
                reason_code="security_provisioning")
        admin.rollback()
        with control_database.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM rag_control.control_service_accounts "
                "WHERE service_account_id = %s", (refused_account,))
            assert cursor.fetchone()[0] == 0

        with pytest.raises(db.ControlPlaneDenied):
            db.issue_service_account(
                runtime, operator_key_version=1,
                operator_digest=b"p" * 32, tenant_id=tenant,
                account_id=refused_account, credential_digest=b"r" * 32,
                scopes=("rag.query",),
                account_expires_at=now + timedelta(days=30),
                credential_expires_at=now + timedelta(days=7),
                reason_code="security_provisioning")
        runtime.rollback()

        assert db.issue_service_account(
            control_database, operator_key_version=1,
            operator_digest=b"p" * 32, tenant_id=tenant,
            account_id=account, credential_digest=b"x" * 32,
            scopes=("documents.read", "rag.query"),
            account_expires_at=now + timedelta(days=30),
            credential_expires_at=now + timedelta(days=7),
            reason_code="security_provisioning") == 1
        control_database.commit()

    # The retire-first order is safe only because both writes share one
    # transaction. A duplicate digest aborts the insert and rollback restores
    # the old active credential instead of stranding the account.
        with pytest.raises(db.ControlPlaneRefused):
            db.rotate_service_account(
                control_database, operator_key_version=1,
                operator_digest=b"p" * 32, tenant_id=tenant,
                account_id=account, expected_revision=1,
                credential_digest=b"x" * 32,
                credential_expires_at=now + timedelta(days=8),
                reason_code="scheduled_rotation")
        control_database.rollback()
        assert db.resolve_service_account(
            runtime, account, 1, b"x" * 32) is not None

        assert db.rotate_service_account(
            control_database, operator_key_version=1,
            operator_digest=b"p" * 32, tenant_id=tenant,
            account_id=account, expected_revision=1,
            credential_digest=b"y" * 32,
            credential_expires_at=now + timedelta(days=8),
            reason_code="scheduled_rotation") == 2
        control_database.commit()
        assert db.resolve_service_account(
            runtime, account, 1, b"x" * 32) is None
        assert db.resolve_service_account(
            runtime, account, 2, b"y" * 32) is not None
        with pytest.raises(db.ControlPlaneConflict):
            db.rotate_service_account(
                control_database, operator_key_version=1,
                operator_digest=b"p" * 32, tenant_id=tenant,
                account_id=account, expected_revision=1,
                credential_digest=b"z" * 32,
                credential_expires_at=now + timedelta(days=8),
                reason_code="scheduled_rotation")
        control_database.rollback()

        assert db.revoke_service_account(
            admin, operator_key_version=1,
            operator_digest=b"p" * 32, tenant_id=tenant,
            account_id=account, expected_revision=2,
            reason_code="access_removed") == 3
        admin.commit()
        assert db.resolve_service_account(
            runtime, account, 2, b"y" * 32) is None

        for connection, statement in (
                (admin, "SELECT * FROM rag_control.control_service_accounts"),
                (admin, "SELECT nextval('rag_control."
                        "control_service_account_events_sequence_id_seq')"),
                (admin, "SELECT * FROM rag_control."
                        "control_resolve_identity(1, decode(repeat('00', 32),"
                        " 'hex'))")):
            with pytest.raises(Exception) as forbidden:
                with connection.cursor() as cursor:
                    cursor.execute(statement)
            connection.rollback()
            assert type(forbidden.value).__name__ == "InsufficientPrivilege"
    finally:
        admin.close()
        runtime.close()

    with control_database.cursor() as cursor:
        cursor.execute(
            "SELECT action, expected_revision, resulting_revision, "
            "octet_length(request_digest), octet_length(resulting_fact_digest) "
            "FROM rag_control.control_service_account_events "
            "WHERE service_account_id = %s ORDER BY sequence_id", (account,))
        assert cursor.fetchall() == [
            ("service_account_issue", None, 1, 32, 32),
            ("service_account_rotate", 1, 2, 32, 32),
            ("service_account_revoke", 2, 3, 32, 32),
        ]

        other_tenant = uuid.uuid4()
        cursor.execute(
            "INSERT INTO rag_control.control_tenants "
            "(tenant_id, lifecycle, deployment_profile) "
            "VALUES (%s, 'provisioning', 'enterprise')", (other_tenant,))
        with pytest.raises(Exception) as forged_event:
            cursor.execute(
                "INSERT INTO rag_control.control_service_account_events "
                "(operator_id, target_tenant_id, service_account_id, action, "
                "reason_code, resulting_revision, request_digest, "
                "resulting_fact_digest) VALUES "
                "(%s, %s, %s, 'service_account_revoke', 'access_removed', "
                "4, %s, %s)",
                (security, other_tenant, account, b"a" * 32, b"b" * 32))
    control_database.rollback()
    assert type(forged_event.value).__name__ == "ForeignKeyViolation"


def test_role_readiness_rejects_later_function_privilege_drift(
        control_database):
    passwords = _install_lifecycle_roles(control_database)
    with control_database.cursor() as cursor:
        cursor.execute(
            "CREATE OR REPLACE FUNCTION rag_control.control_rogue_probe() "
            "RETURNS integer LANGUAGE sql AS 'SELECT 1'")
        cursor.execute(
            "REVOKE ALL ON FUNCTION rag_control.control_rogue_probe() "
            "FROM PUBLIC")
        cursor.execute(
            "GRANT EXECUTE ON FUNCTION rag_control.control_rogue_probe() "
            "TO rag_control_runtime, rag_control_admin, rag_control_redeemer")
    control_database.commit()
    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_runtime_connection(runtime)
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_admin_connection(admin)
    finally:
        runtime.close()
        admin.close()
    with control_database.cursor() as cursor:
        cursor.execute(
            "DROP FUNCTION rag_control.control_rogue_probe()")
    control_database.commit()


def test_service_account_approvals_create_no_credential_and_revoke_cancels(
        control_database, monkeypatch):
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "a" * 32)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 32)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 32)
    passwords = _install_lifecycle_roles(control_database)
    tenant = uuid.uuid4()
    issue_account = uuid.uuid4()
    active_account = uuid.uuid4()
    issue_approval = uuid.uuid4()
    rotate_approval = uuid.uuid4()
    replacement_approval = uuid.uuid4()
    active_digest = uuid.uuid4().bytes + uuid.uuid4().bytes
    now = datetime.now(timezone.utc)
    with control_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control.control_regions "
            "(region_code, state) VALUES ('approval-test', 'active') "
            "ON CONFLICT (region_code) DO UPDATE SET state = 'active'")
        cursor.execute(
            "INSERT INTO rag_control.control_tenants "
            "(tenant_id, lifecycle, deployment_profile) "
            "VALUES (%s, 'active', 'enterprise')", (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_routes "
            "(tenant_id, route_kind, region_code, connection_ref, state) "
            "VALUES (%s, 'shared_rls', 'approval-test', %s, 'active')",
            (tenant, f"route:{tenant.hex}"))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_quotas VALUES "
            "(%s, 10, 2, 2, 1000, 10, 1, 1, 1000, 'declared', 1)",
            (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operators "
            "(role, state) VALUES ('platform_security', 'active') "
            "RETURNING operator_id")
        operator = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operator_digests "
            "(operator_id, key_version, digest, state) "
            "VALUES (%s, 3, %s, 'active')", (operator, b"k" * 32))
    control_database.commit()
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        db._configure_admin_connection(admin)
        issue = db.approve_service_account_issue(
            admin, operator_key_version=3, operator_digest=b"k" * 32,
            approval_id=issue_approval, tenant_id=tenant,
            account_id=issue_account,
            scopes=("documents.read", "rag.query"),
            account_expires_at=now + timedelta(days=30),
            credential_expires_at=now + timedelta(days=7),
            expected_policy_revision=1,
            reason_code="security_provisioning")
        admin.commit()
        assert issue.approval_revision == 1
        assert db.list_redeemable_service_account_approvals(
            control_database, tenant) == (issue,)
        with control_database.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM rag_control.control_service_accounts "
                "WHERE service_account_id = %s", (issue_account,))
            assert cursor.fetchone()[0] == 0
        assert db.cancel_service_account_approval(
            admin, operator_key_version=3, operator_digest=b"k" * 32,
            approval_id=issue_approval, tenant_id=tenant,
            account_id=issue_account, expected_approval_revision=1,
            reason_code="approval_cancelled") == 2
        admin.commit()
        assert db.list_redeemable_service_account_approvals(
            control_database, tenant) == ()

        assert db.issue_service_account(
            control_database, operator_key_version=3,
            operator_digest=b"k" * 32,
            tenant_id=tenant, account_id=active_account,
            credential_digest=active_digest, scopes=("rag.query",),
            account_expires_at=now + timedelta(days=30),
            credential_expires_at=now + timedelta(days=7),
            reason_code="security_provisioning") == 1
        control_database.commit()
        rotation = db.approve_service_account_rotation(
            admin, operator_key_version=3, operator_digest=b"k" * 32,
            approval_id=rotate_approval, tenant_id=tenant,
            account_id=active_account, expected_account_revision=1,
            credential_expires_at=now + timedelta(days=8),
            expected_policy_revision=1,
            reason_code="scheduled_rotation")
        admin.commit()
        assert rotation.action == "rotate"
        with pytest.raises(db.ControlPlaneConflict):
            db.approve_service_account_rotation(
                admin, operator_key_version=3, operator_digest=b"k" * 32,
                approval_id=uuid.uuid4(), tenant_id=tenant,
                account_id=active_account, expected_account_revision=1,
                credential_expires_at=now + timedelta(days=8),
                expected_policy_revision=1,
                reason_code="scheduled_rotation")
        admin.rollback()
        with control_database.cursor() as cursor:
            cursor.execute(
                "UPDATE rag_control.control_service_account_approvals "
                "SET created_at = statement_timestamp() - interval '16 minutes', "
                "expires_at = statement_timestamp() - interval '1 minute' "
                "WHERE approval_id = %s", (rotate_approval,))
        control_database.commit()
        replacement = db.approve_service_account_rotation(
            admin, operator_key_version=3, operator_digest=b"k" * 32,
            approval_id=replacement_approval, tenant_id=tenant,
            account_id=active_account, expected_account_revision=1,
            credential_expires_at=now + timedelta(days=8),
            expected_policy_revision=1,
            reason_code="scheduled_rotation")
        admin.commit()
        assert replacement.approval_revision == 1
        assert db.revoke_service_account(
            admin, operator_key_version=3, operator_digest=b"k" * 32,
            tenant_id=tenant, account_id=active_account,
            expected_revision=1, reason_code="access_removed") == 2
        admin.commit()
    finally:
        admin.close()
    with control_database.cursor() as cursor:
        cursor.execute(
            "SELECT state, approval_revision FROM "
            "rag_control.control_service_account_approvals "
            "WHERE approval_id = %s", (rotate_approval,))
        assert cursor.fetchone() == ("cancelled", 2)
        cursor.execute(
            "SELECT action, actor_kind, reason_code, approval_revision FROM "
            "rag_control.control_service_account_approval_events "
            "WHERE approval_id = %s ORDER BY sequence_id", (rotate_approval,))
        assert cursor.fetchall() == [
            ("approval_created", "platform_security", "scheduled_rotation", 1),
            ("approval_cancelled", "system", "approval_expired", 2),
        ]
        cursor.execute(
            "SELECT state, approval_revision FROM "
            "rag_control.control_service_account_approvals "
            "WHERE approval_id = %s", (replacement_approval,))
        assert cursor.fetchone() == ("cancelled", 2)
        cursor.execute(
            "SELECT action, actor_kind, reason_code FROM "
            "rag_control.control_service_account_approval_events "
            "WHERE approval_id = %s ORDER BY sequence_id",
            (replacement_approval,))
        assert cursor.fetchall() == [
            ("approval_created", "platform_security", "scheduled_rotation"),
            ("approval_cancelled", "platform_security",
             "service_account_revoked"),
        ]


def test_offline_authority_consumes_one_exact_approval_and_replay_fails(
        control_database, monkeypatch):
    monkeypatch.setenv("CONTROL_AUDIT_HMAC_SECRET", "a" * 32)
    monkeypatch.setenv("CONTROL_IDENTITY_HMAC_SECRET", "i" * 32)
    monkeypatch.setenv("CONTROL_SERVICE_ACCOUNT_HMAC_SECRET", "s" * 32)
    monkeypatch.setenv("OIDC_SESSION_SECRET", "o" * 32)
    passwords = _install_lifecycle_roles(control_database)
    tenant = uuid.uuid4()
    account = uuid.uuid4()
    approval_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    with control_database.cursor() as cursor:
        cursor.execute(
            "INSERT INTO rag_control.control_regions "
            "(region_code, state) VALUES ('redeem-test', 'active') "
            "ON CONFLICT (region_code) DO UPDATE SET state = 'active'")
        cursor.execute(
            "INSERT INTO rag_control.control_tenants "
            "(tenant_id, lifecycle, deployment_profile) "
            "VALUES (%s, 'active', 'enterprise')", (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_routes "
            "(tenant_id, route_kind, region_code, connection_ref, state) "
            "VALUES (%s, 'shared_rls', 'redeem-test', %s, 'active')",
            (tenant, f"route:{tenant.hex}"))
        cursor.execute(
            "INSERT INTO rag_control.control_tenant_quotas VALUES "
            "(%s, 10, 2, 2, 1000, 10, 1, 1, 1000, 'declared', 1)",
            (tenant,))
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operators "
            "(role, state) VALUES ('platform_security', 'active') "
            "RETURNING operator_id")
        operator = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO rag_control.control_platform_operator_digests "
            "(operator_id, key_version, digest, state) "
            "VALUES (%s, 5, %s, 'active')", (operator, b"v" * 32))
    control_database.commit()
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    untrusted = _connect_as_role(
        "rag_control_redeemer", passwords["rag_control_redeemer"])
    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    try:
        db._configure_admin_connection(admin)
        db._configure_runtime_connection(runtime)
        with pytest.raises(db.ControlPlaneDenied):
            db.issue_service_account(
                admin, operator_key_version=5, operator_digest=b"v" * 32,
                tenant_id=tenant, account_id=account,
                credential_digest=b"x" * 32, scopes=("rag.query",),
                account_expires_at=now + timedelta(days=30),
                credential_expires_at=now + timedelta(days=7),
                reason_code="security_provisioning")
        admin.rollback()
        approval = db.approve_service_account_issue(
            admin, operator_key_version=5, operator_digest=b"v" * 32,
            approval_id=approval_id, tenant_id=tenant, account_id=account,
            scopes=("documents.read", "rag.query"),
            account_expires_at=now + timedelta(days=30),
            credential_expires_at=now + timedelta(days=7),
            expected_policy_revision=1,
            reason_code="security_provisioning")
        admin.commit()
        with pytest.raises(db.ControlPlaneRefused):
            db.list_redeemable_service_account_approvals(untrusted, tenant)
        untrusted.rollback()
        with pytest.raises(db.ControlPlaneDenied):
            db.redeem_service_account_approval(
                untrusted, approval, tenant_actor_digest=b"u" * 32,
                org_policy_epoch=1, credential_digest=b"z" * 32)
        untrusted.rollback()
        assert db.list_redeemable_service_account_approvals(
            control_database, tenant) == (approval,)
        with control_database.cursor() as cursor:
            cursor.execute(
                "UPDATE rag_control.control_tenants SET policy_revision = 2 "
                "WHERE tenant_id = %s", (tenant,))
        control_database.commit()
        assert db.list_redeemable_service_account_approvals(
            control_database, tenant) == ()
        with pytest.raises(db.ControlPlaneConflict):
            db.redeem_service_account_approval(
                control_database, approval, tenant_actor_digest=b"t" * 32,
                org_policy_epoch=9, credential_digest=b"b" * 32)
        control_database.rollback()
        with control_database.cursor() as cursor:
            cursor.execute(
                "UPDATE rag_control.control_tenants SET policy_revision = 1 "
                "WHERE tenant_id = %s", (tenant,))
        control_database.commit()
        def attempt_redemption(credential):
            import psycopg

            candidate = psycopg.connect(DSN)
            try:
                result = db.redeem_service_account_approval(
                    candidate, approval, tenant_actor_digest=b"t" * 32,
                    org_policy_epoch=9, credential_digest=credential)
                candidate.commit()
                return "redeemed", credential, result.account_revision
            except db.ControlPlaneConflict:
                candidate.rollback()
                return "conflict", credential, None
            finally:
                candidate.close()

        with ThreadPoolExecutor(max_workers=2) as workers:
            attempts = list(workers.map(
                attempt_redemption, (b"c" * 32, b"d" * 32)))
        assert sorted(item[0] for item in attempts) == ["conflict", "redeemed"]
        winner = next(item for item in attempts if item[0] == "redeemed")
        assert winner[2] == 1
        assert db.resolve_service_account(runtime, account, 1, winner[1])
        runtime.rollback()
        with pytest.raises(db.ControlPlaneConflict):
            db.redeem_service_account_approval(
                control_database, approval, tenant_actor_digest=b"t" * 32,
                org_policy_epoch=9, credential_digest=b"e" * 32)
        control_database.rollback()
    finally:
        runtime.close()
        untrusted.close()
        admin.close()
    with control_database.cursor() as cursor:
        cursor.execute(
            "SELECT state, approval_revision, redeemed_at IS NOT NULL "
            "FROM rag_control.control_service_account_approvals "
            "WHERE approval_id = %s", (approval_id,))
        assert cursor.fetchone() == ("redeemed", 2, True)
        cursor.execute(
            "SELECT action, actor_kind, prior_state, resulting_state, "
            "prior_revision, approval_revision, org_policy_epoch "
            "FROM rag_control.control_service_account_approval_events "
            "WHERE approval_id = %s ORDER BY sequence_id", (approval_id,))
        assert cursor.fetchall() == [
            ("approval_created", "platform_security", None, "approved",
             None, 1, None),
            ("approval_redeemed", "tenant_org_admin", "approved", "redeemed",
             1, 2, 9),
        ]
        cursor.execute(
            "SELECT action, request_digest, resulting_fact_digest FROM "
            "rag_control.control_service_account_approval_events "
            "WHERE approval_id = %s ORDER BY sequence_id", (approval_id,))
        sealed = cursor.fetchall()
        assert len(sealed) == 2
        assert all(len(row[1]) == 32 and len(row[2]) == 32 for row in sealed)
        assert sealed[0][1:] != sealed[1][1:]
        cursor.execute(
            "SELECT operator_id, actor_kind, tenant_actor_digest, "
            "org_policy_epoch FROM rag_control.control_service_account_events "
            "WHERE service_account_id = %s AND action = "
            "'service_account_issue'", (account,))
        assert cursor.fetchone() == (operator, "tenant_org_admin", b"t" * 32, 9)
        with pytest.raises(Exception) as rewritten:
            cursor.execute(
                "UPDATE rag_control.control_service_account_approval_events "
                "SET approval_revision = 1 WHERE approval_id = %s",
                (approval_id,))
    control_database.rollback()
    assert type(rewritten.value).__name__ == "ObjectNotInPrerequisiteState"
    with control_database.cursor() as cursor:
        with pytest.raises(Exception) as deleted:
            cursor.execute(
                "DELETE FROM "
                "rag_control.control_service_account_approval_events "
                "WHERE approval_id = %s", (approval_id,))
    control_database.rollback()
    assert type(deleted.value).__name__ == "ObjectNotInPrerequisiteState"


def test_role_readiness_rejects_set_role_and_membership_power(
        control_database):
    from psycopg import sql

    passwords = _install_lifecycle_roles(control_database)
    assumed_runtime = _owner_as_role("rag_control_runtime")
    assumed_admin = _owner_as_role("rag_control_admin")
    try:
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_runtime_connection(assumed_runtime)
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_admin_connection(assumed_admin)
    finally:
        assumed_runtime.close()
        assumed_admin.close()

    probe_role = "rag_control_escalation_probe"
    with control_database.cursor() as cursor:
        cursor.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(
            sql.Identifier(probe_role)))
        cursor.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(
            sql.Identifier(probe_role)))
        for role in passwords:
            cursor.execute(sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(probe_role), sql.Identifier(role)))
    control_database.commit()
    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_runtime_connection(runtime)
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_admin_connection(admin)
    finally:
        runtime.close()
        admin.close()
    with control_database.cursor() as cursor:
        for role in passwords:
            cursor.execute(sql.SQL("REVOKE {} FROM {}").format(
                sql.Identifier(probe_role), sql.Identifier(role)))
        cursor.execute(sql.SQL("DROP ROLE {}").format(
            sql.Identifier(probe_role)))
    control_database.commit()


def test_role_readiness_rejects_database_create_power(control_database):
    from psycopg import sql

    passwords = _install_lifecycle_roles(control_database)
    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        db._configure_runtime_connection(runtime)
        db._configure_admin_connection(admin)
    finally:
        runtime.close()
        admin.close()

    with control_database.cursor() as cursor:
        for role in passwords:
            cursor.execute(sql.SQL(
                "GRANT CREATE ON DATABASE {} TO {}"
            ).format(sql.Identifier(control_database.info.dbname),
                     sql.Identifier(role)))
    control_database.commit()
    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_runtime_connection(runtime)
        with pytest.raises(db.ControlPlaneRefused):
            db._configure_admin_connection(admin)
    finally:
        runtime.close()
        admin.close()
    with control_database.cursor() as cursor:
        for role in passwords:
            cursor.execute(sql.SQL(
                "REVOKE CREATE ON DATABASE {} FROM {}"
            ).format(sql.Identifier(control_database.info.dbname),
                     sql.Identifier(role)))
    control_database.commit()


def test_existing_v2_service_account_table_upgrades_to_v3(
        control_database):
    schema_sql = (
        Path(__file__).resolve().parent.parent / "pipeline" / "control" /
        "schema.sql").read_text(encoding="utf-8")
    with control_database.cursor() as cursor:
        cursor.execute(
            "DROP TABLE rag_control.control_service_account_events")
        cursor.execute(
            "DROP INDEX rag_control.control_service_account_tenant_identity")
        cursor.execute(schema_sql)
        cursor.execute(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = 'rag_control' "
            "AND indexname = 'control_service_account_tenant_identity'")
        assert cursor.fetchone()[0] == 1
        cursor.execute(
            "SELECT count(*) FROM information_schema.table_constraints "
            "WHERE constraint_schema = 'rag_control' "
            "AND table_name = 'control_service_account_events' "
            "AND constraint_type = 'FOREIGN KEY'")
        assert cursor.fetchone()[0] == 2
    control_database.commit()


def test_v3_shaped_control_schema_upgrades_through_init_schema(
        control_database):
    with control_database.cursor() as cursor:
        cursor.execute(
            "DROP FUNCTION rag_control.control_redeem_service_account_issue("
            "uuid,uuid,uuid,bigint,bytea,bigint,bytea,bytea,bytea)")
        cursor.execute(
            "DROP FUNCTION rag_control.control_redeem_service_account_rotation("
            "uuid,uuid,uuid,bigint,bytea,bigint,bytea,bytea,bytea)")
        cursor.execute(
            "DROP FUNCTION rag_control.control_cancel_service_account_approval("
            "integer,bytea,uuid,uuid,uuid,bigint,text)")
        cursor.execute(
            "DROP FUNCTION rag_control."
            "control_list_redeemable_service_account_approvals(uuid,integer)")
        cursor.execute(
            "DROP FUNCTION rag_control.control_approve_service_account_issue("
            "integer,bytea,uuid,uuid,uuid,text[],timestamptz,timestamptz,"
            "bigint,text,bytea,bytea)")
        cursor.execute(
            "DROP FUNCTION rag_control.control_approve_service_account_rotation("
            "integer,bytea,uuid,uuid,uuid,bigint,timestamptz,bigint,text,"
            "bytea,bytea)")
        cursor.execute(
            "DROP FUNCTION rag_control."
            "control_expire_service_account_approval(uuid)")
        cursor.execute(
            "DROP FUNCTION rag_control.control_lock_service_account(uuid)")
        cursor.execute(
            "DROP TABLE rag_control.control_service_account_approval_events")
        cursor.execute(
            "DROP TABLE rag_control.control_service_account_approvals")
        cursor.execute(
            "DROP FUNCTION "
            "rag_control.control_seal_service_account_approval_event()")
        for constraint in (
                "control_service_account_events_actor_shape_check",
                "control_service_account_events_epoch_check",
                "control_service_account_events_actor_digest_check",
                "control_service_account_events_actor_kind_check"):
            cursor.execute(
                "ALTER TABLE rag_control.control_service_account_events "
                f"DROP CONSTRAINT IF EXISTS {constraint}")
        cursor.execute(
            "ALTER TABLE rag_control.control_service_account_events "
            "DROP COLUMN org_policy_epoch, DROP COLUMN tenant_actor_digest, "
            "DROP COLUMN actor_kind")
        cursor.execute(
            "DROP TRIGGER control_schema_state_monotonic ON "
            "rag_control.control_schema_state")
        cursor.execute(
            "DELETE FROM rag_control.control_schema_history "
            "WHERE schema_version = 4")
        cursor.execute(
            "UPDATE rag_control.control_schema_state SET schema_version = 3, "
            "schema_sha256 = repeat('3', 64)")
        cursor.execute(
            "INSERT INTO rag_control.control_schema_history "
            "(schema_version, schema_sha256) VALUES (3, repeat('3', 64)) "
            "ON CONFLICT (schema_version) DO UPDATE SET "
            "schema_sha256 = EXCLUDED.schema_sha256")
    control_database.commit()

    db.init_schema(control_database)
    db.init_schema(control_database)
    expected_digest = hashlib.sha256(
        (Path(__file__).resolve().parent.parent / "pipeline" / "control" /
         "schema.sql").read_bytes()).hexdigest()
    with control_database.cursor() as cursor:
        cursor.execute(
            "SELECT schema_version, schema_sha256 FROM "
            "rag_control.control_schema_state")
        assert cursor.fetchone() == (4, expected_digest)
        cursor.execute(
            "SELECT array_agg(schema_version ORDER BY schema_version), "
            "max(schema_sha256) FILTER (WHERE schema_version = 4) "
            "FROM rag_control.control_schema_history")
        history, v4_digest = cursor.fetchone()
        assert history[-2:] == [3, 4]
        assert v4_digest == expected_digest
        cursor.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'rag_control' AND table_name IN ("
            "'control_service_account_approvals', "
            "'control_service_account_approval_events')")
        assert cursor.fetchone()[0] == 2
        cursor.execute(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = 'rag_control' "
            "AND indexname = 'control_one_pending_account_approval'")
        assert cursor.fetchone()[0] == 1
        cursor.execute(
            "SELECT count(*) FROM information_schema.table_constraints "
            "WHERE constraint_schema = 'rag_control' "
            "AND table_name = 'control_service_account_approval_events' "
            "AND constraint_type = 'FOREIGN KEY'")
        assert cursor.fetchone()[0] == 2
        signatures = (
            "control_lock_service_account(uuid)",
            "control_expire_service_account_approval(uuid)",
            "control_approve_service_account_issue(integer,bytea,uuid,uuid,"
            "uuid,text[],timestamptz,timestamptz,bigint,text,bytea,bytea)",
            "control_approve_service_account_rotation(integer,bytea,uuid,"
            "uuid,uuid,bigint,timestamptz,bigint,text,bytea,bytea)",
            "control_cancel_service_account_approval(integer,bytea,uuid,uuid,"
            "uuid,bigint,text)",
            "control_list_redeemable_service_account_approvals(uuid,integer)",
            "control_redeem_service_account_issue(uuid,uuid,uuid,bigint,"
            "bytea,bigint,bytea,bytea,bytea)",
            "control_redeem_service_account_rotation(uuid,uuid,uuid,bigint,"
            "bytea,bigint,bytea,bytea,bytea)",
            "control_seal_service_account_approval_event()",
        )
        for signature in signatures:
            qualified = "rag_control." + signature
            cursor.execute(
                "SELECT to_regprocedure(%s), "
                "has_function_privilege('public', to_regprocedure(%s), "
                "'EXECUTE')", (qualified, qualified))
            procedure, public_execute = cursor.fetchone()
            assert procedure is not None
            assert public_execute is False
        cursor.execute(
            "SELECT event_manipulation, trigger_name FROM "
            "information_schema.triggers WHERE event_object_schema = "
            "'rag_control' AND event_object_table = "
            "'control_service_account_approval_events' ORDER BY 1, 2")
        assert cursor.fetchall() == [
            ("DELETE",
             "control_service_account_approval_events_immutable_write"),
            ("INSERT", "control_service_account_approval_events_seal"),
            ("UPDATE",
             "control_service_account_approval_events_immutable_write"),
        ]
    passwords = _install_lifecycle_roles(control_database)
    runtime = _connect_as_role(
        "rag_control_runtime", passwords["rag_control_runtime"])
    admin = _connect_as_role(
        "rag_control_admin", passwords["rag_control_admin"])
    try:
        db._configure_runtime_connection(runtime)
        db._configure_admin_connection(admin)
    finally:
        runtime.close()
        admin.close()
