"""Real PostgreSQL proof for the content-free routing control plane."""
from datetime import datetime, timedelta, timezone
import os
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
