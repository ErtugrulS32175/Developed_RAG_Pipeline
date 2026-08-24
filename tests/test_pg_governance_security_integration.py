"""Real PostgreSQL attacks against governance history and service authority."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_OPERATIONS_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_OPERATIONS_PG_DSN is absent")

TENANT_A = uuid.UUID("7a000000-0000-0000-0000-000000000001")
TENANT_B = uuid.UUID("7b000000-0000-0000-0000-000000000001")
POSITION_A = uuid.UUID("7a000000-0000-0000-0000-000000000010")
POSITION_B = uuid.UUID("7b000000-0000-0000-0000-000000000010")
ACTOR_A = uuid.UUID("7a000000-0000-0000-0000-000000000020")
SUBJECT_A = uuid.UUID("7a000000-0000-0000-0000-000000000021")
REVIEWER_A = uuid.UUID("7a000000-0000-0000-0000-000000000022")
ACTOR_B = uuid.UUID("7b000000-0000-0000-0000-000000000020")
INTERACTION = uuid.UUID("7a000000-0000-0000-0000-000000000030")
CASE = uuid.UUID("7a000000-0000-0000-0000-000000000031")
AUDIT_EVENT = uuid.UUID("7a000000-0000-0000-0000-000000000040")
REVIEW_EVENT = uuid.UUID("7a000000-0000-0000-0000-000000000041")


@pytest.fixture
def governance_database():
    import psycopg
    from psycopg import sql

    suffix = uuid.uuid4().hex[:12]
    schema = "ragtest_governance_" + suffix
    owner_role = "ragtest_governance_owner_" + suffix
    request_role = "ragtest_governance_request_" + suffix
    password = "governance-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    owner = None
    requester = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(owner_role), sql.Literal(password)))
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(request_role), sql.Literal(password)))
            cur.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(owner_role)))
        owner = psycopg.connect(DSN, user=owner_role, password=password)
        requester = psycopg.connect(
            DSN, user=request_role, password=password)
        for conn in (owner, requester):
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(schema)))
            conn.commit()

        db.set_tenant_context(owner, TENANT_A, service=True)
        db.init_schema(owner)
        with owner.cursor() as cur:
            cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                sql.Identifier(schema), sql.Identifier(request_role)))
            cur.execute(sql.SQL(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES "
                "IN SCHEMA {} TO {}").format(
                    sql.Identifier(schema), sql.Identifier(request_role)))
            cur.execute(sql.SQL(
                "REVOKE ALL ON rag_context_secrets, "
                "org_identity_tenant_bindings FROM {}").format(
                    sql.Identifier(request_role)))
            cur.execute(sql.SQL(
                "REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, "
                "TRIGGER ON rag_schema_state, rag_schema_history FROM {}"
            ).format(sql.Identifier(request_role)))
            cur.executemany(
                "INSERT INTO org_tenants (id, name) VALUES (%s, %s)",
                [(TENANT_A, "Tenant A"), (TENANT_B, "Tenant B")])
            cur.executemany(
                "INSERT INTO org_positions "
                "(id, tenant_id, parent_id, title, kind, "
                "can_monitor_descendants, protected_from_monitoring) "
                "VALUES (%s, %s, NULL, %s, 'root', true, true)",
                [(POSITION_A, TENANT_A, "Root A"),
                 (POSITION_B, TENANT_B, "Root B")])
            cur.executemany(
                "INSERT INTO org_identities (id, issuer, subject) "
                "VALUES (%s, 'governance-test', %s)",
                [(ACTOR_A, "actor-a"), (SUBJECT_A, "subject-a"),
                 (REVIEWER_A, "reviewer-a"), (ACTOR_B, "actor-b")])
            cur.executemany(
                "INSERT INTO org_architects (tenant_id, identity_id, active) "
                "VALUES (%s, %s, true)",
                [(TENANT_A, ACTOR_A), (TENANT_A, SUBJECT_A),
                 (TENANT_A, REVIEWER_A), (TENANT_B, ACTOR_B)])
            cur.execute(
                "INSERT INTO review_interactions "
                "(id, tenant_id, actor_id, ref_digest, outcome, "
                "citation_count, policy_epoch_at_creation) "
                "VALUES (%s, %s, %s, %s, 'answered', 1, 1)",
                (INTERACTION, TENANT_A, SUBJECT_A, b"g" * 32))
            cur.execute(
                "INSERT INTO review_cases "
                "(id, tenant_id, interaction_id, subject_actor_id, "
                "trigger_code, state, revision, reviewer_id, "
                "resolution_code, decided_at) "
                "VALUES (%s, %s, %s, %s, 'user_feedback', 'resolved', 2, "
                "%s, 'no_issue', now())",
                (CASE, TENANT_A, INTERACTION, SUBJECT_A, REVIEWER_A))
            cur.execute(
                "INSERT INTO review_case_events "
                "(id, tenant_id, case_id, subject_actor_id, reviewer_id, "
                "base_revision, resulting_revision, decision, "
                "resolution_code) VALUES (%s, %s, %s, %s, %s, 1, 2, "
                "'resolved', 'no_issue')",
                (REVIEW_EVENT, TENANT_A, CASE, SUBJECT_A, REVIEWER_A))
            cur.execute(
                "INSERT INTO org_audit_events "
                "(id, tenant_id, actor_id, subject_id, action, reason_code, "
                "decision, request_id) VALUES (%s, %s, %s, %s, "
                "'monitor_view', 'management_duty', 'allowed', "
                "'governance-seed')",
                (AUDIT_EVENT, TENANT_A, ACTOR_A, SUBJECT_A))
        owner.commit()
        yield owner, requester
    finally:
        if requester is not None:
            requester.close()
        if owner is not None:
            owner.close()
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cur.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(request_role)))
                cur.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(owner_role)))
        finally:
            admin.close()


@pytest.mark.parametrize(("table", "row_id", "statement"), [
    ("org_audit_events", AUDIT_EVENT,
     "UPDATE org_audit_events SET request_id = 'governance-mutated' "
     "WHERE id = %s"),
    ("org_audit_events", AUDIT_EVENT,
     "DELETE FROM org_audit_events WHERE id = %s"),
    ("review_case_events", REVIEW_EVENT,
     "UPDATE review_case_events SET resolution_code = 'corrected' "
     "WHERE id = %s"),
    ("review_case_events", REVIEW_EVENT,
     "DELETE FROM review_case_events WHERE id = %s"),
])
def test_governance_history_is_immutable_even_to_the_table_owner(
        governance_database, table, row_id, statement):
    owner, _requester = governance_database
    db.set_tenant_context(owner, TENANT_A, service=True)
    with pytest.raises(Exception, match="immutable"):
        with owner.cursor() as cur:
            cur.execute(statement, (row_id,))
        owner.commit()
    owner.rollback()
    with owner.cursor() as cur:
        cur.execute("SELECT count(*) FROM " + table + " WHERE id = %s",
                    (row_id,))
        assert cur.fetchone()[0] == 1


@pytest.mark.parametrize("foreign_field", ["actor_id", "subject_id"])
def test_an_audit_event_cannot_attribute_a_foreign_tenant_identity(
        governance_database, foreign_field):
    owner, _requester = governance_database
    values = {"actor_id": ACTOR_A, "subject_id": SUBJECT_A}
    values[foreign_field] = ACTOR_B
    db.set_tenant_context(owner, TENANT_A, service=True)
    with pytest.raises(Exception):
        with owner.cursor() as cur:
            cur.execute(
                "INSERT INTO org_audit_events "
                "(id, tenant_id, actor_id, subject_id, action, reason_code, "
                "decision, request_id) VALUES (%s, %s, %s, %s, "
                "'monitor_view', 'management_duty', 'denied', %s)",
                (uuid.uuid4(), TENANT_A, values["actor_id"],
                 values["subject_id"], "foreign-" + foreign_field))
        owner.commit()
    owner.rollback()


def test_a_request_role_cannot_enable_service_access_with_the_guc_alone(
        governance_database):
    owner, requester = governance_database
    assert db.runtime_role_is_safe(owner) is False
    assert db.runtime_role_is_safe(requester) is True
    with requester.cursor() as cur:
        cur.execute("SELECT set_config('rag.service', '1', false)")
        cur.execute("SELECT rag_service_access()")
        assert cur.fetchone()[0] is False


def test_a_runtime_role_cannot_rewrite_control_plane_authority(
        governance_database):
    _owner, requester = governance_database
    with pytest.raises(Exception, match="permission denied"):
        with requester.cursor() as cur:
            cur.execute("SELECT * FROM org_identity_tenant_bindings")
    requester.rollback()
    with pytest.raises(Exception, match="permission denied"):
        with requester.cursor() as cur:
            cur.execute(
                "UPDATE rag_schema_state SET schema_version = 999 "
                "WHERE singleton = true")
    requester.rollback()


def test_only_a_process_signed_context_opens_the_tenant_rls_view(
        governance_database):
    _owner, requester = governance_database
    with requester.cursor() as cur:
        cur.execute("SELECT set_config('rag.tenant_id', %s, false)",
                    (str(TENANT_A),))
        cur.execute("SELECT count(*) FROM org_tenants")
        assert cur.fetchone()[0] == 0
    requester.rollback()

    db.set_tenant_context(requester, TENANT_A)
    with requester.cursor() as cur:
        cur.execute("SELECT id FROM org_tenants")
        assert cur.fetchall() == [(TENANT_A,)]


def test_one_identity_cannot_be_active_in_two_tenants(
        governance_database):
    owner, _requester = governance_database
    db.set_tenant_context(owner, TENANT_A, service=True)
    with pytest.raises(Exception, match="another tenant"):
        with owner.cursor() as cur:
            cur.execute(
                "INSERT INTO org_architects (tenant_id, identity_id, active) "
                "VALUES (%s, %s, true)", (TENANT_B, ACTOR_A))
        owner.commit()
    owner.rollback()
