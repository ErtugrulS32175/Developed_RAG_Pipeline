"""Real PostgreSQL proof for actor-bound feedback and hierarchy review RLS."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_REVIEW_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_REVIEW_PG_DSN is absent")
TENANT = uuid.UUID("76000000-0000-0000-0000-000000000001")
ROOT = uuid.UUID("76000000-0000-0000-0000-000000000010")
MANAGER_A = uuid.UUID("76000000-0000-0000-0000-000000000011")
MANAGER_B = uuid.UUID("76000000-0000-0000-0000-000000000012")
LEAF_A = uuid.UUID("76000000-0000-0000-0000-000000000013")
LEAF_B = uuid.UUID("76000000-0000-0000-0000-000000000014")
PROTECTED = uuid.UUID("76000000-0000-0000-0000-000000000015")
REVIEWER = uuid.UUID("76000000-0000-0000-0000-000000000020")
SUBJECT = uuid.UUID("76000000-0000-0000-0000-000000000021")
OTHER = uuid.UUID("76000000-0000-0000-0000-000000000022")
EXECUTIVE = uuid.UUID("76000000-0000-0000-0000-000000000023")


@pytest.fixture
def review_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_review_" + uuid.uuid4().hex[:12]
    role = "ragtest_review_role_" + uuid.uuid4().hex[:12]
    password = "review-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cur.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        conn = psycopg.connect(DSN, user=role, password=password)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
        conn.commit()
        db.set_tenant_context(conn, TENANT, service=True)
        db.init_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO org_tenants (id, name) VALUES (%s, 'Tenant')",
                (TENANT,))
            cur.executemany(
                "INSERT INTO org_positions "
                "(id, tenant_id, parent_id, title, kind, "
                "can_monitor_descendants, protected_from_monitoring) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [
                    (ROOT, TENANT, None, "Root", "root", True, True),
                    (MANAGER_A, TENANT, ROOT, "Manager A", "manager", True, False),
                    (MANAGER_B, TENANT, ROOT, "Manager B", "manager", True, False),
                    (LEAF_A, TENANT, MANAGER_A, "Leaf A", "member", False, False),
                    (LEAF_B, TENANT, MANAGER_B, "Leaf B", "member", False, False),
                    (PROTECTED, TENANT, ROOT, "Executive", "manager", True, True),
                ])
            cur.executemany(
                "INSERT INTO org_identities (id, issuer, subject) "
                "VALUES (%s, 'open-webui', %s)",
                [(REVIEWER, "reviewer"), (SUBJECT, "subject"),
                 (OTHER, "other"), (EXECUTIVE, "executive")])
            cur.executemany(
                "INSERT INTO org_memberships "
                "(tenant_id, identity_id, position_id, display_label, "
                "app_role, state) VALUES (%s, %s, %s, %s, 'reader', 'active')",
                [(TENANT, REVIEWER, MANAGER_A, "Reviewer"),
                 (TENANT, SUBJECT, LEAF_A, "Subject"),
                 (TENANT, OTHER, LEAF_B, "Other"),
                 (TENANT, EXECUTIVE, PROTECTED, "Executive")])
        conn.commit()
        yield conn
    finally:
        if conn is not None:
            conn.close()
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cur.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        finally:
            admin.close()


def _open_case(conn, actor, digest):
    interaction = uuid.uuid4()
    db.set_tenant_context(conn, TENANT, actor_id=actor)
    db.create_review_interaction(
        conn, interaction_id=interaction, actor_id=actor,
        ref_digest=digest, outcome="answered", citation_count=2)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_setting('rag.actor_id', true), "
            "current_setting('rag.service', true), count(*) "
            "FROM review_interactions")
        assert cur.fetchone() == (str(actor), "0", 1)
        cur.execute(
            "SELECT ref_digest = %s, actor_id = %s, "
            "EXISTS (SELECT 1 FROM org_memberships WHERE identity_id = %s), "
            "EXISTS (SELECT 1 FROM org_identities WHERE id = %s) "
            "FROM review_interactions",
            (digest, actor, actor, actor))
        assert cur.fetchone() == (True, True, True, True)
        cur.execute(
            "SELECT interaction.id FROM review_interactions interaction "
            "JOIN org_memberships membership "
            "ON membership.tenant_id = interaction.tenant_id "
            "AND membership.identity_id = interaction.actor_id "
            "JOIN org_identities identity ON identity.id = interaction.actor_id "
            "WHERE interaction.ref_digest = %s AND interaction.actor_id = %s "
            "AND membership.state = 'active' AND identity.state = 'active'",
            (digest, actor))
        assert cur.fetchone() == (interaction,)
    result = db.submit_review_feedback(
        conn, actor_id=actor, ref_digest=digest,
        verdict="not_helpful", reason_code="incorrect")
    assert result == {"revision": 1, "review_open": True}
    return interaction


def test_feedback_is_actor_bound_idempotent_and_opens_one_case(review_database):
    conn = review_database
    digest = b"a" * 32
    interaction = _open_case(conn, SUBJECT, digest)
    again = db.submit_review_feedback(
        conn, actor_id=SUBJECT, ref_digest=digest,
        verdict="not_helpful", reason_code="incorrect")
    assert again == {"revision": 1, "review_open": True}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM review_cases WHERE interaction_id = %s",
            (interaction,))
        assert cur.fetchone()[0] == 1

    db.set_tenant_context(conn, TENANT, actor_id=OTHER)
    with pytest.raises(db.ReviewAccessRefused):
        db.submit_review_feedback(
            conn, actor_id=OTHER, ref_digest=digest,
            verdict="helpful", reason_code=None)


def test_queue_uses_current_branch_and_absolute_protection(review_database):
    conn = review_database
    _open_case(conn, SUBJECT, b"s" * 32)
    _open_case(conn, OTHER, b"o" * 32)
    _open_case(conn, EXECUTIVE, b"e" * 32)

    db.set_tenant_context(conn, TENANT, actor_id=REVIEWER)
    rows = db.list_review_cases(conn, reviewer_id=REVIEWER, limit=20)
    assert [row["display_label"] for row in rows] == ["Subject"]
    assert not ({"question", "answer", "passage", "source_path"}
                & set(rows[0]))

    db.set_tenant_context(conn, TENANT, actor_id=SUBJECT)
    assert db.list_review_cases(conn, reviewer_id=SUBJECT, limit=20) == []


def test_suspended_membership_revokes_feedback_and_review_authority(
        review_database):
    conn = review_database
    digest = b"r" * 32
    _open_case(conn, SUBJECT, digest)
    db.set_tenant_context(conn, TENANT, service=True)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'suspended' "
            "WHERE tenant_id = %s AND identity_id = %s", (TENANT, SUBJECT))
    conn.commit()
    db.set_tenant_context(conn, TENANT, actor_id=SUBJECT)
    with pytest.raises(db.ReviewAccessRefused):
        db.submit_review_feedback(
            conn, actor_id=SUBJECT, ref_digest=digest,
            verdict="helpful", reason_code=None)


def test_decision_rechecks_revision_policy_epoch_and_fresh_hierarchy(
        review_database):
    conn = review_database
    _open_case(conn, SUBJECT, b"d" * 32)
    db.set_tenant_context(conn, TENANT, actor_id=REVIEWER)
    row = db.list_review_cases(conn, reviewer_id=REVIEWER, limit=1)[0]
    with pytest.raises(db.ReviewConflict):
        db.decide_review_case(
            conn, reviewer_id=REVIEWER, case_id=row["id"],
            expected_revision=row["revision"] + 1,
            expected_policy_epoch=row["policy_epoch"],
            decision="resolved", resolution_code="corrected",
            reason_code="management_duty", request_id="12345678")
    db.set_tenant_context(conn, TENANT, service=True)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_tenants SET policy_epoch = policy_epoch + 1 "
            "WHERE id = %s RETURNING policy_epoch", (TENANT,))
        current_policy_epoch = cur.fetchone()[0]
    conn.commit()
    db.set_tenant_context(conn, TENANT, actor_id=REVIEWER)
    with pytest.raises(db.ReviewConflict):
        db.decide_review_case(
            conn, reviewer_id=REVIEWER, case_id=row["id"],
            expected_revision=row["revision"],
            expected_policy_epoch=row["policy_epoch"],
            decision="resolved", resolution_code="corrected",
            reason_code="management_duty", request_id="12345678")
    result = db.decide_review_case(
        conn, reviewer_id=REVIEWER, case_id=row["id"],
        expected_revision=row["revision"],
        expected_policy_epoch=current_policy_epoch,
        decision="resolved", resolution_code="corrected",
        reason_code="management_duty", request_id="12345678")
    assert result["state"] == "resolved" and result["revision"] == 2
    with pytest.raises(db.ReviewAccessRefused):
        db.decide_review_case(
            conn, reviewer_id=REVIEWER, case_id=row["id"],
            expected_revision=2, expected_policy_epoch=row["policy_epoch"],
            decision="resolved", resolution_code="corrected",
            reason_code="management_duty", request_id="12345679")


def test_suspended_reviewer_loses_queue_and_decision_authority(review_database):
    conn = review_database
    _open_case(conn, SUBJECT, b"v" * 32)
    db.set_tenant_context(conn, TENANT, actor_id=REVIEWER)
    row = db.list_review_cases(conn, reviewer_id=REVIEWER, limit=1)[0]
    db.set_tenant_context(conn, TENANT, service=True)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE org_memberships SET state = 'suspended' "
            "WHERE tenant_id = %s AND identity_id = %s", (TENANT, REVIEWER))
    conn.commit()
    db.set_tenant_context(conn, TENANT, actor_id=REVIEWER)
    assert db.list_review_cases(conn, reviewer_id=REVIEWER, limit=20) == []
    with pytest.raises(db.ReviewAccessRefused):
        db.decide_review_case(
            conn, reviewer_id=REVIEWER, case_id=row["id"],
            expected_revision=row["revision"],
            expected_policy_epoch=row["policy_epoch"],
            decision="resolved", resolution_code="corrected",
            reason_code="management_duty", request_id="12345678")


def test_rls_blocks_direct_cross_actor_case_reads(review_database):
    conn = review_database
    _open_case(conn, OTHER, b"x" * 32)
    db.set_tenant_context(conn, TENANT, actor_id=REVIEWER)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM review_cases")
        assert cur.fetchone()[0] == 0
