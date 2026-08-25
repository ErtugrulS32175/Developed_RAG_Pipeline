"""Real PostgreSQL proof for monotonic legacy-to-current migration."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_OPERATIONS_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_OPERATIONS_PG_DSN is absent")
V4_DIGEST = "f8119c5cf0c4661d21946b19ee1bf67094f07d15249ccd9841178fb70312fba5"


@pytest.fixture
def migration_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_v5_" + uuid.uuid4().hex[:12]
    admin = psycopg.connect(DSN, autocommit=True)
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(
                sql.Identifier(schema)))
        conn = psycopg.connect(DSN)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
        conn.commit()
        yield conn
    finally:
        if conn is not None:
            conn.close()
        try:
            with admin.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
        finally:
            admin.close()


def _legacy_receipt(conn, *, version=4, digest=V4_DIGEST):
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE org_audit_events ("
            "id uuid PRIMARY KEY, tenant_id uuid NOT NULL, actor_id uuid NOT NULL, "
            "subject_id uuid, action text NOT NULL CHECK (action IN ("
            "'monitor_view', 'topology_read', 'topology_change', "
            "'access_preview')), reason_code text NOT NULL CHECK (reason_code IN ("
            "'management_duty', 'security_review', 'system_operation', "
            "'policy_preview')), decision text NOT NULL CHECK (decision IN ("
            "'allowed', 'denied')), request_id text NOT NULL CHECK ("
            "length(request_id) BETWEEN 8 AND 64), "
            "created_at timestamptz NOT NULL DEFAULT now())")
        cur.execute(
            "CREATE TABLE rag_schema_state ("
            "singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), "
            "schema_version integer NOT NULL, schema_sha256 text NOT NULL, "
            "applied_at timestamptz NOT NULL DEFAULT now())")
        cur.execute(
            "CREATE TABLE rag_schema_history ("
            "schema_version integer PRIMARY KEY, schema_sha256 text NOT NULL, "
            "applied_at timestamptz NOT NULL DEFAULT now())")
        cur.execute(
            "INSERT INTO rag_schema_history (schema_version, schema_sha256) "
            "VALUES (%s, %s)", (version, digest))
        cur.execute(
            "INSERT INTO rag_schema_state "
            "(singleton, schema_version, schema_sha256) VALUES (true, %s, %s)",
            (version, digest))
    conn.commit()


def test_v4_receipt_advances_to_current_and_repeated_init_is_idempotent(
        migration_database):
    conn = migration_database
    _legacy_receipt(conn)

    db.init_schema(conn)
    assert db.schema_is_current(conn)
    version, digest = db.expected_schema_state()
    assert version == 9 and digest != V4_DIGEST
    with conn.cursor() as cur:
        cur.execute(
            "SELECT schema_version, schema_sha256 FROM rag_schema_history "
            "ORDER BY schema_version")
        assert cur.fetchall() == [(4, V4_DIGEST), (9, digest)]
        for table in ("review_interactions", "review_feedback", "review_cases",
                      "review_case_events"):
            cur.execute("SELECT to_regclass(%s)", (table,))
            assert cur.fetchone()[0] == table
        cur.execute(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = 'org_audit_events'::regclass "
            "AND conname = 'org_audit_events_action_check'")
        action_check = cur.fetchone()[0]
        assert "review_queue_view" in action_check
        assert "review_decision" in action_check
        cur.execute(
            "SELECT count(*) FROM pg_trigger "
            "WHERE tgrelid = 'rag_schema_state'::regclass "
            "AND tgname = 'rag_schema_state_monotonic' AND NOT tgisinternal")
        assert cur.fetchone()[0] == 1

    db.init_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_schema_history")
        assert cur.fetchone()[0] == 2


def test_database_trigger_rolls_back_a_legacy_downgrade_transaction(
        migration_database):
    conn = migration_database
    _legacy_receipt(conn)
    db.init_schema(conn)
    with pytest.raises(Exception, match="schema downgrade refused"):
        with conn.cursor() as cur:
            # Simulates old CREATE/REPLACE work before the legacy binary's final
            # state upsert. The trigger must roll the entire transaction back.
            cur.execute("CREATE TABLE legacy_downgrade_probe (id integer)")
            cur.execute(
                "UPDATE rag_schema_state SET schema_version = 4, "
                "schema_sha256 = %s WHERE singleton = true", (V4_DIGEST,))
        conn.commit()
    conn.rollback()
    assert db.schema_is_current(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('legacy_downgrade_probe')")
        assert cur.fetchone()[0] is None


def test_same_version_wrong_digest_is_refused_before_product_ddl(
        migration_database):
    conn = migration_database
    _legacy_receipt(conn, version=db.SCHEMA_VERSION, digest="0" * 64)
    with pytest.raises(RuntimeError, match="digest"):
        db.init_schema(conn)
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass(current_schema() || '.review_interactions')")
        assert cur.fetchone()[0] is None
