"""Real PostgreSQL proof for monotonic v5-to-current schema migration."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_EVAL_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_EVAL_PG_DSN is absent")
V5_DIGEST = "693b479dfe51f603cccb8034f1e53567f3183d643bbe5cc2aac418c8489ae95e"


@pytest.fixture
def migration_database():
    import psycopg
    from psycopg import sql

    schema = "ragtest_v6_" + uuid.uuid4().hex[:12]
    role = "ragtest_v6_role_" + uuid.uuid4().hex[:12]
    password = "eval-migration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    conn = None
    try:
        with admin.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cur.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        conn = psycopg.connect(DSN, user=role, password=password)
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
                cur.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        finally:
            admin.close()


def _legacy_receipt(conn, *, version=5, digest=V5_DIGEST):
    with conn.cursor() as cur:
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


def test_v5_receipt_advances_to_current_with_forced_rls_and_is_idempotent(
        migration_database):
    conn = migration_database
    _legacy_receipt(conn)
    db.init_schema(conn)

    version, digest = db.expected_schema_state()
    assert version == db.SCHEMA_VERSION == 11
    assert digest != V5_DIGEST
    assert db.schema_is_current(conn)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT schema_version, schema_sha256 FROM rag_schema_history "
            "ORDER BY schema_version")
        assert cur.fetchall() == [(5, V5_DIGEST), (11, digest)]
        for table in ("eval_datasets", "eval_dataset_versions", "eval_cases",
                      "eval_dataset_events"):
            cur.execute("SELECT to_regclass(%s)", (table,))
            assert cur.fetchone()[0] == table
            cur.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE oid = %s::regclass", (table,))
            assert cur.fetchone() == (True, True)
        cur.execute(
            "SELECT tgname FROM pg_trigger "
            "WHERE tgrelid = 'eval_dataset_versions'::regclass "
            "AND NOT tgisinternal")
        assert "eval_versions_immutable" in {row[0] for row in cur.fetchall()}
        cur.execute(
            "SELECT tgname FROM pg_trigger "
            "WHERE tgrelid = 'eval_cases'::regclass AND NOT tgisinternal")
        assert "eval_cases_immutable" in {row[0] for row in cur.fetchall()}
        cur.execute(
            "SELECT tgname FROM pg_trigger "
            "WHERE tgrelid = 'eval_dataset_events'::regclass "
            "AND NOT tgisinternal")
        assert "eval_events_immutable" in {row[0] for row in cur.fetchall()}
        cur.execute(
            "SELECT tablename, policyname FROM pg_policies "
            "WHERE schemaname = current_schema() AND tablename LIKE 'eval_%'")
        policies = set(cur.fetchall())
        assert {
            ("eval_datasets", "eval_datasets_read"),
            ("eval_datasets", "eval_datasets_insert"),
            ("eval_dataset_versions", "eval_versions_read"),
            ("eval_cases", "eval_cases_read"),
            ("eval_dataset_events", "eval_events_read"),
            ("eval_dataset_events", "eval_events_insert"),
        } <= policies

    db.init_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM rag_schema_history")
        assert cur.fetchone()[0] == 2


def test_database_trigger_rolls_back_a_v5_binary_downgrade_transaction(
        migration_database):
    conn = migration_database
    _legacy_receipt(conn)
    db.init_schema(conn)
    with pytest.raises(Exception, match="schema downgrade refused"):
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE v5_downgrade_probe (id integer)")
            cur.execute(
                "UPDATE rag_schema_state SET schema_version = 5, "
                "schema_sha256 = %s WHERE singleton = true", (V5_DIGEST,))
        conn.commit()
    conn.rollback()
    assert db.schema_is_current(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('v5_downgrade_probe')")
        assert cur.fetchone()[0] is None


def test_same_version_wrong_digest_is_refused_before_eval_product_ddl(
        migration_database):
    conn = migration_database
    _legacy_receipt(conn, version=db.SCHEMA_VERSION, digest="0" * 64)
    with pytest.raises(RuntimeError, match="digest"):
        db.init_schema(conn)
    conn.rollback()
    with conn.cursor() as cur:
        for table in ("eval_datasets", "eval_dataset_versions", "eval_cases",
                      "eval_dataset_events"):
            cur.execute("SELECT to_regclass(current_schema() || '.' || %s)",
                        (table,))
            assert cur.fetchone()[0] is None
