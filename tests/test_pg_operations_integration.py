"""Real PostgreSQL proof for versioned, repeatable schema migration."""
import os
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_OPERATIONS_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_OPERATIONS_PG_DSN is absent")


@pytest.fixture(scope="module")
def migrated_connection():
    import psycopg
    from psycopg import sql

    schema_name = f"ragtest_operations_{uuid.uuid4().hex[:12]}"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(
                sql.Identifier(schema_name)))
        connection = psycopg.connect(DSN)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema_name)))
        connection.commit()
        yield connection
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema_name)))
        finally:
            admin.close()


def test_real_migration_is_repeatable_and_readiness_refuses_digest_drift(
        migrated_connection):
    db.init_schema(migrated_connection)
    assert db.schema_is_current(migrated_connection)

    with migrated_connection.cursor() as cursor:
        cursor.execute(
            "UPDATE rag_schema_state SET schema_sha256 = %s "
            "WHERE singleton = true", ("0" * 64,))
    migrated_connection.commit()
    assert not db.schema_is_current(migrated_connection)

    with pytest.raises(RuntimeError, match="digest"):
        db.init_schema(migrated_connection)
    migrated_connection.rollback()
    assert not db.schema_is_current(migrated_connection)
