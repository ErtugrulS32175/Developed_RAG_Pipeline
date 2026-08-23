"""Real PostgreSQL/pgvector proof for document-scoped hybrid retrieval.

The normal suite skips this module because pretending a missing server passed
would be a false integration claim. Set both variables to make it a hard gate::

    RAGTEST_SCOPE_GATE=1
    RAGTEST_SCOPE_PG_DSN=postgresql://.../disposable_database

Every run creates and later drops a private schema. It never names or changes
``public.documents`` or ``public.chunks``.
"""
import os
from pathlib import Path
import uuid

import pytest

from pipeline.index import db


DSN = os.getenv("RAGTEST_SCOPE_PG_DSN", "").strip()
GATE = os.getenv("RAGTEST_SCOPE_GATE", "").strip() == "1"

if GATE and not DSN:
    raise RuntimeError(
        "RAGTEST_SCOPE_GATE=1 but RAGTEST_SCOPE_PG_DSN is missing")

pytestmark = pytest.mark.skipif(
    not DSN,
    reason="RAGTEST_SCOPE_PG_DSN is absent: real scope SQL was not checked",
)

INSIDE = uuid.UUID("11111111-1111-1111-1111-111111111111")
OUTSIDE = uuid.UUID("22222222-2222-2222-2222-222222222222")
UNKNOWN = uuid.UUID("33333333-3333-3333-3333-333333333333")


@pytest.fixture(scope="module")
def real_scope_connection():
    import psycopg
    from pgvector import SparseVector, Vector
    from pgvector.psycopg import register_vector
    from psycopg import sql

    schema_name = f"ragtest_scope_{uuid.uuid4().hex[:12]}"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(
                    sql.Identifier(schema_name)))

        connection = psycopg.connect(DSN)
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(schema_name)))
            schema_path = Path(db.__file__).with_name("schema.sql")
            cursor.execute(schema_path.read_text(encoding="utf-8"))
        connection.commit()
        register_vector(connection)

        dense_inside = Vector([0.0, 1.0] + [0.0] * 1022)
        dense_outside = Vector([1.0] + [0.0] * 1023)
        sparse_inside = SparseVector({1: 1.0}, db.SPARSE_DIM)
        sparse_outside = SparseVector({0: 2.0}, db.SPARSE_DIM)
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO documents "
                "(id, filename, file_type, status, active_generation) "
                "VALUES (%s, %s, 'pdf', 'ready', 1)",
                [(INSIDE, "inside.pdf"), (OUTSIDE, "outside.pdf")],
            )
            cursor.executemany(
                "INSERT INTO chunks "
                "(id, document_id, type, text, source_tag, page, headings, "
                "dense, sparse, generation) "
                "VALUES (%s, %s, 'text', %s, %s, 1, '[]'::jsonb, "
                "%s, %s, 1)",
                [
                    (uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                     INSIDE, "inside", "inside:1", dense_inside,
                     sparse_inside),
                    (uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
                     OUTSIDE, "outside", "outside:1", dense_outside,
                     sparse_outside),
                ],
            )
        connection.commit()
        yield connection
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(schema_name)))
        finally:
            admin.close()


def _search(connection, document_ids=None):
    query = [1.0] + [0.0] * 1023
    return db.hybrid_search(
        connection,
        query,
        [0],
        [1.0],
        top_k=1,
        document_ids=document_ids,
    )


def test_scope_is_applied_before_the_real_database_top_k(
        real_scope_connection):
    unscoped = _search(real_scope_connection)
    scoped = _search(real_scope_connection, [INSIDE])

    assert [row["filename"] for row in unscoped] == ["outside.pdf"]
    assert [row["filename"] for row in scoped] == ["inside.pdf"]


def test_an_unknown_real_database_id_never_widens_the_scope(
        real_scope_connection):
    assert _search(real_scope_connection, [UNKNOWN]) == []


def test_real_filename_resolution_is_exact_ordered_and_parameterised(
        real_scope_connection):
    assert db.filenames_for_documents(
        real_scope_connection, [OUTSIDE, INSIDE, OUTSIDE]) == [
            "inside.pdf", "outside.pdf"]


def test_the_real_schema_contains_the_measured_scope_index(
        real_scope_connection):
    with real_scope_connection.cursor() as cursor:
        cursor.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = current_schema() "
            "AND indexname = 'chunks_document_id_idx'")
        row = cursor.fetchone()
    assert row is not None
    assert "USING btree (document_id)" in row[0]


def test_archive_hides_a_document_before_real_ranking_and_restore_returns_it(
        real_scope_connection):
    archived = db.set_document_archived(real_scope_connection, OUTSIDE, True)
    try:
        assert archived["archived"] is True
        assert archived["archived_at"] is not None
        assert [row["filename"] for row in _search(real_scope_connection)] == [
            "inside.pdf"]
        assert db.filenames_for_documents(
            real_scope_connection, [OUTSIDE]) == []
        assert db.active_document_filenames(real_scope_connection) == [
            "inside.pdf"]
        assert [row["filename"] for row in db.list_documents(
            real_scope_connection, limit=10, offset=0)] == ["inside.pdf"]
        archived_rows = db.list_documents(
            real_scope_connection, limit=10, offset=0, archived=True)
        assert [row["filename"] for row in archived_rows] == ["outside.pdf"]
    finally:
        restored = db.set_document_archived(
            real_scope_connection, OUTSIDE, False)
    assert restored["archived"] is False
    assert restored["archived_at"] is None
    assert [row["filename"] for row in _search(real_scope_connection)] == [
        "outside.pdf"]


def test_a_real_active_lease_blocks_archive_without_changing_the_row(
        real_scope_connection):
    attempt_id = uuid.UUID("44444444-4444-4444-4444-444444444444")
    candidate_id = uuid.UUID("55555555-5555-5555-5555-555555555555")
    with real_scope_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO attempts "
            "(attempt_id, document_id, candidate_id, candidate_sha, "
            "observed_active) VALUES (%s, %s, %s, %s, 1)",
            (attempt_id, INSIDE, candidate_id, "kurgu-ozet"),
        )
        cursor.execute(
            "UPDATE documents SET attempt_id = %s WHERE id = %s",
            (attempt_id, INSIDE),
        )
    real_scope_connection.commit()
    try:
        with pytest.raises(db.DocumentLifecycleConflict):
            db.set_document_archived(real_scope_connection, INSIDE, True)
        real_scope_connection.rollback()
        with real_scope_connection.cursor() as cursor:
            cursor.execute(
                "SELECT archived_at FROM documents WHERE id = %s", (INSIDE,))
            assert cursor.fetchone()[0] is None
    finally:
        with real_scope_connection.cursor() as cursor:
            cursor.execute(
                "UPDATE documents SET attempt_id = NULL WHERE id = %s",
                (INSIDE,),
            )
            cursor.execute(
                "DELETE FROM attempts WHERE attempt_id = %s", (attempt_id,))
        real_scope_connection.commit()


def test_a_real_archived_row_cannot_mint_an_ingest_lease(
        real_scope_connection):
    db.set_document_archived(real_scope_connection, OUTSIDE, True)
    try:
        with pytest.raises(db.DocumentLifecycleConflict):
            db.begin_attempt(
                real_scope_connection, OUTSIDE, owner="kurgu-worker")
        with real_scope_connection.cursor() as cursor:
            cursor.execute(
                "SELECT attempt_id FROM documents WHERE id = %s", (OUTSIDE,))
            assert cursor.fetchone()[0] is None
    finally:
        real_scope_connection.rollback()
        db.set_document_archived(real_scope_connection, OUTSIDE, False)


def test_snapshot_retrieval_lock_orders_a_concurrent_real_archive(
        real_scope_connection):
    import psycopg
    from psycopg import sql

    with real_scope_connection.cursor() as cursor:
        cursor.execute("SELECT current_schema()")
        schema_name = cursor.fetchone()[0]
    real_scope_connection.commit()

    other = psycopg.connect(DSN)
    try:
        with other.cursor() as cursor:
            cursor.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(schema_name)))
            cursor.execute("SET lock_timeout = '200ms'")
        other.commit()

        assert db.lock_retrieval_filenames(
            real_scope_connection, [INSIDE]) == ["inside.pdf"]
        with pytest.raises(psycopg.errors.LockNotAvailable):
            db.set_document_archived(other, INSIDE, True)
        other.rollback()

        # Releasing the retrieval transaction makes the same transition
        # succeed; there was contention, not a permanently invalid request.
        real_scope_connection.rollback()
        changed = db.set_document_archived(other, INSIDE, True)
        assert changed["archived"] is True
        restored = db.set_document_archived(other, INSIDE, False)
        assert restored["archived"] is False
    finally:
        real_scope_connection.rollback()
        other.close()
