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


def _remove_versioned_document_fixture(connection, document_id):
    """Remove exactly this test-owned graph, restoring the module fixture.

    Production history is immutable.  This private disposable schema needs a
    narrower test teardown so later unscoped assertions still see only their
    two baseline documents.  Exact trigger names are disabled transactionally,
    then restored before commit; an error rolls the whole teardown back.
    """
    disabled = (
        ("documents", "documents_record_version_activation"),
        ("document_version_events", "document_version_events_immutable"),
        ("document_version_builds", "document_version_builds_immutable"),
        ("document_versions", "document_versions_immutable"),
    )
    try:
        with connection.cursor() as cursor:
            for table, trigger in disabled:
                cursor.execute(
                    f"ALTER TABLE {table} DISABLE TRIGGER {trigger}")
            cursor.execute(
                "DELETE FROM document_version_events WHERE document_id = %s",
                (document_id,))
            cursor.execute("DELETE FROM chunks WHERE document_id = %s",
                           (document_id,))
            cursor.execute(
                "DELETE FROM document_version_builds WHERE document_id = %s",
                (document_id,))
            cursor.execute("DELETE FROM attempts WHERE document_id = %s",
                           (document_id,))
            cursor.execute("DELETE FROM ingest_jobs WHERE document_id = %s",
                           (document_id,))
            cursor.execute(
                "UPDATE documents SET active_version_id = NULL, "
                "active_generation = 0 WHERE id = %s", (document_id,))
            cursor.execute(
                "DELETE FROM document_versions WHERE document_id = %s",
                (document_id,))
            cursor.execute("DELETE FROM documents WHERE id = %s",
                           (document_id,))
            for table, trigger in reversed(disabled):
                cursor.execute(
                    f"ALTER TABLE {table} ENABLE TRIGGER {trigger}")
    except Exception:
        connection.rollback()
        raise
    connection.commit()


def test_scope_is_applied_before_the_real_database_top_k(
        real_scope_connection):
    unscoped = _search(real_scope_connection)
    scoped = _search(real_scope_connection, [INSIDE])

    assert [row["filename"] for row in unscoped] == ["outside.pdf"]
    assert [row["filename"] for row in scoped] == ["inside.pdf"]


def test_an_unknown_real_database_id_never_widens_the_scope(
        real_scope_connection):
    assert _search(real_scope_connection, [UNKNOWN]) == []


def test_two_ready_versions_are_retained_and_rollback_moves_both_rankings(
        real_scope_connection):
    """Promotion retains v1; activation rolls dense and sparse back together."""
    from pgvector import SparseVector, Vector

    filename = "rollback-" + uuid.uuid4().hex[:10] + ".pdf"
    sha_one, sha_two, sha_unready = "1" * 64, "2" * 64, "3" * 64

    def stage(sha, *, replace=False):
        document, version, _name = db.stage_candidate(
            real_scope_connection, filename, "pdf", content_sha256=sha,
            allow_replace=replace)
        assert db.finalize_candidate_publication(
            real_scope_connection, document, version)
        return document, version

    def build(document, version, sha, text, dense_head, sparse_head):
        attempt = db.begin_attempt(real_scope_connection, document)
        assert attempt.candidate_id == version
        generation = db.allocate_generation(
            real_scope_connection, document, attempt)
        chunk_id = uuid.uuid4()
        dense = Vector([dense_head] + [0.0] * 1023)
        sparse = SparseVector({0: sparse_head}, db.SPARSE_DIM)
        with real_scope_connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO chunks "
                "(id, document_id, version_id, type, text, source_tag, page, "
                "headings, dense, sparse, generation) VALUES "
                "(%s, %s, %s, 'text', %s, %s, 1, '[]'::jsonb, %s, %s, %s)",
                (chunk_id, document, version, text, text + ":1", dense,
                 sparse, generation))
        real_scope_connection.commit()
        db.promote_generation(
            real_scope_connection, document, generation,
            expected_active=attempt.observed_active,
            manifest_ids={chunk_id}, content_sha256=sha,
            candidate_id=version, attempt_id=attempt.attempt_id)
        return generation, chunk_id

    document, version_one = stage(sha_one)
    generation_one, chunk_one = build(
        document, version_one, sha_one, "version-one", 1.0, 1.0)
    same_document, version_two = stage(sha_two, replace=True)
    assert same_document == document
    generation_two, chunk_two = build(
        document, version_two, sha_two, "version-two", 2.0, 2.0)

    with real_scope_connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, version_id, generation FROM chunks "
            "WHERE document_id = %s ORDER BY generation", (document,))
        assert cursor.fetchall() == [
            (chunk_one, uuid.UUID(version_one), generation_one),
            (chunk_two, uuid.UUID(version_two), generation_two),
        ]

    def visible_texts():
        rows = db.hybrid_search(
            real_scope_connection, [1.0] + [0.0] * 1023,
            [0], [1.0], top_k=5, document_ids=[document])
        return [row["text"] for row in rows]

    assert visible_texts() == ["version-two"]
    with real_scope_connection.cursor() as cursor:
        cursor.execute("SELECT revision FROM documents WHERE id = %s",
                       (document,))
        revision_before_rollback = cursor.fetchone()[0]

    rollback = db.activate_document_version(
        real_scope_connection, document, version_one,
        revision_before_rollback, verified_source_sha256=sha_one)
    assert rollback == {
        "document_id": document,
        "active_version_id": version_one,
        "active_generation": generation_one,
        "revision": revision_before_rollback + 1,
        "changed": True,
    }
    assert visible_texts() == ["version-one"]

    with pytest.raises(db.DocumentVersionConflict):
        db.activate_document_version(
            real_scope_connection, document, version_two,
            revision_before_rollback, verified_source_sha256=sha_two)
    real_scope_connection.rollback()

    same_document, version_unready = stage(sha_unready, replace=True)
    assert same_document == document
    assert db.activate_document_version(
        real_scope_connection, document, version_unready,
        rollback["revision"],
        verified_source_sha256=sha_unready) is None
    _remove_versioned_document_fixture(real_scope_connection, document)


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
    candidate_sha = "5" * 64
    with real_scope_connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO document_versions "
            "(id, tenant_id, document_id, version_number, content_sha256) "
            "SELECT %s, tenant_id, id, 1, %s FROM documents WHERE id = %s",
            (candidate_id, candidate_sha, INSIDE),
        )
        cursor.execute(
            "INSERT INTO attempts "
            "(attempt_id, document_id, candidate_id, version_id, "
            "candidate_sha, observed_active) VALUES (%s, %s, %s, %s, %s, 1)",
            (attempt_id, INSIDE, candidate_id, candidate_id, candidate_sha),
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


def test_real_collection_and_tag_scopes_intersect_before_retrieval(
        real_scope_connection):
    finance = db.create_collection(real_scope_connection, "Finance")
    same = db.create_collection(real_scope_connection, "  FINANCE  ")
    research = db.create_collection(real_scope_connection, "Research")
    assert same["collection_id"] == finance["collection_id"]
    assert same["name"] == "Finance"

    try:
        assert db.set_collection_document(
            real_scope_connection, finance["collection_id"], INSIDE, True)
        assert db.set_collection_document(
            real_scope_connection, research["collection_id"], OUTSIDE, True)
        assert db.replace_document_tags(
            real_scope_connection, INSIDE, ["Urgent", "Finance", "urgent"]
        )["tags"] == ["Finance", "Urgent"]
        db.replace_document_tags(real_scope_connection, OUTSIDE, ["Finance"])
        assert [(row["name"], row["document_count"])
                for row in db.list_tags(real_scope_connection)] == [
                    ("Finance", 2), ("Urgent", 1)]

        assert db.resolve_document_scope(
            real_scope_connection,
            collection_ids=(finance["collection_id"],
                            research["collection_id"]),
            tags=("FINANCE", "urgent")) == (str(INSIDE),)
        assert db.resolve_document_scope(
            real_scope_connection, document_ids=(OUTSIDE,),
            collection_ids=(finance["collection_id"],
                            research["collection_id"]),
            tags=("finance",)) == (str(OUTSIDE),)

        inventory = db.list_documents(
            real_scope_connection, limit=10, offset=0,
            collection_id=finance["collection_id"], tag="FINANCE")
        assert [row["document_id"] for row in inventory] == [str(INSIDE)]

        db.set_document_archived(real_scope_connection, INSIDE, True)
        assert db.resolve_document_scope(
            real_scope_connection,
            collection_ids=(finance["collection_id"],)) == ()
        assert db.list_documents(
            real_scope_connection, limit=10, offset=0,
            collection_id=finance["collection_id"]) == []
    finally:
        db.set_document_archived(real_scope_connection, INSIDE, False)
        db.replace_document_tags(real_scope_connection, INSIDE, [])
        db.replace_document_tags(real_scope_connection, OUTSIDE, [])
        for tag in db.list_tags(real_scope_connection):
            db.delete_tag(real_scope_connection, tag["tag_id"])
        db.delete_collection(real_scope_connection, finance["collection_id"])
        db.delete_collection(real_scope_connection, research["collection_id"])

    assert db.get_document(real_scope_connection, INSIDE) is not None
