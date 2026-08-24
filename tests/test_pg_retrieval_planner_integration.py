"""Real PostgreSQL proof for actor-bound retrieval planning authority.

The ordinary suite skips this module when no disposable server is supplied.
CI makes the check mandatory with the existing scope gate; a missing DSN must
never look like a passing PostgreSQL integration test.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import uuid

import pytest

from pipeline.index import db
from pipeline.retrieval import query


DSN = os.getenv("RAGTEST_SCOPE_PG_DSN", "").strip()
GATE = os.getenv("RAGTEST_SCOPE_GATE", "").strip() == "1"

if GATE and not DSN:
    raise RuntimeError(
        "RAGTEST_SCOPE_GATE=1 but RAGTEST_SCOPE_PG_DSN is missing")

pytestmark = pytest.mark.skipif(
    not DSN,
    reason="RAGTEST_SCOPE_PG_DSN is absent: planner RLS was not checked",
)

TENANT_A = uuid.UUID("a6000000-0000-4000-8000-000000000001")
TENANT_B = uuid.UUID("b6000000-0000-4000-8000-000000000001")
ACTOR_A = uuid.UUID("a6000000-0000-4000-8000-000000000010")
ACTOR_B = uuid.UUID("b6000000-0000-4000-8000-000000000010")
UNKNOWN = uuid.UUID("c6000000-0000-4000-8000-000000000099")


class _ConnectionLease:
    """Let query.retrieve borrow the fixture connection without closing it."""

    def __init__(self, connection):
        self._connection = connection

    def __enter__(self):
        return self._connection

    def __exit__(self, _kind, _error, _traceback):
        return False


class _OneConnectionPool:
    def __init__(self, connection):
        self._connection = connection

    def connection(self):
        return _ConnectionLease(self._connection)


def _publish_version(connection, filename, sha, text, *, replace=False):
    """Build one genuine retained version through the public DB seams."""
    from pgvector import SparseVector, Vector

    document_id, version_id, _stored = db.stage_candidate(
        connection, filename, "pdf", content_sha256=sha,
        allow_replace=replace)
    assert db.finalize_candidate_publication(
        connection, document_id, version_id)
    attempt = db.begin_attempt(connection, document_id, owner="planner-test")
    generation = db.allocate_generation(connection, document_id, attempt)
    chunk_id = uuid.uuid4()
    db.upsert_chunks(connection, [{
        "id": chunk_id,
        "document_id": document_id,
        "type": "text",
        "text": text,
        "source_tag": text + ":1",
        "page": 1,
        "headings": [],
        "table_data": None,
        "dense": Vector([1.0] + [0.0] * 1023),
        "sparse": SparseVector({0: 1.0}, db.SPARSE_DIM),
        "generation": generation,
        "content_key": str(uuid.uuid4()),
        "embedding_fingerprint": "planner-test-v1",
    }], attempt)
    assert db.promote_generation(
        connection, document_id, generation,
        expected_active=attempt.observed_active,
        manifest_ids={chunk_id}, content_sha256=sha,
        candidate_id=version_id, attempt_id=attempt.attempt_id) >= 0
    return {
        "document_id": document_id,
        "version_id": version_id,
        "generation": generation,
        "chunk_id": str(chunk_id),
        "text": text,
    }


@pytest.fixture(scope="module")
def planner_database():
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql

    schema = "ragtest_planner_" + uuid.uuid4().hex[:12]
    role = "ragtest_planner_role_" + uuid.uuid4().hex[:10]
    password = "planner-integration-only"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public")
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)))
            cursor.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(schema), sql.Identifier(role)))
        connection = psycopg.connect(DSN, user=role, password=password)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema)))
            cursor.execute(Path(db.__file__).with_name("schema.sql").read_text(
                encoding="utf-8"))
            cursor.execute(
                "INSERT INTO rag_context_secrets (singleton, secret) "
                "VALUES (true, %s)", (db._context_secret(),))
        connection.commit()
        register_vector(connection)

        db.set_tenant_context(connection, TENANT_A, actor_id=ACTOR_A)
        old_a = _publish_version(
            connection, "tenant-a.pdf", "1" * 64, "tenant-a-old")
        current_a = _publish_version(
            connection, "tenant-a.pdf", "2" * 64, "tenant-a-current",
            replace=True)

        db.set_tenant_context(connection, TENANT_B, actor_id=ACTOR_B)
        current_b = _publish_version(
            connection, "tenant-b.pdf", "3" * 64, "tenant-b-current")

        yield {
            "conn": connection,
            "admin_dsn": DSN,
            "schema": schema,
            "role": role,
            "password": password,
            "old_a": old_a,
            "current_a": current_a,
            "current_b": current_b,
        }
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema)))
                cursor.execute(sql.SQL("DROP ROLE {}").format(
                    sql.Identifier(role)))
        finally:
            admin.close()


@contextmanager
def _actor_binding(tenant, actor):
    token = db.bind_execution_tenant(tenant, actor_id=actor)
    try:
        yield
    finally:
        db.reset_execution_tenant(token)


def _wire_real_retrieval(monkeypatch, world):
    monkeypatch.setattr(db, "get_pool", lambda: _OneConnectionPool(
        world["conn"]))
    monkeypatch.setattr(
        query, "embed_dense", lambda _question: [1.0] + [0.0] * 1023)
    monkeypatch.setattr(
        query, "embed_sparse", lambda _question: ([0], [1.0]))


def _assert_actor_context(connection, tenant, actor):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT rag_effective_tenant(), rag_effective_actor(), "
            "rag_service_access()")
        assert cursor.fetchone() == (tenant, actor, False)


def test_scope_resolution_and_hybrid_search_share_one_actor_transaction(
        monkeypatch, planner_database):
    import psycopg

    world = planner_database
    connection = world["conn"]
    _wire_real_retrieval(monkeypatch, world)
    real_resolve = db.resolve_document_scope
    real_search = db.hybrid_search
    seen = []

    def resolve(bound, **scope):
        _assert_actor_context(bound, TENANT_A, ACTOR_A)
        seen.append(("resolve", id(bound), bound.info.transaction_status))
        return real_resolve(bound, **scope)

    def search(bound, *args, **kwargs):
        _assert_actor_context(bound, TENANT_A, ACTOR_A)
        seen.append(("search", id(bound), bound.info.transaction_status,
                     kwargs["document_ids"]))
        return real_search(bound, *args, **kwargs)

    monkeypatch.setattr(db, "resolve_document_scope", resolve)
    monkeypatch.setattr(db, "hybrid_search", search)
    with _actor_binding(TENANT_A, ACTOR_A):
        rows = query.retrieve(
            "bounded question",
            document_ids=(world["current_a"]["document_id"],))

    assert [row["text"] for row in rows] == ["tenant-a-current"]
    assert [item[0] for item in seen] == ["resolve", "search"]
    assert {item[1] for item in seen} == {id(connection)}
    assert all(item[2] != psycopg.pq.TransactionStatus.IDLE for item in seen)
    assert seen[1][3] == (world["current_a"]["document_id"],)


@pytest.mark.parametrize("offered", ["unknown", "foreign"])
def test_an_explicit_scope_resolving_empty_never_becomes_whole_corpus(
        monkeypatch, planner_database, offered):
    world = planner_database
    _wire_real_retrieval(monkeypatch, world)
    offered_id = (
        str(UNKNOWN) if offered == "unknown"
        else world["current_b"]["document_id"]
    )
    monkeypatch.setattr(
        db, "hybrid_search",
        lambda *_args, **_kwargs: pytest.fail(
            "an empty resolved scope reached hybrid search"),
    )
    with _actor_binding(TENANT_A, ACTOR_A):
        rows = query.retrieve("bounded question", document_ids=(offered_id,))

    assert rows == []


def test_a_mixed_scope_can_return_visible_rows_but_never_tenant_b(
        monkeypatch, planner_database):
    world = planner_database
    _wire_real_retrieval(monkeypatch, world)
    with _actor_binding(TENANT_A, ACTOR_A):
        rows = query.retrieve(
            "bounded question",
            document_ids=(world["current_a"]["document_id"],
                          world["current_b"]["document_id"]),
        )

    assert [row["text"] for row in rows] == ["tenant-a-current"]
    assert all(str(row["document_id"])
               != world["current_b"]["document_id"] for row in rows)


def test_unscoped_actor_retrieval_is_still_tenant_closed(
        monkeypatch, planner_database):
    world = planner_database
    _wire_real_retrieval(monkeypatch, world)
    with _actor_binding(TENANT_A, ACTOR_A):
        rows = query.retrieve("bounded question")

    assert [row["text"] for row in rows] == ["tenant-a-current"]
    assert all(str(row["document_id"])
               != world["current_b"]["document_id"] for row in rows)


def test_current_version_generation_is_locked_through_the_real_search(
        monkeypatch, planner_database):
    import psycopg
    from psycopg import sql

    world = planner_database
    _wire_real_retrieval(monkeypatch, world)
    real_search = db.hybrid_search
    blocked = []

    def search(connection, *args, **kwargs):
        rows = real_search(connection, *args, **kwargs)
        contender = psycopg.connect(
            world["admin_dsn"], user=world["role"],
            password=world["password"])
        try:
            with contender.cursor() as cursor:
                cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(world["schema"])))
                cursor.execute("SET lock_timeout = '150ms'")
            contender.commit()
            db.set_tenant_context(contender, TENANT_A, actor_id=ACTOR_A)
            with pytest.raises(psycopg.errors.LockNotAvailable):
                db.set_document_archived(
                    contender, world["current_a"]["document_id"], True)
            contender.rollback()
            blocked.append(True)
        finally:
            contender.close()
        return rows

    monkeypatch.setattr(db, "hybrid_search", search)
    with _actor_binding(TENANT_A, ACTOR_A):
        rows = query.retrieve(
            "bounded question",
            document_ids=(world["current_a"]["document_id"],))

    assert blocked == [True]
    assert [row["text"] for row in rows] == ["tenant-a-current"]
    assert world["old_a"]["text"] not in {row["text"] for row in rows}
    assert {str(row["version_id"]) for row in rows} == {
        world["current_a"]["version_id"]}
