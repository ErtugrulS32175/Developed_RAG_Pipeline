import uuid

import pytest

from pipeline.index import db


DOCUMENT_ID = "00000000-0000-0000-0000-0000000000d0"
VERSION_ID = "00000000-0000-0000-0000-0000000000e0"
TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


class ScriptedCursor:
    def __init__(self, one=(), all_rows=()):
        self.ones = list(one)
        self.all_rows = list(all_rows)
        self.calls = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, statement, params=None):
        self.calls.append((statement, params))

    def fetchone(self):
        return None if not self.ones else self.ones.pop(0)

    def fetchall(self):
        return [dict(row) if isinstance(row, dict) else row
                for row in self.all_rows]


class ScriptedConnection:
    def __init__(self, one=(), all_rows=()):
        self.cur = ScriptedCursor(one, all_rows)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, row_factory=None):
        return self.cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _document(**overrides):
    row = {
        "tenant_id": TENANT_ID,
        "active_version_id": None,
        "active_generation": 0,
        "revision": 4,
        "attempt_id": None,
        "archived_at": None,
    }
    row.update(overrides)
    return row


def test_current_schema_integrates_versions_builds_and_all_bindings():
    schema = db.Path(db.__file__).with_name("schema.sql").read_text("utf-8")

    assert db.SCHEMA_VERSION == 9
    assert "CREATE TABLE IF NOT EXISTS document_versions" in schema
    assert "CREATE TABLE IF NOT EXISTS document_version_builds" in schema
    assert "CREATE TABLE IF NOT EXISTS document_version_events" in schema
    for table in ("document_versions", "document_version_builds",
                  "document_version_events"):
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in schema
    assert "UNIQUE (tenant_id, document_id, version_number)" in schema
    assert "documents_active_version_fk" in schema
    assert "BEFORE UPDATE OR DELETE ON document_versions" in schema
    assert "BEFORE UPDATE OR DELETE ON document_version_builds" in schema
    assert "BEFORE UPDATE OR DELETE ON document_version_events" in schema
    assert "ALTER TABLE attempts ADD COLUMN IF NOT EXISTS version_id" in schema
    assert "ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS version_id" in schema
    assert "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS version_id" in schema


class StageCursor(ScriptedCursor):
    def execute(self, statement, params=None):
        super().execute(statement, params)
        if statement.startswith("SELECT filename"):
            self.ones.append(None)
        elif statement.startswith("INSERT INTO documents"):
            self.ones.append((DOCUMENT_ID, params["cid"], params["filename"]))


def test_stage_candidate_mints_the_version_in_its_own_transaction():
    conn = ScriptedConnection()
    conn.cur = StageCursor()

    document_id, candidate_id, name = db.stage_candidate(
        conn, "kurgu.pdf", "pdf", "a" * 64)

    assert document_id == DOCUMENT_ID and name == "kurgu.pdf"
    statements = "\n".join(sql for sql, _params in conn.cur.calls)
    version_call = next((sql, params) for sql, params in conn.cur.calls
                        if "INSERT INTO document_versions" in sql)
    assert "last_version_number + 1" in version_call[0]
    assert "%(sha)s::text IS NOT NULL" in version_call[0]
    assert version_call[1]["candidate"] == candidate_id
    assert "SELECT %(candidate)s::uuid" in version_call[0]
    assert "INSERT INTO document_version_events" not in statements
    assert conn.commits == 1 and conn.rollbacks == 0


def test_version_listing_is_parameterized_bounded_ready_and_identity_safe():
    conn = ScriptedConnection(all_rows=[{
        "id": uuid.UUID(VERSION_ID), "version_number": 8,
        "created_at": "then", "is_active": True, "index_ready": True,
        "revision": 11,
    }])
    rows = db.list_document_versions(
        conn, DOCUMENT_ID, limit=2, before_version_number=9)
    sql, params = conn.cur.calls[0]
    assert params == {"document": DOCUMENT_ID, "limit": 3, "before": 9}
    assert "version_number < %(before)s" in sql
    assert "ORDER BY v.version_number DESC, v.id DESC" in sql
    assert "content_sha256" not in sql
    assert "document_version_builds" in sql
    assert "tenant" not in params
    assert rows == [{"version_id": VERSION_ID, "version_number": 8,
                     "created_at": "then", "is_active": True,
                     "index_ready": True, "document_revision": 11}]


def test_source_digest_is_an_internal_parameterized_proof_input():
    conn = ScriptedConnection(one=[("a" * 64,)])
    assert db.document_version_source_digest(
        conn, DOCUMENT_ID, VERSION_ID) == "a" * 64
    sql, params = conn.cur.calls[0]
    assert "SELECT content_sha256 FROM document_versions" in sql
    assert params == (DOCUMENT_ID, VERSION_ID)


@pytest.mark.parametrize("limit,before", [(0, None), (101, None), (1, 0)])
def test_version_listing_rejects_bad_page_values_without_sql(limit, before):
    conn = ScriptedConnection()
    with pytest.raises(ValueError):
        db.list_document_versions(
            conn, DOCUMENT_ID, limit=limit,
            before_version_number=before)
    assert conn.cur.calls == []


def test_activation_cas_switches_every_retrieval_authority_to_a_ready_build():
    conn = ScriptedConnection(one=[
        _document(), {"has_active_job": False},
        {"id": uuid.UUID(VERSION_ID), "content_sha256": "b" * 64,
         "generation": 7, "chunk_count": 3}, {"revision": 5},
    ])
    result = db.activate_document_version(
        conn, DOCUMENT_ID, VERSION_ID, 4,
        verified_source_sha256="b" * 64)
    assert result == {"document_id": DOCUMENT_ID,
                      "active_version_id": VERSION_ID,
                      "active_generation": 7,
                      "revision": 5, "changed": True}
    update, params = next((sql, params) for sql, params in conn.cur.calls
                          if sql.startswith("UPDATE documents SET"))
    assert "active_version_id = %(version)s" in update
    assert "active_generation = %(generation)s" in update
    assert "active_content_sha = %(sha)s" in update
    assert "status = 'done'" in update and "status_note = NULL" in update
    assert "revision = %(expected)s" in update
    assert params["generation"] == 7 and params["sha"] == "b" * 64
    assert not any("INSERT INTO document_version_events" in sql
                   for sql, _params in conn.cur.calls)
    assert conn.commits == 1 and conn.rollbacks == 0


def test_activation_refuses_stale_busy_archived_or_unready_states():
    stale = ScriptedConnection(one=[_document(revision=6)])
    with pytest.raises(db.DocumentVersionConflict, match="revizyon"):
        db.activate_document_version(
            stale, DOCUMENT_ID, VERSION_ID, 4,
            verified_source_sha256="b" * 64)
    assert stale.rollbacks == 1

    busy = ScriptedConnection(one=[_document(), {"has_active_job": True}])
    with pytest.raises(db.DocumentLifecycleConflict, match="job"):
        db.activate_document_version(
            busy, DOCUMENT_ID, VERSION_ID, 4,
            verified_source_sha256="b" * 64)
    assert busy.rollbacks == 1

    archived = ScriptedConnection(one=[_document(archived_at="then")])
    with pytest.raises(db.DocumentLifecycleConflict, match="arsivlenmis"):
        db.activate_document_version(
            archived, DOCUMENT_ID, VERSION_ID, 4,
            verified_source_sha256="b" * 64)

    unready = ScriptedConnection(one=[
        _document(), {"has_active_job": False}, None,
    ])
    assert db.activate_document_version(
        unready, DOCUMENT_ID, VERSION_ID, 4,
        verified_source_sha256="b" * 64) is None
    assert unready.commits == 0 and unready.rollbacks == 1

    wrong_source = ScriptedConnection(one=[
        _document(), {"has_active_job": False},
        {"id": uuid.UUID(VERSION_ID), "content_sha256": "b" * 64,
         "generation": 7, "chunk_count": 3},
    ])
    with pytest.raises(db.DocumentVersionConflict, match="kaynak kaniti"):
        db.activate_document_version(
            wrong_source, DOCUMENT_ID, VERSION_ID, 4,
            verified_source_sha256="c" * 64)
    assert wrong_source.commits == 0 and wrong_source.rollbacks == 1


class PromotionCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []
        self.one = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, statement, params=None):
        self.conn.calls.append((statement, params))
        if statement.startswith("SELECT id FROM chunks"):
            self.rows = [("chunk-1",)]
        elif statement.startswith("SELECT status FROM attempts"):
            self.one = (None,)
        elif statement.startswith("UPDATE documents"):
            self.one = (DOCUMENT_ID,)
        elif statement.startswith("UPDATE attempts"):
            self.one = ("attempt",)
        elif statement.startswith("DELETE FROM chunks"):
            self.rowcount = 0

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.one


class PromotionConnection:
    def __init__(self):
        self.calls = []
        self.commits = 0

    def cursor(self):
        return PromotionCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def test_promotion_binds_version_and_retains_every_ready_generation():
    conn = PromotionConnection()
    db.promote_generation(
        conn, DOCUMENT_ID, 7, 0, {"chunk-1"}, "a" * 64,
        VERSION_ID, "00000000-0000-0000-0000-0000000000f0")
    manifest_sql = conn.calls[0][0]
    attempt_lock_sql = conn.calls[1][0]
    update_sql = conn.calls[2][0]
    delete_sql = conn.calls[-1][0]
    assert "version_id = %s::uuid" in manifest_sql
    assert attempt_lock_sql.endswith("FOR UPDATE")
    assert "active_version_id = %(cid)s::uuid" in update_sql
    assert "revision = revision + 1" in update_sql
    assert "document_version_builds" in delete_sql
    assert "version_id IS DISTINCT FROM" in delete_sql
    assert conn.commits == 1


def test_native_dense_and_sparse_queries_share_exact_version_generation_gate():
    conn = ScriptedConnection()
    db.hybrid_search(conn, [0.0], [], [], top_k=1)
    assert len(conn.cur.calls) == 3
    lock_sql, lock_params = conn.cur.calls[0]
    assert lock_sql.endswith("ORDER BY id FOR SHARE")
    assert lock_params == ()
    statements = [sql for sql, _params in conn.cur.calls[-2:]]
    gate = "d.active_version_id IS NULL AND c.version_id IS NULL"
    for sql in statements:
        assert "c.generation = d.active_generation" in sql
        assert gate in sql
        assert "c.version_id = d.active_version_id" in sql


def test_snapshot_scope_key_names_the_exact_version_and_generation():
    scope_key = (str(TENANT_ID) + ":" + DOCUMENT_ID + ":" + VERSION_ID
                 + ":7")
    conn = ScriptedConnection(all_rows=[(scope_key,)])
    assert db.lock_retrieval_scope_keys(conn, (DOCUMENT_ID,)) == [scope_key]
    sql = conn.cur.calls[0][0]
    assert "COALESCE(active_version_id::text, 'legacy')" in sql
    assert "active_generation::text" in sql


def test_public_jobs_expose_the_bound_version_without_a_second_identity():
    row = {
        "id": uuid.uuid4(), "document_id": uuid.uuid4(),
        "candidate_id": uuid.UUID(VERSION_ID),
        "version_id": uuid.UUID(VERSION_ID), "status": "queued",
        "attempt_count": 0, "created_at": "then", "started_at": None,
        "finished_at": None, "outcome_note": None,
    }
    public = db._public_job(row)
    assert public["version_id"] == public["candidate_id"] == VERSION_ID
