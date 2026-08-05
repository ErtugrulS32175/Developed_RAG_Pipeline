"""One failed statement must cost one request, never the process.

The API used to cache a single module-level connection with no rollback and no
reconnect: any database error left it in a failed transaction and every later
request died until a restart. These tests lock the replacement contract at OUR
seam -- a connection is borrowed per request, returned on both exit paths, and
a failure in one request cannot leak state into the next. The pool's own
commit/rollback behaviour belongs to psycopg_pool; what is ours is that every
caller goes through it and holds nothing across requests.
"""
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api


class FakePool:
    """Counts checkouts and returns, and hands out a fresh connection each
    time -- the properties a caching bug would violate."""

    def __init__(self):
        self.handed_out = []
        self.returned = 0

    @contextmanager
    def connection(self):
        conn = object()
        self.handed_out.append(conn)
        try:
            yield conn
        finally:
            self.returned += 1


@pytest.fixture
def pooled(monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(api.db, "get_pool", lambda: pool)
    monkeypatch.setattr(api.db, "init_schema", lambda conn: None)
    monkeypatch.setattr(api, "_schema_ready", False)
    return pool


def _headers():
    return {"Authorization": f"Bearer {api.API_KEY}"} if api.API_KEY else {}


def test_a_failed_request_does_not_poison_the_next(pooled, monkeypatch):
    calls = []

    def get_document(_conn, document_id):
        calls.append(document_id)
        if len(calls) == 1:
            raise RuntimeError("KURGU_VERITABANI_HATASI")
        return {"id": document_id, "filename": "kurgu.pdf",
                "file_type": "pdf", "status": "done"}

    monkeypatch.setattr(api.db, "get_document", get_document)
    client = TestClient(api.app, raise_server_exceptions=False)

    first = client.get("/documents/kurgu-belge-kimligi", headers=_headers())
    second = client.get("/documents/kurgu-belge-kimligi", headers=_headers())

    assert first.status_code == 500
    assert second.status_code == 200
    assert second.json()["status"] == "done"
    # the failing request's connection went back to the pool, and the second
    # request got its own -- nothing was cached across the failure
    assert len(pooled.handed_out) == 2
    assert pooled.handed_out[0] is not pooled.handed_out[1]
    assert pooled.returned == 2


def test_every_request_borrows_and_returns_its_own_connection(pooled, monkeypatch):
    monkeypatch.setattr(
        api.db, "get_document",
        lambda _conn, document_id: {"id": document_id, "filename": "kurgu.pdf",
                                    "file_type": "pdf", "status": "pending"})
    client = TestClient(api.app)

    for _ in range(3):
        assert client.get("/documents/kurgu-id", headers=_headers()).status_code == 200

    assert len(pooled.handed_out) == 3
    assert len(set(map(id, pooled.handed_out))) == 3
    assert pooled.returned == 3


def test_schema_init_runs_once_not_per_request(pooled, monkeypatch):
    ran = []
    monkeypatch.setattr(api.db, "init_schema", lambda conn: ran.append(1))
    monkeypatch.setattr(
        api.db, "get_document",
        lambda _conn, document_id: {"id": document_id, "filename": "kurgu.pdf",
                                    "file_type": "pdf", "status": "done"})
    client = TestClient(api.app)

    client.get("/documents/kurgu-id", headers=_headers())
    client.get("/documents/kurgu-id", headers=_headers())

    assert ran == [1]


def test_oversized_upload_is_refused_before_disk_and_database(
        pooled, monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "UPLOAD_MAX_BYTES", 10)
    upserts = []
    monkeypatch.setattr(api.db, "upsert_document",
                        lambda *a, **k: upserts.append(a) or "kurgu-id")

    response = TestClient(api.app).post(
        "/documents/upload", headers=_headers(),
        files={"file": ("kurgu.pdf", b"X" * 11, "application/pdf")})

    assert response.status_code == 413
    assert list(upload_dir.iterdir()) == []
    assert upserts == []
    # the cap refused the body without ever needing a connection
    assert pooled.handed_out == []


def test_upload_under_the_cap_still_works(pooled, monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "UPLOAD_MAX_BYTES", 10)
    monkeypatch.setattr(api.db, "upsert_document", lambda *a, **k: "kurgu-id")

    response = TestClient(api.app).post(
        "/documents/upload", headers=_headers(),
        files={"file": ("kurgu.pdf", b"X" * 10, "application/pdf")})

    assert response.status_code == 200
    assert (upload_dir / "kurgu.pdf").read_bytes() == b"X" * 10
    assert pooled.returned == 1


def test_lifespan_closes_the_pool_and_clears_the_global(monkeypatch):
    """Controlled shutdown must not depend on the OS reclaiming sockets."""
    from pipeline.index import db

    class ClosablePool:
        closed = False

        def close(self):
            self.closed = True

    fake = ClosablePool()
    monkeypatch.setattr(db, "_pool", fake)
    with TestClient(api.app):
        pass
    assert fake.closed
    assert db._pool is None


def test_retrieve_borrows_from_the_pool_per_query(monkeypatch):
    from pipeline.index import db
    from pipeline.retrieval import query

    pool = FakePool()
    seen = []
    monkeypatch.setattr(db, "get_pool", lambda: pool)
    monkeypatch.setattr(query, "embed_dense", lambda q: [0.0])
    monkeypatch.setattr(query, "embed_sparse", lambda q: ([1], [1.0]))
    monkeypatch.setattr(
        db, "hybrid_search",
        lambda conn, *a, **k: seen.append(conn) or [])

    query.retrieve("kurgu soru")
    query.retrieve("kurgu soru")

    assert seen == pool.handed_out
    assert len(set(map(id, seen))) == 2
    assert pooled_returns_match(pool)


def pooled_returns_match(pool):
    return pool.returned == len(pool.handed_out)
