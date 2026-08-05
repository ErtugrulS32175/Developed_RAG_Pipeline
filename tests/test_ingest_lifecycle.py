"""The ingest connection closes exactly once on EVERY exit path.

The original code closed it on two paths and leaked it on the rest: an error
in schema init, in the metadata reads, or in the finalisation left
closed=False. These tests lock the whole-lifetime try/finally by injecting a
failure into each previously-leaking stage and counting close() calls.
"""
import pytest

from pipeline.index import ingest


class FakeConn:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1

    def cursor(self):
        conn = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, *a, **k):
                return None

            def fetchone(self):
                return (0,)

        return Cursor()


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A document whose parse yields nothing, so no embedding service is hit."""
    conn = FakeConn()
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    monkeypatch.setattr(ingest, "route_and_parse", lambda path: [])
    monkeypatch.setattr(ingest.db, "get_conn", lambda: conn)
    monkeypatch.setattr(ingest.db, "init_schema", lambda c: None)
    monkeypatch.setattr(ingest.db, "upsert_document", lambda *a, **k: "kurgu-id")
    monkeypatch.setattr(ingest.db, "existing_chunk_ids", lambda *a: set())
    monkeypatch.setattr(ingest.db, "delete_stale_chunks", lambda *a: 0)
    monkeypatch.setattr(ingest.db, "set_document_status", lambda *a, **k: None)
    return conn, source


def test_schema_init_failure_still_closes_the_connection(wired, monkeypatch):
    conn, source = wired
    monkeypatch.setattr(ingest.db, "init_schema",
                        lambda c: (_ for _ in ()).throw(RuntimeError("KURGU")))
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert conn.close_calls == 1


def test_finalisation_failure_still_closes_the_connection(wired, monkeypatch):
    conn, source = wired
    monkeypatch.setattr(ingest.db, "delete_stale_chunks",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("KURGU")))
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert conn.close_calls == 1


def test_the_clean_path_closes_the_connection_once(wired):
    conn, source = wired
    ingest.main(str(source))
    assert conn.close_calls == 1
