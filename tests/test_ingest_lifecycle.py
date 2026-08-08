"""The ingest connection closes exactly once on EVERY exit path.

The original code closed it on two paths and leaked it on the rest: an error
in schema init, in the metadata reads, or in the finalisation left
closed=False. These tests lock the whole-lifetime try/finally by injecting a
failure into each previously-leaking stage and counting close() calls.
"""
import hashlib

import pytest

from pipeline.index import ingest
from pipeline.index.attempt_contract import IngestAttempt


class FakeConn:
    """Counts BOTH ends of a connection's life.

    Package 3B split the run into a short binding step (resolve the
    document, take the attempt) and the long indexing step, so one run
    now borrows more than one connection. The invariant these tests
    exist for is unchanged and is stated exactly: every connection that
    is opened is closed. Counting closes alone would have quietly become
    a claim about how many connections a run happens to use."""

    def __init__(self):
        self.close_calls = 0
        self.open_calls = 0

    def close(self):
        self.close_calls += 1

    def opened(self):
        self.open_calls += 1
        return self

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
    """A minimal one-part parse with faked embeddings and db writes."""
    conn = FakeConn()
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    monkeypatch.setattr(
        ingest, "route_and_parse",
        lambda path: ([("page1:scanned", ("text", "Kurgu icerik parcasi."))],
                      []))
    monkeypatch.setattr(ingest, "embed_sparse", lambda text: ([1], [0.5]))
    monkeypatch.setattr(ingest, "embed_dense", lambda text: [0.0])
    monkeypatch.setattr(ingest.db, "get_conn",
                        lambda: conn.opened())
    monkeypatch.setattr(ingest.db, "init_schema", lambda c: None)
    monkeypatch.setattr(
        ingest.db, "upsert_document",
        lambda *a, **k: ("kurgu-id", "kurgu-aday-kimligi", "kurgu.pdf"))
    # Package 3C: the core asks "am I still the current candidate?" once,
    # between the parse and the first write, so the row must carry the
    # candidate this fixture's attempt was minted with.
    monkeypatch.setattr(ingest.db, "get_document",
                        lambda *a: {"active_generation": 0,
                                    "candidate_id": "kurgu-aday"})
    monkeypatch.setattr(ingest.db, "allocate_generation", lambda *a: 1)
    monkeypatch.setattr(ingest.db, "existing_content_keys", lambda *a: set())
    monkeypatch.setattr(ingest.db, "upsert_chunks", lambda *a: None)
    monkeypatch.setattr(ingest.db, "copy_chunks_into_generation",
                        lambda *a, **k: 0)
    monkeypatch.setattr(ingest.db, "promote_generation",
                        lambda *a, **k: 0)
    # Package 3B: the core runs under an attempt, so the fixture opens
    # one. These tests are about the CONNECTION's lifetime, so the
    # attempt seams are the thinnest thing that keeps the run moving.
    monkeypatch.setattr(
        ingest.db, "begin_attempt",
        lambda _conn, document_id, owner=None: IngestAttempt(
            attempt_id="kurgu-deneme-1", document_id=document_id,
            candidate_id="kurgu-aday", candidate_sha=hashlib.sha256(
                b"KURGU_PDF").hexdigest(), observed_active=0),
        raising=False)
    monkeypatch.setattr(ingest.db, "record_attempt_outcome",
                        lambda *a, **k: True, raising=False)
    monkeypatch.setattr(ingest.db, "heartbeat_attempt",
                        lambda *a, **k: True, raising=False)
    monkeypatch.setattr(
        ingest.db, "lookup_document",
        lambda *a: {"id": "kurgu-id", "filename": "kurgu.pdf",
                    "candidate_id": "kurgu-aday"})
    monkeypatch.setattr(ingest.db, "set_document_status", lambda *a, **k: None)
    return conn, source


def test_schema_init_failure_still_closes_the_connection(wired, monkeypatch):
    conn, source = wired
    monkeypatch.setattr(ingest.db, "init_schema",
                        lambda c: (_ for _ in ()).throw(RuntimeError("KURGU")))
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert conn.close_calls == conn.open_calls > 0


def test_finalisation_failure_still_closes_the_connection(wired, monkeypatch):
    conn, source = wired
    monkeypatch.setattr(
        ingest.db, "promote_generation",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("KURGU")))
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert conn.close_calls == conn.open_calls > 0


def test_the_clean_path_closes_every_connection_it_opens(wired):
    conn, source = wired
    ingest.main(str(source))
    assert conn.close_calls == conn.open_calls > 0


def test_an_empty_parse_refuses_and_still_closes_once(wired, monkeypatch):
    """Round 15: a parse yielding nothing with no recorded failure used to
    delete the document's healthy rows and stamp it done. It now refuses --
    and the refusal path must close the connection exactly once too."""
    conn, source = wired
    monkeypatch.setattr(ingest, "route_and_parse", lambda path: ([], []))
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert conn.close_calls == conn.open_calls > 0
