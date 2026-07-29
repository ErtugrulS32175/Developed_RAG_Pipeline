"""Ingest survives a transient failure instead of throwing the run away.

Before this, one hiccup from the embedding service killed an ingest that had
already embedded hundreds of chunks, and the next attempt started from zero.
"""
import pytest

from pipeline import ingest_router as ir


def _c(text, tag="page7:native"):
    return {"type": "text", "text": text, "source_tag": tag, "page": 7, "headings": []}


# --- ids are derived from content, which is what makes resuming possible ---

def test_the_same_chunk_always_gets_the_same_id():
    a = ir._chunk_id("doc-1", _c("zeta gamma"), 3)
    b = ir._chunk_id("doc-1", _c("zeta gamma"), 3)
    assert a == b


def test_different_text_gets_a_different_id():
    a = ir._chunk_id("doc-1", _c("zeta"), 3)
    b = ir._chunk_id("doc-1", _c("gamma"), 3)
    assert a != b


def test_the_same_text_in_a_different_position_is_a_different_chunk():
    """A phrase repeated on two pages is two chunks, not one."""
    assert ir._chunk_id("doc-1", _c("zeta"), 3) != ir._chunk_id("doc-1", _c("zeta"), 4)
    assert (ir._chunk_id("doc-1", _c("zeta", "page7:native"), 3)
            != ir._chunk_id("doc-1", _c("zeta", "page8:native"), 3))


def test_two_documents_never_share_a_chunk_id():
    assert ir._chunk_id("doc-1", _c("zeta"), 3) != ir._chunk_id("doc-2", _c("zeta"), 3)


def test_the_id_is_a_valid_uuid():
    import uuid
    uuid.UUID(ir._chunk_id("doc-1", _c("zeta"), 3))


# --- retry ---

def test_a_transient_failure_is_retried(monkeypatch):
    monkeypatch.setattr(ir.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("servis mesgul")
        return "sonuc"

    assert ir._retry(flaky, attempts=4, backoff=1) == "sonuc"
    assert calls["n"] == 3


def test_a_persistent_failure_still_raises(monkeypatch):
    """Retrying must not turn a real outage into a silent hang."""
    monkeypatch.setattr(ir.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ConnectionError("servis kapali")

    with pytest.raises(ConnectionError):
        ir._retry(broken, attempts=3, backoff=1)
    assert calls["n"] == 3


def test_a_call_that_works_is_not_retried(monkeypatch):
    monkeypatch.setattr(ir.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fine():
        calls["n"] += 1
        return 42

    assert ir._retry(fine, attempts=4, backoff=1) == 42
    assert calls["n"] == 1


def test_backoff_grows_between_attempts(monkeypatch):
    """A fixed short delay would hammer a service that is still starting up."""
    waits = []
    monkeypatch.setattr(ir.time, "sleep", waits.append)

    def broken():
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        ir._retry(broken, attempts=4, backoff=2)
    assert waits == [1, 2, 4]
