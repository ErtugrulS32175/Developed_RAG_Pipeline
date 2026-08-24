"""Package-1 regression tests: the publication service fails CLOSED.

Three ways it did not, each found by probing the implementation rather
than reading it, and each pinned here so it cannot come back:

  * a REFUSED finalisation returned the success triple, so the caller
    believed a publication had happened that the database declined;
  * the central filename check lost protections the API door already
    had -- device names, trailing dot, trailing space -- and the CLI is
    about to start coming through this same door;
  * the lock release never read ``pg_advisory_unlock``'s answer, so a
    pooled session could keep a lock the code believed it had released.

Every invisible character in this file is written as an escape, never
literally: fixtures that carry the very characters a scanner hunts have
tripped that scanner more than once.
"""
import hashlib

import pytest

from pipeline.index import db, publication
from pipeline.index.attempt_contract import (
    CandidateState,
    CandidateSuperseded,
)
from pipeline.index.publication import UnsafeCanonicalName

BODY = b"KURGU_YAYIN_GOVDESI"
SHA = hashlib.sha256(BODY).hexdigest()
DOCUMENT_ID = "10000000-0000-4000-8000-000000000001"
VERSION_ID = "20000000-0000-4000-8000-000000000002"


class _Row:
    """The bit of document state these tests observe."""

    def __init__(self, canonical="kurgu.pdf"):
        self.canonical = canonical
        self.state = None
        self.finalize_calls = 0


@pytest.fixture
def wired(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir)
    row = _Row()

    from contextlib import contextmanager

    @contextmanager
    def lock(_conn, _filename):
        yield

    def stage(_conn, filename, file_type, content_sha256=None,
              allow_replace=False):
        row.state = CandidateState.STAGED
        return DOCUMENT_ID, VERSION_ID, row.canonical

    monkeypatch.setattr(db, "document_publish_lock", lock)
    monkeypatch.setattr(db, "stage_candidate", stage)
    return row, upload_dir


def test_a_refused_finalisation_is_not_a_successful_publication(
        wired, monkeypatch):
    """The disk moved and the database declined to follow. Returning the
    success triple there tells the caller a publication happened that
    did not: the two halves must fail together or not at all."""
    row, upload_dir = wired

    def refusing_finalize(_conn, _document_id, _candidate_id):
        row.finalize_calls += 1
        return False                      # a newer candidate was staged

    monkeypatch.setattr(db, "finalize_candidate_publication",
                        refusing_finalize)

    with pytest.raises(CandidateSuperseded):
        publication.publish_candidate(object(), "kurgu.pdf", "pdf", BODY)

    assert row.finalize_calls == 1
    assert row.state != CandidateState.PUBLISHED


def test_an_accepted_finalisation_still_publishes(wired, monkeypatch):
    """The refusal must not become a wall: the ordinary path still ends
    in a published candidate and the bytes on disk."""
    row, upload_dir = wired

    def accepting_finalize(_conn, _document_id, _candidate_id):
        row.finalize_calls += 1
        row.state = CandidateState.PUBLISHED
        return True

    monkeypatch.setattr(db, "finalize_candidate_publication",
                        accepting_finalize)

    result = publication.publish_candidate(object(), "kurgu.pdf", "pdf", BODY)

    assert result == (DOCUMENT_ID, VERSION_ID, "kurgu.pdf")
    assert row.state == CandidateState.PUBLISHED
    assert publication.read_version_source(
        upload_dir, db.DEFAULT_TENANT_ID, DOCUMENT_ID, VERSION_ID,
        expected_sha256=SHA) == BODY
    assert not (upload_dir / "kurgu.pdf").exists()


# --- the portable filename rules, at the CENTRAL door -------------------

@pytest.mark.parametrize(
    ("hostile", "why"),
    [
        ("NUL", "aygit adi"),
        ("nul.pdf", "uzantili aygit adi"),
        ("PRN.pdf", "aygit adi"),
        ("COM1", "aygit adi"),
        ("kurgu.pdf.", "sonda nokta"),
        ("kurgu.pdf ", "sonda bosluk"),
        (" kurgu.pdf", "basta bosluk"),
        ("kurgu" + chr(0x00a0) + "belge.pdf", "NBSP"),
        ("kurgu" + chr(0x200b) + "belge.pdf", "ZWSP"),
        ("kurgu" + chr(0x202f) + "belge.pdf", "NNBSP"),
        ("...", "nokta adi"),
        ("kurgu\tbelge.pdf", "kontrol karakteri"),
    ],
)
def test_the_central_check_refuses_every_hostile_spelling(wired, monkeypatch,
                                                          hostile, why):
    """Windows resolves device names to devices and strips trailing dots
    and spaces; Unicode spaces make one name display as another. Each of
    these reached disk through the central publisher after the API's own
    door stopped being the only entrance."""
    row, upload_dir = wired
    row.canonical = hostile
    monkeypatch.setattr(db, "finalize_candidate_publication",
                        lambda *a: pytest.fail(
                            "reddedilmesi gereken ad finalize'a ulasti"))

    with pytest.raises(UnsafeCanonicalName):
        publication.publish_candidate(object(), "kurgu.pdf", "pdf", BODY)

    assert list(upload_dir.iterdir()) == [], (
        f"{why}: reddedilen ad icin dosya yazildi")


@pytest.mark.parametrize(
    "benign", ["kurgu.pdf", "kurgu belge.pdf", "kurgu-2.pdf", "KURGU.PDF"])
def test_the_central_check_accepts_ordinary_names(wired, monkeypatch, benign):
    """The rules must not become a wall either: an ordinary name --
    including one with an interior space -- still publishes."""
    row, upload_dir = wired
    row.canonical = benign
    monkeypatch.setattr(db, "finalize_candidate_publication",
                        lambda *a: True)

    publication.publish_candidate(object(), "kurgu.pdf", "pdf", BODY)

    assert publication.read_version_source(
        upload_dir, db.DEFAULT_TENANT_ID, DOCUMENT_ID, VERSION_ID,
        expected_sha256=SHA) == BODY
    assert not (upload_dir / benign).exists()


def test_new_publications_never_call_the_legacy_flat_replace(wired,
                                                             monkeypatch):
    _row, upload_dir = wired
    monkeypatch.setattr(db, "finalize_candidate_publication",
                        lambda *args: True)
    monkeypatch.setattr(
        publication.os, "replace",
        lambda *args: pytest.fail("legacy flat-file replace was called"))

    publication.publish_candidate(object(), "kurgu.pdf", "pdf", BODY)

    assert publication.verify_version_source(
        upload_dir, db.DEFAULT_TENANT_ID, DOCUMENT_ID, VERSION_ID,
        expected_sha256=SHA).size == len(BODY)


# --- the lock release reads its own answer ------------------------------

class _LockConn:
    def __init__(self, unlock_result):
        self.unlock_result = unlock_result
        self.events = []

    def cursor(self):
        conn = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                conn.events.append("unlock" if "unlock" in sql else "lock")

            def fetchone(self):
                return conn.unlock_result

        return Cursor()

    def commit(self):
        self.events.append("commit")

    def rollback(self):
        self.events.append("rollback")

    def close(self):
        self.events.append("close")


@pytest.mark.parametrize("answer", [(False,), None],
                         ids=["unlock_false", "cevap_yok"])
def test_an_unproven_unlock_closes_and_RAISES(answer):
    """``pg_advisory_unlock`` returns FALSE when this session did not hold
    the lock: a statement that ran without error and released nothing.
    Ignoring the answer left a pooled session holding the lock while the
    code believed it had let go.

    Closing the connection is only half the fix. An earlier version
    raised inside the release path and then caught its own exception one
    line below, so a successful body still returned SUCCESS while the
    lock leaked -- the caller had no way to know. Both halves now."""
    conn = _LockConn(unlock_result=answer)
    with pytest.raises(db.PublishLockNotReleased):
        with db.document_publish_lock(conn, "kurgu.pdf"):
            pass
    assert "close" in conn.events, (
        "kanitlanamayan unlock sonrasi baglanti kapatilmadi")


def test_a_failing_body_keeps_its_own_error_when_the_unlock_also_fails():
    """The lock problem must not overwrite the reason the caller actually
    needs to see: the body's failure propagates, and the connection is
    still closed."""
    conn = _LockConn(unlock_result=(False,))

    class KurguHatasi(RuntimeError):
        pass

    with pytest.raises(KurguHatasi):
        with db.document_publish_lock(conn, "kurgu.pdf"):
            raise KurguHatasi("govde hatasi")
    assert "close" in conn.events


def test_a_proven_unlock_keeps_the_connection():
    conn = _LockConn(unlock_result=(True,))
    with db.document_publish_lock(conn, "kurgu.pdf"):
        pass
    assert "close" not in conn.events
    assert conn.events.count("unlock") == 1
