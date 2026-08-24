"""Three defects that shared one shape: the system knowing less than it said.

A truncated embedding said nothing, a half-parsed document said "done", and
two different files wearing one basename said they were the same document.
Each fix is locked here: the loud cap, the "partial" status, the hash-guarded
upsert and the upload path that announces replacement.
"""
import hashlib

import pytest

from pipeline.index import db, embeddings, ingest
from pipeline.index.attempt_contract import (
    AttemptAlreadyRunning,
    AttemptOutcome,
    CandidateNotPublished,
    ExitCode,
    IngestAttempt,
)


# --- C7: the dense-embedding cap is higher, configurable and LOUD -----------

def _capture_embed(monkeypatch):
    sent = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [0.0]}]}

    def fake_post(url, json=None, timeout=None):
        sent["input"] = json["input"]
        return FakeResponse()

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    return sent


def test_text_over_the_old_silent_cap_now_embeds_whole(monkeypatch):
    """The old cap was text[:2000], silent: a 3000-char table chunk had its
    lower rows invisible to dense retrieval while BM25 saw all of them."""
    sent = _capture_embed(monkeypatch)
    text = "kurgu " * 500  # 3000 chars: over the old cap, under the new
    embeddings.embed_dense(text)
    assert sent["input"] == text


def test_truncation_still_guards_the_window_but_says_so(monkeypatch, capsys):
    sent = _capture_embed(monkeypatch)
    text = "k" * (embeddings.EMBED_MAX_CHARS + 500)
    embeddings.embed_dense(text)
    assert len(sent["input"]) == embeddings.EMBED_MAX_CHARS
    assert "kirpildi" in capsys.readouterr().out


def test_the_cap_is_an_env_knob_not_a_constant(monkeypatch):
    sent = _capture_embed(monkeypatch)
    monkeypatch.setattr(embeddings, "EMBED_MAX_CHARS", 100)
    embeddings.embed_dense("k" * 300)
    assert len(sent["input"]) == 100


def test_a_broken_cap_falls_back_instead_of_zeroing(capsys):
    """Round 14: EMBED_MAX_CHARS=0 truncated every text to EMPTY and sent
    blank inputs to the service -- a broken knob silently disabling the
    feature it tunes. Anything not a positive integer falls back loudly."""
    assert embeddings._resolve_embed_cap("0") == 8000
    assert embeddings._resolve_embed_cap("-5") == 8000
    assert embeddings._resolve_embed_cap("bozuk") == 8000
    assert "gecersiz" in capsys.readouterr().out
    assert embeddings._resolve_embed_cap("500") == 500


# --- C9: the filename key is guarded by the content hash --------------------

class _FakeCursor:
    """Mirrors the upsert contract: the refusal decision is made by the
    database inside the guarded statement's own WHERE clause, so the fake
    evaluates the same arms -- allow flag, no hash, served hash, recorded
    candidate, fresh row -- and answers None exactly where PostgreSQL
    would return no row. The row lookup that feeds ``fresh`` and the
    canonical spelling happens UNDER the advisory lock, which the fake
    verifies by statement order."""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.statements.append((sql, params))
        existing = self.conn.existing
        if sql.startswith("SELECT pg_advisory"):
            self._result = (True,)
        elif sql.startswith("SELECT filename"):
            self._result = (
                (existing["filename"], existing["has_chunks"])
                if existing else None)
        elif sql.startswith("INSERT INTO documents"):
            offered = params["sha"]
            active = existing.get("active_content_sha") if existing else None
            candidate = existing.get("content_sha256") if existing else None
            ok = (
                params["allow"]
                or offered is None
                or (active is not None and active == offered)
                or (candidate is not None and candidate == offered)
                or (active is None and params["fresh"])
            )
            if ok:
                # candidate_id follows the statement's CASE: kept when the
                # hash is unchanged (and one exists), minted otherwise
                prior_cid = (existing.get("candidate_id")
                             if existing else None)
                if offered is None or (
                        existing is not None
                        and candidate == offered and prior_cid):
                    cid = prior_cid
                else:
                    cid = params["cid"]
                self._result = ("kurgu-belge-id", cid, params["filename"])
            else:
                self._result = None
        else:
            self._result = (0,)  # e.g. the final chunk-count query

    def fetchone(self):
        return self._result


class _FakeConn:
    def __init__(self, existing=None):
        self.existing = existing
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _served(filename="kurgu.pdf", active="a" * 64, candidate=None,
            has_chunks=True):
    """A pre-existing document row as the gate sees it: what is SERVED
    (active), what last knocked (candidate), whether any rows exist."""
    return {"filename": filename, "active_content_sha": active,
            "content_sha256": candidate if candidate is not None else active,
            "has_chunks": has_chunks}


def test_same_name_different_content_is_refused():
    """The collision that used to lose data: a second file with the same
    basename merged into the first document's row, then delete_stale_chunks
    removed the first file's chunks. The refusal now lives INSIDE the
    statement (round 14: check-then-write raced concurrent ingests), so a
    refused write surfaces as no returned row, a rollback and no commit."""
    conn = _FakeConn(existing=_served())
    with pytest.raises(ValueError):
        db.upsert_document(conn, "kurgu.pdf", "pdf", content_sha256="b" * 64)
    assert conn.commits == 0
    assert conn.rollbacks == 1
    # the guard must be in the SQL itself, not in a preceding SELECT
    insert_sql, _ = next(
        (sql, p) for sql, p in conn.statements if sql.startswith("INSERT"))
    assert "WHERE" in insert_sql
    assert not any(sql.startswith("SELECT id, content_sha256")
                   for sql, _ in conn.statements)


def test_the_advisory_lock_is_taken_before_any_read_and_is_casefolded():
    """Round 17: the lock key was the exact spelling, but Windows resolves
    every spelling to ONE file -- two spellings held two locks over it.
    The lock is FIRST (the row read under it feeds the gate) and its key
    is casefolded so every spelling serialises on the same lock."""
    conn = _FakeConn(existing=_served())
    db.upsert_document(conn, "KURGU.pdf", "pdf", content_sha256="a" * 64)
    first_sql, first_params = conn.statements[0]
    assert first_sql.startswith("SELECT pg_advisory_xact_lock")
    assert first_params == ("kurgu.pdf",)


def test_a_recased_name_lands_on_the_existing_row_not_a_sibling():
    """The stored spelling is canonical: uploading Kurgu.pdf over kurgu.pdf
    must hit the existing unique key, not insert a second row over the
    same Windows file."""
    conn = _FakeConn(existing=_served(filename="kurgu.pdf"))
    db.upsert_document(conn, "Kurgu.PDF", "pdf", content_sha256="a" * 64)
    _, params = next(
        (sql, p) for sql, p in conn.statements if sql.startswith("INSERT"))
    assert params["filename"] == "kurgu.pdf"


def test_same_name_same_content_proceeds():
    conn = _FakeConn(existing=_served())
    doc, _cid, _name = db.upsert_document(conn, "kurgu.pdf", "pdf",
                                          content_sha256="a" * 64)
    assert doc == "kurgu-belge-id"
    assert conn.commits == 1


def test_an_unchanged_candidate_keeps_its_identity():
    """Round 18: the id is the RUN identity. Re-knocking the SAME bytes
    must not invalidate an in-flight run of those bytes -- the identity
    changes only when the candidate bytes change."""
    existing = _served()
    existing["candidate_id"] = "eski-aday-kimligi"
    conn = _FakeConn(existing=existing)
    _doc, cid, _name = db.upsert_document(conn, "kurgu.pdf", "pdf",
                                          content_sha256="a" * 64)
    assert cid == "eski-aday-kimligi"


def test_a_changed_candidate_mints_a_fresh_identity():
    """...and a knock that CHANGES the bytes mints a new one, which is what
    makes a superseded run's promotion CAS fail loudly."""
    existing = _served()
    existing["candidate_id"] = "eski-aday-kimligi"
    conn = _FakeConn(existing=existing)
    _doc, cid, _name = db.upsert_document(conn, "kurgu.pdf", "pdf",
                                          content_sha256="b" * 64,
                                          allow_replace=True)
    assert cid is not None
    assert cid != "eski-aday-kimligi"


def test_a_serving_legacy_row_refuses_unmatched_content():
    """Round 17: active_content_sha=NULL used to be an open door -- a
    legacy migration's serving rows accepted arbitrary different bytes
    with no replace authority. "I cannot compare" is fail-closed now."""
    conn = _FakeConn(existing=_served(active=None))
    with pytest.raises(ValueError):
        db.upsert_document(conn, "kurgu.pdf", "pdf", content_sha256="b" * 64)
    # explicit authority still opens it
    conn = _FakeConn(existing=_served(active=None))
    db.upsert_document(conn, "kurgu.pdf", "pdf", content_sha256="b" * 64,
                       allow_replace=True)


def test_a_row_with_no_served_chunks_accepts_new_content():
    """A row that serves NOTHING (failed first attempt, fresh migration)
    loses nothing by replacement -- refusing it would strand the name."""
    conn = _FakeConn(existing=_served(active=None, has_chunks=False))
    doc, _cid, _name = db.upsert_document(conn, "kurgu.pdf", "pdf",
                                          content_sha256="b" * 64)
    assert doc == "kurgu-belge-id"


def test_an_authorised_candidate_carries_replace_authority_to_ingest():
    """Round 17: upload with ?replace=true recorded the new candidate, but
    the process step's upsert had no authority and refused the very bytes
    the operator just authorised. The candidate column can only be written
    THROUGH this gate, so matching it IS proof of prior authority -- no
    global environment flag needed."""
    conn = _FakeConn(existing=_served(active="a" * 64, candidate="b" * 64))
    doc, _cid, _name = db.upsert_document(conn, "kurgu.pdf", "pdf",
                                          content_sha256="b" * 64)
    assert doc == "kurgu-belge-id"


def test_a_legacy_row_without_a_hash_proceeds_and_backfills():
    conn = _FakeConn(existing=_served(active=None, has_chunks=False))
    db.upsert_document(conn, "kurgu.pdf", "pdf", content_sha256="a" * 64)
    insert_sql, params = next(
        (sql, p) for sql, p in conn.statements if sql.startswith("INSERT"))
    assert params["sha"] == "a" * 64
    assert "COALESCE" in insert_sql  # a later hashless call cannot erase it


def test_explicit_replacement_is_allowed_and_records_the_new_hash():
    conn = _FakeConn(existing=_served())
    db.upsert_document(conn, "kurgu.pdf", "pdf", content_sha256="b" * 64,
                       allow_replace=True)
    assert conn.commits == 1


def test_a_missing_hash_keeps_the_old_behaviour():
    # callers that cannot hash (none in-repo today) must not break rows
    conn = _FakeConn(existing=_served())
    db.upsert_document(conn, "kurgu.pdf", "pdf")
    assert conn.commits == 1


# --- Round 17: promotion verifies MEMBERSHIP, and nothing is never promotable

class _GenCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.statements.append(sql)
        if sql.startswith("SELECT id FROM chunks"):
            self._rows = [(i,) for i in self.conn.staged]
        elif sql.startswith("SELECT status FROM attempts"):
            self._one = (None,)
        elif sql.startswith("SELECT candidate_id, attempt_id"):
            self._one = ("kurgu-aday", "kurgu-deneme")
        elif sql.startswith("UPDATE documents"):
            self._one = ("kurgu-id",) if self.conn.cas_ok else None
        elif sql.startswith("UPDATE attempts"):
            self._one = ("kurgu-deneme",)
        elif sql.startswith("DELETE FROM chunks"):
            self.rowcount = self.conn.stale

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._one


class _GenConn:
    def __init__(self, staged, cas_ok=True, stale=0):
        self.staged = staged
        self.cas_ok = cas_ok
        self.stale = stale
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _GenCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_an_empty_manifest_is_refused_before_any_sql():
    """Promoting nothing is deleting everything wearing a success status;
    the refusal happens before the database is even consulted."""
    conn = _GenConn(staged=[])
    with pytest.raises(ValueError, match="bos manifest"):
        db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                              manifest_ids=set(), content_sha256="a" * 64,
                              candidate_id="kurgu-aday",

                              attempt_id="kurgu-deneme")
    assert conn.statements == []


def test_a_promotion_without_the_served_hash_is_refused():
    conn = _GenConn(staged=["k1"])
    with pytest.raises(ValueError, match="ozet"):
        db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                              manifest_ids={"k1"}, content_sha256=None,
                              candidate_id="kurgu-aday",

                              attempt_id="kurgu-deneme")
    assert conn.statements == []


def test_a_promotion_without_a_candidate_identity_is_refused():
    """Round 18: the active pointer alone is not a run identity -- a run
    that cannot say WHICH candidate it was ingesting may not promote."""
    conn = _GenConn(staged=["k1"])
    with pytest.raises(ValueError, match="aday kimligi"):
        db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                              manifest_ids={"k1"}, content_sha256="a" * 64,
                              candidate_id=None, attempt_id="kurgu-deneme")
    assert conn.statements == []


def test_a_promotion_without_a_run_identity_is_refused():
    """Package 3B: a run that cannot say WHICH ATTEMPT it is may not
    promote either. Two runs of one candidate are indistinguishable
    without it, and a promotion nobody can trace to a run is a promotion
    nobody can hold to account."""
    conn = _GenConn(staged=["k1"])
    with pytest.raises(ValueError, match="deneme kimligi"):
        db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                              manifest_ids={"k1"}, content_sha256="a" * 64,
                              candidate_id="kurgu-aday", attempt_id=None)
    assert conn.statements == []


def test_the_promotion_cas_binds_the_candidate_identity():
    """Round 18, the P0's closing clause: the UPDATE's own WHERE carries
    BOTH the observed active pointer and the bound candidate id, so a run
    whose candidate was superseded mid-flight cannot promote even though
    the pointer never moved."""
    conn = _GenConn(staged=["k1"])
    db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                          manifest_ids={"k1"}, content_sha256="a" * 64,
                          candidate_id="kurgu-aday",

                          attempt_id="kurgu-deneme")
    update = next(s for s in conn.statements if s.startswith("UPDATE"))
    assert "active_generation = %(expected)s" in update
    assert "candidate_id = %(cid)s" in update


def test_same_count_wrong_rows_no_longer_promotes():
    """Round 17: the manifest was a COUNT, and any same-sized set of wrong
    rows passed it. Membership refuses missing and foreign rows alike."""
    conn = _GenConn(staged=["k1", "yabanci"])
    with pytest.raises(ValueError, match="1 eksik, 1 yabanci"):
        db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                              manifest_ids={"k1", "k2"},
                              content_sha256="a" * 64,
                              candidate_id="kurgu-aday",

                              attempt_id="kurgu-deneme")
    assert conn.rollbacks == 1
    assert not any(s.startswith("UPDATE") for s in conn.statements)


def test_an_exact_manifest_promotes_and_sweeps_in_order():
    conn = _GenConn(staged=["k1", "k2"], stale=3)
    removed = db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                                    manifest_ids={"k1", "k2"},
                                    content_sha256="a" * 64,
                                    candidate_id="kurgu-aday",

                                    attempt_id="kurgu-deneme")
    assert removed == 3
    kinds = [s.split()[0] for s in conn.statements]
    # SELECT (manifest) - SELECT attempt (the trigger's parent proof) -
    # UPDATE documents (the three-part CAS plus the lease release) - UPDATE
    # attempts (this run's terminal DONE) - DELETE (the sweep). Package 3B
    # added the attempt closure: closing the
    # attempt is part of the same success, not a follow-up.
    assert kinds == ["SELECT", "SELECT", "UPDATE", "UPDATE", "DELETE"]
    assert conn.commits == 1


def test_a_lost_cas_race_raises_and_sweeps_nothing():
    conn = _GenConn(staged=["k1"], cas_ok=False)
    with pytest.raises(ValueError, match="es zamanli"):
        db.promote_generation(conn, "kurgu-id", 7, expected_active=2,
                              manifest_ids={"k1"}, content_sha256="a" * 64,
                              candidate_id="kurgu-aday",

                              attempt_id="kurgu-deneme")
    assert not any(s.startswith("DELETE") for s in conn.statements)
    assert conn.rollbacks == 1


def test_a_guarded_status_stamp_reports_whether_it_applied():
    """Round 17: a run that lost the promotion race must not rewrite the
    winner's 'done'. The stamp carries the run's observed active pointer
    in its own WHERE clause and says whether it landed."""
    class _StampCursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params):
            self.conn.seen = params
            applies = ((params["expected"] is None
                        or params["expected"] == self.conn.active)
                       and (params["cid"] is None
                            or params["cid"] == self.conn.candidate))
            self._one = ("kurgu-id",) if applies else None

        def fetchone(self):
            return self._one

    class _StampConn:
        def __init__(self, active, candidate="kurgu-aday"):
            self.active = active
            self.candidate = candidate
            self.seen = None

        def cursor(self):
            return _StampCursor(self)

        def commit(self):
            pass

    conn = _StampConn(active=3)
    assert db.set_document_status(conn, "kurgu-id", "error",
                                  expected_active=3) is True
    # the pointer moved: someone else promoted, the stamp must NOT land
    assert db.set_document_status(conn, "kurgu-id", "error",
                                  expected_active=2) is False
    # round 18: the candidate is the second half of the run identity --
    # a stamp bound to a superseded candidate lands nowhere either
    assert db.set_document_status(conn, "kurgu-id", "error",
                                  expected_active=3,
                                  candidate_id="kurgu-aday") is True
    assert db.set_document_status(conn, "kurgu-id", "error",
                                  expected_active=3,
                                  candidate_id="baska-aday") is False
    # unguarded callers keep the old semantics
    assert db.set_document_status(conn, "kurgu-id", "done") is True


# --- C8: a half-parsed document is "partial", never "done" ------------------

@pytest.fixture
def wired(monkeypatch, tmp_path):
    conn = _FakeConn()
    statuses = []
    promotions = []
    outcomes = []
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    monkeypatch.setattr(
        ingest, "route_and_parse",
        lambda path: ([("page1:scanned", ("text", "Kurgu icerik parcasi."))],
                      []))
    monkeypatch.setattr(ingest, "embed_sparse", lambda text: ([1], [0.5]))
    monkeypatch.setattr(ingest, "embed_dense", lambda text: [0.0])
    monkeypatch.setattr(ingest.db, "get_conn", lambda: conn)
    monkeypatch.setattr(ingest.db, "init_schema", lambda c: None)
    monkeypatch.setattr(
        ingest.db, "upsert_document",
        lambda *a, **k: ("kurgu-id", "kurgu-aday-kimligi", "kurgu.pdf"))
    monkeypatch.setattr(
        ingest.db, "lookup_document",
        lambda *a: {"id": "kurgu-id", "filename": "kurgu.pdf",
                    "candidate_id": "kurgu-aday-kimligi",
                    "content_sha256": hashlib.sha256(
                        b"KURGU_PDF").hexdigest(),
                    "active_generation": 2})
    # Package 3C: the core takes a cheap early-out between the parse and
    # the first write -- "is this still the current candidate?" -- so the
    # row this fixture hands back has to carry the candidate the attempt
    # was minted with. Without it every run here looked fenced.
    monkeypatch.setattr(ingest.db, "get_document",
                        lambda *a: {"active_generation": 2,
                                    "candidate_id": "kurgu-aday-kimligi"})
    monkeypatch.setattr(ingest.db, "allocate_generation", lambda *a: 7)
    monkeypatch.setattr(ingest.db, "existing_content_keys",
                        lambda *a: set())
    monkeypatch.setattr(ingest.db, "upsert_chunks", lambda *a: None)
    monkeypatch.setattr(ingest.db, "copy_chunks_into_generation",
                        lambda *a, **k: 0)
    monkeypatch.setattr(
        ingest.db, "promote_generation",
        lambda _conn, _id, gen, expected_active, manifest_ids,
        content_sha256, candidate_id, attempt_id: promotions.append(
            (gen, expected_active, len(manifest_ids), candidate_id)) or 0)
    monkeypatch.setattr(
        ingest.db, "set_document_status",
        lambda _conn, _id, status, note=None, expected_active=None,
        candidate_id=None: statuses.append((status, note)) or True)
    # Package 3B: the core runs UNDER AN ATTEMPT, so the fixture supplies
    # one. The verdict of a failed or partial run lands on the attempt --
    # `outcomes` -- and never on the document, which is why `statuses`
    # stays empty in the tests below where it used to carry the stamp.
    monkeypatch.setattr(
        ingest.db, "begin_attempt",
        lambda _conn, document_id, owner=None: IngestAttempt(
            attempt_id="kurgu-deneme-1", document_id=document_id,
            candidate_id="kurgu-aday-kimligi",
            candidate_sha=hashlib.sha256(b"KURGU_PDF").hexdigest(),
            observed_active=2),
        raising=False)
    monkeypatch.setattr(
        ingest.db, "record_attempt_outcome",
        lambda _conn, attempt, status, note=None: outcomes.append(
            (attempt.attempt_id, status, note)) or True, raising=False)
    monkeypatch.setattr(ingest.db, "heartbeat_attempt",
                        lambda _conn, _attempt: True, raising=False)
    conn.close = lambda: None
    return statuses, promotions, source, outcomes


def test_a_clean_parse_promotes_the_next_generation(wired):
    """Rounds 15+16: completion is a PROMOTION of an immutably allocated
    generation, carrying the observed active pointer (for the CAS) and the
    manifest size (for the row-count check) -- never an end-of-run
    delete."""
    statuses, promotions, source, outcomes = wired
    verdict = ingest.main(str(source))
    # allocated gen, observed active, rows, bound candidate identity
    assert promotions == [(7, 2, 1, "kurgu-aday-kimligi")]
    assert statuses == []             # done/cleanup live inside promotion
    # Package 3C: the run REPORTS its own verdict. Its caller cannot read
    # it anywhere else -- see the partial case, where the document row is
    # deliberately left untouched.
    assert verdict == (AttemptOutcome.DONE, None)


def test_lost_pages_stage_but_never_promote(wired, monkeypatch, capsys):
    """Rounds 14+15: a partial parse deletes nothing AND promotes nothing.
    The staged rows wait; retrieval keeps serving the last complete
    generation; WHAT failed persists in the row."""
    statuses, promotions, source, outcomes = wired
    failures = [{"kaynak": "page2:scanned", "asama": "sayfa",
                 "hata": "baglanti kurulamadi"}]
    monkeypatch.setattr(
        ingest, "route_and_parse",
        lambda path: ([("page1:scanned", ("text", "Kurgu icerik parcasi."))],
                      list(failures)))
    verdict = ingest.main(str(source))
    assert promotions == []      # the active pointer never moved
    # THE SEAM THE API READS. A partial run leaves the document row
    # alone, so an endpoint that stamped `processing` and then read that
    # row back concluded the run had never finished -- 500, and a healthy
    # generation relabelled `error`. The verdict travels in the RETURN
    # VALUE, and this is where that is pinned on the REAL core.
    assert verdict[0] == AttemptOutcome.PARTIAL
    assert "page2:scanned" in verdict[1]
    # Package 3B: the verdict belongs to the ATTEMPT. The document's
    # status describes the SERVED version, and a partial run did not
    # make that version any worse -- so it is not stamped at all.
    assert statuses == []
    assert len(outcomes) == 1
    _attempt_id, status, note = outcomes[0]
    assert status == AttemptOutcome.PARTIAL
    assert "page2:scanned" in note      # persisted, machine-readable
    out = capsys.readouterr().out
    assert "KISMI" in out and "aktif surum degismedi" in out


def test_an_empty_parse_is_an_error_not_a_wipe(wired, monkeypatch):
    """Round 15, the auditor's probe verbatim: route_and_parse -> ([], [])
    used to delete the document's existing rows and stamp it done. Now it
    refuses, marks error, and touches neither generation."""
    statuses, promotions, source, outcomes = wired
    monkeypatch.setattr(ingest, "route_and_parse", lambda path: ([], []))
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert promotions == []
    assert statuses == []               # the document is not stamped
    assert [(status, note) for _a, status, note in outcomes] == [
        (AttemptOutcome.ERROR, "parse kullanilabilir icerik uretmedi")]


def test_an_all_whitespace_parse_cannot_promote_an_empty_generation(
        wired, monkeypatch):
    """Round 17: counting PARTS was not enough. A parser that returned a
    part whose every chunk was whitespace staged ZERO rows, passed the
    size-0 manifest (0 == 0), promoted, and the sweep deleted the healthy
    index. Usable text is what gets counted now, and the refusal is
    identical to the empty-parse one."""
    statuses, promotions, source, outcomes = wired
    monkeypatch.setattr(
        ingest, "route_and_parse",
        lambda path: ([("page1:scanned", ("text", "bos"))], []))
    monkeypatch.setattr(
        ingest, "chunk_plain_text",
        lambda text, tag: [{"type": "text", "text": "   ",
                            "source_tag": tag, "page": 1, "headings": []}])
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert promotions == []
    assert statuses == []               # the document is not stamped
    assert [(status, note) for _a, status, note in outcomes] == [
        (AttemptOutcome.ERROR, "parse kullanilabilir icerik uretmedi")]


def test_the_parser_is_given_a_private_snapshot_of_the_hashed_bytes(
        wired, monkeypatch):
    """Round 17, the ABA probe: hashing the path and parsing the path were
    two separate reads. The file held content B while the parser read it
    and content A again for the re-hash -- chunks from B promoted under
    hash(A). No re-hash schedule can close that; parsing a PRIVATE copy of
    the exact bytes that were hashed can, and does."""
    statuses, promotions, source, outcomes = wired
    seen = {}

    def recording_parse(path):
        seen["path"] = str(path)
        seen["bytes_at_parse"] = ingest.Path(path).read_bytes()
        return ([("page1:scanned", ("text", "Kurgu icerik parcasi."))], [])

    monkeypatch.setattr(ingest, "route_and_parse", recording_parse)
    ingest.main(str(source))
    assert seen["path"] != str(source)          # never the mutable original
    assert seen["bytes_at_parse"] == b"KURGU_PDF"  # exactly what was hashed
    assert ingest.Path(seen["path"]).name == "kurgu.pdf"  # same identity
    assert promotions  # and the run itself completed normally


def test_an_aba_swap_of_the_source_no_longer_reaches_the_index(
        wired, monkeypatch):
    """The probe verbatim: mid-parse the source flips to other bytes and
    back. With the snapshot, whatever happens to the original during the
    run is irrelevant -- the parsed bytes ARE the hashed bytes."""
    statuses, promotions, source, outcomes = wired
    seen = {}

    def swapping_parse(path):
        source.write_bytes(b"BASKA_ICERIK")     # the mid-parse swap...
        seen["bytes_at_parse"] = ingest.Path(path).read_bytes()
        source.write_bytes(b"KURGU_PDF")        # ...and the swap back
        return ([("page1:scanned", ("text", "Kurgu icerik parcasi."))], [])

    monkeypatch.setattr(ingest, "route_and_parse", swapping_parse)
    ingest.main(str(source))
    assert seen["bytes_at_parse"] == b"KURGU_PDF"
    assert promotions == [(7, 2, 1, "kurgu-aday-kimligi")]


def test_a_tampered_snapshot_refuses_and_preserves_the_index(
        wired, monkeypatch):
    """The snapshot re-hash guards OUR OWN machinery: if the private copy
    itself changes under the parser (corrupt volume, interfering scanner),
    the run refuses instead of promoting bytes nobody hashed."""
    statuses, promotions, source, outcomes = wired

    def corrupting_parse(path):
        ingest.Path(path).write_bytes(b"BOZULMUS_ANLIK")
        return ([("page1:scanned", ("text", "Kurgu icerik parcasi."))], [])

    monkeypatch.setattr(ingest, "route_and_parse", corrupting_parse)
    with pytest.raises(RuntimeError):
        ingest.main(str(source))
    assert promotions == []
    # the snapshot moved before any attempt work began, so there is
    # nothing to stamp anywhere: the run refuses and the index is intact
    assert statuses == []


def test_a_bound_ingest_refuses_a_disk_that_left_the_candidate(
        wired, monkeypatch):
    """Round 18, the P0's first gate: a process step bound to a recorded
    candidate must refuse BEFORE parsing when the disk no longer carries
    those bytes -- a newer upload landed, or the file was touched."""
    statuses, promotions, source, outcomes = wired
    called = []
    monkeypatch.setattr(ingest, "route_and_parse",
                        lambda path: called.append(path) or ([], []))
    with pytest.raises(RuntimeError, match="kayitli adayla uyusmuyor"):
        ingest.main(str(source),
                    expected_candidate=("kurgu-aday-kimligi", "f" * 64))
    assert called == []        # refused before any parsing
    assert promotions == []


def test_a_bound_ingest_refuses_a_superseded_candidate_identity(
        wired, monkeypatch):
    """...and when the hash still matches but the RECORDED candidate id
    moved on (a same-bytes re-upload is the only way that happens without
    a hash change -- or a probe forging state), the run cancels."""
    statuses, promotions, source, outcomes = wired
    monkeypatch.setattr(
        ingest.db, "lookup_document",
        lambda *a: {"id": "kurgu-id", "filename": "kurgu.pdf",
                    "candidate_id": "baska-aday-kimligi",
                    "active_generation": 2})
    with pytest.raises(RuntimeError, match="aday kimligi"):
        ingest.main(str(source),
                    expected_candidate=(
                        "kurgu-aday-kimligi",
                        hashlib.sha256(b"KURGU_PDF").hexdigest()))
    assert promotions == []


def test_a_bound_ingest_adopts_the_row_without_knocking_the_gate(
        wired, monkeypatch):
    """The P0's mechanism was the process step RE-RECORDING the old hash
    as candidate (legitimately, via the active arm) over a newer upload.
    Bound mode therefore never calls the upsert at all -- it adopts the
    row it was bound to and promotes under that identity."""
    statuses, promotions, source, outcomes = wired
    knocked = []
    monkeypatch.setattr(
        ingest.db, "upsert_document",
        lambda *a, **k: knocked.append(1) or ("x", "y", "z"))
    ingest.main(str(source),
                expected_candidate=(
                    "kurgu-aday-kimligi",
                    hashlib.sha256(b"KURGU_PDF").hexdigest()))
    assert knocked == []       # the candidate gate was never re-knocked
    assert promotions == [(7, 2, 1, "kurgu-aday-kimligi")]


def test_the_publish_lock_releases_safely_on_an_aborted_body():
    """Round 18: a body that left the transaction aborted made the unlock
    statement fail too -- the primary error was masked and the POOLED
    session kept the lock. The release path now rolls back first; if the
    unlock still cannot be proven, the connection is closed outright so
    the server drops the lock."""
    class _LockConn:
        def __init__(self, unlock_breaks=False):
            self.unlock_breaks = unlock_breaks
            self.aborted = False
            self.events = []

        def cursor(self):
            conn = self

            class Cursor:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, sql, params=None):
                    if conn.aborted or (conn.unlock_breaks
                                        and "unlock" in sql):
                        conn.events.append("execute-hata")
                        raise RuntimeError("InFailedSqlTransaction")
                    conn.events.append(
                        "unlock" if "unlock" in sql else "lock")

                def fetchone(self):
                    return (True,)

            return Cursor()

        def commit(self):
            self.events.append("commit")

        def rollback(self):
            self.aborted = False
            self.events.append("rollback")

        def close(self):
            self.events.append("close")

    # an aborted body: rollback clears the state, the unlock still runs,
    # and the PRIMARY error is what propagates
    conn = _LockConn()
    with pytest.raises(RuntimeError, match="birincil"):
        with db.document_publish_lock(conn, "kurgu.pdf"):
            conn.aborted = True
            raise RuntimeError("birincil hata")
    assert "rollback" in conn.events
    assert "unlock" in conn.events
    assert "close" not in conn.events

    # the unlock itself failing: the connection is CLOSED so the server
    # releases the lock -- and the primary error still propagates
    conn = _LockConn(unlock_breaks=True)
    with pytest.raises(RuntimeError, match="birincil"):
        with db.document_publish_lock(conn, "kurgu.pdf"):
            raise RuntimeError("birincil hata")
    assert "close" in conn.events


@pytest.mark.parametrize("where", ["parser", "chunker"])
def test_an_early_phase_crash_closes_the_attempt_and_frees_the_lease(
        wired, monkeypatch, where):
    """The parse runs WITHOUT a connection -- it is minutes long and must
    not hold a pooled one -- so a parser or chunker that raised used to
    leave the lease held until it expired. Nobody could retry the
    document for that whole window because of a run that had already
    given up.

    Three things at once: the ORIGINAL failure reaches the caller, the
    attempt is closed ERROR, and the note carries the exception's TYPE
    and nothing else (a parser message can contain a path, a fragment of
    the document or a service endpoint)."""
    statuses, promotions, source, outcomes = wired
    private = "OZEL_KURGU_PARSE_AYRINTISI"

    class KurguParseHatasi(RuntimeError):
        pass

    def explode(*_a, **_k):
        raise KurguParseHatasi(private)

    if where == "parser":
        monkeypatch.setattr(ingest, "route_and_parse", explode)
    else:
        monkeypatch.setattr(ingest, "_chunks_from_parts", explode)

    with pytest.raises(KurguParseHatasi):
        ingest.main(str(source))

    assert promotions == []
    assert statuses == []                      # the document is untouched
    assert len(outcomes) == 1
    _attempt_id, status, note = outcomes[0]
    assert status == AttemptOutcome.ERROR
    assert note == "KurguParseHatasi"
    assert private not in str(outcomes), (
        "ham istisna metni deneme kaydina kopyalandi")


def test_a_partial_run_does_not_report_success_when_its_record_is_broken(
        wired, monkeypatch):
    """Package 3B: ``_close_attempt`` swallows a FENCE and a LOST LEASE --
    refusals a displaced run expected -- and nothing else.

    Catching the whole AttemptError family once meant that a run whose
    attempt record could not be closed AT ALL still printed "partial
    completed" and returned normally. That is the same shape of lie the
    attempt record exists to prevent, so the inconsistency propagates."""
    from pipeline.index.attempt_contract import (
        AttemptFenced,
        AttemptRecordInconsistent,
    )

    statuses, promotions, source, outcomes = wired
    monkeypatch.setattr(
        ingest, "route_and_parse",
        lambda path: ([("page1:scanned", ("text", "Kurgu icerik parcasi."))],
                      [{"kaynak": "page2:scanned", "asama": "sayfa",
                        "hata": "kurgu hata"}]))

    def broken_record(*_a, **_k):
        raise AttemptRecordInconsistent("deneme kaydi kapatilamadi")

    monkeypatch.setattr(ingest.db, "record_attempt_outcome", broken_record)
    with pytest.raises(AttemptRecordInconsistent):
        ingest.main(str(source))
    assert promotions == []

    # ...while a fence, which a displaced run expects, is still swallowed
    monkeypatch.setattr(
        ingest.db, "record_attempt_outcome",
        lambda *_a, **_k: (_ for _ in ()).throw(AttemptFenced("cevrildi")))
    ingest.main(str(source))          # returns normally


def test_the_core_never_knocks_the_candidate_gate(wired, monkeypatch):
    """Package 3B, rule 7: candidates are recorded by PUBLICATION only.

    This test used to assert the opposite -- that the run knocked the
    gate with its own hash -- which is exactly the behaviour the CLI used
    to revert a newer authorised upload with. The core now binds to an
    attempt someone else opened and never touches the gate."""
    statuses, _promotions, source, outcomes = wired
    knocked = []
    monkeypatch.setattr(
        ingest.db, "upsert_document",
        lambda *a, **k: knocked.append(a) or ("x", "y", "z"))

    ingest.main(str(source))

    assert knocked == [], "cekirdek aday kapisini caldi (rule 7)"


def test_embedding_reuse_is_gated_on_the_exact_fingerprint(
        wired, monkeypatch):
    """Round 18: the content key speaks only about the TEXT, and a model
    change rode stale vectors into the new generation on a text match.
    The reuse lookup now carries the embedding fingerprint, and every
    written row records it."""
    from pipeline.index import embeddings

    statuses, promotions, source, outcomes = wired
    seen = {}
    written = []
    monkeypatch.setattr(
        ingest.db, "existing_content_keys",
        lambda _conn, _doc, fingerprint=None:
        seen.update(fingerprint=fingerprint) or set())
    monkeypatch.setattr(
        ingest.db, "upsert_chunks",
        lambda _conn, batch, _attempt: written.extend(batch))
    ingest.main(str(source))
    assert seen["fingerprint"] == embeddings.embedding_fingerprint()
    assert all(row["embedding_fingerprint"]
               == embeddings.embedding_fingerprint() for row in written)
    assert written  # the fixture parse produced at least one row


def test_the_fingerprint_changes_with_every_contract_knob(monkeypatch):
    """Dense model, truncation cap, sparse language: each knob must move
    the fingerprint, or its change would silently reuse stale vectors."""
    from pipeline.index import embeddings

    base = embeddings.embedding_fingerprint()
    monkeypatch.setattr(embeddings, "EMBED_MODEL_NAME", "kurgu/baska-model")
    changed_model = embeddings.embedding_fingerprint()
    monkeypatch.undo()
    monkeypatch.setattr(embeddings, "EMBED_MAX_CHARS", 123)
    changed_cap = embeddings.embedding_fingerprint()
    monkeypatch.undo()
    monkeypatch.setattr(embeddings, "BM25_LANGUAGE", "english")
    changed_language = embeddings.embedding_fingerprint()
    assert len({base, changed_model, changed_cap, changed_language}) == 4
    assert embeddings.embedding_fingerprint.__call__  # deterministic api
    monkeypatch.undo()
    assert embeddings.embedding_fingerprint() == base


# --- 3C: the CLI answers with a CODE, and the code says which refusal ------

@pytest.mark.parametrize(
    ("refusal", "expected"),
    [
        (CandidateNotPublished("aday evrede kaldi"),
         ExitCode.CANDIDATE_NOT_PUBLISHED),
        (AttemptAlreadyRunning("lease baskasinda"),
         ExitCode.ATTEMPT_UNAVAILABLE),
    ],
    ids=["yayimlanmamis_aday", "canli_lease"],
)
def test_the_cli_answers_a_refused_lease_with_its_own_frozen_code(
        monkeypatch, tmp_path, refusal, expected):
    """Two of the five frozen codes can only come from ``begin_attempt``:
    a candidate that never finished publishing, and a document another
    run is already holding. Neither is a crash and neither is the other
    -- a caller that retries a busy document must not be told its upload
    is broken, and a caller whose upload really is half-published must
    not be told to wait for a run that does not exist.

    Both must also leave the CORE untouched: a refusal that still parsed
    the file would have spent the whole cost of the run before deciding
    not to do it."""
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    reached = []

    class FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(ingest.db, "get_conn", lambda: FakeConn())
    monkeypatch.setattr(ingest.db, "init_schema", lambda _conn: None)
    monkeypatch.setattr(
        ingest.publication, "publish_candidate",
        lambda *a, **k: ("kurgu-id", "kurgu-aday-1", "kurgu.pdf"))

    def refusing_begin(*_a, **_k):
        raise refusal

    monkeypatch.setattr(ingest.db, "begin_attempt", refusing_begin,
                        raising=False)
    monkeypatch.setattr(ingest, "ingest_attempt",
                        lambda *a, **k: reached.append("core"))

    assert ingest.cli_main([str(source)]) == expected
    assert reached == []


def test_the_cli_refuses_a_command_line_it_does_not_understand(tmp_path):
    """A usage error is NOT one of the frozen decisions, and it must not
    borrow one of their codes: exit 2 means "different content, no
    --replace", and a mistyped command answering 2 would send an operator
    looking for a conflict that never existed."""
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    for argv in ([], [str(source), "extra.pdf"], [str(source), "--kurgu"]):
        code = ingest.cli_main(argv)
        assert code == ingest.USAGE_EXIT
        assert code not in (ExitCode.OK, ExitCode.CANDIDATE_CONFLICT,
                            ExitCode.CANDIDATE_NOT_PUBLISHED,
                            ExitCode.ATTEMPT_UNAVAILABLE,
                            ExitCode.ATTEMPT_LOST)


def test_a_snapshot_failure_closes_the_attempt_its_caller_already_took(
        monkeypatch, tmp_path):
    """The API takes the lease BEFORE calling, so that a candidate still
    being published can be answered 409 rather than discovered halfway
    through an ingest. That makes every failure from the call onwards the
    LEASE HOLDER's problem -- including the snapshot itself, which sits
    before the core is even entered.

    An unreadable path or a full temp volume used to just raise: the
    attempt stayed open and the document could not be retried by anyone
    until the lease EXPIRED, because of a run that never started. The
    note carries the exception TYPE and nothing else -- an OS error
    message names the path it failed on."""
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    attempt = IngestAttempt(
        attempt_id="kurgu-deneme-1", document_id="kurgu-id",
        candidate_id="kurgu-aday", candidate_sha="a" * 64,
        observed_active=3)
    outcomes = []
    entered_core = []
    private = "OZEL_KURGU_ANLIK_GORUNTU_AYRINTISI"

    class FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(ingest.db, "get_conn", lambda: FakeConn())
    monkeypatch.setattr(
        ingest.db, "record_attempt_outcome",
        lambda _conn, att, status, note=None: outcomes.append(
            (att.attempt_id, status, note)) or True, raising=False)
    monkeypatch.setattr(ingest, "ingest_attempt",
                        lambda *a, **k: entered_core.append("core"))

    def failing_snapshot(_path):
        raise OSError(private)

    monkeypatch.setattr(ingest, "_snapshot_of", failing_snapshot)

    with pytest.raises(OSError):
        ingest.main(str(source), attempt=attempt)

    assert entered_core == []
    assert outcomes == [("kurgu-deneme-1", AttemptOutcome.ERROR, "OSError")]
    assert private not in str(outcomes)


def test_a_snapshot_failure_without_a_lease_closes_nothing(monkeypatch,
                                                           tmp_path):
    """The mirror image, and the reason the fix is conditional: a caller
    that has NOT taken a lease has no attempt to close. Opening a
    connection to record a verdict for a run that was never begun would
    be a write with no subject."""
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(b"KURGU_PDF")
    opened = []

    monkeypatch.setattr(ingest.db, "get_conn",
                        lambda: opened.append("conn"))
    monkeypatch.setattr(ingest, "_snapshot_of",
                        lambda _path: (_ for _ in ()).throw(OSError("KURGU")))

    with pytest.raises(OSError):
        ingest.main(str(source))
    assert opened == []
