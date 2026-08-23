"""RED PHASE -- LOCAL half of the attempt/publication contract.

Written BEFORE any fix, deliberately. What is checked HERE is what is
local: the module split, the CLI's wiring and its flags, the error kinds
Python reports, and the API's answer in the publish gap.

WHAT IS DELIBERATELY NOT HERE. Database behaviour -- the gate arms, the
lease, the promotion CAS, the crash windows -- is checked against a real
server in ``tests/test_pg_attempt_integration.py``. An earlier draft
asserted those rules against an in-memory model after merely checking
that the production seam existed; an EMPTY production function would
have turned every one of them green. A model can pin intent, never
behaviour, so the model no longer stands in for the database anywhere.

TWO CALLABLES, NOT ONE. An earlier draft also asked the documented entry
point both to refuse running without an attempt and to publish-then-begin
one -- two opposite behaviours from one function, which can never both be
green. The contract splits them (see attempt_contract): ``cli_main``
publishes and begins; ``ingest_attempt`` refuses to run unbound.
"""
import hashlib
import inspect
import threading
from pathlib import Path

import pytest

from pipeline.index import db, ingest
from pipeline.index.attempt_contract import (
    CandidateConflict,
    CandidateNotPublished,
    CandidateState,
    ExitCode,
    IngestAttempt,
)

A_BYTES = b"KURGU_SURUM_A"
B_BYTES = b"KURGU_SURUM_B"
SHA_A = hashlib.sha256(A_BYTES).hexdigest()
SHA_B = hashlib.sha256(B_BYTES).hexdigest()


class Violations:
    """Collects every broken requirement so one fix cannot hide the rest."""

    def __init__(self):
        self.entries = []

    def require(self, ok, message):
        if not ok:
            self.entries.append(message)
        return bool(ok)

    def assert_none(self):
        if self.entries:
            raise AssertionError(
                f"{len(self.entries)} sozlesme ihlali:\n  - "
                + "\n  - ".join(self.entries))


def _run_capturing(callable_):
    try:
        return None, callable_()
    except BaseException as error:      # noqa: BLE001 -- the kind is the test
        return error, None


def run_cli(path, replace=False):
    """The DOCUMENTED CLI entry point, whichever it currently is.

    Once the split lands this is ``cli_main(argv)``; until then it is the
    old ``main(path)``. The CLI's contract is tested through this and the
    core's through ``ingest_attempt`` -- never both through one callable."""
    cli_main = getattr(ingest, "cli_main", None)
    if cli_main is not None:
        argv = [str(path)] + (["--replace"] if replace else [])
        return cli_main(argv)
    return ingest.main(str(path))


class DocumentStore:
    """In-memory model of the row the API and CLI seams talk to.

    It exists to let the LOCAL tests observe what production wrote -- not
    to stand in for the database's decisions. Where a rule is a database
    rule, the assertion lives in the PostgreSQL module instead."""

    def __init__(self, filename="kurgu.pdf"):
        self.row = {
            "id": "kurgu-belge-id",
            "filename": filename,
            "status": "pending",
            "content_sha256": None,
            "candidate_id": None,
            "candidate_state": None,
            "active_generation": 0,
            "last_generation": 0,
            "active_content_sha": None,
            "attempt_id": None,
        }
        self.has_chunks = False
        self.minted = 0
        self.promotions = []
        self.stamps = []
        self.attempt_outcomes = {}
        self.begin_attempt_calls = []
        self.begin_attempt_errors = []
        self.ingest_calls = []
        self.chunk_writes = []

    def upsert_document(self, _conn, filename, file_type, status="processing",
                        content_sha256=None, allow_replace=False):
        row = self.row
        accepted = (
            allow_replace
            or content_sha256 is None
            or row["active_content_sha"] == content_sha256
            or row["content_sha256"] == content_sha256
            or (row["active_content_sha"] is None and not self.has_chunks)
        )
        if not accepted:
            raise CandidateConflict(
                "ayni dosya adi farkli icerikle zaten kayitli")
        if content_sha256 is None:
            candidate_id = row["candidate_id"]
        elif (row["content_sha256"] == content_sha256
              and row["candidate_id"] is not None):
            candidate_id = row["candidate_id"]
        else:
            self.minted += 1
            candidate_id = f"kurgu-aday-{self.minted}"
            row["candidate_state"] = CandidateState.STAGED
        row["status"] = status
        row["candidate_id"] = candidate_id
        if content_sha256 is not None:
            row["content_sha256"] = content_sha256
        return row["id"], candidate_id, row["filename"]

    def stage_candidate(self, _conn, filename, file_type,
                        content_sha256=None, allow_replace=False):
        row = self.row
        if not (allow_replace or content_sha256 is None
                or row["content_sha256"] == content_sha256
                or (row["active_content_sha"] is None
                    and not self.has_chunks)):
            raise CandidateConflict("farkli icerik acik yetki ister")
        if content_sha256 is not None and (
                row["content_sha256"] != content_sha256
                or row["candidate_id"] is None):
            self.minted += 1
            row["candidate_id"] = f"kurgu-aday-{self.minted}"
            row["content_sha256"] = content_sha256
            row["candidate_state"] = CandidateState.STAGED
            row["attempt_id"] = None
        # THREE values, per the frozen seam: the canonical filename is
        # what the publisher's disk target is derived from. A two-value
        # model would break a correct implementation with "too many
        # values to unpack" -- a false RED.
        return row["id"], row["candidate_id"], row["filename"]

    def finalize_candidate_publication(self, _conn, document_id,
                                       candidate_id):
        if self.row["candidate_id"] != candidate_id:
            return False
        self.row["candidate_state"] = CandidateState.PUBLISHED
        return True

    def begin_attempt(self, _conn, document_id, owner="kurgu-worker"):
        self.begin_attempt_calls.append((document_id, owner))
        if self.row["candidate_state"] != CandidateState.PUBLISHED:
            error = CandidateNotPublished("aday henuz yayimlanmadi")
            self.begin_attempt_errors.append(error)
            raise error
        attempt = IngestAttempt(
            attempt_id=f"kurgu-deneme-{len(self.begin_attempt_calls)}",
            document_id=self.row["id"],
            candidate_id=self.row["candidate_id"],
            candidate_sha=self.row["content_sha256"],
            observed_active=self.row["active_generation"],
        )
        self.row["attempt_id"] = attempt.attempt_id
        return attempt

    def heartbeat_attempt(self, _conn, attempt):
        """Package 3B wired a heartbeat into the core's long embed loop:
        it EXTENDS the lease and grants nothing -- the right to write is
        checked inside each write's own transaction."""
        if self.row["attempt_id"] != attempt.attempt_id:
            from pipeline.index.attempt_contract import AttemptLeaseLost

            raise AttemptLeaseLost("lease devralindi")
        return True

    def record_attempt_outcome(self, _conn, attempt, status, note=None):
        if self.row["attempt_id"] != attempt.attempt_id:
            from pipeline.index.attempt_contract import AttemptLeaseLost

            raise AttemptLeaseLost("lease devralindi")
        self.attempt_outcomes[attempt.attempt_id] = {
            "status": status, "note": note,
            "observed_active": attempt.observed_active,
        }
        return True

    def lookup_document(self, _conn, filename):
        if filename.casefold() != self.row["filename"].casefold():
            return None
        return dict(self.row)

    def get_document(self, _conn, document_id):
        return dict(self.row) if document_id == self.row["id"] else None

    def allocate_generation(self, _conn, _document_id, _attempt=None):
        # Package 3B: every write seam carries the attempt, because a
        # heartbeat a moment ago is not authority now.
        self.row["last_generation"] += 1
        return self.row["last_generation"]

    def promote_generation(self, _conn, document_id, generation,
                           expected_active, manifest_ids, content_sha256,
                           candidate_id, attempt_id=None):
        row = self.row
        if (row["active_generation"] != expected_active
                or row["candidate_id"] != candidate_id):
            raise ValueError("aktif nesil ya da aday kimligi farkli")
        row.update({"active_generation": generation,
                    "active_content_sha": content_sha256, "status": "done",
                    "attempt_id": None})
        self.promotions.append((generation, content_sha256, candidate_id))
        return 0

    def set_document_status(self, _conn, document_id, status, note=None,
                            expected_active=None, candidate_id=None):
        row = self.row
        if (expected_active is not None
                and row["active_generation"] != expected_active):
            return False
        if candidate_id is not None and row["candidate_id"] != candidate_id:
            return False
        row["status"] = status
        self.stamps.append((status, expected_active, candidate_id))
        return True

    def snapshot(self):
        return {"row": dict(self.row), "stamps": len(self.stamps),
                "promotions": len(self.promotions),
                "chunks": len(self.chunk_writes),
                "outcomes": dict(self.attempt_outcomes)}


@pytest.fixture
def store_and_source(monkeypatch, tmp_path):
    store = DocumentStore()
    source = tmp_path / "kurgu.pdf"
    source.write_bytes(A_BYTES)
    # THE CLI TESTS BELOW REALLY PUBLISH. Once the CLI became a
    # first-class publisher, running these fixtures put 13 real bytes of
    # invented content into whatever UPLOAD_DIR pointed at -- which by
    # default is ./data/uploads, the operator's own document directory.
    # It landed there, and the leak scanner then failed CLOSED on a file
    # it could not read, exactly as it should have: a test that writes
    # into the data tree does not just make a mess, it disables the tool
    # that guards it.
    publication = _import_publication()
    if publication is not None:
        published_to = tmp_path / "uploads"
        published_to.mkdir(exist_ok=True)
        monkeypatch.setattr(publication, "UPLOAD_DIR", published_to,
                            raising=False)
        # `ingest.publication` is the same module object today. Binding
        # it too costs nothing and keeps the redirect honest if the
        # import style ever changes.
        monkeypatch.setattr(ingest.publication, "UPLOAD_DIR", published_to,
                            raising=False)

    class FakeConn:
        """Enough of a connection for BOTH halves of the CLI's chain.

        REPAIRED IN PACKAGE 3C, and this is the one change made to this
        frozen file -- recorded here so it can be audited in one look.

        This fake was written for the CORE, which reaches the database
        only through seams the fixture patches, so `cursor` and `close`
        were all it ever needed. The CLI's contract sends it somewhere
        else: through the shared publication service, which takes a
        database SESSION lock. That lock commits, rolls back, and READS
        the unlock's answer instead of assuming it -- a deliberate
        fail-closed choice, because `pg_advisory_unlock` returns FALSE
        when the session never held the lock. Against a fake with no
        transaction methods it died on `commit`; once given them it read
        the hard-coded `(0,)` as "released nothing" and refused, exactly
        as it should have. Five contract tests could therefore never
        reach the code they were written for.

        What is added is the ability to ANSWER, nothing else: no
        assertion is added, weakened or removed, and no production
        behaviour is accommodated -- a real server answers this unlock
        with a boolean, never with 0."""

        def close(self):
            pass

        def commit(self):
            pass

        def rollback(self):
            pass

        def cursor(self):
            class Cursor:
                sql = ""

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, sql="", *a, **k):
                    self.sql = str(sql)

                def fetchone(self):
                    if "advisory_unlock" in self.sql:
                        return (True,)      # the lock WAS released
                    return (0,)             # the core's row count

            return Cursor()

    monkeypatch.setattr(ingest, "embed_sparse", lambda text: ([1], [0.5]))
    monkeypatch.setattr(ingest, "embed_dense", lambda text: [0.0])
    monkeypatch.setattr(ingest.db, "get_conn", lambda: FakeConn())
    monkeypatch.setattr(ingest.db, "init_schema", lambda c: None)
    monkeypatch.setattr(ingest.db, "existing_content_keys",
                        lambda *a, **k: set())
    monkeypatch.setattr(
        ingest.db, "upsert_chunks",
        lambda _conn, batch, _attempt: store.chunk_writes.extend(batch))
    monkeypatch.setattr(ingest.db, "copy_chunks_into_generation",
                        lambda *a, **k: 0)
    for name in ("upsert_document", "lookup_document", "get_document",
                 "allocate_generation", "promote_generation",
                 "set_document_status", "stage_candidate",
                 "finalize_candidate_publication", "begin_attempt",
                 "heartbeat_attempt", "record_attempt_outcome"):
        monkeypatch.setattr(ingest.db, name, getattr(store, name),
                            raising=False)
    return store, source


def _parse_with(reached, during=None, failures=()):
    def parse(_path):
        reached["parse"] = reached.get("parse", 0) + 1
        if during is not None:
            during()
            reached["during"] = reached.get("during", 0) + 1
        return ([("page1:scanned", ("text", "Kurgu icerik parcasi."))],
                list(failures))
    return parse


# =====================================================================
# DEFECT 1 -- the documented CLI path promotes a stale snapshot
# =====================================================================

def test_the_documented_cli_entry_point_cannot_promote_a_stale_snapshot(
        store_and_source, monkeypatch):
    """The CLI parses a snapshot of version A while an authorised upload
    publishes B. Today it then knocks the candidate gate itself with A's
    hash -- which equals the SERVED hash, a legitimate arm -- mints a new
    candidate id over B's, and promotes the stale snapshot.

    Rules 2 and 7. After the fence NOTHING this run touched may differ:
    the assertion is a full snapshot, not a spot check."""
    store, source = store_and_source
    store.row.update({
        "content_sha256": SHA_A, "candidate_id": "kurgu-aday-A",
        "candidate_state": CandidateState.PUBLISHED,
        "active_generation": 1, "active_content_sha": SHA_A,
        "status": "done",
    })
    store.has_chunks = True
    reached, before = {}, {}

    def authorised_upload_lands():
        store.row.update({"content_sha256": SHA_B,
                          "candidate_id": "kurgu-aday-B",
                          "candidate_state": CandidateState.PUBLISHED,
                          "status": "pending"})
        source.write_bytes(B_BYTES)
        before["snapshot"] = store.snapshot()

    monkeypatch.setattr(ingest, "route_and_parse",
                        _parse_with(reached, during=authorised_upload_lands))

    error, code = _run_capturing(lambda: run_cli(source))

    v = Violations()
    v.require(reached.get("during") == 1,
              "es zamanli upload geri cagrisina hic ulasilmadi: senaryo "
              "kurulmadi, bu testin yesili anlamsiz olurdu")
    # the CLI ANSWERS a fence, it does not crash on one. An earlier
    # version demanded an AttemptFenced traceback here, which would have
    # FAILED a correct CLI that caught it and returned the frozen code.
    v.require(error is None,
              f"CLI cevrilmeyi istisna olarak firlatti: {error!r} "
              f"(refuzal bir DONUS DEGERIDIR)")
    v.require(code == ExitCode.ATTEMPT_LOST,
              f"beklenen cikis kodu {ExitCode.ATTEMPT_LOST}, gelen: {code!r}")
    after, baseline = store.snapshot(), before["snapshot"]
    for field in ("candidate_id", "content_sha256", "candidate_state",
                  "active_generation", "active_content_sha", "status",
                  "last_generation", "attempt_id"):
        v.require(after["row"][field] == baseline["row"][field],
                  f"cevrilen kosu satiri degistirdi: {field} "
                  f"{baseline['row'][field]!r} -> {after['row'][field]!r}")
    for field in ("stamps", "promotions", "chunks"):
        v.require(after[field] == baseline[field],
                  f"cevrilen kosu yazdi: {field} {baseline[field]} -> "
                  f"{after[field]}")
    v.require(source.read_bytes() == B_BYTES,
              "disk yeni adaydan ayrildi: ingest diski yazmamali")
    v.assert_none()


def test_a_refused_candidate_offer_is_reported_as_a_domain_conflict():
    """WHICH offers the database refuses is an SQL property, tested against
    real PostgreSQL (rule 10) THROUGH ``stage_candidate`` -- the one gate
    a candidate may enter by. A fake cursor re-implementing the
    acceptance formula could not detect a fix to that formula, so this
    test does not try.

    What is local: when the database HAS refused (no row returned), the
    code must roll back and report a precise ``CandidateConflict`` -- not
    a bare ValueError that any other failure also raises. It subclasses
    ValueError, so callers translating refusals into 409 keep working.

    Driven through the FROZEN seam, not the legacy one: a suite that
    exercises ``upsert_document`` stays green while stage_candidate is
    entirely broken, which is a false green about the P0 itself."""
    class RefusingCursor:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            self.conn.statements.append(sql.split()[0])
            if sql.startswith("SELECT filename"):
                self._result = ("kurgu.pdf", True)
            elif sql.lstrip().upper().startswith("INSERT"):
                self._result = None          # the database REFUSED
            else:
                self._result = (True,)

        def fetchone(self):
            return self._result

    class Conn:
        def __init__(self):
            self.statements, self.commits, self.rollbacks = [], 0, 0

        def cursor(self):
            return RefusingCursor(self)

        def commit(self):
            self.commits += 1

        def rollback(self):
            self.rollbacks += 1

    stage = getattr(db, "stage_candidate", None)
    v = Violations()
    if not v.require(stage is not None,
                     "db.stage_candidate dikisi yok: rule 10 donmus kapi "
                     "uzerinden sinanamiyor"):
        v.assert_none()
        return

    conn = Conn()
    error, _ = _run_capturing(
        lambda: stage(conn, "kurgu.pdf", "pdf", content_sha256=SHA_A))

    v.require(isinstance(error, CandidateConflict),
              f"beklenen CandidateConflict, gelen: {error!r}")
    v.require(conn.rollbacks == 1,
              f"reddedilen teklif geri alinmadi: {conn.rollbacks} rollback")
    v.require(conn.commits == 0,
              f"reddedilen teklif commit etti: {conn.commits}")
    v.assert_none()


# =====================================================================
# DEFECT 2 -- a losing run stamps its verdict on the winner's document
# =====================================================================

@pytest.mark.parametrize(
    ("failures", "expected_stamp"),
    [
        ((), "error"),
        (({"kaynak": "page2:scanned", "asama": "sayfa",
           "hata": "kurgu baglanti hatasi"},), "partial"),
    ],
    ids=["kaybeden_error", "kaybeden_partial"],
)
def test_a_losing_run_cannot_restamp_the_winners_document(
        store_and_source, monkeypatch, failures, expected_stamp):
    """Two runs of the SAME candidate are indistinguishable to every guard:
    both carry that candidate_id, and the loser re-reads the active
    generation AFTER parsing, so it re-binds to what the winner just
    promoted. Its verdict then matches the guard and lands on a healthy
    served generation -- as ``error`` when it crashes and as ``partial``
    when pages were merely lost, the same corruption in a gentler word.

    Rules 1 and 5. This loser still HOLDS its lease; what it may not do
    is write its verdict onto the document."""
    store, source = store_and_source
    store.row.update({"content_sha256": SHA_A,
                      "candidate_id": "kurgu-aday-A",
                      "candidate_state": CandidateState.PUBLISHED})
    reached = {}

    def a_concurrent_run_wins():
        store.row.update({"active_generation": 1,
                          "active_content_sha": SHA_A,
                          "last_generation": 1, "status": "done"})

    monkeypatch.setattr(
        ingest, "route_and_parse",
        _parse_with(reached, during=a_concurrent_run_wins, failures=failures))
    if not failures:
        def exploding_write(*_a):
            reached["injected_failure"] = (
                reached.get("injected_failure", 0) + 1)
            raise RuntimeError("kurgu yazma hatasi")

        monkeypatch.setattr(ingest.db, "upsert_chunks", exploding_write)

    error, _ = _run_capturing(
        lambda: ingest.main(str(source),
                            expected_candidate=("kurgu-aday-A", SHA_A)))

    v = Violations()
    v.require(reached.get("during") == 1,
              "kazanan kosunun terfisi hic gerceklesmedi: senaryo kurulmadi")
    if failures:
        v.require(error is None, f"kismi kosu istisna atmamaliydi: {error!r}")
    else:
        v.require(reached.get("injected_failure") == 1,
                  "enjekte edilen yazma hatasina hic ulasilmadi: kaybeden "
                  "kosu hic yazmaya kalkismamis, senaryo kurulmadi")
        v.require(isinstance(error, RuntimeError),
                  f"enjekte edilen hata yutuldu: {error!r}")
    v.require(store.row["active_generation"] == 1,
              "kazananin nesli yerinden oynadi")
    v.require(store.row["active_content_sha"] == SHA_A,
              "kazananin servis ettigi baytlar degisti")
    v.require(store.row["status"] == "done",
              f"kaybeden kosu kazananin saglam neslini "
              f"'{store.row['status']}' olarak etiketledi")
    v.require(
        (expected_stamp, 1, "kurgu-aday-A") not in store.stamps,
        f"kaybedenin '{expected_stamp}' damgasi kazananin nesline yazildi")
    v.assert_none()


# =====================================================================
# DEFECT 3 -- the upload's two publication moments can contradict
# =====================================================================

def test_a_process_in_the_publish_gap_is_refused_without_touching_state(
        monkeypatch, tmp_path):
    """The upload commits the candidate row and writes the disk file at two
    different moments. A process request issued in between reads a
    candidate whose bytes are not on disk yet, refuses, and stamps the
    document error -- while the upload returns 200 pending.

    Frozen: while candidate_state is STAGED the process endpoint answers
    HTTP 409 THROUGH begin_attempt raising ``CandidateNotPublished``,
    without calling ingest and without changing any document or attempt
    state. A 404/500/503 is also "a refusal" and must NOT pass: the code
    AND the path are the contract. Also frozen: the document's status
    describes the SERVED version, so an upload does not move it, while
    the RESPONSE describes the candidate."""
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    from pipeline.api import app as api

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "kurgu.pdf").write_bytes(A_BYTES)
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    # REPAIRED IN PACKAGE 3C, second and last change to this frozen file.
    # Rule 13 moved the destination out of the endpoint and into the
    # publication service, so the directory this test has to redirect is
    # the SERVICE's -- redirecting only the endpoint's copy left the
    # bytes landing in the real upload directory while the assertions
    # below read a temporary one. Plumbing only: no assertion changes.
    publication = _import_publication()
    if publication is not None:
        monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir,
                            raising=False)

    @contextmanager
    def fake_conn():
        yield object()

    @contextmanager
    def publish_lock(_conn, _filename):
        yield

    store = DocumentStore()
    store.row.update({"content_sha256": SHA_A, "candidate_id": "kurgu-aday-1",
                      "candidate_state": CandidateState.PUBLISHED,
                      "status": "done", "active_generation": 1,
                      "active_content_sha": SHA_A})
    store.has_chunks = True

    def bound_ingest(path, expected_candidate=None, attempt=None):
        store.ingest_calls.append((path, expected_candidate, attempt))
        sha = attempt.candidate_sha if attempt else expected_candidate[1]
        with open(path, "rb") as handle:
            disk = hashlib.sha256(handle.read()).hexdigest()
        if disk != sha:
            raise RuntimeError("disk icerigi kayitli adayla uyusmuyor")

    monkeypatch.setattr(api, "db_conn", fake_conn)
    monkeypatch.setattr(api.db, "active_ingest_job",
                        lambda _conn, _document_id: None)
    monkeypatch.setattr(api.db, "document_publish_lock", publish_lock)
    for name in ("upsert_document", "lookup_document", "get_document",
                 "set_document_status", "stage_candidate",
                 "finalize_candidate_publication", "begin_attempt",
                 "record_attempt_outcome"):
        monkeypatch.setattr(api.db, name, getattr(store, name), raising=False)
    monkeypatch.setattr(api.ingest, "main", bound_ingest)

    in_the_gap, process_done = threading.Event(), threading.Event()
    real_replace = api.os.replace

    def replace_after_the_process_ran(src, dst):
        in_the_gap.set()                 # the row is committed, disk is not
        process_done.wait(timeout=5)
        return real_replace(src, dst)

    monkeypatch.setattr(api.os, "replace", replace_after_the_process_ran)

    client = TestClient(api.app)
    headers = ({"Authorization": f"Bearer {api.API_KEY}"}
               if api.API_KEY else {})
    outcome, before = {}, store.snapshot()

    def uploader():
        outcome["upload"] = client.post(
            "/documents/upload?replace=true", headers=headers,
            files={"file": ("kurgu.pdf", B_BYTES, "application/pdf")})

    def processor():
        outcome["gap_reached"] = in_the_gap.wait(timeout=5)
        outcome["process"] = client.post(
            f"/documents/{store.row['id']}/process", headers=headers)
        process_done.set()

    threads = [threading.Thread(target=uploader),
               threading.Thread(target=processor)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    v = Violations()
    v.require(all(not thread.is_alive() for thread in threads),
              "thread'ler tamamlanmadi: sonuc okunamaz")
    v.require(outcome.get("gap_reached") is True,
              "yayin araligina hic girilmedi: senaryo kurulmadi")
    v.require(outcome["upload"].status_code == 200,
              f"upload basarisiz: {outcome['upload'].status_code}")
    v.require(outcome["upload"].json().get("status") == "pending",
              "upload yanitinda aday durumu 'pending' degil")
    process = outcome.get("process")
    v.require(process is not None and process.status_code == 409,
              f"araliktaki process 409 dondurmeliydi, dondurdugu: "
              f"{None if process is None else process.status_code}")
    v.require(len(store.begin_attempt_calls) == 1,
              f"process begin_attempt'i cagirmadi (409 baska bir yoldan "
              f"gelmis olabilir): {len(store.begin_attempt_calls)} cagri")
    v.require(any(isinstance(e, CandidateNotPublished)
                  for e in store.begin_attempt_errors),
              "409 CandidateNotPublished yolundan gelmedi")
    v.require(store.ingest_calls == [],
              f"yayimlanmamis aday icin ingest cagrildi: "
              f"{len(store.ingest_calls)} kez")
    v.require(store.row["status"] == before["row"]["status"],
              f"belge durumu degisti: {before['row']['status']} -> "
              f"{store.row['status']} (belge durumu SERVIS EDILEN surumu "
              f"anlatir; upload onu oynatmamali)")
    v.require(store.attempt_outcomes == {},
              "yayimlanmamis aday icin attempt sonucu yazildi")
    v.require(store.row["candidate_state"] == CandidateState.PUBLISHED,
              f"upload bitti ama aday yayimlanmis degil: "
              f"{store.row['candidate_state']}")
    disk_sha = hashlib.sha256(
        (upload_dir / "kurgu.pdf").read_bytes()).hexdigest()
    v.require(disk_sha == store.row["content_sha256"],
              "disk hash'i kayitli adayla eslesmiyor")
    v.assert_none()


# =====================================================================
# THE MODULE SPLIT AND THE CLI -- rule 7
# =====================================================================

def test_core_ingest_requires_an_attempt(store_and_source):
    """``ingest_attempt(snapshot, attempt)`` is the ONLY way into the
    index, and the attempt is mandatory. This is a different callable
    from the CLI entry point on purpose: one function cannot both refuse
    to run unbound and publish-then-begin."""
    core = getattr(ingest, "ingest_attempt", None)
    v = Violations()
    if v.require(core is not None,
                 "ingest.ingest_attempt dikisi yok: cekirdek/CLI ayrimi "
                 "henuz yapilmadi"):
        parameters = inspect.signature(core).parameters
        v.require("attempt" in parameters,
                  f"ingest_attempt bir attempt almiyor: {list(parameters)}")
        v.require(parameters["attempt"].default is inspect.Parameter.empty,
                  "attempt zorunlu degil: varsayilani var, yani bagsiz "
                  "cagrilabiliyor")
    v.assert_none()


def _import_publication():
    try:
        from pipeline.index import publication
    except ImportError:
        return None
    return publication


class Call:
    """One recorded seam call: its arguments AND what it returned.

    Recording only the inputs was a false RED: the chain test then
    compared the ingest_attempt argument against ``begin_attempt``'s
    INPUT tuple, so a correct CLI handing over the returned attempt
    failed. What the chain must preserve is the RETURNED object."""

    def __init__(self, name, args, kwargs):
        self.name, self.args, self.kwargs = name, args, kwargs
        self.result = None
        self.error = None


def _record_chain(monkeypatch, store, publication):
    order = []
    targets = [(db, "stage_candidate"),
               (db, "finalize_candidate_publication"),
               (db, "begin_attempt"),
               (db, "upsert_document"),
               (ingest, "ingest_attempt")]
    if publication is not None:
        targets.append((publication, "publish_candidate"))
    for module, name in targets:
        original = getattr(store, name, None) or getattr(module, name, None)

        def recorder(*args, _name=name, _original=original, **kwargs):
            call = Call(_name, args, kwargs)
            order.append(call)
            if _original is None:
                raise AssertionError(f"{_name} yok")
            try:
                call.result = _original(*args, **kwargs)
            except BaseException as error:      # noqa: BLE001
                call.error = error
                raise
            return call.result

        monkeypatch.setattr(module, name, recorder, raising=False)
    return order


def _handed_attempt(call):
    """The attempt passed to ingest_attempt(snapshot, attempt)."""
    if "attempt" in call.kwargs:
        return call.kwargs["attempt"]
    return call.args[1] if len(call.args) > 1 else None


def test_the_cli_runs_the_whole_chain_with_one_attempt(store_and_source,
                                                       monkeypatch):
    """publish_candidate -> begin_attempt -> ingest_attempt, in that order,
    and the attempt handed to the core is THE VERY OBJECT begin_attempt
    RETURNED -- identity, not a lookalike.

    An earlier version only looked for the first two names, so a CLI that
    published, began an attempt and then raised before indexing anything
    passed. Success without the core call is not success."""
    store, source = store_and_source
    store.row.update({"candidate_state": CandidateState.PUBLISHED})
    publication = _import_publication()
    order = _record_chain(monkeypatch, store, publication)
    monkeypatch.setattr(ingest, "route_and_parse", _parse_with({}))

    error, code = _run_capturing(lambda: run_cli(source))

    called = [call.name for call in order]
    v = Violations()
    v.require(publication is not None,
              "pipeline.index.publication modulu yok: ortak yayin servisi "
              "henuz ayrilmadi")
    v.require("upsert_document" not in called,
              "CLI aday kapisini dogrudan caldi (rule 7)")
    for name in ("publish_candidate", "begin_attempt", "ingest_attempt"):
        v.require(name in called, f"CLI {name} cagirmadi; cagrilar: {called}")
    if {"publish_candidate", "begin_attempt", "ingest_attempt"} <= set(called):
        v.require(called.index("publish_candidate")
                  < called.index("begin_attempt")
                  < called.index("ingest_attempt"),
                  f"zincir sirasi yanlis: {called}")
        began = next(c for c in order if c.name == "begin_attempt")
        handed = _handed_attempt(
            next(c for c in order if c.name == "ingest_attempt"))
        v.require(handed is began.result,
                  "cekirdege begin_attempt'in DONDURDUGU attempt "
                  "verilmedi (kimlik karsilastirmasi)")
    v.require(error is None,
              f"CLI sozlesme reddini istisna olarak firlatti: {error!r} "
              f"(refuzal bir DONUS DEGERIDIR)")
    v.require(code == ExitCode.OK,
              f"basarili zincir {ExitCode.OK} dondurmeliydi, gelen: {code!r}")
    v.assert_none()


@pytest.mark.parametrize(
    ("replace", "expected_code", "label"),
    [(False, ExitCode.CANDIDATE_CONFLICT, "yetkisiz degisim"),
     (True, ExitCode.OK, "acik yetkiyle degisim")],
)
def test_the_cli_replace_flag_is_behavioural_and_answers_with_exit_codes(
        store_and_source, monkeypatch, replace, expected_code, label):
    """Different content is accepted only with an explicit --replace, and
    the CLI ANSWERS WITH A CODE rather than a traceback. The flag is RUN,
    not grepped: a source comment mentioning --replace would satisfy a
    textual check while the CLI silently replaced everything."""
    store, source = store_and_source
    store.row.update({"content_sha256": SHA_A, "candidate_id": "kurgu-aday-A",
                      "candidate_state": CandidateState.PUBLISHED,
                      "active_generation": 1, "active_content_sha": SHA_A,
                      "status": "done"})
    store.has_chunks = True
    source.write_bytes(B_BYTES)                  # DIFFERENT content
    monkeypatch.setattr(ingest, "route_and_parse", _parse_with({}))

    v = Violations()
    if not v.require(getattr(ingest, "cli_main", None) is not None,
                     "ingest.cli_main dikisi yok: bayrak davranissal olarak "
                     "sinanamiyor"):
        v.assert_none()
        return

    before = store.snapshot()
    error, code = _run_capturing(lambda: run_cli(source, replace=replace))
    v.require(error is None,
              f"{label}: CLI istisna firlatti, kod dondurmedi: {error!r}")
    v.require(code == expected_code,
              f"{label}: beklenen cikis kodu {expected_code}, gelen {code!r}")
    if expected_code == ExitCode.CANDIDATE_CONFLICT:
        v.require(store.snapshot()["row"]["content_sha256"]
                  == before["row"]["content_sha256"],
                  "yetkisiz kosu adayi degistirdi")
    else:
        v.require(store.row["content_sha256"] == SHA_B,
                  "acik --replace yeni adayi kaydetmedi")
    v.assert_none()


def test_there_is_no_implicit_environment_replacement(store_and_source,
                                                      monkeypatch):
    """Replacement authority comes from the caller, never from a global
    environment flag granting it to every document at once. Checked both
    ways: the name must be gone from the module, AND setting it must not
    change the CLI's answer."""
    store, source = store_and_source
    store.row.update({"content_sha256": SHA_A, "candidate_id": "kurgu-aday-A",
                      "candidate_state": CandidateState.PUBLISHED,
                      "active_generation": 1, "active_content_sha": SHA_A,
                      "status": "done"})
    store.has_chunks = True
    source.write_bytes(B_BYTES)
    monkeypatch.setattr(ingest, "route_and_parse", _parse_with({}))
    monkeypatch.setenv("INGEST_ALLOW_REPLACE", "1")

    v = Violations()
    body = Path(ingest.__file__).read_text(encoding="utf-8")
    v.require("INGEST_ALLOW_REPLACE" not in body,
              "ortam degiskeniyle ortuk replacement hala kaynakta var")
    if getattr(ingest, "cli_main", None) is not None:
        _error, code = _run_capturing(lambda: run_cli(source, replace=False))
        v.require(code == ExitCode.CANDIDATE_CONFLICT,
                  f"ortam degiskeni acik bayragin yerine gecti: kod {code!r}")
    else:
        v.require(False, "ingest.cli_main dikisi yok")
    v.assert_none()


def test_the_publisher_writes_where_the_database_says_not_where_asked(
        monkeypatch, tmp_path):
    """Rule 13, BEHAVIOURALLY: the database canonicalises a re-cased name,
    and the bytes must follow the ROW, not the request. A parameter
    blacklist was the earlier check and it let ``disk_location`` through;
    an exact signature plus a real call cannot be worked around by
    naming."""
    publication = _import_publication()
    v = Violations()
    if not v.require(publication is not None,
                     "pipeline.index.publication modulu yok"):
        v.assert_none()
        return
    publish = getattr(publication, "publish_candidate", None)
    if not v.require(publish is not None,
                     "publication.publish_candidate yok"):
        v.assert_none()
        return

    problem = _signature_matches(
        publish, ("conn", "filename", "file_type", "body",
                  "allow_replace", "tenant_id"),
        "publish_candidate")
    v.require(problem is None, f"publish_candidate: {problem}")

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir, raising=False)
    # a canonical name that CANNOT be derived from the request by any
    # string rule: a publisher that lowercases what it was given writes
    # the wrong file and fails
    canonical = "kurgu-kanonik.pdf"
    store = DocumentStore(filename=canonical)
    counts = {"stage": 0, "finalize": 0, "lock": 0}

    def canonicalising_stage(_conn, filename, file_type,
                             content_sha256=None, allow_replace=False):
        counts["stage"] += 1
        document_id, candidate_id, _name = store.stage_candidate(
            _conn, canonical, file_type, content_sha256=content_sha256,
            allow_replace=allow_replace)
        return document_id, candidate_id, canonical

    def counting_finalize(_conn, document_id, candidate_id):
        counts["finalize"] += 1
        return store.finalize_candidate_publication(_conn, document_id,
                                                    candidate_id)

    from contextlib import contextmanager

    @contextmanager
    def counting_lock(_conn, _filename):
        counts["lock"] += 1
        yield

    monkeypatch.setattr(db, "stage_candidate", canonicalising_stage,
                        raising=False)
    monkeypatch.setattr(db, "finalize_candidate_publication",
                        counting_finalize, raising=False)
    monkeypatch.setattr(db, "document_publish_lock", counting_lock,
                        raising=False)

    class Conn:
        """A connection that supports the lock the real publisher takes --
        handing it a bare object() would fail a CORRECT publisher."""

        def cursor(self):
            raise AssertionError("yayin servisi ham SQL calistirdi")

        def commit(self):
            pass

        def rollback(self):
            pass

    error, result = _run_capturing(
        lambda: publish(Conn(), "KURGU.PDF", "pdf", B_BYTES,
                        allow_replace=True))

    v.require(error is None, f"yayin servisi hata verdi: {error!r}")
    v.require(counts == {"stage": 1, "finalize": 1, "lock": 1},
              f"dikisler tam birer kez cagrilmadi: {counts}")
    written = sorted(p.name for p in upload_dir.iterdir())
    v.require(written == [canonical],
              f"baytlar kanonik hedefe yazilmadi (ya da gecici dosya "
              f"kaldi): {written}")
    if written == [canonical]:
        v.require((upload_dir / canonical).read_bytes() == B_BYTES,
                  "kanonik hedefe YANLIS baytlar yazildi")
    v.require(store.row["candidate_state"] == CandidateState.PUBLISHED,
              f"aday yayimlanmis degil: {store.row['candidate_state']}")
    v.require(isinstance(result, tuple) and len(result) == 3
              and result[2] == canonical,
              f"yayin servisi kanonik adi dondurmedi: {result!r}")
    v.assert_none()


@pytest.mark.parametrize(
    "hostile",
    ["../disari.pdf", "alt/disari.pdf", "/mutlak.pdf",
     # Windows spellings: a POSIX-only check passes every one of these
     # while the write still escapes on the platform this runs on
     "..\\disari.pdf", "C:\\disari.pdf", "\\\\sunucu\\pay\\disari.pdf"])
def test_the_publisher_revalidates_the_canonical_name(monkeypatch, tmp_path,
                                                      hostile):
    """The canonical name comes from the database, which makes it TRUSTED
    INPUT -- and trusted input is still input. A row carrying a path
    escape must not become a write outside the upload directory: the name
    is re-checked for basename-ness and containment BEFORE any byte
    lands.

    "Before any byte" is the part the earlier version did not test: it
    only checked that two particular final names did not appear, so a
    publisher that wrote the whole body to a temp file and THEN raised
    passed. The upload directory must be entirely empty afterwards --
    no final file, no temp file -- and the candidate must not have been
    marked published."""
    publication = _import_publication()
    v = Violations()
    if not v.require(publication is not None,
                     "pipeline.index.publication modulu yok"):
        v.assert_none()
        return
    publish = getattr(publication, "publish_candidate", None)
    if not v.require(publish is not None,
                     "publication.publish_candidate yok"):
        v.assert_none()
        return

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir, raising=False)
    store = DocumentStore(filename="kurgu.pdf")

    def hostile_stage(_conn, filename, file_type, content_sha256=None,
                      allow_replace=False):
        document_id, candidate_id, _name = store.stage_candidate(
            _conn, "kurgu.pdf", file_type, content_sha256=content_sha256,
            allow_replace=allow_replace)
        return document_id, candidate_id, hostile

    from contextlib import contextmanager

    @contextmanager
    def lock(_conn, _filename):
        yield

    monkeypatch.setattr(db, "stage_candidate", hostile_stage, raising=False)
    monkeypatch.setattr(db, "finalize_candidate_publication",
                        store.finalize_candidate_publication, raising=False)
    monkeypatch.setattr(db, "document_publish_lock", lock, raising=False)

    error, _result = _run_capturing(
        lambda: publish(object(), "kurgu.pdf", "pdf", A_BYTES,
                        allow_replace=True))

    v.require(error is not None,
              f"yol kacisi tasiyan kanonik ad kabul edildi: {hostile!r}")
    # NOTHING may have been written -- not the final file, not a temp
    # file the publisher meant to rename later
    leftovers = sorted(p.name for p in upload_dir.iterdir())
    v.require(leftovers == [],
              f"upload dizini bos degil (gecici dosya da sayilir): "
              f"{leftovers}")
    escaped = [p for p in tmp_path.rglob("*")
               if p.is_file() and p.parent != upload_dir]
    v.require(not escaped,
              f"upload dizini disina yazildi: {[str(p) for p in escaped]}")
    v.require(store.row["candidate_state"] != CandidateState.PUBLISHED,
              "reddedilen yayin adayi PUBLISHED yapti")
    v.assert_none()


def test_the_api_publishes_only_through_the_shared_service(monkeypatch,
                                                           tmp_path):
    """The API must not keep its own copy of the publication sequence: one
    service, one lock order, one crash-window recovery.

    Checked by CALLING the endpoint, not by grepping the source: a
    comment mentioning publish_candidate satisfied the earlier text
    check. The shared publisher must be called exactly once and the
    low-level seams (stage/finalize/os.replace) exactly zero times by
    the API itself."""
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    publication = _import_publication()
    v = Violations()
    if not v.require(publication is not None,
                     "pipeline.index.publication modulu yok"):
        v.assert_none()
        return

    from pipeline.api import app as api

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)

    @contextmanager
    def fake_conn():
        yield object()

    @contextmanager
    def publish_lock(_conn, _filename):
        yield

    store = DocumentStore()
    counts = {"publish": 0, "stage": 0, "finalize": 0, "replace": 0,
              "lock": 0, "upsert": 0}

    def counted_publish(_conn, filename, file_type, body,
                        allow_replace=False, tenant_id=db.DEFAULT_TENANT_ID):
        counts["publish"] += 1
        document_id, candidate_id, canonical = store.stage_candidate(
            None, filename, file_type,
            content_sha256=hashlib.sha256(body).hexdigest(),
            allow_replace=allow_replace)
        store.finalize_candidate_publication(None, document_id, candidate_id)
        (upload_dir / canonical).write_bytes(body)
        return document_id, candidate_id, canonical

    # patched on BOTH the service module and any alias the endpoint may
    # have imported directly: patching only the module would miss a
    # `from ... import publish_candidate` and fail a correct API
    for target in (publication, api):
        if hasattr(target, "publish_candidate"):
            monkeypatch.setattr(target, "publish_candidate", counted_publish)
    monkeypatch.setattr(publication, "publish_candidate", counted_publish)
    monkeypatch.setattr(api, "db_conn", fake_conn)

    @contextmanager
    def counting_lock(_conn, _filename):
        counts["lock"] += 1
        yield

    monkeypatch.setattr(api.db, "document_publish_lock", counting_lock)
    for name, key in (("stage_candidate", "stage"),
                      ("finalize_candidate_publication", "finalize"),
                      ("upsert_document", "upsert")):
        def counting(*_a, _key=key, **_k):
            counts[_key] += 1
            raise AssertionError(f"API dusuk seviye dikisi cagirdi: {_key}")

        monkeypatch.setattr(api.db, name, counting, raising=False)
    for name in ("lookup_document", "get_document", "set_document_status"):
        monkeypatch.setattr(api.db, name, getattr(store, name),
                            raising=False)

    def counting_replace(src, dst):
        counts["replace"] += 1
        raise AssertionError("API kendi disk yayinini yapti")

    monkeypatch.setattr(api.os, "replace", counting_replace)

    client = TestClient(api.app)
    headers = ({"Authorization": f"Bearer {api.API_KEY}"}
               if api.API_KEY else {})
    response = client.post(
        "/documents/upload", headers=headers,
        files={"file": ("kurgu.pdf", A_BYTES, "application/pdf")})

    v.require(counts["publish"] == 1,
              f"ortak yayin servisi tam bir kez cagrilmadi: "
              f"{counts['publish']}")
    v.require(counts["stage"] == 0 and counts["finalize"] == 0,
              f"API dusuk seviye yayin dikislerini cagirdi: {counts}")
    v.require(counts["upsert"] == 0,
              "API hala legacy aday kapisini caliyor (upsert_document)")
    v.require(counts["replace"] == 0,
              "API hala kendi disk yayinini yapiyor (os.replace)")
    v.require(counts["lock"] == 0,
              f"yayin kilidi API'de aliniyor: kilit yayin servisinindir "
              f"({counts['lock']} kez)")
    v.require(response.status_code == 200,
              f"upload basarisiz: {response.status_code}")
    v.assert_none()


def _signature_matches(function, expected, seam_name):
    """Exact ordered signature: names, order, kind, and which parameters
    may carry a default.

    The required-parameter count is keyed on the SEAM NAME we looked the
    function up by -- an earlier version keyed it on ``function.__name__``,
    so a function whose internal name differed from its seam name got a
    required count of zero and could make every parameter optional.

    ``allow_replace`` is checked by name wherever it appears: its default
    may only be False. A default of True hands every caller the
    replacement authority the whole rule exists to withhold."""
    if not callable(function):
        return f"callable degil: {type(function).__name__}"
    parameters = list(inspect.signature(function).parameters.values())
    names = [p.name for p in parameters]
    if names != list(expected):
        return f"imza birebir degil: beklenen {list(expected)}, bulunan {names}"
    for parameter in parameters:
        if parameter.kind is not parameter.POSITIONAL_OR_KEYWORD:
            return f"{parameter.name} konumsal-veya-anahtar degil"
    if seam_name not in _REQUIRED_PREFIX:
        return f"dikis icin zorunlu parametre sayisi tanimsiz: {seam_name}"
    for parameter in parameters[:_REQUIRED_PREFIX[seam_name]]:
        if parameter.default is not inspect.Parameter.empty:
            return f"{parameter.name} zorunlu olmali, varsayilani var"
    for parameter in parameters:
        if parameter.name == "allow_replace" and parameter.default is not False:
            return (f"allow_replace varsayilani False olmali, bulunan "
                    f"{parameter.default!r}: acik yetki kurali bozulur")
    return None


_REQUIRED_PREFIX = {
    "stage_candidate": 3,
    "finalize_candidate_publication": 3,
    "begin_attempt": 2,
    "heartbeat_attempt": 2,
    "record_attempt_outcome": 3,
    "ingest_attempt": 2,
    "cli_main": 1,
    "publish_candidate": 4,
}


def test_the_frozen_seams_have_exact_signatures():
    """The seam SURFACE -- existence, callability, exact parameter order,
    and required parameters without defaults. What each seam DOES is
    checked against real PostgreSQL; asserting behaviour against a model
    here would pass an empty implementation."""
    publication = _import_publication()
    expected = [
        # returns (document_id, candidate_id, canonical_filename): the
        # publisher's disk target comes from that third value
        (db, "stage_candidate",
         ("conn", "filename", "file_type", "content_sha256",
          "allow_replace")),
        (db, "finalize_candidate_publication",
         ("conn", "document_id", "candidate_id")),
        (db, "begin_attempt",
         ("conn", "document_id", "owner", "ingest_job_id",
          "ingest_job_worker")),
        (db, "heartbeat_attempt", ("conn", "attempt")),
        (db, "record_attempt_outcome",
         ("conn", "attempt", "status", "note")),
        (ingest, "ingest_attempt", ("snapshot", "attempt")),
        (ingest, "cli_main", ("argv",)),
    ]
    v = Violations()
    for module, name, parameters in expected:
        function = getattr(module, name, None)
        if not v.require(function is not None,
                         f"{module.__name__}.{name} dikisi yok"):
            continue
        problem = _signature_matches(function, parameters, name)
        v.require(problem is None, f"{name}: {problem}")
    if v.require(publication is not None,
                 "pipeline.index.publication modulu yok"):
        publish = getattr(publication, "publish_candidate", None)
        if v.require(publish is not None,
                     "publication.publish_candidate yok"):
            problem = _signature_matches(
                publish,
                ("conn", "filename", "file_type", "body",
                 "allow_replace", "tenant_id"),
                "publish_candidate")
            v.require(problem is None, f"publish_candidate: {problem}")
    v.assert_none()
