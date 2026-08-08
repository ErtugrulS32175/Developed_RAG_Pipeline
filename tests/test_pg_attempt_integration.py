"""PERMANENT PostgreSQL integration tests for the attempt contract.

Not a scratch script, and the ONLY place database behaviour is asserted.
The candidate gate, the lease and the promotion CAS are SQL properties: a
fake cursor that re-implements the acceptance formula cannot tell a fixed
statement from a broken one, and a model of the rules would pass an empty
implementation. So the claims that live in SQL are made HERE.

ISOLATION -- this module never touches anything it did not create. Every
run creates a schema named ``ragtest_p0_<random>``; every connection is
pinned to it with ``search_path``, so the unqualified table names in
schema.sql and in db.py resolve INSIDE it. Teardown rolls back first and
then drops that schema over a SEPARATE connection, so a failure during
setup cannot leave the drop running inside a poisoned transaction.
``public.documents`` and ``public.chunks`` are never named, never
dropped, never written -- a fixture that DROPs tables in whatever
database the DSN points at is not made safe by a docstring asking for a
throwaway server.

RUNNING THEM -- one official command::

    scripts/p0_gate.sh

which creates a disposable cluster, exports the DSN and runs this module
with the gate enabled. Manually::

    RAGTEST_PG_TEST_DSN=postgresql://postgres@127.0.0.1:55433/p0probe \\
    RAGTEST_P0_GATE=1 python -m pytest tests/test_pg_attempt_integration.py

THE OFFICIAL P0 GATE requires these tests to RUN: with
``RAGTEST_P0_GATE=1`` a missing DSN is a FAILURE, not a skip. The gate
may not be passed by a suite that quietly declined to check the half of
the contract living in the database.

DECLARED DEVIATION, one, narrow, and unconditional: the two vector-typed
columns are ALWAYS created as text here. Nothing in the candidate or
attempt contract reads them, and making the P0 gate depend on whether
this particular server has pgvector installed would make the gate's
answer depend on something the contract does not care about. The fixture
asserts that the substitution left the contract's own DDL untouched.
``db.get_conn`` registers the pgvector type, so connections are made with
psycopg directly: only the FACTORY is substituted, and every db.py
function under test takes a connection and runs its real SQL.
"""
import hashlib
import os
import re
import threading
import time
import uuid
from pathlib import Path

import pytest

from pipeline.index import db
from pipeline.index.attempt_contract import (
    AttemptAlreadyRunning,
    AttemptFenced,
    AttemptLeaseLost,
    AttemptOutcome,
    AttemptOutcomeNotWritable,
    AttemptRecordInconsistent,
    CandidateConflict,
    CandidateNotPublished,
    CandidateState,
)

DSN = os.getenv("RAGTEST_PG_TEST_DSN", "").strip()
GATE = os.getenv("RAGTEST_P0_GATE", "").strip() == "1"

if GATE and not DSN:
    raise RuntimeError(
        "RAGTEST_P0_GATE=1 ama RAGTEST_PG_TEST_DSN yok: resmi P0 kapisi "
        "veritabani iddialarini ATLAYARAK gecilemez")

pytestmark = pytest.mark.skipif(
    not DSN,
    reason="RAGTEST_PG_TEST_DSN tanimli degil: gercek PostgreSQL iddialari "
           "KONTROL EDILMEDI (atlanmak, gecmek degildir)")

A_BYTES = b"KURGU_SURUM_A"
B_BYTES = b"KURGU_SURUM_B"
SHA_A = hashlib.sha256(A_BYTES).hexdigest()
SHA_B = hashlib.sha256(B_BYTES).hexdigest()

CONTRACT_DDL_TOKENS = ("candidate_id", "active_generation", "last_generation",
                       "active_content_sha", "content_sha256", "content_key")


def _probe_schema() -> str:
    """schema.sql with the vector columns typed as text -- always, so the
    gate's verdict never depends on a server-side extension the contract
    does not use."""
    raw = (Path(db.__file__).parent / "schema.sql").read_text(
        encoding="utf-8")
    body = raw
    for pattern, replacement in (
        (r"CREATE EXTENSION IF NOT EXISTS vector;",
         "-- [P0 KAPISI] vektor tipleri text'e ikame edildi"),
        (r"dense\s+vector\(1024\) NOT NULL", "dense       text NOT NULL"),
        (r"sparse\s+sparsevec\(999999937\) NOT NULL",
         "sparse      text NOT NULL"),
    ):
        body = re.sub(pattern, replacement, body)
    for token in CONTRACT_DDL_TOKENS:
        assert raw.count(token) == body.count(token), (
            f"ikame sozlesme DDL'ine dokundu: {token}")
    return body


def _raw_connect():
    import psycopg

    return psycopg.connect(DSN)


@pytest.fixture(scope="module")
def isolated_schema():
    """A private schema for this run; dropped whole at the end, over a
    connection that cannot be inside a failed transaction."""
    name = f"ragtest_p0_{uuid.uuid4().hex[:12]}"
    assert name != "public"
    setup = _raw_connect()
    try:
        with setup.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{name}"')
        setup.commit()
        with setup.cursor() as cur:
            cur.execute(f'SET search_path TO "{name}"')
            cur.execute(_probe_schema())
        setup.commit()
        with setup.cursor() as cur:
            cur.execute("SELECT count(*) FROM information_schema.tables "
                        "WHERE table_schema = %s AND table_name IN "
                        "('documents', 'chunks')", (name,))
            assert cur.fetchone()[0] == 2, "izole schema kurulmadi"
        setup.commit()
        yield {"schema": name}
    finally:
        try:
            setup.rollback()          # a failed setup poisons this one
        except Exception:
            pass
        try:
            setup.close()
        except Exception:
            pass
        cleanup = _raw_connect()      # a SEPARATE, clean connection drops
        try:
            with cleanup.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{name}" CASCADE')
            cleanup.commit()
        finally:
            cleanup.close()


def _connect(schema):
    connection = _raw_connect()
    with connection.cursor() as cur:
        cur.execute(f'SET search_path TO "{schema}"')
    connection.commit()
    return connection


@pytest.fixture
def connect(isolated_schema):
    """A factory: concurrency tests need MORE THAN ONE connection, and a
    single shared one would serialise the very race under test."""
    opened = []

    def factory():
        connection = _connect(isolated_schema["schema"])
        opened.append(connection)
        return connection

    yield factory
    for connection in opened:
        try:
            connection.close()
        except Exception:
            pass


@pytest.fixture
def conn(connect):
    return connect()


@pytest.fixture
def filename():
    return f"kurgu-{uuid.uuid4().hex[:10]}.pdf"


def seam(name):
    return getattr(db, name, None)


def _has_column(conn, table, column):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "AND column_name = %s", (table, column))
        return cur.fetchone()[0] > 0


def _insert_chunk(conn, document_id, generation, chunk_id=None):
    chunk_id = chunk_id or str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunks (id, document_id, type, text, source_tag, "
            "page, dense, sparse, generation, content_key) VALUES "
            "(%s, %s, 'text', 'kurgu parca', 'page1:native', 1, '', '', "
            "%s, %s)",
            (chunk_id, document_id, generation, str(uuid.uuid4())))
    conn.commit()
    return chunk_id


def _served_row(conn, filename, served_sha, candidate_sha,
                candidate_state=CandidateState.PUBLISHED):
    """A document SERVING one version, with a candidate recorded.

    ``candidate_state`` is written only WHEN THE COLUMN EXISTS: the
    contract requires it and one test says so plainly, but a missing
    column must not make every other test fail during setup instead of
    on its own claim."""
    document_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, filename, file_type, status, "
            "content_sha256, candidate_id, active_generation, "
            "last_generation, active_content_sha) VALUES "
            "(%s, %s, 'pdf', 'done', %s, %s, 1, 1, %s)",
            (document_id, filename, candidate_sha, str(uuid.uuid4()),
             served_sha))
    conn.commit()
    if candidate_state is not None and _has_column(conn, "documents",
                                                   "candidate_state"):
        with conn.cursor() as cur:
            cur.execute("UPDATE documents SET candidate_state = %s "
                        "WHERE id = %s", (candidate_state, document_id))
        conn.commit()
    _insert_chunk(conn, document_id, 1)
    return document_id


def _row(conn, document_id, *fields):
    """Read the named columns, skipping any the schema does not have yet."""
    present = [f for f in fields if _has_column(conn, "documents", f)]
    if not present:
        return {}
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(present)} FROM documents "
                    f"WHERE id = %s", (document_id,))
        return dict(zip(present, cur.fetchone()))


FULL = ("status", "content_sha256", "candidate_id", "active_generation",
        "active_content_sha", "candidate_state", "attempt_id",
        "attempt_owner", "attempt_expires_at")


def _full_snapshot(conn, document_id):
    """EVERY column of every row the promotion could touch, as jsonb.

    A hand-picked column list cannot support a "nothing changed" claim --
    it only says the columns someone remembered did not change. to_jsonb
    of the whole row does, and it keeps working when the implementation
    adds the contract's missing columns."""
    snapshot = {}
    with conn.cursor() as cur:
        cur.execute("SELECT to_jsonb(d) FROM documents d WHERE id = %s",
                    (document_id,))
        row = cur.fetchone()
        snapshot["document"] = row[0] if row else None
        cur.execute("SELECT to_jsonb(c) FROM chunks c WHERE document_id = %s "
                    "ORDER BY id", (document_id,))
        snapshot["chunks"] = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'attempts'")
        if cur.fetchone()[0]:
            cur.execute("SELECT to_jsonb(a) FROM attempts a "
                        "WHERE document_id = %s ORDER BY attempt_id",
                        (document_id,))
            snapshot["attempts"] = [r[0] for r in cur.fetchall()]
    return snapshot


# =====================================================================
# PACKAGE 1 -- the candidate gate, through the FROZEN seam
# =====================================================================

def test_rule_10_stage_candidate_refuses_the_served_hash_while_another_is_pending(
        conn, filename):
    """Version A is SERVED, candidate B is recorded. An offer of A without
    replace authority must be REFUSED: accepting it is exactly how a stale
    CLI run reverted a newer authorised upload.

    Driven through ``stage_candidate``, the one gate a candidate may
    enter by. An earlier suite drove the legacy ``upsert_document``, so a
    completely broken stage_candidate would have stayed green -- a false
    green about the P0 itself."""
    stage = seam("stage_candidate")
    assert stage is not None, "db.stage_candidate dikisi yok"
    document_id = _served_row(conn, filename, served_sha=SHA_A,
                              candidate_sha=SHA_B)
    before = _full_snapshot(conn, document_id)

    with pytest.raises(CandidateConflict):
        stage(conn, filename, "pdf", content_sha256=SHA_A)

    assert _full_snapshot(conn, document_id) == before, (
        "reddedilen teklif satiri degistirdi")


def test_rule_10_explicit_replacement_still_passes(conn, filename):
    """The refusal must not become a wall: with authority it proceeds."""
    stage = seam("stage_candidate")
    assert stage is not None, "db.stage_candidate dikisi yok"
    document_id = _served_row(conn, filename, served_sha=SHA_A,
                              candidate_sha=SHA_B)
    stage(conn, filename, "pdf", content_sha256=SHA_A, allow_replace=True)
    assert _row(conn, document_id, "content_sha256")["content_sha256"] == SHA_A


# =====================================================================
# PACKAGE 2 -- the attempt lifecycle
# =====================================================================

def test_rule_9_two_truly_concurrent_begin_attempts_leave_one_holder(
        connect, filename):
    """TWO connections release a barrier together and call begin_attempt at
    the same moment. Sequential calls would prove only that the second
    sees the first's committed row; the contract is about the race."""
    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    document_id = _served_row(connect(), filename, SHA_A, SHA_A)

    barrier = threading.Barrier(2)
    results = []

    def racer():
        connection = connect()
        barrier.wait(timeout=10)
        try:
            results.append(("ok", begin(connection, document_id)))
        except BaseException as error:      # noqa: BLE001
            results.append(("hata", error))

    threads = [threading.Thread(target=racer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    assert all(not thread.is_alive() for thread in threads)
    kinds = [kind for kind, _ in results]
    assert kinds.count("ok") == 1, f"tam olarak bir sahip olmali: {results}"
    refused = [value for kind, value in results if kind == "hata"]
    assert isinstance(refused[0], AttemptAlreadyRunning), refused[0]


def test_rule_12_the_lease_expires_on_the_database_clock(conn, filename):
    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    first = begin(conn, document_id)
    _expire(conn, document_id)
    second = begin(conn, document_id)
    assert second.attempt_id != first.attempt_id


def test_rule_12_the_workers_clock_has_no_authority(conn, filename,
                                                    monkeypatch):
    """Skewing the WORKER's clock an hour forward must not expire anything:
    the lease is compared against the DATABASE clock, so a live lease
    stays live and a second begin_attempt is still refused.

    EVERY client-side clock source is skewed, not just ``time.time`` --
    an implementation reading datetime.now(), utcnow() or monotonic()
    would have walked past a single-source patch."""
    import datetime as datetime_module

    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    begin(conn, document_id)

    offset = 3600.0
    real_time, real_monotonic = time.time, time.monotonic
    real_datetime = datetime_module.datetime
    skew = datetime_module.timedelta(seconds=offset)

    class SkewedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz) + skew

        @classmethod
        def utcnow(cls):
            return real_datetime.utcnow() + skew

    # Patch the modules AND any alias the implementation imported into
    # its own namespace: `from time import time` or `from datetime import
    # datetime` in db.py would walk straight past a module-level patch,
    # which is how a single-source skew proves nothing.
    for module in (time, datetime_module, db):
        if getattr(module, "time", None) is not None and module is not db:
            monkeypatch.setattr(module, "time",
                                lambda: real_time() + offset, raising=False)
        if getattr(module, "monotonic", None) is not None:
            monkeypatch.setattr(module, "monotonic",
                                lambda: real_monotonic() + offset,
                                raising=False)
        if getattr(module, "datetime", None) is not None:
            monkeypatch.setattr(module, "datetime", SkewedDatetime,
                                raising=False)
    if getattr(db, "time", None) is not None:
        monkeypatch.setattr(db, "time", time, raising=False)

    with pytest.raises(AttemptAlreadyRunning):
        begin(conn, document_id)


def _expire(conn, document_id):
    with conn.cursor() as cur:
        cur.execute("UPDATE documents SET attempt_expires_at = now() - "
                    "interval '1 second' WHERE id = %s", (document_id,))
    conn.commit()


def test_rule_6_takeover_closes_the_old_attempt_and_silences_its_worker(
        conn, filename):
    """At takeover the SYSTEM closes the displaced attempt as SUPERSEDED --
    the record is not left dangling and is not written by the displaced
    worker, which from that moment may write nothing at all."""
    begin, record = seam("begin_attempt"), seam("record_attempt_outcome")
    assert begin is not None and record is not None, (
        "db.begin_attempt / db.record_attempt_outcome dikisi yok")
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    displaced = begin(conn, document_id)

    # the manifest is staged BEFORE the takeover, while this attempt
    # still has authority, so the refusal below can only be the attempt
    # CAS -- not a manifest mismatch wearing its clothes. (Staging it
    # afterwards is now impossible anyway: package 3B made every write
    # seam refuse a displaced worker, which is strictly stronger.)
    generation = db.allocate_generation(conn, document_id, displaced)
    manifest = {_insert_chunk(conn, document_id, generation)}

    _expire(conn, document_id)
    begin(conn, document_id)                      # takeover

    with conn.cursor() as cur:
        cur.execute("SELECT status FROM attempts WHERE attempt_id = %s",
                    (displaced.attempt_id,))
        closed = cur.fetchone()
    assert closed and closed[0] == AttemptOutcome.SUPERSEDED, (
        "devralma eski denemeyi SUPERSEDED olarak kapatmadi")

    with pytest.raises(AttemptLeaseLost):
        record(conn, displaced, AttemptOutcome.ERROR)
    with pytest.raises(AttemptLeaseLost):
        db.promote_generation(conn, document_id, generation,
                              expected_active=displaced.observed_active,
                              manifest_ids=manifest, content_sha256=SHA_A,
                              candidate_id=displaced.candidate_id,
                              attempt_id=displaced.attempt_id)


def test_rule_5_a_lease_holder_records_its_own_failure_before_promotion(
        conn, filename):
    """A single-holder lease means there is no second run to lose a race
    to: the ordinary failure is "the holder failed before promoting". It
    writes its verdict against its OWN attempt, with the observed_active
    it captured at start, and the document's status is untouched.

    (An earlier version forged a "winner" with raw SQL while the lease
    was still held -- a state the protocol cannot produce.)"""
    begin, record = seam("begin_attempt"), seam("record_attempt_outcome")
    assert begin is not None and record is not None, (
        "db.begin_attempt / db.record_attempt_outcome dikisi yok")
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    before = _row(conn, document_id, "status", "active_generation")
    holder = begin(conn, document_id)

    record(conn, holder, AttemptOutcome.ERROR, "kurgu hata")

    with conn.cursor() as cur:
        cur.execute("SELECT status, observed_active FROM attempts "
                    "WHERE attempt_id = %s", (holder.attempt_id,))
        status, observed = cur.fetchone()
    assert status == AttemptOutcome.ERROR
    assert observed == holder.observed_active, (
        "sonuc baslangictaki aktif nesli tasimiyor")
    after = _row(conn, document_id, "status", "active_generation")
    assert after == before, "kaybeden belgenin durumunu degistirdi"


@pytest.mark.parametrize("verdict",
                         [AttemptOutcome.ERROR, AttemptOutcome.PARTIAL])
def test_rule_14_a_terminal_verdict_releases_the_lease(conn, filename,
                                                       verdict):
    """A run that is over must hold nothing: recording ERROR or PARTIAL
    clears attempt_id, owner and expires_at in the SAME statement, and the
    next attempt may begin immediately rather than waiting out a lease
    nobody is using."""
    begin, record = seam("begin_attempt"), seam("record_attempt_outcome")
    assert begin is not None and record is not None, "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    finished = begin(conn, document_id)

    record(conn, finished, verdict, "kurgu not")

    row = _row(conn, document_id, "attempt_id", "attempt_owner",
               "attempt_expires_at")
    assert row["attempt_id"] is None, f"{verdict} lease'i birakmadi"
    assert row["attempt_owner"] is None, "lease sahibi temizlenmedi"
    assert row["attempt_expires_at"] is None, "lease suresi temizlenmedi"
    fresh = begin(conn, document_id)          # no waiting for expiry
    assert fresh.attempt_id != finished.attempt_id
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM attempts WHERE attempt_id = %s",
                    (finished.attempt_id,))
        assert cur.fetchone()[0] == verdict


@pytest.mark.parametrize("verdict",
                         [AttemptOutcome.ERROR, AttemptOutcome.PARTIAL])
@pytest.mark.parametrize("fault_point",
                         ["lease_temizligi", "deneme_kaydi"])
def test_rule_14_the_verdict_and_the_release_are_one_transaction(
        connect, isolated_schema, filename, verdict, fault_point):
    """Reading the successful end state cannot tell "one statement" from
    "commit one half, then the other in a second transaction".

    BOTH orderings need a fault point, or one of them walks through.
    Injecting only on the lease clearing lets an implementation that
    CLEARS THE LEASE FIRST and commits, then writes the verdict
    separately, blow up before touching anything -- snapshot unchanged,
    test green, contract broken. Injecting only on the attempt record
    lets the mirror-image ordering through. Together they close both."""
    begin, record = seam("begin_attempt"), seam("record_attempt_outcome")
    assert begin is not None and record is not None, "dikisler yok"
    schema = isolated_schema["schema"]
    worker, observer = connect(), connect()
    document_id = _served_row(worker, filename, SHA_A, SHA_A)
    attempt = begin(worker, document_id)
    before = _full_snapshot(observer, document_id)

    _install_fault(worker, schema, fault_point)
    fired_before = _fault_firings(observer, schema)
    try:
        with pytest.raises(Exception) as caught:
            record(worker, attempt, verdict, "kurgu not")
    finally:
        try:
            worker.rollback()
        except Exception:
            pass

    observer.rollback()
    fired_after = _fault_firings(observer, schema)
    _remove_fault(connect(), fault_point)

    assert getattr(caught.value, "sqlstate", None) == FAULT_SQLSTATE, (
        f"enjekte edilen ariza degil: {caught.value!r} -- sonuc yazimi "
        f"{fault_point} noktasina hic ulasmamis olabilir")
    assert fired_after > fired_before, (
        f"{fault_point} tetigine hic ulasilmadi: sonuc yazimi sozlesmenin "
        f"gerektirdigi iki yazimdan birini yapmiyor (kural 14)")
    assert _full_snapshot(observer, document_id) == before, (
        f"{verdict} yazimi yarida iz birakti ({fault_point}): sonuc ve "
        f"lease birakma tek islem degil")


@pytest.mark.parametrize(
    ("forbidden", "why"),
    [
        (AttemptOutcome.DONE, "terfiye ait"),
        (AttemptOutcome.SUPERSEDED, "sisteme ait"),
        (None, "sonuc degil"),
        ("kurgu_bilinmeyen_sonuc", "sozlesmede yok"),
    ],
)
def test_a_worker_cannot_record_an_outcome_that_is_not_its_to_write(
        connect, filename, forbidden, why):
    """A worker records its OWN verdict and nothing else.

    ``done`` belongs to the promotion, which writes it together with the
    swap and the lease release; ``superseded`` belongs to the system, at
    takeover or fencing. A worker able to write either could mark itself
    successful with nothing promoted, or close its own attempt as though
    it had been displaced -- and an unknown value would leave every
    later reader guessing. All four are refused BEFORE any statement
    runs, so the document, the attempt and the lease are untouched."""
    begin, record = seam("begin_attempt"), seam("record_attempt_outcome")
    assert begin is not None and record is not None, "dikisler yok"
    worker, observer = connect(), connect()
    document_id = _served_row(worker, filename, SHA_A, SHA_A)
    holder = begin(worker, document_id)
    before = _full_snapshot(observer, document_id)

    with pytest.raises(AttemptOutcomeNotWritable):
        record(worker, holder, forbidden, "kurgu not")

    observer.rollback()
    assert _full_snapshot(observer, document_id) == before, (
        f"reddedilen sonuc ({why}) yine de iz birakti")
    # and the lease is still the holder's: a refusal is not a release
    assert str(_row(worker, document_id, "attempt_id")["attempt_id"]) == (
        holder.attempt_id)


def test_the_database_refuses_an_unknown_attempt_status_too(conn, filename):
    """The rule lives in the schema as well as in the code: a caller that
    reaches the table by another path still cannot store a status the
    contract does not know."""
    import psycopg

    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    holder = begin(conn, document_id)
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.cursor() as cur:
            cur.execute("UPDATE attempts SET status = %s WHERE attempt_id = %s",
                        ("kurgu_bilinmeyen_sonuc", holder.attempt_id))
    conn.rollback()


@pytest.mark.parametrize(
    "write_seam",
    ["allocate_generation", "upsert_chunks", "copy_chunks_into_generation"])
def test_every_write_seam_refuses_a_displaced_worker(connect, filename,
                                                     write_seam):
    """A takeover lands immediately before each write, and the write must
    not happen.

    A heartbeat that succeeded a moment ago is NOT authority: between its
    commit and the next statement a lease can be taken over, and an
    earlier version wrote a whole batch in that window before finding
    out. Every write seam therefore checks the attempt inside its OWN
    transaction, under the row lock, so there is no gap between deciding
    and doing."""
    begin = seam("begin_attempt")
    target = seam(write_seam)
    assert begin is not None and target is not None, "dikisler yok"
    worker, observer = connect(), connect()
    document_id = _served_row(worker, filename, SHA_A, SHA_A)
    displaced = begin(worker, document_id)

    # the takeover happens with the displaced worker about to write
    _expire(worker, document_id)
    begin(observer, document_id)
    before = _full_snapshot(observer, document_id)

    row = {
        "id": str(uuid.uuid4()), "document_id": document_id,
        "type": "text", "text": "kurgu parca", "source_tag": "page1:native",
        "page": 1, "headings": [], "table_data": None,
        "dense": "", "sparse": "", "generation": 9,
        "content_key": str(uuid.uuid4()),
        "embedding_fingerprint": "kurgu-parmak-izi",
    }
    calls = {
        "allocate_generation": lambda: target(worker, document_id, displaced),
        "upsert_chunks": lambda: target(worker, [row], displaced),
        "copy_chunks_into_generation": lambda: target(worker, [row],
                                                      displaced),
    }
    with pytest.raises(AttemptLeaseLost):
        calls[write_seam]()

    worker.rollback()
    observer.rollback()
    assert _full_snapshot(observer, document_id) == before, (
        f"{write_seam}: devrilen worker yine de yazdi")


def test_the_fence_lands_at_stage_and_empties_the_lease(conn, filename):
    """Rule 2 and the closure protocol: the fence is at STAGE, not at
    publish -- between them the live attempt would be indexing bytes
    nobody will serve. The displaced attempt goes terminal SUPERSEDED and
    the lease fields are cleared together with it."""
    begin, stage = seam("begin_attempt"), seam("stage_candidate")
    assert begin is not None and stage is not None, "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    live = begin(conn, document_id)

    stage(conn, filename, "pdf", content_sha256=SHA_B, allow_replace=True)

    row = _row(conn, document_id, "candidate_state", "attempt_id",
               "attempt_owner", "attempt_expires_at")
    assert row["candidate_state"] == CandidateState.STAGED, (
        "fence yayin asamasinda beklendi; STAGED'de olmaliydi")
    assert row["attempt_id"] is None, "cevrilen lease temizlenmedi"
    assert row["attempt_owner"] is None, "lease sahibi temizlenmedi"
    assert row["attempt_expires_at"] is None, "lease suresi temizlenmedi"
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM attempts WHERE attempt_id = %s",
                    (live.attempt_id,))
        assert cur.fetchone()[0] == AttemptOutcome.SUPERSEDED


def test_a_new_attempt_waits_for_the_new_candidate_to_be_published(
        conn, filename):
    """The fence EMPTIES the lease, it does not hand it to the newcomer:
    a staged candidate is still not processable."""
    begin, stage = seam("begin_attempt"), seam("stage_candidate")
    finalize = seam("finalize_candidate_publication")
    assert None not in (begin, stage, finalize), "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    begin(conn, document_id)
    _document_id, candidate_id, _canonical = stage(
        conn, filename, "pdf", content_sha256=SHA_B,
        allow_replace=True)
    with pytest.raises(CandidateNotPublished):
        begin(conn, document_id)
    finalize(conn, document_id, candidate_id)
    fresh = begin(conn, document_id)
    assert fresh.candidate_sha == SHA_B


def test_rule_2_a_fenced_attempt_can_write_nothing(conn, filename):
    begin, stage = seam("begin_attempt"), seam("stage_candidate")
    record = seam("record_attempt_outcome")
    assert None not in (begin, stage, record), "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    live = begin(conn, document_id)
    stage(conn, filename, "pdf", content_sha256=SHA_B, allow_replace=True)
    with pytest.raises(AttemptFenced):
        record(conn, live, AttemptOutcome.ERROR)


def test_rule_3_an_idempotent_upload_does_not_cancel_a_live_attempt(
        conn, filename):
    begin, stage = seam("begin_attempt"), seam("stage_candidate")
    assert begin is not None and stage is not None, "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    live = begin(conn, document_id)
    stage(conn, filename, "pdf", content_sha256=SHA_A)
    row = _row(conn, document_id, "candidate_id", "status", "attempt_id")
    assert str(row["candidate_id"]) == str(live.candidate_id)
    assert row["status"] == "done", "ayni baytlar belge durumunu oynatti"
    assert str(row["attempt_id"]) == str(live.attempt_id), (
        "ayni baytlar canli denemeyi iptal etti")


# --- heartbeat: strictly forward, revivable, and losable ------------------

def test_heartbeat_moves_the_expiry_strictly_forward(conn, filename):
    """``after >= before`` would accept a heartbeat that does nothing."""
    begin, beat = seam("begin_attempt"), seam("heartbeat_attempt")
    assert begin is not None and beat is not None, "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    attempt = begin(conn, document_id)
    before = _row(conn, document_id, "attempt_expires_at")
    time.sleep(0.05)
    assert beat(conn, attempt) is True
    after = _row(conn, document_id, "attempt_expires_at")
    assert after["attempt_expires_at"] > before["attempt_expires_at"], (
        "heartbeat sureyi ILERI tasimadi: no-op bir heartbeat")


def test_an_expired_but_untaken_lease_can_be_revived_by_its_holder(
        conn, filename):
    """Expiry makes a lease TAKEABLE; it does not by itself transfer
    ownership. Until someone takes it, the holder may reclaim it."""
    begin, beat = seam("begin_attempt"), seam("heartbeat_attempt")
    assert begin is not None and beat is not None, "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    attempt = begin(conn, document_id)
    _expire(conn, document_id)
    assert beat(conn, attempt) is True
    with pytest.raises(AttemptAlreadyRunning):
        begin(conn, document_id)          # revived: no longer takeable


def test_heartbeat_after_a_takeover_raises_lease_lost(conn, filename):
    begin, beat = seam("begin_attempt"), seam("heartbeat_attempt")
    assert begin is not None and beat is not None, "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    displaced = begin(conn, document_id)
    _expire(conn, document_id)
    begin(conn, document_id)
    with pytest.raises(AttemptLeaseLost):
        beat(conn, displaced)


def test_heartbeat_on_a_fenced_attempt_raises_fenced(conn, filename):
    begin, beat = seam("begin_attempt"), seam("heartbeat_attempt")
    stage = seam("stage_candidate")
    assert None not in (begin, beat, stage), "dikisler yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    live = begin(conn, document_id)
    stage(conn, filename, "pdf", content_sha256=SHA_B, allow_replace=True)
    with pytest.raises(AttemptFenced):
        beat(conn, live)


def test_a_heartbeat_racing_a_takeover_has_exactly_one_winner(
        connect, filename):
    """Both reach for the same expired lease at the same instant: either
    the holder revives it and the newcomer is refused, or the newcomer
    takes it and the holder's heartbeat raises. Never both, never
    neither."""
    begin, beat = seam("begin_attempt"), seam("heartbeat_attempt")
    assert begin is not None and beat is not None, "dikisler yok"
    setup = connect()
    document_id = _served_row(setup, filename, SHA_A, SHA_A)
    holder = begin(setup, document_id)
    _expire(setup, document_id)

    barrier = threading.Barrier(2)
    outcome = {}

    def reviver():
        connection = connect()
        barrier.wait(timeout=10)
        try:
            outcome["heartbeat"] = ("ok", beat(connection, holder))
        except BaseException as error:      # noqa: BLE001
            outcome["heartbeat"] = ("hata", error)

    def taker():
        connection = connect()
        barrier.wait(timeout=10)
        try:
            outcome["takeover"] = ("ok", begin(connection, document_id))
        except BaseException as error:      # noqa: BLE001
            outcome["takeover"] = ("hata", error)

    threads = [threading.Thread(target=reviver),
               threading.Thread(target=taker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    # Exactly TWO shapes are acceptable. "one ok and one error" was the
    # earlier check and it passed on heartbeat=False plus an unrelated
    # RuntimeError -- neither of which is either party winning.
    from pipeline.index.attempt_contract import IngestAttempt

    heartbeat_kind, heartbeat_value = outcome["heartbeat"]
    takeover_kind, takeover_value = outcome["takeover"]
    # bool() on BOTH sides: `taker_won` used to be whatever
    # getattr(..., None) returned, and `False != None` is True, so an
    # invalid takeover result passed the XOR
    holder_won = bool(heartbeat_kind == "ok" and heartbeat_value is True
                      and takeover_kind == "hata"
                      and isinstance(takeover_value, AttemptAlreadyRunning))
    taker_won = bool(takeover_kind == "ok"
                     and isinstance(takeover_value, IngestAttempt)
                     and takeover_value.attempt_id
                     and heartbeat_kind == "hata"
                     and isinstance(heartbeat_value, AttemptLeaseLost))
    assert holder_won != taker_won, (
        f"tam olarak biri kazanmali ve sonuclar sozlesmeye uymali: "
        f"{outcome}")

    setup.rollback()
    row = _row(setup, document_id, "attempt_id", "attempt_owner",
               "attempt_expires_at")
    expected = (holder.attempt_id if holder_won
                else takeover_value.attempt_id)
    assert str(row["attempt_id"]) == str(expected), (
        f"veritabanindaki lease kazananla uyusmuyor: "
        f"{row['attempt_id']} != {expected}")
    assert row["attempt_owner"] is not None, "kazananin sahibi yazilmadi"
    assert row["attempt_expires_at"] is not None, "kazananin suresi yok"
    if taker_won:
        with setup.cursor() as cur:
            cur.execute("SELECT status FROM attempts WHERE attempt_id = %s",
                        (holder.attempt_id,))
            assert cur.fetchone()[0] == AttemptOutcome.SUPERSEDED, (
                "devrilen sahip terminal SUPERSEDED yapilmadi")


# =====================================================================
# PACKAGE 3 -- promotion atomicity, proven by fault injection
# =====================================================================

FAULT_SQLSTATE = "KG001"

# WHERE the fault is injected decides WHICH boundary is proven. Injecting
# only on the stale sweep measures "document update ... sweep" and says
# nothing about a broken implementation that committed the sweep and then
# wrote the attempt's terminal record in a SECOND transaction. Both an
# EARLY and a LATE point are exercised.
FAULT_POINTS = {
    "supurme": ("chunks", "BEFORE DELETE"),
    "deneme_kaydi": ("attempts", "BEFORE INSERT OR UPDATE"),
    "lease_temizligi": ("documents", "BEFORE UPDATE"),
}


def _install_fault(conn, schema, point):
    """A trigger that COUNTS its firing on a sequence -- sequences are
    non-transactional, so the count survives the rollback and proves the
    fault was actually reached -- and then raises with a UNIQUE sqlstate,
    so "some exception happened" cannot stand in for "the injected fault
    happened"."""
    table, when = FAULT_POINTS[point]
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = %s", (table,))
        assert cur.fetchone()[0] == 1, (
            f"ariza noktasi icin tablo yok: {table} -- sozlesmenin DDL'i "
            f"eksik")
        cur.execute(f'CREATE SEQUENCE IF NOT EXISTS "{schema}".kurgu_sayaci')
        cur.execute(
            f'CREATE OR REPLACE FUNCTION "{schema}".kurgu_arizasi() '
            f"RETURNS trigger AS $$ BEGIN "
            f"PERFORM nextval('\"{schema}\".kurgu_sayaci'); "
            f"RAISE EXCEPTION 'kurgu enjekte ariza' USING ERRCODE = "
            f"'{FAULT_SQLSTATE}'; END; $$ LANGUAGE plpgsql")
        cur.execute(f"DROP TRIGGER IF EXISTS kurgu_ariza_tetigi ON {table}")
        cur.execute(f"CREATE TRIGGER kurgu_ariza_tetigi {when} ON {table} "
                    f"FOR EACH ROW EXECUTE FUNCTION "
                    f'"{schema}".kurgu_arizasi()')
    conn.commit()


def _fault_firings(conn, schema):
    with conn.cursor() as cur:
        cur.execute(f'SELECT last_value, is_called FROM "{schema}".'
                    f"kurgu_sayaci")
        last, called = cur.fetchone()
    return last if called else 0


def _remove_fault(conn, point="supurme"):
    table = FAULT_POINTS[point][0]
    with conn.cursor() as cur:
        cur.execute(f"DROP TRIGGER IF EXISTS kurgu_ariza_tetigi ON {table}")
    conn.commit()


@pytest.mark.parametrize("fault_point", ["supurme", "deneme_kaydi"])
def test_rule_4_a_failure_mid_promotion_leaves_nothing_behind(
        connect, isolated_schema, filename, fault_point):
    """Rule 4, proven rather than read: a fault raised INSIDE the promotion
    must leave the document row, the chunk rows, the lease fields and the
    attempt record identical to the pre-promotion snapshot -- compared as
    WHOLE ROWS (to_jsonb), from a SEPARATE connection, because the failing
    one may still hold an aborted transaction.

    Three things this test refuses to be satisfied by, all of which the
    earlier version accepted: any exception at all (now the trigger's
    unique sqlstate), a promotion that never reached the injected write
    (now counted on a non-transactional sequence), and a comparison over
    a hand-picked column list (now every column of every row).

    Reading a successful end state proves nothing about atomicity: an
    implementation that commits between the steps passes that reading."""
    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    schema = isolated_schema["schema"]
    worker, observer = connect(), connect()
    document_id = _served_row(worker, filename, SHA_A, SHA_A)
    attempt = begin(worker, document_id)
    generation = db.allocate_generation(worker, document_id, attempt)
    manifest = {_insert_chunk(worker, document_id, generation)}
    before = _full_snapshot(observer, document_id)

    _install_fault(worker, schema, fault_point)
    fired_before = _fault_firings(observer, schema)
    try:
        with pytest.raises(Exception) as caught:
            db.promote_generation(worker, document_id, generation,
                                  expected_active=attempt.observed_active,
                                  manifest_ids=manifest,
                                  content_sha256=SHA_A,
                                  candidate_id=attempt.candidate_id,
                                  attempt_id=attempt.attempt_id)
    finally:
        try:
            worker.rollback()
        except Exception:
            pass

    observer.rollback()               # fresh read, not a cached snapshot
    fired_after = _fault_firings(observer, schema)
    _remove_fault(connect(), fault_point)

    assert getattr(caught.value, "sqlstate", None) == FAULT_SQLSTATE, (
        f"enjekte edilen ariza degil, baska bir hata: {caught.value!r} "
        f"-- terfi sozlesme geregi yazmaya hic ulasmamis olabilir")
    assert fired_after > fired_before, (
        f"ariza tetigine ({fault_point}) hic ulasilmadi: terfi sozlesmenin "
        f"gerektirdigi yazimi ayni islemde yapmiyor")
    assert _full_snapshot(observer, document_id) == before, (
        f"yarida kalan terfi iz birakti ({fault_point}): terfi tek islem "
        f"degil")


@pytest.mark.parametrize("damage", ["kayit_yok", "zaten_terminal"])
def test_a_promotion_that_cannot_close_its_attempt_commits_nothing(
        connect, filename, damage):
    """Rule 4's third limb, enforced rather than attempted.

    The document row still says the lease is this attempt's -- nobody
    fenced it and nobody took it over -- but the attempt record cannot be
    closed: it is missing, or it is already terminal. An earlier version
    ran that UPDATE without checking the affected row, so the document
    swapped, the lease cleared and the old generations were DELETED while
    the attempt was never closed, and the whole thing committed. A
    promotion that fails a third of what it claims is not a partial
    success; it is a corrupt record, and it must leave nothing behind."""
    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    worker, observer = connect(), connect()
    document_id = _served_row(worker, filename, SHA_A, SHA_A)
    attempt = begin(worker, document_id)
    generation = db.allocate_generation(worker, document_id, attempt)
    manifest = {_insert_chunk(worker, document_id, generation)}

    with worker.cursor() as cur:
        if damage == "kayit_yok":
            cur.execute("DELETE FROM attempts WHERE attempt_id = %s",
                        (attempt.attempt_id,))
        else:
            cur.execute("UPDATE attempts SET status = %s WHERE attempt_id = %s",
                        (AttemptOutcome.ERROR, attempt.attempt_id))
    worker.commit()

    before = _full_snapshot(observer, document_id)
    with pytest.raises(AttemptRecordInconsistent):
        db.promote_generation(worker, document_id, generation,
                              expected_active=attempt.observed_active,
                              manifest_ids=manifest, content_sha256=SHA_A,
                              candidate_id=attempt.candidate_id,
                              attempt_id=attempt.attempt_id)

    observer.rollback()
    assert _full_snapshot(observer, document_id) == before, (
        f"kapanamayan deneme ({damage}) yine de iz birakti: terfi tek "
        f"basari degil")


def test_rule_4_a_successful_promotion_clears_the_lease_and_closes_the_attempt(
        conn, filename):
    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A)
    attempt = begin(conn, document_id)
    generation = db.allocate_generation(conn, document_id, attempt)
    manifest = {_insert_chunk(conn, document_id, generation)}
    db.promote_generation(conn, document_id, generation,
                          expected_active=attempt.observed_active,
                          manifest_ids=manifest, content_sha256=SHA_A,
                          candidate_id=attempt.candidate_id,
                          attempt_id=attempt.attempt_id)
    row = _row(conn, document_id, "status", "active_generation", "attempt_id",
               "attempt_owner", "attempt_expires_at")
    assert row["status"] == "done"
    assert row["active_generation"] == generation
    assert row["attempt_id"] is None, "lease birakilmadi"
    assert row["attempt_owner"] is None, "lease sahibi temizlenmedi"
    assert row["attempt_expires_at"] is None, "lease suresi temizlenmedi"
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM attempts WHERE attempt_id = %s",
                    (attempt.attempt_id,))
        assert cur.fetchone()[0] == AttemptOutcome.DONE


# =====================================================================
# crash windows, legacy rows, and the fixture's own safety
# =====================================================================

@pytest.mark.parametrize("crash_point", ["stage_sonrasi", "replace_sonrasi"])
def test_rule_11_publication_crash_windows_recover_idempotently(
        conn, filename, crash_point, tmp_path):
    """The recovery RUNS publish_candidate -- an earlier draft only checked
    that the function existed and then hand-rolled the steps, which tests
    the test rather than the service."""
    from pipeline.index import publication

    stage = seam("stage_candidate")
    assert stage is not None, "db.stage_candidate dikisi yok"
    # the publisher's OWN storage root, not a loose tmp file: pointing
    # them at different places would let a publisher that never touches
    # the disk pass, and could write into the real upload directory
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir, raising=False)
    disk = upload_dir / filename
    disk.write_bytes(A_BYTES)

    try:
        document_id, candidate_id, _canonical = stage(
            conn, filename, "pdf", content_sha256=SHA_B,
            allow_replace=True)
        if crash_point == "replace_sonrasi":
            disk.write_bytes(B_BYTES)
        assert _row(conn, document_id,
                    "candidate_state")["candidate_state"] == (
            CandidateState.STAGED)

        publication.publish_candidate(conn, filename, "pdf", B_BYTES,
                                      allow_replace=True)
    finally:
        monkeypatch.undo()

    row = _row(conn, document_id, "candidate_state", "candidate_id")
    assert row["candidate_state"] == CandidateState.PUBLISHED
    assert str(row["candidate_id"]) == str(candidate_id), (
        "idempotent tekrar yeni bir aday kimligi uretti")
    assert disk.read_bytes() == B_BYTES, (
        "toparlanma diske dokunmadi: yayin servisi baytlari yaymadi")


def test_rule_8_a_legacy_row_is_not_processable(conn, filename):
    begin = seam("begin_attempt")
    assert begin is not None, "db.begin_attempt dikisi yok"
    document_id = _served_row(conn, filename, SHA_A, SHA_A,
                              candidate_state=None)
    with pytest.raises(CandidateNotPublished):
        begin(conn, document_id)


def test_the_schema_carries_the_attempt_contract_columns(conn):
    """The DDL half of the contract, stated once and explicitly so that no
    other test has to fail during SETUP to report it."""
    missing = [
        f"documents.{column}"
        for column in ("candidate_state", "attempt_id", "attempt_owner",
                       "attempt_expires_at")
        if not _has_column(conn, "documents", column)
    ]
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'attempts'")
        if cur.fetchone()[0] == 0:
            missing.append("attempts tablosu")
    assert not missing, f"sozlesmenin DDL'i eksik: {missing}"


def test_the_isolated_schema_never_touches_public(isolated_schema, conn):
    """The fixture's own safety property, asserted rather than promised."""
    assert isolated_schema["schema"].startswith("ragtest_p0_")
    with conn.cursor() as cur:
        cur.execute("SHOW search_path")
        path = cur.fetchone()[0]
    assert "public" not in path, f"search_path public'i iceriyor: {path}"
    with conn.cursor() as cur:
        cur.execute("SELECT current_schema()")
        assert cur.fetchone()[0] == isolated_schema["schema"]
