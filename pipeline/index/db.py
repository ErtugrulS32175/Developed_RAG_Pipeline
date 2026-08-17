import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from pipeline.index.attempt_contract import (
    WORKER_WRITABLE_OUTCOMES,
    AttemptAlreadyRunning,
    AttemptFenced,
    AttemptLeaseLost,
    AttemptOutcome,
    AttemptOutcomeNotWritable,
    AttemptRecordInconsistent,
    CandidateConflict,
    CandidateNotPublished,
    CandidateState,
    IngestAttempt,
)

load_dotenv()

# No real credential in the default: set PG_DSN in .env. The placeholder
# password intentionally fails auth so a missing .env surfaces loudly
# instead of silently connecting with a committed secret.
PG_DSN = os.getenv("PG_DSN", "postgresql://rag:CHANGE_ME@localhost:5433/ragdb")

# fastembed's Qdrant/bm25 hashes tokens (no fixed vocabulary) into a range that
# exceeds pgvector's sparsevec dimension cap (1_000_000_000). Every sparse index
# is remapped into [1, SPARSE_DIM] before storage or query -- this constant must
# match the `sparsevec(N)` dimension declared in schema.sql exactly.
SPARSE_DIM = 999_999_937


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(PG_DSN)
    register_vector(conn)
    return conn


_pool = None


def _configure_pooled(conn) -> None:
    # register_vector queries pg_type, which opens a transaction; the pool
    # expects a configure hook to hand the connection back idle.
    register_vector(conn)
    conn.commit()


def get_pool():
    """Lazy process-wide connection pool for request-scoped work.

    The API used to cache ONE module-level connection. psycopg serialises
    concurrent statements on it, nothing ever called rollback(), and a single
    failed statement left the connection in a failed transaction -- every later
    request then died with InFailedSqlTransaction until the process restarted.
    A server-side kill (idle timeout, restart) had the same permanent effect.

    The pool closes all four holes at once: each request borrows its own
    connection, `check=` revalidates it on checkout so a dead one is replaced
    instead of served, and the pool's context manager commits on clean exit and
    rolls back on exception. min_size=0 keeps import and startup free of any
    database dependency -- nothing connects until the first checkout.
    """
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(
            PG_DSN,
            min_size=0,
            max_size=int(os.getenv("PG_POOL_MAX", "4")),
            configure=_configure_pooled,
            check=ConnectionPool.check_connection,
        )
    return _pool


def close_pool() -> None:
    """Close the pool explicitly on controlled shutdown.

    The OS reclaims sockets when the process dies, but a reload or a test
    process that never exits keeps the pool's worker thread and any idle
    connections alive; closing makes shutdown deterministic. Safe to call
    when no pool was ever created."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_schema(conn) -> None:
    sql = Path(__file__).parent.joinpath("schema.sql").read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def upsert_document(conn, filename: str, file_type: str,
                    status: str = "processing",
                    content_sha256: str | None = None,
                    allow_replace: bool = False) -> str:
    """Look up a document by filename, reusing its id if it already exists.

    Documents are keyed by filename, and a filename is an AMBIGUOUS key: a
    different file wearing the same basename used to merge into the existing
    row silently, after which the new ingest deleted the old file's chunks
    as "stale" -- cross-document data loss with a "done" status on top. When
    both hashes are known and DIFFER, this refuses unless the caller
    explicitly says replacement is intended (``allow_replace``).

    The advisory lock comes FIRST and its key is CASEFOLDED: Windows
    resolves differently-cased spellings to ONE file, and a lock keyed on
    the exact spelling let two spellings hold two locks -- and two rows --
    over one file. Under the held lock, the existing spelling is looked up
    and REUSED as the canonical name, so a re-cased upload lands on the
    existing row instead of creating a sibling. (No case-insensitive unique
    index: creating one would brick init_schema on any legacy database that
    already carries case-siblings; the lock serialises every writer path
    in this codebase, which is what the index would have enforced.)

    The write itself stays ONE guarded statement evaluated at write time.
    Its arms, in order: explicit replace authority; no hash offered; the
    offered bytes are the ones being SERVED (active_content_sha); the
    offered bytes are the recorded CANDIDATE -- which can only have been
    recorded by passing this same gate, so an upload authorised with
    ``replace`` carries its authority to the ingest that follows without a
    global flag; and finally a row with NO served rows at all, where
    replacement destroys nothing. A row whose active_content_sha is NULL
    but which HAS served rows is a legacy migration -- refusing it
    (fail-closed) is the fix for a probe where such a row accepted
    arbitrary different content with no replace authority at all.

    Every accepted knock that CHANGES the candidate bytes mints a fresh,
    immutable ``candidate_id``; an identical re-knock keeps the existing
    one. The id -- not the hash -- is the run identity everything
    downstream binds to: an audited race had an old ingest re-record the
    OLD hash here (legitimately, via the active arm) over a newer
    authorised upload and then promote, leaving disk, candidate and index
    on three different versions. With the id, that ingest's promotion CAS
    fails loudly instead.

    Returns ``(document_id, candidate_id, canonical_filename)`` -- the
    canonical spelling comes back so the CALLER's disk write can target
    the same file the database is talking about, on case-sensitive
    filesystems too."""
    key = filename.casefold()
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
        cur.execute(
            "SELECT filename, "
            "EXISTS (SELECT 1 FROM chunks WHERE document_id = documents.id) "
            "FROM documents WHERE lower(filename) = lower(%s) "
            "ORDER BY uploaded_at LIMIT 1",
            (filename,))
        existing = cur.fetchone()
        canonical = existing[0] if existing else filename
        fresh = existing is None or not existing[1]
        cur.execute(
            "INSERT INTO documents "
            "(id, filename, file_type, status, content_sha256, candidate_id) "
            "VALUES (%(id)s, %(filename)s, %(file_type)s, %(status)s, "
            "%(sha)s, %(cid)s) "
            "ON CONFLICT (filename) DO UPDATE SET status = %(status)s, "
            "content_sha256 = COALESCE(%(sha)s, documents.content_sha256), "
            "candidate_id = CASE "
            "  WHEN %(sha)s IS NULL THEN documents.candidate_id "
            "  WHEN documents.content_sha256 IS NOT DISTINCT FROM %(sha)s "
            "       AND documents.candidate_id IS NOT NULL "
            "    THEN documents.candidate_id "
            "  ELSE %(cid)s END "
            "WHERE %(allow)s "
            "   OR %(sha)s IS NULL "
            "   OR documents.active_content_sha = %(sha)s "
            "   OR documents.content_sha256 = %(sha)s "
            "   OR (documents.active_content_sha IS NULL AND %(fresh)s) "
            "RETURNING id, candidate_id, filename",
            {"id": str(uuid.uuid4()), "filename": canonical,
             "file_type": file_type, "status": status,
             "sha": content_sha256, "allow": bool(allow_replace),
             "fresh": bool(fresh), "cid": str(uuid.uuid4())},
        )
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise ValueError(
                "ayni dosya adi farkli icerikle zaten HIZMETTE; dosyayi "
                "yeniden adlandir ya da bilincli degistirme icin "
                "replace yetkisi ver"
            )
        document_id, candidate_id, stored_name = row[0], row[1], row[2]
    conn.commit()
    return (str(document_id),
            str(candidate_id) if candidate_id is not None else None,
            str(stored_name))


def stage_candidate(conn, filename: str, file_type: str,
                    content_sha256: str | None = None,
                    allow_replace: bool = False):
    """Record a candidate as STAGED -- the ONE gate a candidate enters by.

    Returns ``(document_id, candidate_id, canonical_filename)``. The
    canonical name is what the publication service derives its disk
    target from: the stored spelling wins, so a re-cased upload lands on
    the existing row AND the existing file rather than creating a second
    one beside it.

    WHAT THIS GATE REFUSES, and why it is not the old one. The previous
    gate had an arm accepting any offer equal to the SERVED bytes. That
    arm is gone. It looked harmless -- "re-ingesting what we already
    serve" -- but it is exactly how a stale run reverted a newer
    authorised upload: a CLI that started before the upload finished
    parsing offered the OLD hash after the upload had recorded the new
    candidate, the arm accepted it, a fresh candidate id was minted over
    the newer one, and the stale snapshot was promoted. Disk, candidate
    and index ended on three different stories with every step reporting
    success. The arms that remain:

      * explicit replacement authority (never implicit, never from an
        environment variable),
      * no hash offered at all,
      * the offer IS the recorded candidate -- the idempotent re-knock,
        which keeps the candidate id and cancels nothing,
      * a row that serves nothing yet, where replacement destroys
        nothing.

    Anything else raises ``CandidateConflict`` and rolls back with every
    column as it was.

    The document's STATUS is not touched here. It describes the SERVED
    version; staging a candidate does not un-index what is being
    answered from. The candidate's own lifecycle lives in
    ``candidate_state``, and the upload response reports the candidate --
    three subjects that were once one column, which is how an upload
    could return "pending" while the row said "error"."""
    key = filename.casefold()
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
        cur.execute(
            "SELECT filename, "
            "EXISTS (SELECT 1 FROM chunks WHERE document_id = documents.id) "
            "FROM documents WHERE lower(filename) = lower(%s) "
            "ORDER BY uploaded_at LIMIT 1",
            (filename,))
        existing = cur.fetchone()
        canonical = existing[0] if existing else filename
        fresh = existing is None or not existing[1]
        # "the offer is the recorded candidate" decides BOTH whether the
        # id is kept and whether the state stays where it is -- written
        # once as a SQL expression so the two can never disagree
        unchanged = ("documents.content_sha256 IS NOT DISTINCT FROM %(sha)s "
                     "AND documents.candidate_id IS NOT NULL")
        cur.execute(
            "INSERT INTO documents "
            "(id, filename, file_type, content_sha256, candidate_id, "
            " candidate_state) "
            "VALUES (%(id)s, %(filename)s, %(file_type)s, %(sha)s, "
            "%(cid)s, %(staged)s) "
            "ON CONFLICT (filename) DO UPDATE SET "
            "content_sha256 = COALESCE(%(sha)s, documents.content_sha256), "
            "candidate_id = CASE "
            "  WHEN %(sha)s IS NULL THEN documents.candidate_id "
            f"  WHEN {unchanged} THEN documents.candidate_id "
            "  ELSE %(cid)s END, "
            "candidate_state = CASE "
            "  WHEN %(sha)s IS NULL THEN documents.candidate_state "
            f"  WHEN {unchanged} THEN documents.candidate_state "
            "  ELSE %(staged)s END, "
            # THE FENCE, in the same statement that changes the
            # candidate: from this moment the live attempt is indexing
            # bytes nobody will serve, and letting it run until the
            # publication finished would leave it a window to promote
            # them. The lease is EMPTIED rather than handed over -- the
            # newcomer may only begin once the new candidate is
            # published. An idempotent re-knock keeps everything.
            "attempt_id = CASE "
            "  WHEN %(sha)s IS NULL THEN documents.attempt_id "
            f"  WHEN {unchanged} THEN documents.attempt_id "
            "  ELSE NULL END, "
            "attempt_owner = CASE "
            "  WHEN %(sha)s IS NULL THEN documents.attempt_owner "
            f"  WHEN {unchanged} THEN documents.attempt_owner "
            "  ELSE NULL END, "
            "attempt_expires_at = CASE "
            "  WHEN %(sha)s IS NULL THEN documents.attempt_expires_at "
            f"  WHEN {unchanged} THEN documents.attempt_expires_at "
            "  ELSE NULL END "
            "WHERE %(allow)s "
            "   OR %(sha)s IS NULL "
            "   OR documents.content_sha256 = %(sha)s "
            "   OR (documents.active_content_sha IS NULL AND %(fresh)s) "
            "RETURNING id, candidate_id, filename",
            {"id": str(uuid.uuid4()), "filename": canonical,
             "file_type": file_type, "sha": content_sha256,
             "cid": str(uuid.uuid4()), "staged": CandidateState.STAGED,
             "allow": bool(allow_replace), "fresh": bool(fresh)},
        )
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise CandidateConflict(
                "ayni dosya adi farkli icerikle zaten kayitli; dosyayi "
                "yeniden adlandir ya da bilincli degistirme icin "
                "replace yetkisi ver")
        document_id, candidate_id, stored_name = row[0], row[1], row[2]
        # A fenced attempt is not left dangling: the SYSTEM closes it
        # here, in the same transaction, because the displaced worker may
        # write nothing from the moment it was fenced.
        #
        # Stated as "every RUNNING attempt of this document whose
        # candidate is no longer the current one", so nothing has to be
        # carried from the pre-read into Python and back: the set is
        # empty exactly when the candidate did not change, which is also
        # what makes an idempotent re-knock cancel nothing. It sweeps a
        # dangling run from an earlier fence too, if one ever survives.
        cur.execute(
            "UPDATE attempts SET status = %(status)s, ended_at = now() "
            "WHERE document_id = %(document)s AND status IS NULL "
            "AND candidate_id IS DISTINCT FROM %(candidate)s::uuid",
            {"status": AttemptOutcome.SUPERSEDED, "document": document_id,
             "candidate": str(candidate_id) if candidate_id else None})
    conn.commit()
    return (str(document_id),
            str(candidate_id) if candidate_id is not None else None,
            str(stored_name))


ATTEMPT_LEASE_SECONDS = int(os.getenv("ATTEMPT_LEASE_SECONDS", "300"))


def _default_owner() -> str:
    """Who holds the lease, for an operator reading the row. Never
    authority: authority is the attempt id, which is a fencing token."""
    import socket

    return f"{socket.gethostname()}/{os.getpid()}"


def begin_attempt(conn, document_id: str, owner: str | None = None):
    """Take the lease on a PUBLISHED candidate and mint the run's identity.

    One transaction does all three things the contract names: verify the
    candidate is published, read the active generation, take the lease.
    ``SELECT ... FOR UPDATE`` is what makes two concurrent callers come
    out with exactly one holder -- the second blocks, then re-reads the
    row the first committed and finds a live lease.

    NO advisory lock is taken here. A process request must never queue
    behind an upload's disk write: a STAGED candidate is refused
    immediately with `CandidateNotPublished`, which the API answers 409,
    rather than waiting for the publication to finish.

    An EXPIRED lease is taken over -- and takeover closes the displaced
    attempt as `SUPERSEDED` right here, because the displaced worker may
    write nothing from this moment on and its record must not dangle.
    Expiry is judged by the DATABASE clock; a worker's own clock has no
    say in whether its lease is still good."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT candidate_state, candidate_id, content_sha256, "
            "active_generation, attempt_id, "
            "(attempt_expires_at IS NOT NULL AND attempt_expires_at > now()) "
            "FROM documents WHERE id = %s FOR UPDATE",
            (document_id,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise ValueError("belge kaydi yok; deneme baslatilamaz")
        (state, candidate_id, candidate_sha, active_generation,
         held_by, lease_is_live) = row
        if state != CandidateState.PUBLISHED or candidate_id is None:
            conn.rollback()
            raise CandidateNotPublished(
                "aday yayimlanmis degil; bu belge simdilik islenemez")
        if held_by is not None and lease_is_live:
            conn.rollback()
            raise AttemptAlreadyRunning(
                "bu belge icin canli bir deneme var; lease sahibi baska "
                "bir kosu")
        if held_by is not None:
            cur.execute(
                "UPDATE attempts SET status = %(status)s, ended_at = now() "
                "WHERE attempt_id = %(attempt)s AND status IS NULL",
                {"status": AttemptOutcome.SUPERSEDED, "attempt": held_by})
        attempt_id = str(uuid.uuid4())
        holder = owner or _default_owner()
        cur.execute(
            "UPDATE documents SET attempt_id = %(attempt)s, "
            "attempt_owner = %(owner)s, "
            "attempt_expires_at = now() + make_interval(secs => %(ttl)s) "
            "WHERE id = %(id)s",
            {"attempt": attempt_id, "owner": holder,
             "ttl": ATTEMPT_LEASE_SECONDS, "id": document_id})
        cur.execute(
            "INSERT INTO attempts (attempt_id, document_id, candidate_id, "
            "candidate_sha, observed_active, owner) "
            "VALUES (%(attempt)s, %(document)s, %(candidate)s, %(sha)s, "
            "%(active)s, %(owner)s)",
            {"attempt": attempt_id, "document": document_id,
             "candidate": str(candidate_id), "sha": candidate_sha,
             "active": int(active_generation or 0), "owner": holder})
    conn.commit()
    return IngestAttempt(
        attempt_id=attempt_id,
        document_id=str(document_id),
        candidate_id=str(candidate_id),
        candidate_sha=candidate_sha,
        observed_active=int(active_generation or 0),
    )


def _authority(cur, attempt):
    """What this attempt is still allowed to do, decided by the row.

    Two different refusals, and the difference matters to the caller: the
    CANDIDATE moved (someone published other bytes -- `AttemptFenced`) or
    the LEASE moved (someone took over after expiry --
    `AttemptLeaseLost`). Both mean "write nothing", but only one of them
    means the work itself was pointless."""
    cur.execute(
        "SELECT candidate_id, attempt_id FROM documents WHERE id = %s "
        "FOR UPDATE", (attempt.document_id,))
    row = cur.fetchone()
    if row is None:
        raise AttemptLeaseLost("belge kaydi yok; bu deneme gecersiz")
    candidate_id, held_by = row
    if str(candidate_id) != str(attempt.candidate_id):
        raise AttemptFenced(
            "aday degisti; bu deneme cevrildi ve hicbir sey yazamaz")
    if str(held_by) != str(attempt.attempt_id):
        raise AttemptLeaseLost(
            "lease devralindi; bu worker artik yazamaz")


def heartbeat_attempt(conn, attempt) -> bool:
    """Push this attempt's expiry forward while the run is alive.

    STRICTLY forward: the new value is at least a millisecond past the
    old one, so an implementation that returns True without moving the
    clock cannot pass for a heartbeat.

    An expired but NOT YET TAKEN lease may be revived by its own holder --
    expiry makes a lease TAKEABLE, it does not by itself transfer
    ownership. Once someone has taken it, this raises."""
    try:
        with conn.cursor() as cur:
            _authority(cur, attempt)
            cur.execute(
                "UPDATE documents SET attempt_expires_at = GREATEST("
                "  now() + make_interval(secs => %(ttl)s), "
                "  attempt_expires_at + interval '1 millisecond') "
                "WHERE id = %(id)s",
                {"ttl": ATTEMPT_LEASE_SECONDS, "id": attempt.document_id})
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return True


def record_attempt_outcome(conn, attempt, status: str,
                           note: str | None = None) -> bool:
    """Write this attempt's terminal verdict AND release its lease -- one
    transaction, never two.

    The verdict goes on the ATTEMPT, never on the document: a run that
    failed did not make the served version worse, and stamping the
    document was how a losing run labelled a healthy index 'error'.

    The lease triple is cleared in the same breath (rule 14). A finished
    attempt that kept its lease would block the next one until expiry --
    a run that is over must hold nothing.

    ONLY ``error`` and ``partial`` may be written here. ``done`` belongs
    to the promotion, which writes it together with the swap and the
    lease release; ``superseded`` belongs to the system, written at
    takeover or fencing. A worker that could write either could mark
    itself successful with nothing promoted, or close its own attempt as
    though it had been displaced. Refused BEFORE any statement runs, so
    a rejected call leaves the document, the attempt and the lease
    exactly as they were -- the database carries the same rule as a
    CHECK constraint, because a guard that lives only in Python is one
    forgotten caller away from being no guard."""
    if status not in WORKER_WRITABLE_OUTCOMES:
        raise AttemptOutcomeNotWritable(
            f"bu sonucu worker yazamaz: {status!r}; yalnizca "
            f"{list(WORKER_WRITABLE_OUTCOMES)} kabul edilir "
            f"('done' terfiye, 'superseded' sisteme aittir)")
    try:
        with conn.cursor() as cur:
            _authority(cur, attempt)
            cur.execute(
                "UPDATE attempts SET status = %(status)s, note = %(note)s, "
                "ended_at = now() WHERE attempt_id = %(attempt)s",
                {"status": status, "note": note,
                 "attempt": attempt.attempt_id})
            cur.execute(
                "UPDATE documents SET attempt_id = NULL, "
                "attempt_owner = NULL, attempt_expires_at = NULL "
                "WHERE id = %(id)s AND attempt_id = %(attempt)s::uuid",
                {"id": attempt.document_id, "attempt": attempt.attempt_id})
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return True


def finalize_candidate_publication(conn, document_id: str,
                                   candidate_id: str) -> bool:
    """STAGED -> PUBLISHED, once the bytes are actually on disk.

    Guarded on the candidate id: if a newer candidate was staged while
    this publication was writing, finalising would mark the NEWER
    candidate published on the OLDER one's bytes. Returns False in that
    case and publishes nothing."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET candidate_state = %(state)s "
            "WHERE id = %(id)s AND candidate_id = %(cid)s::uuid "
            "RETURNING id",
            {"state": CandidateState.PUBLISHED, "id": document_id,
             "cid": candidate_id})
        applied = cur.fetchone() is not None
    conn.commit()
    return applied


def lookup_document(conn, filename: str) -> dict | None:
    """The row a name resolves to, by the same case-insensitive rule the
    upsert canonicalises with -- what a candidate-bound run starts from."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, filename, status, content_sha256, candidate_id, "
            "active_generation, active_content_sha "
            "FROM documents WHERE lower(filename) = lower(%s) "
            "ORDER BY uploaded_at LIMIT 1",
            (filename,))
        row = cur.fetchone()
    if row is None:
        return None
    return {key: (str(value) if key in ("id", "candidate_id")
                  and value is not None else value)
            for key, value in row.items()}


class PublishLockNotReleased(RuntimeError):
    """The publish lock could not be PROVEN released. The connection has
    been closed so the server drops it, and the caller is told -- a
    publication that returned success while a lock leaked would let the
    next one queue behind a fence nobody can lower."""


@contextmanager
def document_publish_lock(conn, filename: str):
    """SESSION-level advisory lock spanning a name's whole publish sequence.

    The transaction-scoped lock inside upsert_document releases at that
    call's own commit -- and the disk write (os.replace) happens AFTER it.
    A concurrent probe drove two uploads through that gap: both returned
    200, the database kept the second content's hash and the disk kept the
    first content's bytes. A session lock survives commits, so the caller
    can hold ONE lock across the database decision AND the disk publish;
    it is keyed on the CASEFOLDED name for the same reason the upsert's
    lock is.

    The RELEASE path is fail-safe, because two probes showed two
    different leaks. A body that left the transaction aborted made the
    unlock statement itself fail, which masked the primary error and
    returned the connection to the pool STILL HOLDING the lock. And
    `pg_advisory_unlock` returns FALSE when this session did not hold the
    lock -- a statement that runs without error and releases nothing --
    so the answer has to be READ, not merely requested.

    What happens now, in each of the three cases:

      * unlock PROVEN: ordinary success, the connection stays open.
      * body succeeded, unlock unproven: the connection is closed (the
        server drops a dead session's advisory locks, the one guaranteed
        release) AND `PublishLockNotReleased` is raised. A caller that
        saw success while a lock leaked would keep publishing behind a
        fence nobody can lower; an earlier version raised here and then
        swallowed it in its own except clause, which is how "we close
        the connection" became the only half that was true.
      * body already failed, unlock unproven: the connection is closed
        and the ORIGINAL failure propagates. The lock problem must not
        overwrite the reason the caller actually needs to see."""
    key = filename.casefold()
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (key,))
    conn.commit()
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        released = False
        try:
            # rollback first: clears any aborted state so the unlock can
            # run at all; a no-op otherwise, since every meaningful write
            # in the body commits itself
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (key,))
                answer = cur.fetchone()
            released = answer is not None and answer[0] is True
            if released:
                conn.commit()
        except Exception:
            released = False
        if not released:
            try:
                conn.close()
            except Exception:
                pass
            if not body_failed:
                raise PublishLockNotReleased(
                    "advisory kilit birakildigi kanitlanamadi; baglanti "
                    "kapatildi")


def get_document(conn, document_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, filename, file_type, uploaded_at, status, "
            "status_note, active_generation, content_sha256, candidate_id "
            "FROM documents WHERE id = %s", (document_id,))
        row = cur.fetchone()
    if row is None:
        return None
    if row.get("candidate_id") is not None:
        row["candidate_id"] = str(row["candidate_id"])
    return row


def list_documents(conn, limit: int, offset: int) -> list[dict]:
    """One page of the document inventory, newest first.

    THE PROJECTION IS THE GATE. Only the columns an inventory may show are
    selected at all, so ``content_sha256`` and ``candidate_id`` -- the
    recorded candidate's bytes and its immutable identity -- never leave
    the database on this path. A caller that needs them asks for a single
    document; a listing is read by anyone holding the API key.

    ORDERING IS TOTAL, not merely "newest first". ``uploaded_at`` is not
    unique -- a batch upload can land several rows inside one clock tick
    -- and a partial order under LIMIT/OFFSET lets the server return the
    same row on two pages while another is never returned at all. The id
    tie-break makes the sequence the same on every page of one scan.

    ONE EXTRA ROW is fetched on purpose: ``limit + 1``. It answers "is
    there another page" from the same scan the page came from, without a
    second COUNT over the whole table and without the window between the
    two that would make the count disagree with the page. The caller gets
    up to ``limit + 1`` rows and decides what to publish -- the sentinel
    row is evidence, not content.
    """
    # A page size is arithmetic, and arithmetic on a value nobody checked
    # is how OFFSET -1 reaches the server. The API rejects these before it
    # borrows a connection; this refuses them for every OTHER caller, so
    # the guard does not live in one endpoint's signature alone.
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit 1 veya daha buyuk bir tamsayi olmali")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset 0 veya daha buyuk bir tamsayi olmali")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, filename, file_type, uploaded_at, status, "
            "status_note, active_generation "
            "FROM documents "
            "ORDER BY uploaded_at DESC, id DESC "
            "LIMIT %(limit)s OFFSET %(offset)s",
            {"limit": limit + 1, "offset": offset})
        rows = cur.fetchall()
    listed = []
    for row in rows:
        # `id` is a uuid object; every other reader of this row is JSON,
        # so it is stringified here under the name the API publishes it by
        row["document_id"] = str(row.pop("id"))
        listed.append(row)
    return listed


def set_document_status(conn, document_id: str, status: str,
                        note: str | None = None,
                        expected_active: int | None = None,
                        candidate_id: str | None = None) -> bool:
    """Terminal status plus its explanation. The note is overwritten on every
    transition on purpose: a 'done' after a repaired re-ingest must not keep
    displaying last week's failure list.

    The RUN IDENTITY guard is two-part: a failing run stamps its verdict
    only while the active pointer still stands where THAT RUN observed it
    (``expected_active``) AND while the candidate it was processing is
    still the recorded one (``candidate_id``). The active pointer alone
    was not an identity: two runs observing the same pointer but bound to
    different candidates could still stamp over each other. A guarded
    stamp that no longer applies returns False and writes nothing: the
    document's current state belongs to whoever moved it."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = %(status)s, "
            "status_note = %(note)s WHERE id = %(id)s "
            "AND (%(expected)s::integer IS NULL "
            "     OR active_generation = %(expected)s) "
            "AND (%(cid)s::uuid IS NULL OR candidate_id = %(cid)s::uuid) "
            "RETURNING id",
            {"status": status, "note": note, "id": document_id,
             "expected": expected_active, "cid": candidate_id})
        applied = cur.fetchone() is not None
    conn.commit()
    return applied


def clear_chunks_for_document(conn, document_id: str) -> None:
    """Delete only this document's previous chunks, so re-ingesting one file
    doesn't wipe out every other document (see the old `clear_chunks`, which
    used to TRUNCATE the whole table on every ingest run)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    conn.commit()


def remap_sparse_index(raw_id: int) -> int:
    return (raw_id % SPARSE_DIM) + 1


def sparse_to_literal(indices, values) -> str:
    remapped = ((remap_sparse_index(i), v) for i, v in zip(indices, values))
    pairs = ",".join(f"{i}:{v}" for i, v in sorted(remapped))
    return f"{{{pairs}}}/{SPARSE_DIM}"


def upsert_chunks(conn, rows: list[dict], attempt) -> None:
    """Insert chunks, skipping any that are already stored.

    `ON CONFLICT DO NOTHING` is what makes an interrupted ingest resumable: ids
    are derived from the content, so re-running writes only what is missing
    instead of failing on the rows that already landed.

    The attempt's authority is checked IN THIS TRANSACTION: a displaced
    worker must not get one more batch in because its last heartbeat was
    a second ago.
    """
    prepared = [
        {**r, "headings": Json(r["headings"]), "table_data": Json(r["table_data"]) if r["table_data"] else None}
        for r in rows
    ]
    try:
        with conn.cursor() as cur:
            _authority(cur, attempt)
            _insert_chunk_rows(cur, prepared)
    except Exception:
        conn.rollback()
        raise
    conn.commit()


def _insert_chunk_rows(cur, prepared) -> None:
    cur.executemany(
        "INSERT INTO chunks (id, document_id, type, text, source_tag, page, headings, table_data, dense, sparse, generation, content_key, embedding_fingerprint) "
        "VALUES (%(id)s, %(document_id)s, %(type)s, %(text)s, %(source_tag)s, %(page)s, %(headings)s, "
        "%(table_data)s, %(dense)s, %(sparse)s::sparsevec, %(generation)s, %(content_key)s, %(embedding_fingerprint)s) "
        "ON CONFLICT (id) DO NOTHING",
        prepared,
    )


def existing_chunk_ids(conn, document_id: str) -> set:
    """Ids already stored for this document -- what a resumed run can skip."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM chunks WHERE document_id = %s", (document_id,))
        return {str(row[0]) for row in cur.fetchall()}


def allocate_generation(conn, document_id: str, attempt) -> int:
    """A generation number no other attempt has ever held, atomically.

    The first design reused active_generation + 1, and two consequences
    followed on probes: consecutive runs merged DIFFERENT contents into one
    staging generation, and two concurrent complete runs promoted two
    contents into one "done" generation without any error. last_generation
    is a counter that only moves forward, incremented inside the database,
    so every attempt stages under a number that is immutably its own.

    The attempt's authority is checked IN THIS TRANSACTION (see
    ``_authority``), not inferred from a heartbeat that succeeded a
    moment ago: between a heartbeat's commit and the next write, a lease
    can be taken over. The row lock the check takes holds until this
    write commits, so there is no window between deciding and doing."""
    try:
        with conn.cursor() as cur:
            _authority(cur, attempt)
            cur.execute(
                "UPDATE documents SET last_generation = last_generation + 1 "
                "WHERE id = %s RETURNING last_generation",
                (document_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError("belge kaydi yok; nesil tahsis edilemedi")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return int(row[0])


def copy_chunks_into_generation(conn, rows, attempt) -> int:
    """Reuse prior rows' VECTORS in this generation -- and nothing else.

    Each row is the CURRENT parse's full payload (type, text, headings,
    table_data, page) minus the vectors; only dense/sparse are pulled from
    a predecessor found by content_key. The key is doc|tag|index|text, so
    the embeddings -- functions of the text alone -- are valid to reuse,
    but the metadata is NOT: an audit probe showed the copy inheriting an
    arbitrary old generation's whole payload (LIMIT 1 over an ambiguous
    set), which could resurrect stale headings or table_data under a
    freshly-parsed identity. The predecessor must ALSO carry this run's
    exact embedding fingerprint: a text match alone let a model change
    ride stale vectors into the new generation. The predecessor is picked
    deterministically (newest generation). Rows are still COPIED, never
    moved: the active generation keeps every one of its own rows."""
    copied = 0
    prepared = [
        {**r, "headings": Json(r["headings"]),
         "table_data": Json(r["table_data"]) if r["table_data"] else None}
        for r in rows
    ]
    try:
        with conn.cursor() as cur:
            _authority(cur, attempt)
            copied = _copy_rows(cur, prepared)
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return copied


def _copy_rows(cur, prepared) -> int:
    copied = 0
    for row in prepared:
        cur.execute(
            "INSERT INTO chunks (id, document_id, type, text, source_tag, "
            "page, headings, table_data, dense, sparse, generation, "
            "content_key, embedding_fingerprint) "
            "SELECT %(id)s, %(document_id)s, %(type)s, %(text)s, "
            "%(source_tag)s, %(page)s, %(headings)s, %(table_data)s, "
            "dense, sparse, %(generation)s, %(content_key)s, "
            "%(embedding_fingerprint)s "
            "FROM chunks WHERE document_id = %(document_id)s "
            "AND content_key = %(content_key)s "
            "AND embedding_fingerprint = %(embedding_fingerprint)s "
            "ORDER BY generation DESC LIMIT 1 "
            "ON CONFLICT (id) DO NOTHING",
            row)
        copied += cur.rowcount
    return copied


def existing_content_keys(conn, document_id: str,
                          embedding_fingerprint: str | None = None) -> set:
    """Content keys with a reusable stored row in any generation -- what a
    new attempt can COPY instead of re-embedding. Reusable means the row
    was embedded under EXACTLY this configuration: a fingerprint mismatch
    (or a legacy NULL) makes the row invisible here, so the chunk is
    re-embedded rather than inherited across a model change."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT content_key FROM chunks "
            "WHERE document_id = %s AND content_key IS NOT NULL "
            "AND embedding_fingerprint = %s",
            (document_id, embedding_fingerprint))
        return {str(row[0]) for row in cur.fetchall()}


def promote_generation(conn, document_id: str, generation: int,
                       expected_active: int, manifest_ids,
                       content_sha256: str, candidate_id: str,
                       attempt_id: str) -> int:
    """Make a COMPLETE staging generation the served one: verify, CAS, sweep.

    The protections, all in one transaction:
      * NON-EMPTY: an empty manifest is refused outright. A parser that
        returned parts but no usable text once sailed a ZERO-row
        generation through a count check (0 == 0) and swept the healthy
        index -- promoting nothing is indistinguishable from deleting
        everything, so nothing is never promotable.
      * MANIFEST BY MEMBERSHIP, not by count: the staged generation must
        hold exactly the IDS the run wrote. A count accepted any same-sized
        set of wrong rows; set equality refuses missing and foreign rows
        alike and says how many of each.
      * The served bytes' hash is REQUIRED: a promotion that cannot say
        which bytes it serves would leave the conflict gate comparing
        against a stale value.
      * THREE-PART COMPARE-AND-SWAP: the active pointer must still stand
        where this run observed it, the recorded candidate must still be
        the one this run was ingesting, AND the lease must still be
        THIS attempt's. The pointer alone was not enough: an audited
        race had an old run promote OLD bytes over a newer authorised
        upload's candidate -- the pointer had not moved, so the single
        CAS passed, and disk, candidate and index ended on three
        versions. The candidate alone is not enough either: two runs of
        the SAME candidate are indistinguishable without the attempt.
        A failed swap is CLASSIFIED rather than lumped together, because
        the three reasons mean different things to the caller: the
        candidate moved (`AttemptFenced` -- the work was pointless), the
        lease moved (`AttemptLeaseLost` -- this worker was displaced), or
        another promotion of the same attempt won the race.
      * The lease is released and the attempt closed as `done` in the
        SAME transaction as the swap: a promotion that swapped and then
        committed before releasing would leave a finished run holding a
        fence. Rule 4 is proven by fault injection at both of those
        writes, not by reading a successful end state.
      * The stale sweep of other generations happens only after all of
        the above.

    ``attempt_id`` is REQUIRED. It was briefly optional while the core
    ingest had not yet been bound to an attempt; that transitional gap is
    closed, and with it the possibility of a promotion nobody can trace
    to a run."""
    ids = {str(item) for item in manifest_ids}
    if not ids:
        raise ValueError(
            "bos manifest terfi edemez: sifir satirli bir nesli aktif "
            "yapmak saglam indeksi silmekle aynidir")
    if not content_sha256:
        raise ValueError(
            "terfi servis edilecek baytlarin ozetini ister; ozetsiz terfi "
            "reddedildi")
    if not candidate_id:
        raise ValueError(
            "terfi bagli oldugu aday kimligini ister; kimliksiz terfi "
            "reddedildi")
    if not attempt_id:
        raise ValueError(
            "terfi bagli oldugu deneme kimligini ister; kimliksiz terfi "
            "reddedildi")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM chunks WHERE document_id = %s "
            "AND generation = %s",
            (document_id, generation))
        staged = {str(row[0]) for row in cur.fetchall()}
        if staged != ids:
            conn.rollback()
            raise ValueError(
                f"manifest uyusmuyor: {len(ids - staged)} eksik, "
                f"{len(staged - ids)} yabanci satir; terfi reddedildi")
        lease_release = (", attempt_id = NULL, attempt_owner = NULL, "
                         "attempt_expires_at = NULL" if attempt_id else "")
        attempt_clause = ("AND attempt_id = %(attempt)s::uuid "
                          if attempt_id else "")
        cur.execute(
            "UPDATE documents SET active_generation = %(gen)s, "
            "active_content_sha = %(sha)s, "
            f"status = 'done', status_note = NULL{lease_release} "
            "WHERE id = %(id)s AND active_generation = %(expected)s "
            "AND candidate_id = %(cid)s::uuid "
            f"{attempt_clause}"
            "RETURNING id",
            {"gen": generation, "sha": content_sha256, "id": document_id,
             "expected": expected_active, "cid": candidate_id,
             "attempt": attempt_id})
        if cur.fetchone() is None:
            _explain_failed_promotion(cur, conn, document_id, candidate_id,
                                      attempt_id)
        if attempt_id:
            # The attempt's DONE is not a courtesy write at the end: rule
            # 4 says the swap, the lease release and this closure are ONE
            # success. Without checking the affected row, a missing or
            # already-terminal attempt record left the document swapped,
            # the lease cleared and the old generations deleted while the
            # attempt was never closed -- a committed transaction that
            # had failed half of what it claimed. Scoped by all three
            # identities so it cannot close some OTHER document's run.
            cur.execute(
                "UPDATE attempts SET status = %(status)s, ended_at = now() "
                "WHERE attempt_id = %(attempt)s::uuid "
                "AND document_id = %(document)s::uuid "
                "AND candidate_id = %(candidate)s::uuid "
                "AND status IS NULL "
                "RETURNING attempt_id",
                {"status": AttemptOutcome.DONE, "attempt": attempt_id,
                 "document": document_id, "candidate": candidate_id})
            if cur.fetchone() is None:
                conn.rollback()
                raise AttemptRecordInconsistent(
                    "belge satiri lease'i bu denemede gosteriyor ama deneme "
                    "kaydi kapatilamadi (kayit yok ya da zaten terminal); "
                    "terfi geri alindi")
        cur.execute(
            "DELETE FROM chunks WHERE document_id = %s AND generation != %s",
            (document_id, generation))
        removed = cur.rowcount
    conn.commit()
    return removed


def _explain_failed_promotion(cur, conn, document_id, candidate_id,
                              attempt_id):
    """Say WHICH of the three identities moved, then refuse.

    "The swap did not apply" is three different situations, and a caller
    that cannot tell them apart cannot react correctly: a fenced run
    should stop because its work is pointless, a displaced worker should
    stop because it no longer owns anything, and a run that merely lost a
    race to another promotion is a bug worth seeing.

    The diagnostic read REFINES the refusal; it never invents a new
    claim. A read that returns nothing is inconclusive, not evidence the
    document vanished -- we were updating that row a statement ago -- so
    it falls through to the generic verdict rather than reporting a
    disappearance nobody observed."""
    cur.execute(
        "SELECT candidate_id, attempt_id FROM documents WHERE id = %s",
        (document_id,))
    row = cur.fetchone()
    conn.rollback()
    if row is not None:
        current_candidate, held_by = row
        if str(current_candidate) != str(candidate_id):
            raise AttemptFenced(
                "aday degisti; bu kosu cevrildi ve terfi ETMEDI")
        if attempt_id and str(held_by) != str(attempt_id):
            raise AttemptLeaseLost(
                "lease artik bu denemenin degil; bu kosu terfi ETMEDI")
    raise ValueError(
        "aktif nesil ya da aday kimligi bu kosunun baglandigindan farkli; "
        "es zamanli bir islem kazandi, bu kosu terfi ETMEDI")


def hybrid_search(conn, dense_vec, sparse_indices, sparse_values, top_k=15, rrf_k=1) -> list[dict]:
    sparse_lit = sparse_to_literal(sparse_indices, sparse_values)
    cols = (
        "c.id, c.type, c.text, c.source_tag, c.page, c.headings, c.table_data, "
        "d.filename"
    )
    # Only the ACTIVE generation is retrievable. Without this filter an
    # audit probe pulled a partial re-ingest's staged rows and an older
    # version's rows into ONE context -- two editions of a document mixed
    # into the same answer. Legacy rows without a document row (NULL join)
    # stay reachable, as before the generation column existed.
    from_clause = (
        "chunks c LEFT JOIN documents d ON c.document_id = d.id "
        "AND c.generation = d.active_generation"
    )
    where_clause = "WHERE c.document_id IS NULL OR d.id IS NOT NULL"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {cols} FROM {from_clause} {where_clause} "
            f"ORDER BY c.dense <=> %s::vector LIMIT %s",
            (dense_vec, top_k),
        )
        dense_ranked = cur.fetchall()

        cur.execute(
            f"SELECT {cols} FROM {from_clause} {where_clause} "
            f"ORDER BY c.sparse <#> %s::sparsevec LIMIT %s",
            (sparse_lit, top_k),
        )
        sparse_ranked = cur.fetchall()

    # Reciprocal Rank Fusion: combine the two rankings into one score per chunk.
    scores: dict = {}
    payloads: dict = {}
    for ranked_list in (dense_ranked, sparse_ranked):
        for rank, row in enumerate(ranked_list, start=1):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
            payloads[rid] = row

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [payloads[rid] for rid, _ in fused]
