import hashlib
import contextvars
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
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
SCHEMA_VERSION = 1
# Stable across schema versions: old and new application revisions must
# serialize against each other during a rolling deploy.
_SCHEMA_LOCK_NAME = "ragtest-schema-migration"

# fastembed's Qdrant/bm25 hashes tokens (no fixed vocabulary) into a range that
# exceeds pgvector's sparsevec dimension cap (1_000_000_000). Every sparse index
# is remapped into [1, SPARSE_DIM] before storage or query -- this constant must
# match the `sparsevec(N)` dimension declared in schema.sql exactly.
SPARSE_DIM = 999_999_937
DEFAULT_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_EXECUTION_TENANT = contextvars.ContextVar(
    "rag_db_execution_tenant", default=(DEFAULT_TENANT_ID, False))


class DocumentLifecycleConflict(RuntimeError):
    """A document with an active ingest lease cannot change lifecycle."""


class IngestJobConflict(RuntimeError):
    """A document already has incompatible queued/running work."""


class IngestJobOwnershipLost(RuntimeError):
    """A worker tried to update a job it no longer owns."""


def set_tenant_context(conn, tenant_id=DEFAULT_TENANT_ID, *, service=False):
    """Bind a connection session to one tenant or to the internal worker.

    Session scope is intentional: publication commits between its stage and
    finalize operations while holding the same connection.  Callers returning
    a pooled connection must clear the setting first; ``api.db_conn`` owns that
    finally block.  Worker connections are short-lived and closed after one
    operation.
    """
    try:
        tenant = uuid.UUID(str(tenant_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("tenant_id gecerli bir UUID olmali") from exc
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('rag.tenant_id', %s, false)",
                    (str(tenant),))
        cur.execute("SELECT set_config('rag.service', %s, false)",
                    ("1" if service else "0",))


def clear_tenant_context(conn):
    """Remove request identity before a pooled connection can be reused."""
    with conn.cursor() as cur:
        cur.execute("RESET rag.tenant_id")
        cur.execute("RESET rag.service")
    conn.commit()


def bind_execution_tenant(tenant_id, *, service=False):
    tenant = uuid.UUID(str(tenant_id))
    return _EXECUTION_TENANT.set((tenant, bool(service)))


def reset_execution_tenant(token):
    _EXECUTION_TENANT.reset(token)


def get_conn(*, service=False) -> psycopg.Connection:
    conn = psycopg.connect(PG_DSN)
    register_vector(conn)
    tenant, inherited_service = _EXECUTION_TENANT.get()
    if service or inherited_service or tenant != DEFAULT_TENANT_ID:
        set_tenant_context(
            conn, tenant, service=bool(service or inherited_service))
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
    schema_path = Path(__file__).parent.joinpath("schema.sql")
    schema_bytes = schema_path.read_bytes()
    schema_sql = schema_bytes.decode("utf-8")
    schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
    with conn.cursor() as cur:
        # One transaction owns both DDL and its receipt. The xact advisory lock
        # is released by commit/rollback even if the process dies midway.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (_SCHEMA_LOCK_NAME,))
        cur.execute(schema_sql)
        cur.execute(
            "CREATE TABLE IF NOT EXISTS rag_schema_state ("
            "singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), "
            "schema_version integer NOT NULL, "
            "schema_sha256 text NOT NULL CHECK (length(schema_sha256) = 64), "
            "applied_at timestamptz NOT NULL DEFAULT now())")
        cur.execute(
            "INSERT INTO rag_schema_state "
            "(singleton, schema_version, schema_sha256, applied_at) "
            "VALUES (true, %s, %s, now()) "
            "ON CONFLICT (singleton) DO UPDATE SET "
            "schema_version = EXCLUDED.schema_version, "
            "schema_sha256 = EXCLUDED.schema_sha256, applied_at = now()",
            (SCHEMA_VERSION, schema_sha256))
    conn.commit()


def expected_schema_state() -> tuple[int, str]:
    raw = Path(__file__).parent.joinpath("schema.sql").read_bytes()
    return SCHEMA_VERSION, hashlib.sha256(raw).hexdigest()


def schema_is_current(conn) -> bool:
    """Exact version+digest readiness check; absence and drift are false."""
    expected = expected_schema_state()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schema_version, schema_sha256 FROM rag_schema_state "
                "WHERE singleton = true")
            row = cur.fetchone()
    except Exception:
        conn.rollback()
        return False
    return row is not None and tuple(row) == expected


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
            "ON CONFLICT (tenant_id, filename) DO UPDATE SET status = %(status)s, "
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
            "ON CONFLICT (tenant_id, filename) DO UPDATE SET "
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


def begin_attempt(conn, document_id: str, owner: str | None = None,
                  ingest_job_id: str | None = None,
                  ingest_job_worker: str | None = None):
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
    if (ingest_job_id is None) != (ingest_job_worker is None):
        raise ValueError("ingest job kimligi ve worker birlikte verilmeli")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT candidate_state, candidate_id, content_sha256, "
            "active_generation, attempt_id, archived_at, "
            "(attempt_expires_at IS NOT NULL AND attempt_expires_at > now()) "
            "FROM documents WHERE id = %s FOR UPDATE",
            (document_id,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise ValueError("belge kaydi yok; deneme baslatilamaz")
        (state, candidate_id, candidate_sha, active_generation,
         held_by, archived_at, lease_is_live) = row
        if archived_at is not None:
            conn.rollback()
            raise DocumentLifecycleConflict(
                "arsivlenmis belge icin ingest denemesi baslatilamaz")
        if state != CandidateState.PUBLISHED or candidate_id is None:
            conn.rollback()
            raise CandidateNotPublished(
                "aday yayimlanmis degil; bu belge simdilik islenemez")
        # The document lock closes the race between the synchronous endpoint
        # and queue insertion. A worker may pass only the live job it already
        # owns; every other caller must observe an entirely idle queue.
        cur.execute(
            "SELECT id, status, worker_id, "
            "(lease_expires_at IS NOT NULL AND lease_expires_at > now()) "
            "FROM ingest_jobs WHERE document_id = %s "
            "AND status IN ('queued', 'running') "
            "ORDER BY created_at, id LIMIT 1 FOR UPDATE",
            (document_id,))
        active_job = cur.fetchone()
        if ingest_job_id is None:
            if active_job is not None:
                conn.rollback()
                raise IngestJobConflict(
                    "etkin ingest job varken dogrudan deneme baslatilamaz")
        elif (active_job is None
              or str(active_job[0]) != str(ingest_job_id)
              or active_job[1] != "running"
              or active_job[2] != ingest_job_worker
              or not active_job[3]):
            conn.rollback()
            raise IngestJobOwnershipLost(
                "ingest job bu worker icin canli ve bagli degil")
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
            "status_note, active_generation, content_sha256, candidate_id, "
            "archived_at "
            "FROM documents WHERE id = %s", (document_id,))
        row = cur.fetchone()
    if row is None:
        return None
    if row.get("candidate_id") is not None:
        row["candidate_id"] = str(row["candidate_id"])
    return row


def _canonical_label(value: str, field: str = "name") -> tuple[str, str]:
    """Return one display spelling and its case-insensitive identity."""
    if not isinstance(value, str):
        raise ValueError(field + " metin olmali")
    display = value.strip()
    if not display:
        raise ValueError(field + " bos olmayan bir metin olmali")
    if any(ord(char) < 32 or ord(char) == 127 for char in display):
        raise ValueError(field + " kontrol karakteri iceremez")
    return display, display.casefold()


def create_collection(conn, name: str) -> dict:
    """Create one logical collection, idempotently across case variants."""
    display, name_key = _canonical_label(name)
    collection_id = str(uuid.uuid4())
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO collections (id, name, name_key) "
            "VALUES (%(id)s, %(name)s, %(name_key)s) "
            "ON CONFLICT (tenant_id, name_key) DO UPDATE SET name = collections.name "
            "RETURNING id, name, created_at",
            {"id": collection_id, "name": display, "name_key": name_key})
        row = cur.fetchone()
    conn.commit()
    return {"collection_id": str(row["id"]), "name": row["name"],
            "created_at": row["created_at"]}


def list_collections(conn) -> list[dict]:
    """List collections with their active membership count."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT c.id, c.name, c.created_at, "
            "COUNT(d.id) FILTER (WHERE d.archived_at IS NULL) AS document_count "
            "FROM collections c "
            "LEFT JOIN collection_documents cd ON cd.collection_id = c.id "
            "LEFT JOIN documents d ON d.id = cd.document_id "
            "GROUP BY c.id, c.name, c.created_at "
            "ORDER BY c.name_key, c.id")
        rows = cur.fetchall()
    return [{"collection_id": str(row["id"]), "name": row["name"],
             "created_at": row["created_at"],
             "document_count": int(row["document_count"])}
            for row in rows]


def list_tags(conn) -> list[dict]:
    """List the shared tag vocabulary with active document counts."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT t.id, t.name, t.created_at, "
            "COUNT(d.id) FILTER (WHERE d.archived_at IS NULL) AS document_count "
            "FROM tags t LEFT JOIN document_tags dt ON dt.tag_id = t.id "
            "LEFT JOIN documents d ON d.id = dt.document_id "
            "GROUP BY t.id, t.name, t.created_at ORDER BY t.name_key, t.id")
        rows = cur.fetchall()
    return [{"tag_id": str(row["id"]), "name": row["name"],
             "created_at": row["created_at"],
             "document_count": int(row["document_count"])}
            for row in rows]


def delete_tag(conn, tag_id: str) -> bool:
    """Delete one tag and its memberships, never any document."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tags WHERE id = %s", (tag_id,))
        deleted = cur.rowcount == 1
    conn.commit()
    return deleted


def delete_collection(conn, collection_id: str) -> bool:
    """Delete organisation metadata, never the member documents."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM collections WHERE id = %s", (collection_id,))
        deleted = cur.rowcount == 1
    conn.commit()
    return deleted


def set_collection_document(conn, collection_id: str, document_id: str,
                            present: bool) -> bool | None:
    """Idempotently attach/detach a document without changing the document."""
    if not isinstance(present, bool):
        raise ValueError("present bir boolean olmali")
    with conn.cursor() as cur:
        # Hold both referenced rows through the membership write.  Otherwise a
        # concurrent collection delete can pass between an EXISTS check and the
        # INSERT and turn an idempotent metadata operation into an FK race.
        cur.execute("SELECT id FROM collections WHERE id = %s FOR KEY SHARE",
                    (collection_id,))
        collection_exists = cur.fetchone() is not None
        cur.execute("SELECT id FROM documents WHERE id = %s FOR KEY SHARE",
                    (document_id,))
        document_exists = cur.fetchone() is not None
        if not collection_exists or not document_exists:
            conn.rollback()
            return None
        if present:
            cur.execute(
                "INSERT INTO collection_documents (collection_id, document_id) "
                "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (collection_id, document_id))
        else:
            cur.execute(
                "DELETE FROM collection_documents "
                "WHERE collection_id = %s AND document_id = %s",
                (collection_id, document_id))
    conn.commit()
    return present


def replace_document_tags(conn, document_id: str, tags) -> dict | None:
    """Replace a document's complete tag set under one document row lock."""
    if not isinstance(tags, (list, tuple)):
        raise ValueError("tags bir liste olmali")
    canonical = {}
    for value in tags:
        display, name_key = _canonical_label(value, "tag")
        canonical.setdefault(name_key, display)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id FROM documents WHERE id = %s FOR UPDATE",
                    (document_id,))
        if cur.fetchone() is None:
            conn.rollback()
            return None
        tag_ids = []
        for name_key, display in sorted(canonical.items()):
            tag_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO tags (id, name, name_key) "
                "VALUES (%(id)s, %(name)s, %(name_key)s) "
                "ON CONFLICT (tenant_id, name_key) DO UPDATE SET name = tags.name "
                "RETURNING id, name",
                {"id": tag_id, "name": display, "name_key": name_key})
            row = cur.fetchone()
            tag_ids.append((str(row["id"]), row["name"]))
        cur.execute("DELETE FROM document_tags WHERE document_id = %s",
                    (document_id,))
        if tag_ids:
            cur.executemany(
                "INSERT INTO document_tags (document_id, tag_id) VALUES (%s, %s)",
                [(document_id, tag_id) for tag_id, _name in tag_ids])
    conn.commit()
    return {"document_id": str(document_id),
            "tags": [name for _tag_id, name in tag_ids]}


def resolve_document_scope(conn, *, document_ids=None, collection_ids=None,
                           tags=None) -> tuple[str, ...] | None:
    """Intersect scope dimensions into active ids; collections ANY, tags ALL."""
    if document_ids is None and collection_ids is None and tags is None:
        return None
    params = {}
    clauses = ["d.archived_at IS NULL"]
    if document_ids is not None:
        if not isinstance(document_ids, (list, tuple)) or not document_ids:
            raise ValueError("document_ids bos olmayan bir liste olmali")
        params["document_ids"] = list(document_ids)
        clauses.append("d.id = ANY(%(document_ids)s::uuid[])")
    if collection_ids is not None:
        if not isinstance(collection_ids, (list, tuple)) or not collection_ids:
            raise ValueError("collection_ids bos olmayan bir liste olmali")
        params["collection_ids"] = list(collection_ids)
        clauses.append(
            "EXISTS (SELECT 1 FROM collection_documents cd "
            "WHERE cd.document_id = d.id "
            "AND cd.collection_id = ANY(%(collection_ids)s::uuid[]))")
    if tags is not None:
        if not isinstance(tags, (list, tuple)) or not tags:
            raise ValueError("tags bos olmayan bir liste olmali")
        tag_keys = sorted({_canonical_label(tag, "tag")[1] for tag in tags})
        params["tag_keys"] = tag_keys
        params["tag_count"] = len(tag_keys)
        clauses.append(
            "(SELECT COUNT(DISTINCT t.name_key) FROM document_tags dt "
            "JOIN tags t ON t.id = dt.tag_id WHERE dt.document_id = d.id "
            "AND t.name_key = ANY(%(tag_keys)s::text[])) = %(tag_count)s")
    with conn.cursor() as cur:
        cur.execute("SELECT d.id FROM documents d WHERE "
                    + " AND ".join(clauses) + " ORDER BY d.id", params)
        return tuple(str(row[0]) for row in cur.fetchall())


def active_document_ids(conn) -> tuple[str, ...]:
    """Return the active corpus visible through the connection's RLS policy."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE archived_at IS NULL ORDER BY id")
        return tuple(str(row[0]) for row in cur.fetchall())


def _job_key_digest(value: str) -> str:
    """Validate an HTTP idempotency token and retain only its digest."""
    if (not isinstance(value, str) or not value or value != value.strip()
            or len(value) > 200
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("idempotency key gecersiz")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _job_note(value) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or not value or len(value) > 100
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("job outcome note gecersiz")
    return value


def _public_job(row) -> dict:
    return {
        "job_id": str(row["id"]),
        "document_id": str(row["document_id"]),
        "candidate_id": str(row["candidate_id"]),
        "status": row["status"],
        "attempt_count": int(row["attempt_count"]),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "outcome_note": row["outcome_note"],
    }


def enqueue_ingest_job(conn, document_id: str, idempotency_key: str) -> dict | None:
    """Persist one candidate-bound job, idempotently and one-active-per-doc."""
    key_digest = _job_key_digest(idempotency_key)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, candidate_id, content_sha256, candidate_state, "
            "archived_at, attempt_id FROM documents WHERE id = %s FOR UPDATE",
            (document_id,))
        document = cur.fetchone()
        if document is None:
            conn.rollback()
            return None
        cur.execute(
            "SELECT * FROM ingest_jobs WHERE document_id = %s "
            "AND idempotency_key_sha256 = %s",
            (document_id, key_digest))
        existing = cur.fetchone()
        if existing is not None:
            conn.commit()
            return _public_job(existing)
        if document["archived_at"] is not None:
            raise DocumentLifecycleConflict(
                "arsivlenmis belge kuyruga alinamaz")
        if (document["candidate_state"] != CandidateState.PUBLISHED
                or document["candidate_id"] is None
                or not document["content_sha256"]):
            raise CandidateNotPublished("yayimlanmis aday olmadan job olusmaz")
        if document["attempt_id"] is not None:
            raise IngestJobConflict("belgenin canli ingest denemesi var")
        cur.execute(
            "SELECT id FROM ingest_jobs WHERE document_id = %s "
            "AND status IN ('queued', 'running')",
            (document_id,))
        if cur.fetchone() is not None:
            raise IngestJobConflict("belgenin etkin ingest job'i var")
        job_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO ingest_jobs "
            "(id, document_id, candidate_id, candidate_sha, "
            "idempotency_key_sha256) "
            "VALUES (%(id)s, %(document)s, %(candidate)s, %(sha)s, %(key)s) "
            "RETURNING *",
            {"id": job_id, "document": document_id,
             "candidate": str(document["candidate_id"]),
             "sha": document["content_sha256"], "key": key_digest})
        row = cur.fetchone()
    conn.commit()
    return _public_job(row)


def get_ingest_job(conn, job_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM ingest_jobs WHERE id = %s", (job_id,))
        row = cur.fetchone()
    return None if row is None else _public_job(row)


def active_ingest_job(conn, document_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM ingest_jobs WHERE document_id = %s "
            "AND status IN ('queued', 'running') ORDER BY created_at, id LIMIT 1",
            (document_id,))
        row = cur.fetchone()
    return None if row is None else _public_job(row)


def claim_ingest_job(conn, worker_id: str, lease_seconds: int = 300,
                     max_attempts: int = 3) -> dict | None:
    """Claim one queued/expired job with SKIP LOCKED for concurrent workers."""
    worker_id, _worker_key = _canonical_label(worker_id, "worker_id")
    if (not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool)
            or lease_seconds < 30 or lease_seconds > 7200):
        raise ValueError("job lease 30..7200 saniye olmali")
    if (not isinstance(max_attempts, int) or isinstance(max_attempts, bool)
            or max_attempts < 1 or max_attempts > 20):
        raise ValueError("max_attempts 1..20 olmali")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "UPDATE ingest_jobs SET status = 'failed', finished_at = now(), "
            "worker_id = NULL, lease_expires_at = NULL, "
            "outcome_note = 'attempts_exhausted' "
            "WHERE status = 'running' AND lease_expires_at <= now() "
            "AND attempt_count >= %s", (max_attempts,))
        cur.execute(
            "WITH next_job AS (SELECT id FROM ingest_jobs "
            "WHERE (status = 'queued' OR (status = 'running' "
            "AND lease_expires_at <= now())) "
            "AND attempt_count < %(max_attempts)s "
            "ORDER BY created_at, id FOR UPDATE SKIP LOCKED LIMIT 1) "
            "UPDATE ingest_jobs j SET status = 'running', "
            "worker_id = %(worker)s, attempt_count = attempt_count + 1, "
            "started_at = COALESCE(started_at, now()), finished_at = NULL, "
            "outcome_note = NULL, "
            "lease_expires_at = now() + make_interval(secs => %(lease)s) "
            "FROM next_job WHERE j.id = next_job.id RETURNING j.*",
            {"worker": worker_id, "lease": lease_seconds,
             "max_attempts": max_attempts})
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        cur.execute(
            "SELECT filename, archived_at, candidate_id, content_sha256 "
            "FROM documents WHERE id = %s", (row["document_id"],))
        document = cur.fetchone()
    conn.commit()
    claimed = _public_job(row)
    claimed.update({
        "tenant_id": str(row["tenant_id"]),
        "filename": None if document is None else document["filename"],
        "archived_at": None if document is None else document["archived_at"],
        "bound_candidate_id": str(row["candidate_id"]),
        "bound_candidate_sha": row["candidate_sha"],
        "current_candidate_id": (None if document is None else
                                 str(document["candidate_id"])),
        "current_candidate_sha": (None if document is None else
                                  document["content_sha256"]),
    })
    return claimed


def heartbeat_ingest_job(conn, job_id: str, worker_id: str,
                         lease_seconds: int = 300) -> bool:
    if (not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool)
            or not 30 <= lease_seconds <= 7200):
        raise ValueError("job lease 30..7200 saniye olmali")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs SET lease_expires_at = "
            "now() + make_interval(secs => %(lease)s) "
            "WHERE id = %(id)s AND status = 'running' "
            "AND worker_id = %(worker)s AND lease_expires_at > now()",
            {"lease": lease_seconds, "id": job_id, "worker": worker_id})
        moved = cur.rowcount == 1
    conn.commit()
    return moved


def finish_ingest_job(conn, job_id: str, worker_id: str, status: str,
                      note: str | None = None) -> bool:
    if status not in {"succeeded", "partial", "failed"}:
        raise ValueError("job terminal status gecersiz")
    note = _job_note(note)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_jobs SET status = %(status)s, finished_at = now(), "
            "worker_id = NULL, lease_expires_at = NULL, outcome_note = %(note)s "
            "WHERE id = %(id)s AND status = 'running' "
            "AND worker_id = %(worker)s AND lease_expires_at > now()",
            {"status": status, "note": note, "id": job_id,
             "worker": worker_id})
        changed = cur.rowcount == 1
    conn.commit()
    return changed


def retry_ingest_job(conn, job_id: str, worker_id: str,
                     note: str, max_attempts: int = 3) -> str:
    """Requeue while budget remains; otherwise close failed."""
    note = _job_note(note)
    if (not isinstance(max_attempts, int) or isinstance(max_attempts, bool)
            or max_attempts < 1 or max_attempts > 20):
        raise ValueError("max_attempts 1..20 olmali")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "UPDATE ingest_jobs SET "
            "status = CASE WHEN attempt_count < %(max)s THEN 'queued' "
            "ELSE 'failed' END, "
            "finished_at = CASE WHEN attempt_count < %(max)s THEN NULL "
            "ELSE now() END, worker_id = NULL, lease_expires_at = NULL, "
            "outcome_note = %(note)s "
            "WHERE id = %(id)s AND status = 'running' "
            "AND worker_id = %(worker)s AND lease_expires_at > now() "
            "RETURNING status",
            {"max": max_attempts, "note": note, "id": job_id,
             "worker": worker_id})
        row = cur.fetchone()
    conn.commit()
    if row is None:
        raise IngestJobOwnershipLost("ingest job lease artik bu worker'in degil")
    return row["status"]


def cancel_ingest_job(conn, job_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM ingest_jobs WHERE id = %s FOR UPDATE",
                    (job_id,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            return None
        if row["status"] == "running":
            raise IngestJobConflict("calisan job iptal edilemez")
        if row["status"] == "queued":
            cur.execute(
                "UPDATE ingest_jobs SET status = 'cancelled', "
                "finished_at = now(), outcome_note = 'cancelled_by_user' "
                "WHERE id = %s RETURNING *", (job_id,))
            row = cur.fetchone()
    conn.commit()
    return _public_job(row)


# The ONE escape character every LIKE pattern in this module is built
# with. It is a literal in the code and is NEVER taken from a caller: an
# operator or an escape character chosen by input would be a fragment of
# the statement chosen by input, which is exactly what parameterization
# exists to prevent. `!` rather than the SQL default backslash, so a
# filename carrying a backslash needs no second layer of reasoning --
# here a backslash is an ordinary character with no meaning at all.
DOCUMENT_SEARCH_ESCAPE = "!"


def escape_like_pattern(value: str) -> str:
    """Neutralise LIKE's own metacharacters in a value meant LITERALLY.

    PARAMETERIZATION AND ESCAPING ANSWER TWO DIFFERENT QUESTIONS, and
    both are required. psycopg adapts a `str` as a VALUE only -- measured:
    percent, underscore, exclamation, a quote and a backslash all travel
    byte-identical, with no quoting and no SQL fragment -- so injection is
    already closed by the parameter. What a parameter does NOT decide is
    what the value MEANS to `LIKE`: there `%` is still "any run of
    characters" and `_` is still "any one character", so a caller
    searching for a literal `%` would otherwise match every row. This
    fixes the PATTERN's meaning; the parameter keeps it a value.

    THE ORDER IS THE WHOLE CONTRACT: the escape character FIRST, then
    `%`, then `_`. Measured: escaping the escape character LAST
    double-escapes what the earlier steps just inserted -- `%` becomes
    `!%` and then `!!%`, which the server reads as a literal `!` followed
    by the wildcard, so a literal-percent search silently finds NOTHING.
    Only this order round-trips.

    Pure: it reads its argument, mutates nothing, and returns a new
    value. There is no length rule here on purpose -- `documents.filename`
    is unbounded `text`, so a cap declared at this seam would refuse a
    value the database itself stores.
    """
    if not isinstance(value, str):
        raise TypeError("escape_like_pattern yalnizca metin alir")
    escape = DOCUMENT_SEARCH_ESCAPE
    return (value
            .replace(escape, escape + escape)
            .replace("%", escape + "%")
            .replace("_", escape + "_"))


def list_documents(conn, limit: int, offset: int,
                   status: str | None = None,
                   file_type: str | None = None,
                   uploaded_after: datetime | None = None,
                   uploaded_before: datetime | None = None,
                   q: str | None = None,
                   archived: bool = False,
                   collection_id: str | None = None,
                   tag: str | None = None) -> list[dict]:
    """One page of the document inventory, newest first.

    SIX FILTERS. `archived=False` is the fail-closed default: the normal
    inventory contains only active rows, while `archived=True` contains only
    archived rows. It is static SQL authority and never a bound value.

    The other five are optional. `status` and `file_type` are exact equality;
    neither column has a closed vocabulary -- `status` is free text with
    no CHECK constraint and `file_type` is whatever suffix the upload
    carried -- so no value set is enforced here: an unknown value simply
    matches nothing. `uploaded_after` and `uploaded_before` bound
    `uploaded_at`, both EXCLUSIVELY (`>` and `<`), so a row sitting
    exactly on a bound falls outside the window and two adjoining
    windows can never both claim it. Any subset may be given and they
    combine with AND. A filter that was not supplied appears NEITHER in
    the statement NOR in the params dict; the WHERE clause is assembled
    only from this function's own static pieces, and every supplied
    value travels as a parameter. Filters narrow the scan BEFORE
    LIMIT/OFFSET, so the page, the offset and the `limit + 1` sentinel
    all describe the filtered sequence.

    `q` IS THE FIFTH OPTIONAL FILTER, AND THE ONLY ONE THAT IS NOT EQUALITY:
    it searches
    `documents.filename` ALONE, case-insensitively, for a LITERAL
    substring -- `ILIKE` against the pattern wrapped in `%` on both
    sides. LITERAL is the part that takes work, because `%` and `_` are
    LIKE's own metacharacters: the raw value goes through
    `escape_like_pattern` first and the clause names the escape
    character, so a search for `%` finds the rows whose name really
    carries one instead of every row in the table. The RAW value never
    enters the statement -- only the transformed pattern enters the
    params dict -- and the operator, the wrapping `%` and the escape
    character are static text here, never anything a caller sent.

    NO LENGTH RULE AND NO CHARACTER RULE IS INVENTED FOR `q`. `filename`
    is unbounded `text`, so a cap declared here would refuse a value the
    column itself stores. And the UPLOAD validator is not reusable as
    one: it rejects slashes, colons, control characters and trailing
    spaces -- every one of them a legitimate thing to SEARCH for, so
    reusing it would silently narrow the search rather than protect
    anything. The only shape asked for is the one the filter needs to
    mean anything: absent, or a non-empty string.

    THE DATE BOUNDS ARRIVE AS DATETIME OBJECTS, NOT TEXT. `uploaded_at`
    is `timestamptz`, psycopg adapts an aware datetime to it natively,
    and an instant is only an instant if it carries an offset -- so this
    seam takes `None` or an AWARE datetime and nothing else. A string is
    refused rather than parsed: turning text into an instant needs a
    timezone policy, that policy belongs to the layer talking to the
    caller, and a second one down here would be a second answer to the
    same question. A naive datetime is refused for the same reason it
    cannot be compared: nobody said which instant it is.

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
    if not isinstance(archived, bool):
        raise ValueError("archived bir boolean olmali")
    # The same courtesy for the filters: the API refuses a malformed one
    # in its own signature, this refuses it for every OTHER caller --
    # before any statement is built, so a refused call executes nothing.
    # `q` is checked HERE, with them, for exactly that reason: the
    # transform below turns a value into a PATTERN, and a transform is
    # arithmetic on a value nobody checked unless the check comes first.
    for name, value in (("status", status), ("file_type", file_type),
                         ("q", q), ("collection_id", collection_id)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(name + " bos olmayan bir metin olmali")
    if tag is not None:
        _display_tag, tag_key = _canonical_label(tag, "tag")
    # A bound that is not an aware datetime is not a bound. `utcoffset()`
    # rather than `tzinfo is not None`: a tzinfo may be attached and still
    # answer None for this instant, which leaves the value exactly as
    # uncomparable as a naive one. Text is refused here rather than
    # parsed -- see the docstring.
    for name, value in (("uploaded_after", uploaded_after),
                        ("uploaded_before", uploaded_before)):
        if value is None:
            continue
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError(
                name + " zaman dilimi bilgisi tasiyan bir datetime olmali")
    # An empty window (equal bounds, both exclusive) and a reversed one
    # can match no row at all, so they are a mistake to report rather than
    # an empty page to hand back. Comparing two AWARE datetimes compares
    # the instants, so the same moment written with two different offsets
    # is refused as equal -- as it should be, whatever the texts looked
    # like.
    if (uploaded_after is not None and uploaded_before is not None
            and uploaded_after >= uploaded_before):
        raise ValueError(
            "uploaded_after, uploaded_before'dan kesin olarak once olmali")
    # Only these static pieces are ever assembled into the statement; the
    # VALUES never are -- each supplied filter adds its clause here and
    # its value to the params dict, and an unsupplied one adds neither.
    clauses = ["archived_at IS NOT NULL" if archived
               else "archived_at IS NULL"]
    params = {"limit": limit + 1, "offset": offset}
    if status is not None:
        clauses.append("status = %(status)s")
        params["status"] = status
    if file_type is not None:
        clauses.append("file_type = %(file_type)s")
        params["file_type"] = file_type
    # The datetime OBJECT goes into params; psycopg adapts it to
    # timestamptz itself. Rendering it to text here would hand the server
    # a string to re-parse under ITS timezone setting -- the one thing the
    # aware requirement above exists to avoid.
    if uploaded_after is not None:
        clauses.append("uploaded_at > %(uploaded_after)s")
        params["uploaded_after"] = uploaded_after
    if uploaded_before is not None:
        clauses.append("uploaded_at < %(uploaded_before)s")
        params["uploaded_before"] = uploaded_before
    # THE SEARCH CLAUSE IS ONE LITERAL STRING -- column, operator and
    # escape character all spelled out, nothing interpolated at all. It
    # used to be assembled with an f-string around the module constant;
    # the value substituted was code-owned and constant, so it was safe,
    # but it was still SQL text being built at runtime and that is the
    # shape a later reader copies. The constant and the clause must
    # therefore agree, and a test pins each of them separately rather
    # than one deriving from the other.
    #
    # What the caller sent reaches the database only as the value of
    # `%(filename_search)s`, wrapped into a substring pattern AFTER
    # escaping -- the two wrapping `%` are ours and are meant as
    # wildcards, every `%` inside the value is the caller's and has been
    # made literal.
    if q is not None:
        clauses.append("filename ILIKE %(filename_search)s ESCAPE '!'")
        params["filename_search"] = "%" + escape_like_pattern(q) + "%"
    if collection_id is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM collection_documents cd "
            "WHERE cd.document_id = documents.id "
            "AND cd.collection_id = %(collection_id)s::uuid)")
        params["collection_id"] = collection_id
    if tag is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM document_tags dt JOIN tags t ON t.id = dt.tag_id "
            "WHERE dt.document_id = documents.id AND t.name_key = %(tag_key)s)")
        params["tag_key"] = tag_key
    where_sql = "WHERE " + " AND ".join(clauses) + " " if clauses else ""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, filename, file_type, uploaded_at, status, "
            "status_note, active_generation, archived_at "
            "FROM documents "
            + where_sql +
            "ORDER BY uploaded_at DESC, id DESC "
            "LIMIT %(limit)s OFFSET %(offset)s",
            params)
        rows = cur.fetchall()
    listed = []
    for row in rows:
        # `id` is a uuid object; every other reader of this row is JSON,
        # so it is stringified here under the name the API publishes it by
        row["document_id"] = str(row.pop("id"))
        listed.append(row)
    return listed


def set_document_archived(conn, document_id: str,
                          archived: bool) -> dict | None:
    """Idempotently archive or restore one document under a row lock.

    An active ingest attempt owns the document's mutable publication state;
    changing lifecycle beside it would let a run publish into an archive, or
    restore a row while a displaced worker still writes. Refuse that race.
    """
    if not isinstance(archived, bool):
        raise ValueError("archived bir boolean olmali")
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, archived_at, attempt_id FROM documents "
            "WHERE id = %s FOR UPDATE", (document_id,))
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        already = row["archived_at"] is not None
        if already == archived:
            # Release the FOR UPDATE transaction here as well.  Idempotent
            # lifecycle calls are complete operations, not locks handed to
            # the caller to remember to close later.
            conn.commit()
            return {"document_id": str(row["id"]),
                    "archived": already,
                    "archived_at": row["archived_at"]}
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM ingest_jobs "
            "WHERE document_id = %s AND status IN ('queued', 'running')) "
            "AS has_active_job",
            (document_id,))
        if cur.fetchone()["has_active_job"]:
            raise DocumentLifecycleConflict(
                "etkin ingest job varken belge yasam dongusu degisemez")
        if row["attempt_id"] is not None:
            raise DocumentLifecycleConflict(
                "aktif ingest denemesi varken belge yasam dongusu degisemez")
        cur.execute(
            "UPDATE documents SET archived_at = "
            "CASE WHEN %(archived)s THEN now() ELSE NULL END "
            "WHERE id = %(id)s RETURNING id, archived_at",
            {"archived": archived, "id": document_id})
        changed = cur.fetchone()
    conn.commit()
    return {"document_id": str(changed["id"]),
            "archived": changed["archived_at"] is not None,
            "archived_at": changed["archived_at"]}


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


def filenames_for_documents(conn, document_ids) -> list[str]:
    """The names a set of document identifiers resolves to. SELECT only.

    `documents.filename` is `text NOT NULL UNIQUE`, so a name identifies a
    document exactly as an id does -- which is what makes it usable as the
    LlamaIndex engine's scope. That index carries `{page, type, filename}`
    in its node metadata and NO identifier at all, so a scope expressed in
    ids has to be resolved to names somewhere; resolving it HERE, against
    the same `documents` table the rest of the system keys on, keeps ONE
    authority for what a document identifier is instead of teaching the
    other engine a second one.

    The identifiers travel as a SINGLE array parameter and the statement is
    closed text -- no identifier is ever assembled into it. An identifier
    matching no row simply contributes no name, so an unknown one NARROWS
    the answer rather than widening it: the caller receives an empty list
    and must read it as an empty scope, never as "no scope".

    The names come back deduplicated and ordered, so a scope that repeats a
    document (or two ids that somehow resolve to one name) produces one
    filter value rather than a repeated one.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT filename FROM documents WHERE id = ANY(%s::uuid[]) "
            "AND archived_at IS NULL",
            (list(document_ids),))
        return sorted({str(row[0]) for row in cur.fetchall()})


def active_document_filenames(conn) -> list[str]:
    """Names currently allowed into the snapshot-backed retrieval engine.

    LlamaIndex stores filename metadata in a snapshot.  Lifecycle state can
    change after that snapshot was built, so its query-time filter must come
    from the live documents table rather than from stale node metadata.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT filename FROM documents "
            "WHERE archived_at IS NULL ORDER BY filename")
        return sorted({str(row[0]) for row in cur.fetchall()})


def lock_retrieval_filenames(conn, document_ids=None) -> list[str]:
    """Lock the active rows that a snapshot-backed retrieval may publish.

    The caller keeps this transaction open through retrieval.  Archive and
    restore use ``FOR UPDATE`` on the same rows, so a lifecycle transition
    either completes before this read (and is excluded) or waits until the
    in-flight answer has finished.  No stale snapshot node crosses the seam.
    """
    params = None
    where = "archived_at IS NULL"
    if document_ids is not None:
        where += " AND id = ANY(%s::uuid[])"
        params = (list(document_ids),)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT filename FROM documents WHERE " + where
            + " ORDER BY filename FOR SHARE",
            params)
        return sorted({str(row[0]) for row in cur.fetchall()})


# THE ONE SCOPE CLAUSE, written once as static text. `hybrid_search` runs
# TWO statements over one code-owned WHERE clause and fuses their rankings
# with RRF; fusion cannot tell which ranking a row arrived on, so scoping
# one statement and not the other would let an out-of-scope candidate into
# the fused result through the unscoped side. Naming the clause once is
# what makes "the same clause on both" a property of the code rather than
# of remembering. No identifier appears in it -- the whole set travels as a
# single array parameter, which is also why a repeated identifier can
# neither widen the scope nor add a second filter.
DOCUMENT_SCOPE_CLAUSE = "c.document_id = ANY(%s::uuid[])"
RRF_CANDIDATE_MULTIPLIER = 4
RRF_CANDIDATE_CEILING = 200


def rrf_candidate_limit(top_k):
    """Bounded breadth for each ranking before reciprocal-rank fusion.

    Pulling only ``top_k`` from each modality makes fusion blind to a chunk
    ranked just below both individual cuts even when the two rankings together
    make it the strongest candidate.  Four times the requested result size is
    enough room for agreement to surface, while the absolute ceiling keeps a
    caller from turning one request into an unbounded materialisation.  A
    request already above the ceiling is never silently shrunk.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k pozitif bir tamsayi olmali")
    return min(top_k * RRF_CANDIDATE_MULTIPLIER,
               max(top_k, RRF_CANDIDATE_CEILING))


def reciprocal_rank_fusion(dense_ranked, sparse_ranked, top_k, rrf_k):
    """Fuse two complete rankings with stable ties and closed identities."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k pozitif bir tamsayi olmali")
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k negatif olmayan bir tamsayi olmali")
    scores, payloads = {}, {}
    for ranked_list in (dense_ranked, sparse_ranked):
        seen = set()
        for rank, row in enumerate(ranked_list, start=1):
            if type(row) is not dict or row.get("id") is None:
                raise ValueError("siralanmis chunk kapali bir id tasimali")
            rid = row["id"]
            if rid in seen:
                raise ValueError("bir ranking ayni chunk kimligini tekrarladi")
            seen.add(rid)
            if rid in payloads and dict(payloads[rid]) != dict(row):
                raise ValueError("iki ranking ayni chunk icin farkli veri tasidi")
            payloads.setdefault(rid, row)
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores, key=lambda rid: (-scores[rid], str(rid)))[:top_k]
    return [payloads[rid] for rid in fused]


def hybrid_search(conn, dense_vec, sparse_indices, sparse_values, top_k=15,
                  rrf_k=1, *, document_ids=None) -> list[dict]:
    """Hybrid retrieval, optionally scoped to a named set of documents.

    `document_ids` is keyword-only with a default, so every existing
    positional call site keeps its meaning. Absent, the statements are the
    ones this function has always sent: no scope clause and no scope
    parameter. Supplied, the clause is ANDed onto BOTH statements, so the
    dense and the sparse ranking are drawn from the same set and nothing
    from outside it can reach the fusion.

    THE SCOPE IS APPLIED IN THE QUERY, not to the candidates it returns. A
    filter after `LIMIT` would answer a scoped question with whatever
    survived an UNSCOPED top-k: a document that really does hold the answer
    would come back empty whenever other documents filled the pool first.

    An empty set is a legitimate scope and matches nothing -- an identifier
    that names no document must never fall back to the whole corpus.
    """
    candidate_limit = rrf_candidate_limit(top_k)
    if type(rrf_k) is not int or rrf_k < 0:
        raise ValueError("rrf_k negatif olmayan bir tamsayi olmali")
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
        "AND c.generation = d.active_generation "
        "AND d.archived_at IS NULL"
    )
    # The generation test is parenthesised because it is an OR: ANDing the
    # scope onto a bare `a OR b` would bind to `b` alone and leave every
    # legacy NULL-document row reachable from OUTSIDE the requested scope.
    where_clause = "WHERE (c.document_id IS NULL OR d.id IS NOT NULL)"
    # A supplied scope adds its clause here and its value to the parameter
    # tuple; an absent one adds NEITHER. The scope parameter comes first in
    # both tuples because its placeholder sits in the WHERE clause, ahead of
    # the ORDER BY and LIMIT placeholders -- these statements are positional
    # `%s`, so the order of the tuple IS the binding.
    scope_params = ()
    if document_ids is not None:
        where_clause += " AND " + DOCUMENT_SCOPE_CLAUSE
        # a list, not a tuple: psycopg adapts a list to an ARRAY, which is
        # what `= ANY(...)` reads, and a tuple to a record, which it does not
        scope_params = (list(document_ids),)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {cols} FROM {from_clause} {where_clause} "
            f"ORDER BY c.dense <=> %s::vector, c.id LIMIT %s",
            (*scope_params, dense_vec, candidate_limit),
        )
        dense_ranked = cur.fetchall()

        cur.execute(
            f"SELECT {cols} FROM {from_clause} {where_clause} "
            f"ORDER BY c.sparse <#> %s::sparsevec, c.id LIMIT %s",
            (*scope_params, sparse_lit, candidate_limit),
        )
        sparse_ranked = cur.fetchall()

    return reciprocal_rank_fusion(dense_ranked, sparse_ranked, top_k, rrf_k)
