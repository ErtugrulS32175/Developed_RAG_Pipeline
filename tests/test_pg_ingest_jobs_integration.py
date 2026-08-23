"""Real PostgreSQL proof for durable ingest-job locking and idempotency."""
import os
from pathlib import Path
import uuid

import pytest

from pipeline.index import db
from pipeline.index.attempt_contract import AttemptOutcome


DSN = os.getenv("RAGTEST_JOB_PG_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not DSN, reason="RAGTEST_JOB_PG_DSN is absent: real job SQL was not checked")

DOCUMENT = uuid.UUID("11111111-1111-1111-1111-111111111111")
CANDIDATE = uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture(scope="module")
def real_job_connection():
    import psycopg
    from psycopg import sql

    schema_name = f"ragtest_jobs_{uuid.uuid4().hex[:12]}"
    admin = psycopg.connect(DSN, autocommit=True)
    connection = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(
                sql.Identifier(schema_name)))
        connection = psycopg.connect(DSN)
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema_name)))
            cursor.execute(Path(db.__file__).with_name("schema.sql").read_text())
            cursor.execute(
                "INSERT INTO documents "
                "(id, filename, file_type, candidate_id, content_sha256, "
                "candidate_state) VALUES (%s, 'alpha.pdf', 'pdf', %s, %s, %s)",
                (DOCUMENT, CANDIDATE, "a" * 64, "published"))
        connection.commit()
        yield connection
    finally:
        if connection is not None:
            connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema_name)))
        finally:
            admin.close()


def test_real_queue_is_idempotent_claimed_once_and_hides_the_raw_key(
        real_job_connection):
    first = db.enqueue_ingest_job(
        real_job_connection, DOCUMENT, "request-alpha")
    same = db.enqueue_ingest_job(
        real_job_connection, DOCUMENT, "request-alpha")
    assert same["job_id"] == first["job_id"]
    with pytest.raises(db.IngestJobConflict):
        db.enqueue_ingest_job(real_job_connection, DOCUMENT, "request-beta")
    real_job_connection.rollback()
    with pytest.raises(db.IngestJobConflict):
        db.begin_attempt(real_job_connection, DOCUMENT)
    real_job_connection.rollback()

    claimed = db.claim_ingest_job(
        real_job_connection, "worker-alpha", lease_seconds=30)
    assert claimed["job_id"] == first["job_id"]
    assert claimed["attempt_count"] == 1
    assert db.claim_ingest_job(
        real_job_connection, "worker-beta", lease_seconds=30) is None
    assert db.heartbeat_ingest_job(
        real_job_connection, first["job_id"], "worker-beta", 30) is False
    assert db.heartbeat_ingest_job(
        real_job_connection, first["job_id"], "worker-alpha", 30) is True
    with pytest.raises(db.IngestJobOwnershipLost):
        db.begin_attempt(
            real_job_connection, DOCUMENT, ingest_job_id=first["job_id"],
            ingest_job_worker="worker-beta")
    real_job_connection.rollback()
    attempt = db.begin_attempt(
        real_job_connection, DOCUMENT, owner="job/worker-alpha",
        ingest_job_id=first["job_id"], ingest_job_worker="worker-alpha")
    assert db.record_attempt_outcome(
        real_job_connection, attempt, AttemptOutcome.ERROR, "test")
    assert db.finish_ingest_job(
        real_job_connection, first["job_id"], "worker-alpha", "succeeded")

    with real_job_connection.cursor() as cursor:
        cursor.execute(
            "SELECT idempotency_key_sha256 FROM ingest_jobs WHERE id = %s",
            (first["job_id"],))
        stored = cursor.fetchone()[0]
    assert stored == db._job_key_digest("request-alpha")
    assert "request-alpha" not in stored


def test_real_queue_cancellation_and_lifecycle_conflict_are_closed(
        real_job_connection):
    queued = db.enqueue_ingest_job(
        real_job_connection, DOCUMENT, "request-gamma")
    with pytest.raises(db.DocumentLifecycleConflict):
        db.set_document_archived(real_job_connection, DOCUMENT, True)
    real_job_connection.rollback()

    cancelled = db.cancel_ingest_job(real_job_connection, queued["job_id"])
    assert cancelled["status"] == "cancelled"
    again = db.cancel_ingest_job(real_job_connection, queued["job_id"])
    assert again["status"] == "cancelled"
    archived = db.set_document_archived(real_job_connection, DOCUMENT, True)
    assert archived["archived"] is True
    with pytest.raises(db.DocumentLifecycleConflict):
        db.enqueue_ingest_job(real_job_connection, DOCUMENT, "request-delta")
    real_job_connection.rollback()
    db.set_document_archived(real_job_connection, DOCUMENT, False)


def test_real_retry_budget_requeues_then_closes_failed(real_job_connection):
    queued = db.enqueue_ingest_job(
        real_job_connection, DOCUMENT, "request-epsilon")
    first = db.claim_ingest_job(
        real_job_connection, "worker-alpha", lease_seconds=30,
        max_attempts=2)
    assert first["job_id"] == queued["job_id"]
    assert db.retry_ingest_job(
        real_job_connection, queued["job_id"], "worker-alpha",
        "OSError", max_attempts=2) == "queued"
    second = db.claim_ingest_job(
        real_job_connection, "worker-beta", lease_seconds=30,
        max_attempts=2)
    assert second["attempt_count"] == 2
    assert db.retry_ingest_job(
        real_job_connection, queued["job_id"], "worker-beta",
        "OSError", max_attempts=2) == "failed"
    assert db.get_ingest_job(
        real_job_connection, queued["job_id"])["status"] == "failed"
