"""Durable ingest-job worker.

The queue decides *what* should run and survives process restarts.  The existing
attempt lease remains the only authority allowed to mutate chunks or publish a
generation.  Job state never substitutes for that fencing token.
"""
import argparse
import os
import socket
import threading
import time
from pathlib import Path

from pipeline.index import db, ingest, publication
from pipeline.index.attempt_contract import (
    AttemptAlreadyRunning,
    AttemptFenced,
    AttemptLeaseLost,
    AttemptOutcome,
    CandidateNotPublished,
)


class StaleIngestJob(RuntimeError):
    """The document no longer carries the candidate bound at enqueue time."""


def _worker_id() -> str:
    return f"{socket.gethostname()}/{os.getpid()}"


def _source_path(upload_dir: Path, filename: str,
                 tenant_id=db.DEFAULT_TENANT_ID) -> Path:
    if (not isinstance(filename, str) or not filename
            or Path(filename).name != filename
            or "/" in filename or "\\" in filename):
        raise StaleIngestJob("stored_filename_invalid")
    try:
        candidate = publication.source_path(upload_dir, filename, tenant_id)
    except (TypeError, ValueError, publication.UnsafeCanonicalName) as exc:
        raise StaleIngestJob("stored_filename_invalid") from exc
    if not candidate.is_file():
        raise StaleIngestJob("source_file_unavailable")
    return candidate


def _heartbeat(stop, lost, job_id, worker_id, lease_seconds):
    interval = max(10, lease_seconds // 3)
    while not stop.wait(interval):
        conn = db.get_conn(service=True)
        try:
            if not db.heartbeat_ingest_job(
                    conn, job_id, worker_id, lease_seconds):
                lost.set()
                return
        except Exception:
            # A failed renewal is indistinguishable from lost ownership to
            # this worker. Fail closed; never publish on an assumed lease.
            lost.set()
            return
        finally:
            conn.close()


def _finish(job_id, worker_id, status, note=None):
    conn = db.get_conn()
    try:
        if not db.finish_ingest_job(conn, job_id, worker_id, status, note):
            raise db.IngestJobOwnershipLost(
                "ingest job terminal yazimi sahiplik kaybetti")
    finally:
        conn.close()


def _retry(job_id, worker_id, note, max_attempts):
    conn = db.get_conn()
    try:
        return db.retry_ingest_job(
            conn, job_id, worker_id, note, max_attempts)
    finally:
        conn.close()


def run_one(*, worker_id=None, upload_dir=None, lease_seconds=300,
            max_attempts=3) -> bool:
    """Claim and execute at most one job; return False when the queue is empty."""
    worker_id = worker_id or _worker_id()
    upload_dir = Path(upload_dir or os.getenv("UPLOAD_DIR", "./data/uploads"))
    conn = db.get_conn(service=True)
    try:
        db.init_schema(conn)
        job = db.claim_ingest_job(
            conn, worker_id, lease_seconds=lease_seconds,
            max_attempts=max_attempts)
    finally:
        conn.close()
    if job is None:
        return False

    job_id = job["job_id"]
    # Cross-tenant privilege ends at the claim. All candidate/chunk/attempt
    # work below runs as the claimed tenant; only the heartbeat thread, which
    # has no inherited ContextVar, uses the narrow service connection needed to
    # renew this already-bound job id.
    tenant_token = db.bind_execution_tenant(job["tenant_id"], service=False)
    attempt = None
    stop = threading.Event()
    lost = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(stop, lost, job_id, worker_id, lease_seconds),
        name="ingest-job-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        if job["archived_at"] is not None:
            raise StaleIngestJob("document_archived")
        if (job["current_candidate_id"] != job["bound_candidate_id"]
                or job["current_candidate_sha"] != job["bound_candidate_sha"]):
            raise StaleIngestJob("candidate_changed")
        conn = db.get_conn()
        try:
            attempt = db.begin_attempt(
                conn, job["document_id"], owner="job/" + worker_id,
                ingest_job_id=job_id, ingest_job_worker=worker_id)
            publication.ensure_bound_version_source(
                conn,
                upload_dir,
                job["tenant_id"],
                job["document_id"],
                job["version_id"],
                job["filename"],
                expected_sha256=job["bound_candidate_sha"],
            )
        finally:
            conn.close()
        verdict = ingest.ingest_version_source(
            upload_dir,
            job["tenant_id"],
            job["document_id"],
            job["version_id"],
            job["filename"],
            attempt,
            expected_sha256=job["bound_candidate_sha"],
        )
        if lost.is_set():
            raise db.IngestJobOwnershipLost("job heartbeat sahipligi kaybetti")
        if (not isinstance(verdict, tuple) or len(verdict) != 2
                or verdict[0] not in {AttemptOutcome.DONE,
                                      AttemptOutcome.PARTIAL}):
            raise RuntimeError("IncompleteIngest")
        status = ("succeeded" if verdict[0] == AttemptOutcome.DONE
                  else "partial")
        _finish(job_id, worker_id, status,
                None if verdict[1] is None else "partial")
    except (StaleIngestJob, CandidateNotPublished,
            publication.VersionSourceMissing,
            publication.VersionSourceCorrupt,
            db.DocumentLifecycleConflict, AttemptFenced) as error:
        if attempt is not None:
            ingest.abandon_attempt(attempt, type(error).__name__)
        _finish(job_id, worker_id, "failed", type(error).__name__)
    except (AttemptAlreadyRunning, AttemptLeaseLost, OSError,
            publication.VersionSourceRefused) as error:
        if attempt is not None:
            ingest.abandon_attempt(attempt, type(error).__name__)
        _retry(job_id, worker_id, type(error).__name__, max_attempts)
    except db.IngestJobOwnershipLost:
        # Another worker owns the durable job now. The attempt fence remains
        # authoritative, but this process must not rewrite the job row.
        if attempt is not None:
            try:
                ingest.abandon_attempt(attempt, "IngestJobOwnershipLost")
            except Exception:
                pass
    except Exception as error:
        if attempt is not None:
            ingest.abandon_attempt(attempt, type(error).__name__)
        _retry(job_id, worker_id, type(error).__name__, max_attempts)
    finally:
        stop.set()
        heartbeat.join(timeout=max(1, min(10, lease_seconds)))
        db.reset_execution_tenant(tenant_token)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run durable ingest jobs")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    while True:
        worked = run_one(lease_seconds=args.lease_seconds,
                         max_attempts=args.max_attempts)
        if args.once:
            return 0
        if not worked:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
