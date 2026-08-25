"""Durable, fail-closed document-content purge worker.

One PostgreSQL transaction owns the selection, the last legal-hold check, the
document row lock, source removal and the tombstone write.  A concurrent hold
therefore happens wholly before the purge (and cancels it) or wholly after the
terminal tombstone (and is refused); it cannot land between the policy check
and irreversible storage work.
"""
from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

from pipeline.index import db, publication


def _worker_id() -> str:
    return f"{socket.gethostname()}/{os.getpid()}"


def run_one(*, worker_id=None, upload_dir=None, max_attempts=5) -> bool:
    """Execute at most one due purge; return False when none is claimable."""
    worker = worker_id or _worker_id()
    object_root = Path(upload_dir or os.getenv("UPLOAD_DIR", "./data/uploads"))
    conn = db.get_conn(service=True)
    try:
        db.require_runtime_ready(conn)
        job = db.claim_document_purge(
            conn, worker, max_attempts=max_attempts)
        if job is None:
            return False
        try:
            publication.purge_document_sources(
                object_root,
                job["tenant_id"],
                job["document_id"],
                job["version_ids"],
                job["filename"],
            )
        except publication.VersionSourceCorrupt:
            db.fail_document_purge(
                conn, job_id=job["id"], worker_id=worker,
                failure_code="storage_refused")
            return True
        except publication.VersionSourceRefused:
            db.fail_document_purge(
                conn, job_id=job["id"], worker_id=worker,
                failure_code="storage_unavailable")
            return True
        db.complete_document_purge(
            conn, job_id=job["id"], worker_id=worker)
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not 1 <= args.max_attempts <= 20:
        parser.error("--max-attempts must be between 1 and 20")
    while True:
        worked = run_one(max_attempts=args.max_attempts)
        if args.once:
            return 0
        if not worked:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
