"""The purge worker crosses storage only after a durable database claim."""
from pathlib import Path

import pytest

from pipeline.index import db, publication, purge_worker


class _Connection:
    def __init__(self):
        self.rollbacks = 0
        self.closed = 0

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed += 1


def _job():
    return {
        "id": "40000000-0000-4000-8000-000000000001",
        "tenant_id": "40000000-0000-4000-8000-000000000002",
        "document_id": "40000000-0000-4000-8000-000000000003",
        "version_ids": (
            "40000000-0000-4000-8000-000000000004",
            "40000000-0000-4000-8000-000000000005",
        ),
        "filename": "fixture.pdf",
    }


@pytest.fixture
def worker_world(monkeypatch, tmp_path):
    conn = _Connection()
    calls = []
    monkeypatch.setattr(db, "get_conn", lambda *, service=False: (
        calls.append(("connect", service)) or conn))
    monkeypatch.setattr(
        db, "require_runtime_ready",
        lambda value: calls.append(("ready", value)))
    return conn, calls, tmp_path


def test_an_empty_queue_never_touches_storage(worker_world, monkeypatch):
    conn, calls, root = worker_world
    monkeypatch.setattr(
        db, "claim_document_purge",
        lambda value, worker, *, max_attempts: (
            calls.append(("claim", value, worker, max_attempts)) or None))
    monkeypatch.setattr(
        publication, "purge_document_sources",
        lambda *args: pytest.fail("empty queue touched storage"))

    assert purge_worker.run_one(
        worker_id="worker-a", upload_dir=root, max_attempts=3) is False
    assert calls == [
        ("connect", True),
        ("ready", conn),
        ("claim", conn, "worker-a", 3),
    ]
    assert conn.closed == 1


def test_storage_success_precedes_the_database_tombstone(
        worker_world, monkeypatch):
    conn, calls, root = worker_world
    job = _job()
    monkeypatch.setattr(
        db, "claim_document_purge",
        lambda *_args, **_kwargs: calls.append("claim") or job)
    monkeypatch.setattr(
        publication, "purge_document_sources",
        lambda object_root, tenant, document, versions, filename:
        calls.append(("storage", Path(object_root), tenant, document,
                      versions, filename)))
    monkeypatch.setattr(
        db, "complete_document_purge",
        lambda value, *, job_id, worker_id:
        calls.append(("complete", value, job_id, worker_id)))
    monkeypatch.setattr(
        db, "fail_document_purge",
        lambda *args, **kwargs: pytest.fail("successful purge was failed"))

    assert purge_worker.run_one(
        worker_id="worker-a", upload_dir=root) is True
    assert calls.index("claim") < next(
        index for index, item in enumerate(calls)
        if isinstance(item, tuple) and item[0] == "storage")
    assert calls[-1] == ("complete", conn, job["id"], "worker-a")
    assert conn.rollbacks == 0
    assert conn.closed == 1


@pytest.mark.parametrize(
    ("error", "failure_code"),
    [
        (publication.VersionSourceCorrupt("fixture"), "storage_refused"),
        (publication.VersionSourceRefused("fixture"), "storage_unavailable"),
    ],
)
def test_storage_refusal_records_only_a_closed_failure_code(
        worker_world, monkeypatch, error, failure_code):
    conn, calls, root = worker_world
    job = _job()
    monkeypatch.setattr(db, "claim_document_purge", lambda *a, **k: job)

    def refuse(*_args):
        raise error

    monkeypatch.setattr(publication, "purge_document_sources", refuse)
    monkeypatch.setattr(
        db, "complete_document_purge",
        lambda *a, **k: pytest.fail("refused storage reached tombstone"))
    monkeypatch.setattr(
        db, "fail_document_purge",
        lambda value, *, job_id, worker_id, failure_code:
        calls.append(("failed", value, job_id, worker_id, failure_code)))

    assert purge_worker.run_one(
        worker_id="worker-a", upload_dir=root) is True
    assert calls[-1] == (
        "failed", conn, job["id"], "worker-a", failure_code)
    assert conn.closed == 1


def test_an_unclassified_failure_rolls_back_and_propagates(
        worker_world, monkeypatch):
    conn, _calls, root = worker_world
    monkeypatch.setattr(db, "claim_document_purge", lambda *a, **k: _job())
    monkeypatch.setattr(
        publication, "purge_document_sources", lambda *a: None)

    def fail_complete(*_args, **_kwargs):
        raise db.PurgeJobOwnershipLost("fixture")

    monkeypatch.setattr(db, "complete_document_purge", fail_complete)

    with pytest.raises(db.PurgeJobOwnershipLost):
        purge_worker.run_one(worker_id="worker-a", upload_dir=root)
    assert conn.rollbacks == 1
    assert conn.closed == 1
