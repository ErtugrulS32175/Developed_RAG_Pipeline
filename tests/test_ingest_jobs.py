"""Persistent ingest jobs: API idempotency, worker ownership and safe retry."""
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api
from pipeline.index import db, job_worker
from pipeline.index.attempt_contract import AttemptOutcome, IngestAttempt


DOCUMENT = "11111111-1111-1111-1111-111111111111"
JOB = "22222222-2222-2222-2222-222222222222"
CANDIDATE = "33333333-3333-3333-3333-333333333333"


def _headers(**extra):
    headers = ({"Authorization": f"Bearer {api.API_KEY}"}
               if api.API_KEY else {})
    headers.update(extra)
    return headers


@contextmanager
def _conn():
    yield object()


def _public(status="queued"):
    return {"job_id": JOB, "document_id": DOCUMENT,
            "candidate_id": CANDIDATE, "status": status,
            "attempt_count": 0, "created_at": None, "started_at": None,
            "finished_at": None, "outcome_note": None}


def _claimed(path, status="running"):
    return {**_public(status), "filename": path.name, "archived_at": None,
            "tenant_id": str(db.DEFAULT_TENANT_ID),
            "bound_candidate_id": CANDIDATE, "bound_candidate_sha": "a" * 64,
            "current_candidate_id": CANDIDATE,
            "current_candidate_sha": "a" * 64}


def test_idempotency_keys_are_validated_and_only_digested():
    digest = db._job_key_digest("request-alpha")
    assert len(digest) == 64
    assert "request-alpha" not in digest
    for invalid in ("", " alpha", "alpha ", "alpha\n", "x" * 201, 7):
        with pytest.raises(ValueError):
            db._job_key_digest(invalid)


def test_enqueue_requires_a_bounded_header_before_borrowing(monkeypatch):
    borrowed = []

    @contextmanager
    def borrowing():
        borrowed.append(True)
        yield object()

    monkeypatch.setattr(api, "db_conn", borrowing)
    client = TestClient(api.app)
    path = f"/documents/{DOCUMENT}/ingest-jobs"
    assert client.post(path, headers=_headers()).status_code == 422
    assert client.post(path, headers=_headers(
        **{"Idempotency-Key": "x" * 201})).status_code == 422
    assert borrowed == []


def test_enqueue_get_and_cancel_routes_publish_closed_job_fields(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(
        api.db, "enqueue_ingest_job",
        lambda _conn, document, key: calls.append((document, key)) or _public())
    monkeypatch.setattr(api.db, "get_ingest_job",
                        lambda _conn, job: _public())
    monkeypatch.setattr(api.db, "cancel_ingest_job",
                        lambda _conn, job: _public("cancelled"))
    client = TestClient(api.app)

    queued = client.post(
        f"/documents/{DOCUMENT}/ingest-jobs",
        headers=_headers(**{"Idempotency-Key": "request-alpha"}))
    read = client.get(f"/ingest-jobs/{JOB}", headers=_headers())
    cancelled = client.delete(f"/ingest-jobs/{JOB}", headers=_headers())

    assert queued.status_code == 202
    assert read.json()["status"] == "queued"
    assert cancelled.json()["status"] == "cancelled"
    assert set(queued.json()) == set(_public())
    assert calls == [(DOCUMENT, "request-alpha")]


def test_synchronous_process_refuses_a_persisted_active_job(
        monkeypatch, tmp_path):
    source = tmp_path / "alpha.pdf"
    source.write_bytes(b"alpha")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(api.db, "get_document", lambda *_args: {
        "id": DOCUMENT, "filename": source.name, "candidate_id": CANDIDATE,
        "content_sha256": "a" * 64, "archived_at": None})
    monkeypatch.setattr(api.db, "active_ingest_job",
                        lambda *_args: _public())
    begun = []
    monkeypatch.setattr(api.db, "begin_attempt",
                        lambda *_args: begun.append(True))

    response = TestClient(api.app).post(
        f"/documents/{DOCUMENT}/process", headers=_headers())
    assert response.status_code == 409
    assert begun == []


def test_synchronous_process_closes_an_enqueue_race_at_attempt_time(
        monkeypatch, tmp_path):
    source = tmp_path / "alpha.pdf"
    source.write_bytes(b"alpha")
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(api.db, "get_document", lambda *_args: {
        "id": DOCUMENT, "filename": source.name, "candidate_id": CANDIDATE,
        "content_sha256": "a" * 64, "archived_at": None})
    monkeypatch.setattr(api.db, "active_ingest_job", lambda *_args: None)
    monkeypatch.setattr(
        api.db, "begin_attempt",
        lambda *_args: (_ for _ in ()).throw(
            db.IngestJobConflict("job arrived after the fast read")))

    response = TestClient(api.app).post(
        f"/documents/{DOCUMENT}/process", headers=_headers())
    assert response.status_code == 409
    assert response.json()["detail"] == (
        "document already has an active ingest job")


class _Connection:
    def close(self):
        pass


def _worker_world(monkeypatch, tmp_path, job=None):
    source = tmp_path / "alpha.pdf"
    source.write_bytes(b"alpha")
    claimed = _claimed(source) if job is None else job
    calls = []
    monkeypatch.setattr(job_worker.db, "get_conn",
                        lambda **_kwargs: _Connection())
    monkeypatch.setattr(job_worker.db, "init_schema", lambda _conn: None)
    monkeypatch.setattr(job_worker.db, "claim_ingest_job",
                        lambda *_args, **_kwargs: claimed)
    monkeypatch.setattr(job_worker, "_heartbeat", lambda *_args: None)
    def begin(*args, **kwargs):
        calls.append(("begin", args[1:], kwargs))
        return IngestAttempt("attempt", DOCUMENT, CANDIDATE, "a" * 64, 0)

    monkeypatch.setattr(job_worker.db, "begin_attempt", begin)
    monkeypatch.setattr(job_worker.db, "finish_ingest_job",
                        lambda _conn, *args: calls.append(("finish", args)) or True)
    monkeypatch.setattr(job_worker.db, "retry_ingest_job",
                        lambda _conn, *args: calls.append(("retry", args)) or "queued")
    monkeypatch.setattr(job_worker.ingest, "abandon_attempt",
                        lambda *args: calls.append(("abandon", args)))
    return source, calls


def test_worker_success_uses_the_existing_attempt_authority(monkeypatch, tmp_path):
    source, calls = _worker_world(monkeypatch, tmp_path)
    ingested = []
    monkeypatch.setattr(
        job_worker.ingest, "main",
        lambda path, attempt=None: ingested.append((path, attempt)) or
        (AttemptOutcome.DONE, None))

    assert job_worker.run_one(worker_id="worker-alpha", upload_dir=tmp_path)
    assert ingested[0][0] == str(source)
    assert isinstance(ingested[0][1], IngestAttempt)
    assert calls == [
        ("begin", (DOCUMENT,), {
            "owner": "job/worker-alpha", "ingest_job_id": JOB,
            "ingest_job_worker": "worker-alpha"}),
        ("finish", (JOB, "worker-alpha", "succeeded", None)),
    ]


def test_stale_candidate_is_failed_without_parsing(monkeypatch, tmp_path):
    stale = _claimed(tmp_path / "alpha.pdf")
    stale["current_candidate_id"] = "44444444-4444-4444-4444-444444444444"
    _source, calls = _worker_world(monkeypatch, tmp_path, stale)
    ingested = []
    monkeypatch.setattr(job_worker.ingest, "main",
                        lambda *_args, **_kwargs: ingested.append(True))

    assert job_worker.run_one(worker_id="worker-alpha", upload_dir=tmp_path)
    assert ingested == []
    assert calls == [("finish", (
        JOB, "worker-alpha", "failed", "StaleIngestJob"))]


def test_transient_worker_failure_requeues_with_only_the_exception_type(
        monkeypatch, tmp_path):
    _source, calls = _worker_world(monkeypatch, tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError("private path and vendor prose")

    monkeypatch.setattr(job_worker.ingest, "main", fail)
    assert job_worker.run_one(worker_id="worker-alpha", upload_dir=tmp_path)
    assert calls[0][0] == "begin"
    assert calls[1][0] == "abandon"
    assert calls[2] == ("retry", (JOB, "worker-alpha", "OSError", 3))
    assert "private path" not in repr(calls)


def test_empty_queue_does_no_ingest_work(monkeypatch, tmp_path):
    monkeypatch.setattr(job_worker.db, "get_conn",
                        lambda **_kwargs: _Connection())
    monkeypatch.setattr(job_worker.db, "init_schema", lambda _conn: None)
    monkeypatch.setattr(job_worker.db, "claim_ingest_job",
                        lambda *_args, **_kwargs: None)
    assert job_worker.run_one(worker_id="worker-alpha", upload_dir=tmp_path) is False


def test_lost_job_ownership_never_retries_or_rewrites_the_job(
        monkeypatch, tmp_path):
    _source, calls = _worker_world(monkeypatch, tmp_path)
    monkeypatch.setattr(
        job_worker.ingest, "main",
        lambda *_args, **_kwargs: (AttemptOutcome.DONE, None))
    monkeypatch.setattr(job_worker.db, "finish_ingest_job",
                        lambda *_args: False)

    assert job_worker.run_one(worker_id="worker-alpha", upload_dir=tmp_path)
    assert [call[0] for call in calls] == ["begin", "abandon"]


def test_a_heartbeat_error_is_treated_as_lost_ownership(monkeypatch):
    class Stop:
        def wait(self, _seconds):
            return False

    lost = __import__("threading").Event()
    monkeypatch.setattr(job_worker.db, "get_conn",
                        lambda **_kwargs: _Connection())
    monkeypatch.setattr(
        job_worker.db, "heartbeat_ingest_job",
        lambda *_args: (_ for _ in ()).throw(OSError("database unavailable")))

    job_worker._heartbeat(Stop(), lost, JOB, "worker-alpha", 30)
    assert lost.is_set()


@pytest.mark.parametrize("value", [True, False, 0, 21, "3"])
def test_retry_budget_is_bounded_before_sql(value):
    class Conn:
        def cursor(self, **_kwargs):
            raise AssertionError("SQL must not run")

    with pytest.raises(ValueError):
        db.retry_ingest_job(
            Conn(), JOB, "worker-alpha", "OSError", max_attempts=value)


def test_source_path_never_escapes_the_upload_root(tmp_path):
    for name in ("../alpha.pdf", "..\\alpha.pdf", "folder/alpha.pdf"):
        with pytest.raises(job_worker.StaleIngestJob):
            job_worker._source_path(Path(tmp_path), name)
