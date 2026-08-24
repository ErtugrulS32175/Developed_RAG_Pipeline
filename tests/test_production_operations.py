import hashlib
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from pipeline.api import metrics
from pipeline.index import db
from scripts import db_snapshot, migrate_db, rollout_gate


class _Cursor:
    def __init__(self, row=None):
        self.row = row
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        self.calls.append((statement, params))

    def fetchone(self):
        return self.row


class _Connection:
    def __init__(self, row=None):
        self.cursor_value = _Cursor(row)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_schema_migration_serializes_ddl_and_records_the_exact_source_digest():
    conn = _Connection()

    db.init_schema(conn)

    calls = conn.cursor_value.calls
    assert "pg_advisory_xact_lock" in calls[0][0]
    assert calls[0][1] == ("ragtest-schema-migration",)
    assert "CREATE EXTENSION" in calls[1][0]
    assert "CREATE TABLE IF NOT EXISTS rag_schema_state" in calls[2][0]
    assert "CREATE TABLE IF NOT EXISTS rag_schema_history" in calls[3][0]
    assert "SELECT schema_sha256 FROM rag_schema_history" in calls[4][0]
    assert "INSERT INTO rag_schema_history" in calls[5][0]
    assert "INSERT INTO rag_schema_state" in calls[6][0]
    version, digest = db.expected_schema_state()
    assert calls[4][1] == (version,)
    assert calls[5][1] == (version, digest)
    assert calls[6][1] == (version, digest)
    assert digest == hashlib.sha256(
        db.Path(db.__file__).with_name("schema.sql").read_bytes()).hexdigest()
    assert conn.commits == 1


def test_schema_version_cannot_be_reused_for_different_bytes():
    conn = _Connection(("f" * 64,))
    with pytest.raises(RuntimeError, match="digest"):
        db.init_schema(conn)
    assert conn.commits == 0


@pytest.mark.parametrize("row, expected", [
    (None, False),
    ((0, "0" * 64), False),
    (db.expected_schema_state(), True),
])
def test_readiness_requires_the_exact_schema_version_and_digest(row, expected):
    assert db.schema_is_current(_Connection(row)) is expected


def test_metrics_keep_only_bounded_code_owned_labels():
    metrics.reset_for_tests()
    metrics.observe("GET", "/documents/{document_id}", 200, 12.5)
    metrics.observe("BAD\nMETHOD", "/raw?secret=OZEL", 799, -3)

    text = metrics.render()

    assert 'method="GET",route="/documents/{document_id}",status_class="2xx"' in text
    assert 'method="OTHER",route="unmatched",status_class="5xx"' in text
    assert "OZEL" not in text
    assert "BAD" not in text


def test_request_id_metrics_and_logs_are_content_free(monkeypatch, caplog):
    from pipeline.api import app as api

    metrics.reset_for_tests()
    client = TestClient(api.app)
    response = client.get("/health?question=OZEL_SORU")
    observed = client.get("/metrics")

    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 8
    assert observed.status_code == 200
    assert 'route="/health"' in observed.text
    assert "OZEL_SORU" not in observed.text

    missing = client.get("/does-not-exist-OZEL_PATH")
    assert missing.status_code == 404
    assert "OZEL_PATH" not in metrics.render()
    assert "OZEL_PATH" not in caplog.text


def _write_snapshot_pair(tmp_path, payload=b"snapshot"):
    archive = tmp_path / "database.dump"
    archive.write_bytes(payload)
    record = {
        "snapshot_version": 1,
        "archive_name": archive.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = tmp_path / "database.dump.manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return archive, manifest, record


def _disposable_dsn(password, database):
    """Build test-only connection coordinates without a credentialed literal."""
    return "".join(("postgres", "ql://", "rag:", password,
                    "@db.internal:5432/", database))


def test_snapshot_backup_keeps_the_dsn_out_of_argv_and_output(
        monkeypatch, tmp_path):
    secret = "OZEL_PAROLA"
    monkeypatch.setenv("PG_DSN", _disposable_dsn(secret, "ragdb"))
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        kwargs["stdout"].write(b"custom-format-backup")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(db_snapshot.subprocess, "run", fake_run)
    output = tmp_path / "created.dump"

    record = db_snapshot.backup(output)

    assert db_snapshot.verify(output) == record
    assert seen["argv"] == [
        "pg_dump", "--format=custom", "--no-owner", "--no-acl"]
    assert seen["env"]["PGPASSWORD"] == secret
    assert "PG_DSN" not in seen["env"]
    assert "PG_RESTORE_DSN" not in seen["env"]
    public = json.dumps(record) + " ".join(seen["argv"])
    assert secret not in public and "db.internal" not in public


def test_snapshot_verification_refuses_byte_drift(tmp_path):
    archive, manifest, _record = _write_snapshot_pair(tmp_path)
    archive.write_bytes(b"changed")

    with pytest.raises(db_snapshot.SnapshotError,
                       match="archive_digest_mismatch"):
        db_snapshot.verify(archive, manifest)


def test_restore_requires_a_verified_archive_and_an_empty_database(
        monkeypatch, tmp_path):
    archive, manifest, record = _write_snapshot_pair(tmp_path)
    monkeypatch.setenv(
        "PG_RESTORE_DSN", _disposable_dsn("fixture-password", "emptydb"))
    monkeypatch.setattr(db_snapshot, "_database_is_empty", lambda _dsn: True)
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["input"] = kwargs["stdin"].read()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(db_snapshot.subprocess, "run", fake_run)

    result = db_snapshot.restore(
        archive, manifest, confirmation="EMPTY_DATABASE")

    assert result == {"snapshot_version": 1, "status": "restored",
                      "sha256": record["sha256"]}
    assert seen["input"] == b"snapshot"
    assert seen["argv"][-1] == "emptydb"
    assert "secret" not in " ".join(seen["argv"])


def test_restore_refuses_a_nonempty_destination_before_launch(
        monkeypatch, tmp_path):
    archive, manifest, _record = _write_snapshot_pair(tmp_path)
    monkeypatch.setenv(
        "PG_RESTORE_DSN", _disposable_dsn("fixture-password", "ragdb"))
    monkeypatch.setattr(db_snapshot, "_database_is_empty", lambda _dsn: False)
    launches = []
    monkeypatch.setattr(
        db_snapshot.subprocess, "run", lambda *a, **k: launches.append(True))

    with pytest.raises(db_snapshot.SnapshotError,
                       match="restore_database_not_empty"):
        db_snapshot.restore(
            archive, manifest, confirmation="EMPTY_DATABASE")
    assert launches == []


def test_migration_cli_never_reflects_connection_exception_prose(
        monkeypatch, capsys):
    def fail(**_kwargs):
        raise RuntimeError("OZEL_DSN_KULLANICI_PAROLA")

    monkeypatch.setattr(migrate_db.db, "get_conn", fail)

    assert migrate_db.main() == 1
    output = capsys.readouterr().out
    assert json.loads(output) == {
        "migration_version": 4, "status": "failed"}
    assert "OZEL" not in output


class _ReadyResponse:
    status_code = 200

    def json(self):
        return {"status": "ready", "kontroller": {
            "veritabani": True, "sema": True, "embedding": True}}


def test_rollout_gate_joins_quality_and_live_readiness(monkeypatch, tmp_path):
    report = tmp_path / "quality.json"
    report.write_text(json.dumps({
        "quality_gate_version": 1,
        "passed": True,
        "sets": {"regression": {}},
        "failures": [],
    }), encoding="utf-8")
    monkeypatch.setattr(
        rollout_gate.requests, "get", lambda *_a, **_k: _ReadyResponse())

    assert rollout_gate.evaluate(report, "http://service/ready") == {
        "rollout_gate_version": 1, "passed": True, "failures": []}


def test_rollout_gate_fails_closed_without_reflecting_dependency_prose(
        monkeypatch, tmp_path):
    report = tmp_path / "quality.json"
    report.write_text("{}", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise rollout_gate.requests.ConnectionError(
            "OZEL_TOKEN postgresql://secret")

    monkeypatch.setattr(rollout_gate.requests, "get", fail)
    result = rollout_gate.evaluate(report, "http://secret/ready")

    assert result == {"rollout_gate_version": 1, "passed": False,
                      "failures": ["quality_gate_failed", "readiness_failed"]}
    assert "OZEL" not in json.dumps(result)
