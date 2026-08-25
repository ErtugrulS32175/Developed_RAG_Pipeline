"""Closed API surface for tenant-bound evaluation dataset governance."""
import asyncio
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import UUID

import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError

from pipeline.api import auth
from pipeline.api import app as api
from pipeline.index import db


TENANT = UUID("86000000-0000-0000-0000-000000000001")
ACTOR = UUID("86000000-0000-0000-0000-000000000002")
DATASET = UUID("86000000-0000-4000-8000-000000000003")
VERSION = UUID("86000000-0000-4000-8000-000000000004")
CASE = "86000000-0000-4000-8000-000000000005"
DIGEST = "a" * 64
NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def _principal(role="editor", *, source="openwebui", architect=False):
    return auth.Principal(
        TENANT, role, ACTOR, source, org_architect=architect)


@contextmanager
def _connection():
    yield object()


def _case(**updates):
    value = {
        "case_key": CASE,
        "q": "closed question",
        "key": "document-key",
        "answer": "closed answer",
        "pages": [1, 3],
        "type": "metin",
    }
    value.update(updates)
    return value


class _Request:
    def __init__(self, value, *, content_length=None):
        self.raw = (value if type(value) is bytes else
                    json.dumps(value, separators=(",", ":")).encode())
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)

    async def stream(self):
        midpoint = len(self.raw) // 2
        yield self.raw[:midpoint]
        yield self.raw[midpoint:]


def test_eval_writer_requires_a_verified_openwebui_editor():
    assert api.require_eval_writer(_principal()) == _principal()
    for principal in (
        _principal("reader"),
        _principal("admin", source="api-key"),
        _principal("reader", architect=True),
    ):
        with pytest.raises(HTTPException) as caught:
            api.require_eval_writer(principal)
        assert caught.value.status_code == 403


def test_eval_request_models_are_closed_and_publish_binds_the_digest():
    with pytest.raises(ValidationError):
        api.EvalDatasetCreateRequest(
            slug="set", label="label", offered_secret="sentinel")
    with pytest.raises(ValidationError):
        api.EvalPublishRequest(
            expected_revision=1, expected_policy_epoch=2,
            expected_draft_sha256="A" * 64)
    body = api.EvalPublishRequest(
        expected_revision=1, expected_policy_epoch=2,
        expected_draft_sha256=DIGEST)
    assert body.expected_draft_sha256 == DIGEST


def test_create_projects_only_the_closed_metadata_wrapper(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "db_conn", _connection)

    def create(_conn, **kwargs):
        seen.update(kwargs)
        return {
            "id": DATASET, "slug": "monthly-set", "label": "Monthly",
            "state": "active", "revision": 1,
            "owner_identity_id": ACTOR,
            "latest_version_id": VERSION, "latest_version_number": 1,
            "latest_version_state": "draft", "latest_version_revision": 1,
            "latest_case_count": 0, "private_value": "must-not-leak",
        }

    monkeypatch.setattr(api.db, "create_eval_dataset", create)
    result = api.eval_dataset_create(
        api.EvalDatasetCreateRequest(slug="monthly-set", label="Monthly"),
        _principal())
    assert seen == {"actor_id": ACTOR, "slug": "monthly-set",
                    "label": "Monthly"}
    assert set(result) == {"dataset"}
    assert set(result["dataset"]) == {
        "dataset_id", "slug", "label", "state", "revision", "versions"}
    assert result["dataset"]["versions"] == [{
        "version_id": VERSION, "version_number": 1, "state": "draft",
        "revision": 1, "case_count": 0,
    }]
    assert "private_value" not in repr(result)


def test_duplicate_dataset_slug_is_a_closed_conflict(monkeypatch):
    monkeypatch.setattr(api, "db_conn", _connection)

    def conflict(*_args, **_kwargs):
        raise db.EvalDatasetConflict("backend prose sentinel")

    monkeypatch.setattr(api.db, "create_eval_dataset", conflict)
    with pytest.raises(HTTPException) as caught:
        api.eval_dataset_create(
            api.EvalDatasetCreateRequest(slug="monthly-set", label="Monthly"),
            _principal())
    assert caught.value.status_code == 409
    assert caught.value.detail == "degerlendirme seti zaten var"
    assert "sentinel" not in caught.value.detail


def test_list_is_metadata_only_and_never_projects_case_content(monkeypatch):
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(api.db, "list_eval_datasets", lambda *_a, **_k: [{
        "id": DATASET, "slug": "set", "label": "Set", "state": "active",
        "revision": 4, "owner_label": "Owner", "policy_epoch": 9,
        "current_version_id": None, "current_version_number": None,
        "latest_version_id": VERSION, "latest_version_number": 2,
        "latest_version_state": "draft", "latest_version_revision": 3,
        "latest_case_count": 7, "q": "sentinel", "answer": "sentinel",
    }])
    result = api.eval_dataset_list(100, _principal("reader"))
    assert set(result) == {"datasets"}
    rendered = repr(result)
    assert "sentinel" not in rendered
    assert "q" not in result["datasets"][0]
    assert result["datasets"][0]["policy_epoch"] == 9


def test_import_is_bounded_before_decode_and_rejects_duplicate_keys(
        monkeypatch):
    touched = []
    monkeypatch.setattr(
        api.db, "replace_eval_cases",
        lambda *_a, **_k: touched.append(True))
    with pytest.raises(HTTPException) as too_large:
        asyncio.run(api.eval_cases_import(
            DATASET, VERSION,
            _Request(b"{}", content_length=api.eval_datasets.MAX_JSON_BYTES + 1),
            _principal()))
    assert too_large.value.status_code == 413
    duplicate = (
        b'{"expected_revision":1,"expected_revision":2,"cases":[]}'
    )
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(api.eval_cases_import(
            DATASET, VERSION, _Request(duplicate), _principal()))
    assert invalid.value.status_code == 422
    assert touched == []


def test_import_passes_a_fresh_canonical_case_and_returns_digest(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "db_conn", _connection)

    def replace(_conn, **kwargs):
        seen.update(kwargs)
        return {"version_number": 1, "state": "draft", "revision": 2,
                "case_count": 1, "content_sha256": DIGEST}

    monkeypatch.setattr(api.db, "replace_eval_cases", replace)
    offered = _case()
    result = asyncio.run(api.eval_cases_import(
        DATASET, VERSION,
        _Request({"expected_revision": 1, "cases": [offered]}),
        _principal()))
    assert seen["actor_id"] == ACTOR
    assert seen["expected_revision"] == 1
    assert seen["cases"][0]["pages"] == (1, 3)
    assert result == {"version": {
        "version_id": VERSION, "version_number": 1, "state": "draft",
        "revision": 2, "case_count": 1, "content_sha256": DIGEST,
    }}


def test_publish_passes_all_three_fences_and_hides_backend_prose(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "db_conn", _connection)

    def publish(_conn, **kwargs):
        seen.update(kwargs)
        raise db.EvalDatasetConflict("secret answer and backend detail")

    monkeypatch.setattr(api.db, "publish_eval_version", publish)
    with pytest.raises(HTTPException) as caught:
        api.eval_version_publish(
            DATASET, VERSION,
            api.EvalPublishRequest(
                expected_revision=2, expected_policy_epoch=9,
                expected_draft_sha256=DIGEST), _principal())
    assert caught.value.status_code == 409
    assert "secret" not in str(caught.value.detail)
    assert seen == {
        "actor_id": ACTOR, "dataset_id": DATASET, "version_id": VERSION,
        "expected_revision": 2, "expected_policy_epoch": 9,
        "expected_draft_sha256": DIGEST,
    }


def test_retire_passes_revision_and_policy_fences_in_a_closed_wrapper(
        monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "db_conn", _connection)

    def retire(_conn, **kwargs):
        seen.update(kwargs)
        return {
            "id": DATASET, "slug": "monthly-set", "label": "Monthly",
            "state": "retired", "revision": 5, "private": "sentinel",
        }

    monkeypatch.setattr(api.db, "retire_eval_dataset", retire)
    result = api.eval_dataset_retire(
        DATASET,
        api.EvalRetireRequest(
            expected_revision=4, expected_policy_epoch=9),
        _principal())
    assert seen == {
        "actor_id": ACTOR, "dataset_id": DATASET,
        "expected_revision": 4, "expected_policy_epoch": 9,
    }
    assert result == {"dataset": {
        "dataset_id": DATASET, "slug": "monthly-set", "label": "Monthly",
        "state": "retired", "revision": 5, "versions": [],
    }}
    assert "sentinel" not in repr(result)


def test_explicit_case_read_sets_no_store_and_uses_fresh_authority(monkeypatch):
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(api.db, "list_eval_versions", lambda *_a, **_k: [{
        "id": VERSION, "version_number": 1, "state": "published",
        "revision": 2, "case_count": 1,
    }])
    monkeypatch.setattr(
        api.db, "read_eval_cases", lambda *_a, **_k: [_case()])
    response = Response()
    result = api.eval_cases_read(
        DATASET, VERSION, response, _principal("reader"))
    assert response.headers["cache-control"] == "no-store"
    assert result == {"dataset_id": DATASET, "version_id": VERSION,
                      "cases": [_case()]}


def test_missing_case_version_is_404_without_reading_content(monkeypatch):
    touched = []
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(api.db, "list_eval_versions", lambda *_a, **_k: [])
    monkeypatch.setattr(
        api.db, "read_eval_cases", lambda *_a, **_k: touched.append(True))
    with pytest.raises(HTTPException) as caught:
        api.eval_cases_read(
            DATASET, VERSION, Response(), _principal("reader"))
    assert caught.value.status_code == 404 and touched == []


def test_eval_routes_never_offer_an_admin_or_architect_bypass():
    source = api.Path(api.__file__).read_text("utf-8")
    eval_surface = source.split(
        "def eval_dataset_list(", 1)[1].split("def list_models(", 1)[0]
    assert "require_admin" not in eval_surface
    assert "require_org_architect" not in eval_surface
    assert "require_org_identity" in eval_surface
    assert "require_eval_writer" in eval_surface


def test_eval_schema_is_forced_rls_and_events_are_content_free_immutable():
    schema = api.Path(db.__file__).with_name("schema.sql").read_text("utf-8")
    for table in ("eval_datasets", "eval_dataset_versions", "eval_cases",
                  "eval_dataset_events"):
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in schema
    event = schema.split(
        "CREATE TABLE IF NOT EXISTS eval_dataset_events", 1)[1].split(
            ");", 1)[0]
    for forbidden in ("question", "expected_answer", "document_key",
                      "source_path", "prompt", "stdout", "stderr"):
        assert forbidden not in event
    assert "CREATE TRIGGER eval_events_immutable" in schema
    assert "rag_can_monitor_identity(owner_identity)" in schema
    assert "octet_length(question) BETWEEN 1 AND 4096" in schema
    assert "octet_length(document_key) BETWEEN 1 AND 16384" in schema
    assert "octet_length(expected_answer) BETWEEN 1 AND 16384" in schema
    assert "rag_eval_pages_valid(pages)" in schema
