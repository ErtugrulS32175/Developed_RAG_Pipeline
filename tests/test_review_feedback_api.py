"""Content-free feedback targets and hierarchy-scoped review endpoints."""
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException

from pipeline.api import auth
from pipeline.api import app as api
from pipeline.index import db
from pipeline.validation.rag.answer_guard import (
    ANSWERED, MODEL_CITATION, REVIEW_REQUIRED, GuardResult, PageCitation,
)


TENANT = UUID("75000000-0000-0000-0000-000000000001")
ACTOR = UUID("75000000-0000-0000-0000-000000000002")
CASE = UUID("75000000-0000-0000-0000-000000000003")
CHUNK = UUID("75000000-0000-0000-0000-000000000004")
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _principal():
    return auth.Principal(
        TENANT, "reader", ACTOR, "openwebui",
        UUID("75000000-0000-0000-0000-000000000005"), False)


@contextmanager
def _connection(value=None):
    yield value if value is not None else object()


def _answered():
    return GuardResult(
        ANSWERED, "checked answer", (),
        (PageCitation(2, MODEL_CITATION, chunk_id=str(CHUNK),
                      document_name="report.pdf"),))


def test_answered_browser_publication_mints_one_actor_bound_opaque_target(
        monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "_browser_evidence_enabled", lambda: True)
    monkeypatch.setattr(api.auth, "current_principal", _principal)
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(
        api.db, "create_review_interaction",
        lambda _conn, **kwargs: captured.update(kwargs))

    reference = api._persist_review_interaction(_answered())

    assert isinstance(reference, str) and len(reference) == 43
    assert api._evidence_digest(reference) == captured["ref_digest"]
    assert captured["actor_id"] == ACTOR
    assert captured["outcome"] == ANSWERED
    assert captured["citation_count"] == 1
    assert str(captured["interaction_id"]) not in reference
    monkeypatch.setattr(
        api.db, "register_evidence_references", lambda *_a, **_k: None)
    citation = api._citation_payload(
        _answered().citations, persist=True, feedback_ref=reference)[0]
    assert citation["feedback_ref"] == reference
    assert str(CHUNK) not in repr(citation)


def test_review_required_opens_a_server_side_case_without_a_browser_token(
        monkeypatch):
    captured = {}
    monkeypatch.setattr(api, "_browser_evidence_enabled", lambda: True)
    monkeypatch.setattr(api.auth, "current_principal", _principal)
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(
        api.db, "create_review_interaction",
        lambda _conn, **kwargs: captured.update(kwargs))
    result = GuardResult(REVIEW_REQUIRED, None, ("closed-code",), ())

    assert api._persist_review_interaction(result) is None
    assert captured["outcome"] == REVIEW_REQUIRED
    assert captured["ref_digest"] is None
    assert captured["citation_count"] == 0


@pytest.mark.parametrize("status", ["api-key", "abstained", "no-citation"])
def test_no_unpersistable_publication_invents_a_feedback_target(
        monkeypatch, status):
    monkeypatch.setattr(
        api, "_browser_evidence_enabled", lambda: status != "api-key")
    called = []
    monkeypatch.setattr(
        api.db, "create_review_interaction",
        lambda *_args, **_kwargs: called.append(True))
    result = (_answered() if status != "no-citation" else
              GuardResult(ANSWERED, "checked", (), ()))
    if status == "abstained":
        result = GuardResult("abstained", "bilmiyorum", (), ())
    assert api._persist_review_interaction(result) is None
    assert called == []


def test_feedback_endpoint_passes_only_closed_values_and_current_actor(
        monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "db_conn", _connection)

    def submit(_conn, **kwargs):
        seen.update(kwargs)
        return {"revision": 2, "review_open": True}

    monkeypatch.setattr(api.db, "submit_review_feedback", submit)
    reference = api._b64url(b"r" * 32)
    response = api.submit_review_feedback(
        api.ReviewFeedbackRequest(
            feedback_ref=reference, verdict="not_helpful",
            reason_code="missing_evidence"),
        _principal())
    assert response == {"status": "recorded", "revision": 2,
                        "review_open": True}
    assert seen == {
        "actor_id": ACTOR,
        "ref_digest": b"r" * 32,
        "verdict": "not_helpful",
        "reason_code": "missing_evidence",
    }


@pytest.mark.parametrize(
    "verdict,reason", [("helpful", "other"), ("not_helpful", None)])
def test_feedback_reason_pairing_is_rejected_before_database(
        monkeypatch, verdict, reason):
    touched = []
    monkeypatch.setattr(
        api.db, "submit_review_feedback",
        lambda *_args, **_kwargs: touched.append(True))
    with pytest.raises(HTTPException) as caught:
        api.submit_review_feedback(
            api.ReviewFeedbackRequest(
                feedback_ref="R" * 43, verdict=verdict,
                reason_code=reason),
            _principal())
    assert caught.value.status_code == 422
    assert touched == []


def test_queue_is_bounded_content_free_and_carries_fences(monkeypatch):
    rows = [{
        "id": CASE, "trigger_code": "user_feedback", "state": "open",
        "revision": 3, "created_at": NOW, "outcome": "answered",
        "citation_count": 2, "display_label": "Team member",
        "position_title": "Analyst", "policy_epoch": 7,
    }]
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(
        api.db, "list_review_cases", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(api.db, "record_org_decision", lambda *_a, **_k: None)
    request = SimpleNamespace(state=SimpleNamespace(request_id="12345678"))

    result = api.review_queue(
        request, limit=20, before_created_at=None, before_id=None,
        reason_code="management_duty", principal=_principal())

    assert result["has_more"] is False and result["next_cursor"] is None
    assert result["cases"] == [{
        "case_id": CASE, "trigger_code": "user_feedback", "state": "open",
        "revision": 3, "created_at": NOW, "outcome": "answered",
        "citation_count": 2, "subject_label": "Team member",
        "position_title": "Analyst", "policy_epoch": 7,
    }]
    assert not ({"question", "answer", "passage", "source_path"}
                & set(result["cases"][0]))


def test_review_decision_maps_revision_conflict_without_backend_prose(
        monkeypatch):
    monkeypatch.setattr(api, "db_conn", _connection)
    monkeypatch.setattr(
        api.db, "decide_review_case",
        lambda *_a, **_k: (_ for _ in ()).throw(
            db.ReviewConflict("stale closed fence")))
    request = SimpleNamespace(state=SimpleNamespace(request_id="12345678"))
    with pytest.raises(HTTPException) as caught:
        api.decide_review_case(
            CASE,
            api.ReviewDecisionRequest(
                expected_revision=1, expected_policy_epoch=2,
                decision="resolved", resolution_code="corrected",
                reason_code="management_duty"),
            request, _principal())
    assert caught.value.status_code == 409


def test_review_schema_has_actor_rls_and_no_content_columns():
    schema = api.Path(db.__file__).with_name("schema.sql").read_text("utf-8")
    for table in ("review_interactions", "review_feedback", "review_cases",
                  "review_case_events"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in schema
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in schema
    for table in ("review_interactions", "review_feedback", "review_cases",
                  "review_case_events"):
        definition = schema.split(
            f"CREATE TABLE IF NOT EXISTS {table}", 1)[1].split(");", 1)[0]
        for forbidden in ("question", "answer", "passage", "source_path",
                          "chat_id", "message_id", "free_text"):
            assert f"\n    {forbidden} " not in definition
    assert "rag_effective_actor()" in schema
    assert "rag_can_monitor_identity" in schema
    assert "target_position.protected_from_monitoring = false" in schema


def test_visible_members_no_longer_lets_root_bypass_absolute_protection():
    source = api.Path(db.__file__).read_text("utf-8")
    function = source.split("def visible_org_members", 1)[1].split(
        "def register_evidence_references", 1)[0]
    assert "position.protected_from_monitoring = false" in function
    assert "viewer_position.kind = 'root'" not in function
