"""OpenWebUI feedback metadata transport and fail-closed Action behavior."""
import asyncio
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CITATIONS = ROOT / "openwebui" / "functions" / "ragtest_citations.py"
ACTION = ROOT / "openwebui" / "functions" / "ragtest_feedback_action.py"
EVIDENCE = "E" * 43
FEEDBACK = "F" * 43


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _citation(feedback=FEEDBACK):
    value = {
        "evidence_ref": EVIDENCE,
        "document_name": "report.pdf",
        "page": 4,
        "source": "model",
    }
    if feedback is not None:
        value["feedback_ref"] = feedback
    return value


def _body(*refs, role="assistant", duplicate=False):
    message = {
        "id": "answer", "role": role,
        "sources": [{
            "source": {"id": EVIDENCE},
            "metadata": [{"ragtest_feedback_ref": ref}],
        } for ref in refs],
    }
    messages = [message]
    if duplicate:
        messages.append(dict(message))
    return {"id": "answer", "messages": messages}


def test_filter_persists_feedback_locator_only_in_source_metadata():
    module = _module(CITATIONS, "ragtest_feedback_metadata")
    source = module.source_event(_citation())
    assert source["data"]["source"] == {
        "name": "report.pdf", "id": EVIDENCE}
    assert source["data"]["metadata"] == [{
        "source": "report.pdf", "name": "report.pdf", "page": 4,
        "ragtest_feedback_ref": FEEDBACK,
    }]
    rendered = repr(source)
    assert "passage" not in rendered and "ticket" not in rendered


def test_legacy_citation_still_renders_without_feedback_metadata():
    module = _module(CITATIONS, "ragtest_feedback_legacy")
    source = module.source_event(_citation(None))
    assert "ragtest_feedback_ref" not in source["data"]["metadata"][0]


def test_conflicting_refs_drop_feedback_but_keep_evidence_sources():
    module = _module(CITATIONS, "ragtest_feedback_conflict")
    emitted = []

    async def emit(value):
        emitted.append(value)

    event = {
        "rag_status": "answered",
        "rag_citations": [_citation("F" * 43), _citation("G" * 43)],
        "choices": [{"finish_reason": "stop"}],
    }
    asyncio.run(module.Filter().stream(event, __event_emitter__=emit))
    assert len(emitted) == 1  # duplicate evidence reference stays deduplicated
    assert "ragtest_feedback_ref" not in emitted[0]["data"]["metadata"][0]
    assert emitted[0]["data"]["source"]["id"] == EVIDENCE


def test_action_extracts_one_unique_marked_ref_from_clicked_assistant_only():
    module = _module(ACTION, "ragtest_feedback_clicked")
    assert module.feedback_reference(_body(FEEDBACK, FEEDBACK)) == FEEDBACK
    assert module.feedback_reference(_body(FEEDBACK, role="user")) is None
    assert module.feedback_reference(_body(FEEDBACK, duplicate=True)) is None
    assert module.feedback_reference(_body(FEEDBACK, "G" * 43)) is None
    assert module.feedback_reference(_body("short")) is None


def test_action_posts_only_closed_helpful_feedback(monkeypatch):
    module = _module(ACTION, "ragtest_feedback_helpful")
    calls = []
    notifications = []

    def request(user, payload):
        calls.append((user, payload))
        return 200, {"status": "recorded", "revision": 1,
                     "review_open": False}

    async def event_call(_event):
        return "1"

    async def emit(event):
        notifications.append(event)

    monkeypatch.setattr(module, "_request", request)
    user = {"id": "reader"}
    result = asyncio.run(module.Action().action(
        _body(FEEDBACK), __user__=user, __event_call__=event_call,
        __event_emitter__=emit))
    assert result is None
    assert calls == [(user, {
        "feedback_ref": FEEDBACK, "verdict": "helpful",
        "reason_code": None,
    })]
    assert notifications[-1]["data"]["type"] == "success"


def test_action_posts_closed_negative_reason_without_message_content(monkeypatch):
    module = _module(ACTION, "ragtest_feedback_negative")
    answers = iter(("2", "2"))
    calls = []

    async def event_call(_event):
        return next(answers)

    def request(_user, payload):
        calls.append(payload)
        return 200, {"status": "recorded", "revision": 2,
                     "review_open": True}

    monkeypatch.setattr(module, "_request", request)
    body = _body(FEEDBACK)
    body["messages"][0]["content"] = "PRIVATE_QUESTION_AND_ANSWER"
    asyncio.run(module.Action().action(
        body, __user__={"id": "reader"}, __event_call__=event_call))
    assert calls == [{
        "feedback_ref": FEEDBACK, "verdict": "not_helpful",
        "reason_code": "missing_evidence",
    }]
    assert "PRIVATE_QUESTION_AND_ANSWER" not in repr(calls)


def test_malformed_or_backend_failure_never_echoes_untrusted_prose(monkeypatch):
    module = _module(ACTION, "ragtest_feedback_failure")
    notifications = []

    async def emit(event):
        notifications.append(event)

    async def event_call(_event):
        return "1"

    def fail(*_args, **_kwargs):
        raise RuntimeError("PRIVATE_BACKEND_DETAIL")

    monkeypatch.setattr(module, "_request", fail)
    asyncio.run(module.Action().action(
        _body(FEEDBACK), __user__={"id": "reader"},
        __event_call__=event_call, __event_emitter__=emit))
    assert "PRIVATE_BACKEND_DETAIL" not in repr(notifications)
    assert notifications[-1]["data"]["type"] == "error"
