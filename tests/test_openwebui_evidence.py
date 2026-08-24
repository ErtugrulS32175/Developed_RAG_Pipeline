"""OpenWebUI citation rendering and freshly-authorized evidence previews."""
import asyncio
import importlib.util
from pathlib import Path

import pytest

from pipeline.api import identity


ROOT = Path(__file__).resolve().parent.parent
CITATIONS = ROOT / "openwebui" / "functions" / "ragtest_citations.py"
ACTION = ROOT / "openwebui" / "functions" / "ragtest_evidence_action.py"
REF_ONE = "A" * 43
REF_TWO = "B" * 43
TICKET = "T" * 43


def _module(path, name, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv(
            "OPENWEBUI_GATEWAY_KEY",
            "evidence-gateway-key-with-more-than-32-bytes")
        monkeypatch.setenv(
            "FORWARD_USER_INFO_HEADER_JWT_SECRET",
            "evidence-identity-key-with-more-than-32-bytes")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _citation(ref=REF_ONE, name="kurgu.pdf", page=7):
    return {
        "evidence_ref": ref,
        "document_name": name,
        "page": page,
        "source": "derived",
    }


def _body(*sources, clicked="answer"):
    return {
        "id": clicked,
        "messages": [
            {"id": "other", "role": "assistant", "sources": [{
                "source": {"id": "Z" * 43, "name": "wrong.pdf"},
            }]},
            {"id": "answer", "role": "assistant", "sources": [{
                "source": {"id": ref, "name": f"source-{index}"},
                "document": ["fixed hint"],
                "metadata": [{"page": index}],
            } for index, ref in enumerate(sources, 1)]},
        ],
    }


def test_filter_emits_the_official_closed_source_shape_once():
    module = _module(CITATIONS, "ragtest_citations_shape")
    emitted = []

    async def emit(value):
        emitted.append(value)

    event = {
        "rag_status": "answered",
        "rag_citations": [_citation()],
        "choices": [{"delta": {"content": "cevap"},
                     "finish_reason": None}],
    }
    filter_ = module.Filter()
    metadata = {"chat_id": "chat", "message_id": "message"}
    returned = asyncio.run(filter_.stream(
        event, __event_emitter__=emit, __metadata__=metadata))
    asyncio.run(filter_.stream(
        event, __event_emitter__=emit, __metadata__=metadata))

    assert returned is event
    assert emitted == [{
        "type": "source",
        "data": {
            "source": {"name": "kurgu.pdf", "id": REF_ONE},
            "document": [module._DOCUMENT_HINT],
            "metadata": [{
                "source": "kurgu.pdf", "name": "kurgu.pdf", "page": 7,
            }],
        },
    }]
    encoded = repr(emitted[0])
    assert "url" not in encoded
    assert "ticket" not in encoded
    assert "passage" not in encoded


def test_filter_fails_closed_on_every_open_or_malformed_citation_shape():
    module = _module(CITATIONS, "ragtest_citations_closed")
    good = _citation()
    invalid = [
        None,
        {**good, "extra": "open"},
        {**good, "evidence_ref": "short"},
        {**good, "document_name": ""},
        {**good, "document_name": "line\nbreak"},
        {**good, "page": True},
        {**good, "page": 0},
        {**good, "source": "invented"},
    ]
    assert all(module.source_event(value) is None for value in invalid)
    assert module.source_event({**good, "document_name": "a" * 500})
    assert module.source_event(
        {**good, "document_name": "a" * 501}) is None


def test_preview_uses_the_same_document_name_bound_as_provenance():
    module = _module(ACTION, "ragtest_evidence_name_bound")
    accepted = {
        "document_name": "a" * 500,
        "page": 1,
        "content_type": "passage",
        "passage": "kurgu",
    }
    assert module._preview(accepted) == ("a" * 500, 1, "kurgu")
    with pytest.raises(RuntimeError, match="contract refused"):
        module._preview({**accepted, "document_name": "a" * 501})


def test_filter_does_not_publish_unanswered_or_unchecked_metadata():
    module = _module(CITATIONS, "ragtest_citations_status")
    emitted = []

    async def emit(value):
        emitted.append(value)

    filter_ = module.Filter()
    for status in (None, "abstained", "review_required"):
        event = {"rag_status": status, "rag_citations": [_citation()]}
        asyncio.run(filter_.stream(event, __event_emitter__=emit))
    assert emitted == []


def test_action_identity_is_accepted_without_openwebui_role_authority(
        monkeypatch):
    module = _module(ACTION, "ragtest_evidence_identity", monkeypatch)
    user = {"id": "owui-subject", "email": "display@example.invalid",
            "name": "Display", "role": "admin"}
    token = module.signed_identity(user, now=1_000)
    verifier = identity.Verifier.configured(
        "evidence-identity-key-with-more-than-32-bytes")
    resolved = verifier.verify(token, now=1_030)
    assert resolved.subject == "owui-subject"
    assert not hasattr(resolved, "role")


def test_action_reads_only_the_clicked_messages_official_sources(monkeypatch):
    module = _module(ACTION, "ragtest_evidence_clicked", monkeypatch)
    assert module.evidence_references(_body(REF_ONE, REF_ONE, REF_TWO)) == (
        REF_ONE, REF_TWO)
    assert module.evidence_references(
        _body(REF_ONE, clicked="missing")) == ()
    assert "Z" * 43 not in module.evidence_references(_body(REF_ONE))


def test_action_uses_two_post_bodies_and_displays_a_transient_preview(
        monkeypatch):
    module = _module(ACTION, "ragtest_evidence_flow", monkeypatch)
    calls = []

    async def proxy(user, path, payload):
        calls.append((user, path, payload))
        if path == "/v1/evidence/tickets":
            return 200, {"ticket": TICKET, "expires_in": 50}
        return 200, {
            "document_name": "kurgu<&>.pdf",
            "page": 7,
            "content_type": "passage",
            "passage": "KANIT <script>calisma()</script>",
        }

    monkeypatch.setattr(module, "_proxy", proxy)
    user = {"id": "reader", "email": "", "name": "Reader",
            "role": "user"}
    prompts = []

    async def event_call(value):
        prompts.append(value)
        return True

    response = asyncio.run(module.Action().action(
        _body(REF_ONE), __user__=user, __event_call__=event_call))

    assert [(path, payload) for _user, path, payload in calls] == [
        ("/v1/evidence/tickets", {"evidence_ref": REF_ONE}),
        ("/v1/evidence/preview", {"ticket": TICKET}),
    ]
    assert all(bound_user is user for bound_user, _path, _payload in calls)
    assert response is None
    assert prompts == [{
        "type": "confirmation",
        "data": {
            "title": "kurgu<&>.pdf — Sayfa 7",
            "message": "KANIT <script>calisma()</script>",
        },
    }]
    assert REF_ONE not in repr(prompts) and TICKET not in repr(prompts)


def test_action_selection_never_exposes_references_to_the_browser(monkeypatch):
    module = _module(ACTION, "ragtest_evidence_select", monkeypatch)
    calls = []
    prompts = []

    async def event_call(value):
        prompts.append(value)
        return "2"

    async def proxy(_user, path, payload):
        calls.append((path, payload))
        if path.endswith("tickets"):
            return 200, {"ticket": TICKET, "expires_in": 50}
        return 200, {"document_name": "ikinci.pdf", "page": 2,
                     "content_type": "passage", "passage": "ikinci kanit"}

    monkeypatch.setattr(module, "_proxy", proxy)
    response = asyncio.run(module.Action().action(
        _body(REF_ONE, REF_TWO), __user__={"id": "reader"},
        __event_call__=event_call))

    assert calls[0] == (
        "/v1/evidence/tickets", {"evidence_ref": REF_TWO})
    assert REF_ONE not in repr(prompts) and REF_TWO not in repr(prompts)
    assert response is None
    assert prompts[-1]["type"] == "confirmation"
    assert prompts[-1]["data"]["message"] == "ikinci kanit"


def test_action_failures_are_fixed_and_do_not_echo_backend_prose(monkeypatch):
    module = _module(ACTION, "ragtest_evidence_failure", monkeypatch)
    notifications = []

    async def emit(value):
        notifications.append(value)

    async def proxy(_user, _path, _payload):
        raise RuntimeError("KURGU_GIZLI_BACKEND_PROSE")

    monkeypatch.setattr(module, "_proxy", proxy)
    async def event_call(_value):
        return True

    result = asyncio.run(module.Action().action(
        _body(REF_ONE), __user__={"id": "reader"},
        __event_emitter__=emit, __event_call__=event_call))
    assert result is None
    assert "KURGU_GIZLI_BACKEND_PROSE" not in repr(notifications)
    assert notifications[0]["data"]["type"] == "error"


def test_action_source_has_no_browser_side_token_transport():
    source = ACTION.read_text(encoding="utf-8")
    assert "localStorage" not in source
    assert '"execute"' not in source
    assert "from fastapi.responses" not in source
    assert "return HTMLResponse(" not in source
    assert "asyncio.wait_for" in source
    assert "_EVENT_CALL_TIMEOUT_SECONDS = 300" in source
    assert "method=\"POST\"" in source
    assert '"/v1/evidence/tickets"' in source
    assert '"/v1/evidence/preview"' in source
    assert "?ticket=" not in source and "?evidence_ref=" not in source
    assert "/{ticket}" not in source and "/{evidence_ref}" not in source
