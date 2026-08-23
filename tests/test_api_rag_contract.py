"""The public RAG boundary publishes decisions, never raw backend strings."""
import json

import pytest
from fastapi.testclient import TestClient

from pipeline.validation.rag.answer_guard import (
    ABSTAINED,
    ANSWERED,
    DERIVED_CITATION,
    REVIEW_REQUIRED,
    GuardResult,
    PageCitation,
)
from pipeline.retrieval.trace import RetrievalTrace, TraceStage


MODELS = [
    ("ragtest-rag", "native"),
    ("ragtest-rag-llamaindex", "llamaindex"),
]


def _headers(api):
    return (
        {"Authorization": f"Bearer {api.API_KEY}"}
        if api.API_KEY else {}
    )


def _request(api, model, stream, include_trace=False):
    return TestClient(api.app).post(
        "/v1/chat/completions",
        headers=_headers(api),
        json={
            "model": model,
            "messages": [{"role": "user", "content": "kurgu soru"}],
            "stream": stream,
            "include_trace": include_trace,
        },
    )


def _public_reply(response, stream):
    if not stream:
        body = response.json()
        return body["rag_status"], body["choices"][0]["message"]["content"]

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    statuses = {payload["rag_status"] for payload in payloads}
    assert len(statuses) == 1
    text = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )
    return statuses.pop(), text


def _public_citations(response, stream):
    if not stream:
        return response.json()["rag_citations"]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    citations = {json.dumps(
        payload["rag_citations"],
        sort_keys=True,
    ) for payload in payloads}
    assert len(citations) == 1
    return json.loads(citations.pop())


def _public_trace(response, stream):
    if not stream:
        return response.json().get("rag_trace")
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    traces = {json.dumps(payload.get("rag_trace"), sort_keys=True)
              for payload in payloads}
    assert len(traces) == 1
    return json.loads(traces.pop())


def _trace():
    return RetrievalTrace(
        trace_id="a" * 32,
        backend="native",
        scope_document_count=2,
        retrieved_count=15,
        reranked_count=10,
        context_passage_count=10,
        stages=(
            TraceStage("retrieve", 7), TraceStage("rerank", 3),
            TraceStage("context", 1), TraceStage("generate", 9),
            TraceStage("validate", 2),
        ),
    )


@pytest.mark.parametrize("stream", [False, True])
def test_trace_is_opt_in_closed_and_identical_across_response_shapes(
        monkeypatch, stream):
    from pipeline.api import app as api

    trace = _trace()
    result = GuardResult(ANSWERED, "Kurgu cevap.", (), trace=trace)
    monkeypatch.setattr(api.rag_backends, "answer_checked",
                        lambda *_args, **_kwargs: result)

    absent = _request(api, "ragtest-rag", stream)
    present = _request(api, "ragtest-rag", stream, include_trace=True)

    assert _public_trace(absent, stream) is None
    assert _public_trace(present, stream) == trace.public()
    assert set(trace.public()) == {
        "trace_id", "backend", "scope_document_count", "retrieved_count",
        "reranked_count", "context_passage_count", "stages_ms",
    }
    forbidden = ("kurgu soru", "Kurgu cevap", "OZEL_DOSYA_ADI.pdf",
                 "11111111-1111-1111-1111-111111111111",
                 "HAM_PASAJ_METNI", "0.91357")
    encoded = json.dumps(trace.public())
    assert not any(value in encoded for value in forbidden)


@pytest.mark.parametrize("stream", [False, True])
def test_requested_trace_is_required_and_malformed_trace_fails_closed(
        monkeypatch, stream):
    from pipeline.api import app as api

    monkeypatch.setattr(
        api.rag_backends, "answer_checked",
        lambda *_args, **_kwargs: GuardResult(ANSWERED, "Kurgu cevap.", ()))
    assert _request(
        api, "ragtest-rag", stream, include_trace=True).status_code == 500

    monkeypatch.setattr(
        api.rag_backends, "answer_checked",
        lambda *_args, **_kwargs: GuardResult(
            ANSWERED, "Kurgu cevap.", (), trace="HAM_KURGU_TRACE"))
    response = _request(api, "ragtest-rag", stream)
    assert response.status_code == 500
    assert "HAM_KURGU_TRACE" not in response.text


@pytest.mark.parametrize(("model", "backend"), MODELS)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("result", "expected_status", "expected_text"),
    [
        (
            GuardResult(ANSWERED, "Kurgu cevap.", ()),
            ANSWERED,
            "Kurgu cevap.",
        ),
        (
            GuardResult(
                ABSTAINED,
                "Bu bilgi mevcut belgelerde bulunamadi.",
                (),
            ),
            ABSTAINED,
            "Bu bilgi mevcut belgelerde bulunamadi.",
        ),
        (
            GuardResult(
                REVIEW_REQUIRED,
                None,
                (("KURGU_GIZLI_TANI", []),),
            ),
            REVIEW_REQUIRED,
            None,
        ),
    ],
)
def test_both_api_shapes_publish_only_the_checked_decision(
        monkeypatch, model, backend, stream, result, expected_status,
        expected_text):
    from pipeline.api import app as api

    calls = []

    def checked(question, backend=None):
        calls.append((question, backend))
        return result

    monkeypatch.setattr(api.rag_backends, "answer_checked", checked)
    monkeypatch.setattr(
        api.rag_backends,
        "answer",
        lambda *_args, **_kwargs: pytest.fail("unchecked path was called"),
    )

    response = _request(api, model, stream)

    assert response.status_code == 200
    status, text = _public_reply(response, stream)
    assert status == expected_status
    if expected_text is None:
        assert text == api.REVIEW_MESSAGE
        assert "KURGU_GIZLI_TANI" not in response.text
    else:
        assert text == expected_text
    assert calls == [("kurgu soru", backend)]


@pytest.mark.parametrize(("model", "_backend"), MODELS)
@pytest.mark.parametrize("stream", [False, True])
def test_derived_citations_are_separate_metadata(
        monkeypatch, model, _backend, stream):
    from pipeline.api import app as api

    result = GuardResult(
        ANSWERED,
        "KURGU_OMEGA_NESNESI yanitidir.",
        (),
        (PageCitation(953761, DERIVED_CITATION),),
    )
    monkeypatch.setattr(
        api.rag_backends,
        "answer_checked",
        lambda *_args, **_kwargs: result,
    )

    response = _request(api, model, stream)

    assert response.status_code == 200
    assert _public_citations(response, stream) == [{
        "page": 953761,
        "source": DERIVED_CITATION,
    }]
    _, text = _public_reply(response, stream)
    assert text == result.answer
    assert "953761" not in text


@pytest.mark.parametrize(("model", "_backend"), MODELS)
@pytest.mark.parametrize("stream", [False, True])
def test_a_raw_checked_backend_value_fails_closed(
        monkeypatch, model, _backend, stream):
    from pipeline.api import app as api

    raw = "DENETLENMEMIS_KURGU_CEVAP"
    monkeypatch.setattr(
        api.rag_backends,
        "answer_checked",
        lambda *_args, **_kwargs: raw,
    )

    response = _request(api, model, stream)

    assert response.status_code == 500
    assert raw not in response.text


@pytest.mark.parametrize("stream", [False, True])
def test_invalid_citation_metadata_fails_closed(monkeypatch, stream):
    from pipeline.api import app as api

    monkeypatch.setattr(
        api.rag_backends,
        "answer_checked",
        lambda *_args, **_kwargs: GuardResult(
            ANSWERED,
            "KURGU_OMEGA_NESNESI yanitidir.",
            (),
            ("gecersiz",),
        ),
    )

    response = _request(api, "ragtest-rag", stream)

    assert response.status_code == 500
    assert "KURGU_OMEGA_NESNESI" not in response.text


@pytest.mark.parametrize(
    ("result", "raw"),
    [
        (
            GuardResult(
                REVIEW_REQUIRED,
                "INCELEMEDE_YAYINLANMAMALI",
                (),
            ),
            "INCELEMEDE_YAYINLANMAMALI",
        ),
        (
            GuardResult(
                ANSWERED,
                "TANILI_CEVAP_YAYINLANMAMALI",
                (("KURGU_TANI", []),),
            ),
            "TANILI_CEVAP_YAYINLANMAMALI",
        ),
        (
            GuardResult(
                ABSTAINED,
                "CEKIMSERLIK_OLMAYAN_METIN",
                (),
            ),
            "CEKIMSERLIK_OLMAYAN_METIN",
        ),
        (
            GuardResult(
                ANSWERED,
                "Bu bilgi mevcut belgelerde bulunamadi.",
                (),
            ),
            "Bu bilgi mevcut belgelerde bulunamadi.",
        ),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_an_inconsistent_checked_result_cannot_publish_its_text(
        monkeypatch, stream, result, raw):
    from pipeline.api import app as api

    monkeypatch.setattr(
        api.rag_backends,
        "answer_checked",
        lambda *_args, **_kwargs: result,
    )

    response = _request(api, "ragtest-rag", stream)

    assert response.status_code == 500
    assert raw not in response.text


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (RuntimeError("OZEL_CALISMA_ZAMANI_AYRINTISI"), 503),
        (ConnectionError("OZEL_BAGLANTI_AYRINTISI"), 502),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_backend_failures_do_not_leak_their_message(
        monkeypatch, caplog, error, status_code, stream):
    from pipeline.api import app as api

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(api.rag_backends, "answer_checked", fail)

    response = _request(api, "ragtest-rag", stream)

    assert response.status_code == status_code
    assert str(error) not in response.text
    assert str(error) not in caplog.text
    assert type(error).__name__ in caplog.text


@pytest.mark.parametrize("content", ["", "   ", "\n\t "])
def test_an_empty_question_is_rejected_before_the_backend(
        monkeypatch, content):
    from pipeline.api import app as api

    calls = []
    monkeypatch.setattr(
        api.rag_backends,
        "answer_checked",
        lambda *_args, **_kwargs: calls.append(True),
    )
    response = TestClient(api.app).post(
        "/v1/chat/completions",
        headers=_headers(api),
        json={
            "model": "ragtest-rag",
            "messages": [{"role": "user", "content": content}],
        },
    )

    assert response.status_code == 400
    assert calls == []
