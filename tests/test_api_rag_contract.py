"""The public RAG boundary publishes decisions, never raw backend strings."""
import json

import pytest
from fastapi.testclient import TestClient

from pipeline.validation.rag.answer_guard import (
    ABSTAINED,
    ANSWERED,
    REVIEW_REQUIRED,
    GuardResult,
)


MODELS = [
    ("ragtest-rag", "native"),
    ("ragtest-rag-llamaindex", "llamaindex"),
]


def _headers(api):
    return (
        {"Authorization": f"Bearer {api.API_KEY}"}
        if api.API_KEY else {}
    )


def _request(api, model, stream):
    return TestClient(api.app).post(
        "/v1/chat/completions",
        headers=_headers(api),
        json={
            "model": model,
            "messages": [{"role": "user", "content": "kurgu soru"}],
            "stream": stream,
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
