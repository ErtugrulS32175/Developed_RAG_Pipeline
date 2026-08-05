"""Local end-to-end contracts for the production API wiring.

External services are replaced at their narrow network/storage seams.  The
application router, backend selection, structured generation, deterministic
guard and public response projection remain real.

Every filename, passage and figure below is invented.
"""
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient


@contextmanager
def _fake_db_conn():
    """Stands in for the pooled per-request connection; the db helpers are
    monkeypatched, so the connection object itself is never touched."""
    yield object()

from pipeline.validation.rag.answer_guard import ANSWERED, REVIEW_REQUIRED


CHUNKS = [
    {
        "filename": "kurgu-belge.pdf",
        "page": 42,
        "type": "text",
        "text": "Zeta uretimi 47 000 birimdir.",
        "headings": [],
        "table_data": None,
    },
]


def _headers(api):
    return (
        {"Authorization": f"Bearer {api.API_KEY}"}
        if api.API_KEY else {}
    )


def _chat(api, model, stream=False):
    return TestClient(api.app).post(
        "/v1/chat/completions",
        headers=_headers(api),
        json={
            "model": model,
            "messages": [{"role": "user", "content": "zeta uretimi nedir?"}],
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


def _wire_retrieval(monkeypatch, backend):
    from pipeline.generation import answer as generation

    if backend == "native":
        from pipeline.retrieval import query

        monkeypatch.setattr(query, "retrieve", lambda _question: CHUNKS)
        monkeypatch.setattr(
            query,
            "rerank",
            lambda _question, chunks: chunks,
        )
    else:
        from pipeline.retrieval import rag_llamaindex

        monkeypatch.setattr(
            rag_llamaindex,
            "retrieve",
            lambda _question: CHUNKS,
        )
    return generation


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        ("ragtest-rag", "native"),
        ("ragtest-rag-llamaindex", "llamaindex"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_public_api_runs_the_real_checked_path(
        monkeypatch, model, backend, stream):
    from pipeline.api import app as api

    generation = _wire_retrieval(monkeypatch, backend)
    seen = []

    def complete(policy, user_content):
        seen.append((policy, user_content))
        return json.dumps({
            "dayanak": [{
                "pasaj": 1,
                "alinti": "Zeta uretimi 47 000 birimdir.",
            }],
            "cevap": "Sayfa 42'ye gore 47 000 birim.",
        })

    monkeypatch.setattr(generation, "complete", complete)

    response = _chat(api, model, stream)

    assert response.status_code == 200
    status, text = _public_reply(response, stream)
    assert status == ANSWERED
    assert text == "Sayfa 42'ye gore 47 000 birim."
    assert len(seen) == 1
    policy, user_content = seen[0]
    assert policy.index("dayanak") < policy.index('"cevap"')
    assert "[P1]" in user_content
    assert "zeta uretimi nedir?" in user_content
    assert "Zeta uretimi" not in policy


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        ("ragtest-rag", "native"),
        ("ragtest-rag-llamaindex", "llamaindex"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_public_api_withholds_a_reply_rejected_by_the_real_guard(
        monkeypatch, model, backend, stream):
    from pipeline.api import app as api

    generation = _wire_retrieval(monkeypatch, backend)
    raw = "Sayfa 42'ye gore 88 000 birim."
    monkeypatch.setattr(
        generation,
        "complete",
        lambda _policy, _user_content: json.dumps({
            "dayanak": [{
                "pasaj": 1,
                "alinti": "Zeta uretimi 47 000 birimdir.",
            }],
            "cevap": raw,
        }),
    )

    response = _chat(api, model, stream)

    assert response.status_code == 200
    status, text = _public_reply(response, stream)
    assert status == REVIEW_REQUIRED
    assert text == api.REVIEW_MESSAGE
    assert raw not in response.text


def _document_api(monkeypatch, tmp_path, *, ingest_status="done",
                  ingest_error=None):
    from pipeline.api import app as api

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "db_conn", _fake_db_conn)

    document_id = "kurgu-belge-kimligi"
    state = {}
    calls = []

    def upsert_document(_conn, filename, file_type, status):
        state[document_id] = {
            "id": document_id,
            "filename": filename,
            "file_type": file_type,
            "status": status,
        }
        return document_id

    def get_document(_conn, wanted):
        value = state.get(wanted)
        return dict(value) if value is not None else None

    def set_document_status(_conn, wanted, status):
        state[wanted]["status"] = status

    def run_ingest(path):
        calls.append(path)
        if ingest_error is not None:
            raise ingest_error
        if ingest_status is not None:
            state[document_id]["status"] = ingest_status

    monkeypatch.setattr(api.db, "upsert_document", upsert_document)
    monkeypatch.setattr(api.db, "get_document", get_document)
    monkeypatch.setattr(api.db, "set_document_status", set_document_status)
    monkeypatch.setattr(api.ingest, "main", run_ingest)
    return api, TestClient(api.app), state, calls, upload_dir


def test_document_upload_process_and_read_use_the_production_routes(
        monkeypatch, tmp_path):
    api, client, state, calls, upload_dir = _document_api(
        monkeypatch,
        tmp_path,
    )

    uploaded = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"KURGU_PDF", "application/pdf")},
    )
    assert uploaded.status_code == 200
    document_id = uploaded.json()["document_id"]
    assert uploaded.json()["status"] == "pending"
    assert (upload_dir / "kurgu-belge.pdf").read_bytes() == b"KURGU_PDF"

    processed = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )
    assert processed.status_code == 200
    assert processed.json() == {"document_id": document_id, "status": "done"}
    assert calls == [str(upload_dir / "kurgu-belge.pdf")]

    read = client.get(
        f"/documents/{document_id}",
        headers=_headers(api),
    )
    assert read.status_code == 200
    assert read.json()["status"] == "done"
    assert state[document_id]["status"] == "done"


def test_process_does_not_claim_done_when_ingest_did_not_finish(
        monkeypatch, tmp_path):
    api, client, state, _calls, upload_dir = _document_api(
        monkeypatch,
        tmp_path,
        ingest_status="processing",
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert state[document_id]["status"] == "error"


def test_missing_uploaded_file_is_generic_and_marks_the_document_error(
        monkeypatch, tmp_path):
    api, client, state, calls, _upload_dir = _document_api(
        monkeypatch,
        tmp_path,
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
    }

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "uploaded file missing"
    assert state[document_id]["status"] == "error"
    assert calls == []
    assert "kurgu-belge.pdf" not in response.text


def test_process_failure_is_generic_and_marks_the_document_error(
        monkeypatch, tmp_path, caplog):
    private = "OZEL_KURGU_INGEST_AYRINTISI"
    api, client, state, _calls, upload_dir = _document_api(
        monkeypatch,
        tmp_path,
        ingest_error=RuntimeError(private),
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert response.json()["detail"] == api.DOCUMENT_PROCESSING_FAILURE_MESSAGE
    assert state[document_id]["status"] == "error"
    assert private not in response.text
    assert private not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize(
    "filename",
    [
        "../disari.pdf",
        "..\\disari.pdf",
        "/disari.pdf",
        "alt/disari.pdf",
        "alt\\disari.pdf",
        "kurgu.pdf.",
        "...",
        "NUL",
        "nul.pdf",
        "PRN.pdf",
    ],
)
def test_upload_rejects_a_path_or_noncanonical_filename_before_writing(
        monkeypatch, tmp_path, filename):
    api, client, _state, calls, upload_dir = _document_api(
        monkeypatch,
        tmp_path,
    )

    response = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": (filename, b"KURGU_PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert calls == []
    assert not (tmp_path / "disari.pdf").exists()
    assert list(upload_dir.iterdir()) == []


def test_rejected_alias_cannot_overwrite_an_existing_upload(
        monkeypatch, tmp_path):
    api, client, state, calls, upload_dir = _document_api(
        monkeypatch,
        tmp_path,
    )
    original = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu.pdf", b"ILK_KURGU_PDF", "application/pdf")},
    )
    alias = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu.pdf.", b"IKINCI_KURGU_PDF", "application/pdf")},
    )

    assert original.status_code == 200
    assert alias.status_code == 400
    assert (upload_dir / "kurgu.pdf").read_bytes() == b"ILK_KURGU_PDF"
    assert len(state) == 1
    assert calls == []


@pytest.mark.parametrize(
    "filename",
    [
        "C:\\disari.pdf",
        "kurgu.pdf\n",
        "kurgu\x00.pdf",
        "kurgu.pdf:ek",
        "kurgu.pdf.",
        "...",
        "NUL",
        "nul.pdf",
        "CON",
        "PRN.pdf",
        "AUX",
        "com1",
        "COM¹.txt",
        "LPT9.txt",
        "NUL .pdf",
        "..",
        None,
    ],
)
def test_unsafe_filename_is_rejected_before_path_construction(filename):
    from fastapi import HTTPException
    from pipeline.api import app as api

    with pytest.raises(HTTPException) as error:
        api._safe_upload_filename(filename)

    assert error.value.status_code == 400


def test_unhandled_dependency_error_never_copies_its_message_to_logs(
        monkeypatch, caplog):
    from pipeline.api import app as api

    private = "OZEL_KURGU_BAGLANTI_AYRINTISI"
    monkeypatch.setattr(api, "db_conn", _fake_db_conn)

    def fail(_conn, _document_id):
        raise RuntimeError(private)

    monkeypatch.setattr(api.db, "get_document", fail)
    response = TestClient(
        api.app,
        raise_server_exceptions=False,
    ).get(
        "/documents/kurgu-belge-kimligi",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert private not in response.text
    assert private not in caplog.text
    assert "RuntimeError" in caplog.text
    assert '"iz":' in caplog.text
    assert '"fonksiyon": "fail"' in caplog.text
