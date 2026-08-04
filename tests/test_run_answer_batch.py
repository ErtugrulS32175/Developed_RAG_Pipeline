"""The rented-GPU preflight must itself fail safely and readably."""
import json
import sys
from types import SimpleNamespace

import requests

from eval.answer import run_answer_batch


def _valid_question():
    return {
        "q": "KURGU_OMEGA_NESNESI nedir?",
        "key": "731641 birim",
        "answer": "KURGU_OMEGA_NESNESI 731641 birimdir.",
        "pages": [953761],
    }


def test_preflight_reports_a_dead_llm_endpoint_instead_of_crashing(
        monkeypatch, tmp_path):
    import psycopg

    from pipeline.generation import answer as generation

    question_dir = tmp_path / "questions"
    question_dir.mkdir()
    (question_dir / "kurgu.json").write_text(
        json.dumps([_valid_question()]),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_answer_batch, "QUESTION_DIR", question_dir)
    monkeypatch.setattr(
        generation,
        "LLM_API_URL",
        "http://127.0.0.1:9/v1/chat/completions",
    )
    monkeypatch.setattr(generation, "LLM_API_KEY", "")
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            psycopg.OperationalError("kurgu veritabani")
        ),
    )

    def dead_get(*_args, **_kwargs):
        raise requests.ConnectionError("kurgu olu uc nokta")

    monkeypatch.setattr(requests, "get", dead_get)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.ConnectionError("kurgu yerel servis")
        ),
    )

    problems, notes = run_answer_batch.preflight(["kurgu"])

    assert any("1 soru" in note for note in notes)
    assert "calistirilacak setler: kurgu" in notes
    llm_problems = [problem for problem in problems
                    if problem.startswith("LLM'e ulasilamadi")]
    assert len(llm_problems) == 1
    assert "127.0.0.1:9/v1" in llm_problems[0]
    assert "[ConnectionError]" in llm_problems[0]


def test_discovery_excludes_auxiliary_json_that_is_not_scoreable(
        monkeypatch, tmp_path):
    (tmp_path / "kurgu.json").write_text(
        json.dumps([_valid_question()]),
        encoding="utf-8",
    )
    (tmp_path / "yardimci.json").write_text(
        json.dumps([{"q": "KURGU_OMEGA_NESNESI nedir?"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_answer_batch, "QUESTION_DIR", tmp_path)

    assert run_answer_batch.discover_sets() == ["kurgu"]


def test_schema_preflight_refuses_an_explicit_invalid_set_before_services(
        monkeypatch, tmp_path):
    (tmp_path / "yardimci.json").write_text(
        json.dumps([{"q": "KURGU_OMEGA_NESNESI nedir?"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_answer_batch, "QUESTION_DIR", tmp_path)

    problems, notes = run_answer_batch.preflight(["yardimci"])

    assert any("eksik alanlar" in problem for problem in problems)
    assert notes == ["calistirilacak setler: yardimci"]


def test_preflight_uses_generation_model_and_auth(monkeypatch, tmp_path):
    import psycopg

    from pipeline.generation import answer as generation

    monkeypatch.setattr(run_answer_batch, "QUESTION_DIR", tmp_path)
    monkeypatch.setattr(
        generation,
        "LLM_API_URL",
        "http://generation.test/v1/chat/completions",
    )
    monkeypatch.setattr(generation, "LLM_MODEL_NAME", "kurgu-model")
    monkeypatch.setattr(generation, "LLM_API_KEY", "kurgu-anahtar")
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            psycopg.OperationalError("kurgu veritabani")
        ),
    )

    seen = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "kurgu-model"}]}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.ConnectionError("kurgu yerel servis")
        ),
    )

    problems, notes = run_answer_batch.preflight([])

    assert seen == {
        "url": "http://generation.test/v1/models",
        "headers": {"Authorization": "Bearer kurgu-anahtar"},
    }
    assert not any("LLM'e ulasilamadi" in problem for problem in problems)
    assert any("LLM: kurgu-model" in note for note in notes)


def test_run_set_invokes_the_post_split_answer_module(monkeypatch, tmp_path):
    commands = []

    def fake_run(command):
        commands.append(command)
        (tmp_path / "rag_answers_kurgu.json").write_text(
            json.dumps({"ozet": {}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_answer_batch.subprocess, "run", fake_run)

    result = run_answer_batch.run_set(
        "kurgu",
        rerank_to=7,
        top_k=11,
        backend="native",
        out_dir=tmp_path,
        structured=True,
    )

    assert result["_arsiv"] == "rag_answers_kurgu_k7.json"
    assert commands == [[
        sys.executable,
        "-m",
        "eval.answer.rag_answer_eval",
        "--set",
        "kurgu",
        "--top-k",
        "11",
        "--rerank",
        "--rerank-to",
        "7",
        "--backend",
        "native",
        "--out-dir",
        str(tmp_path),
        "--structured",
    ]]
