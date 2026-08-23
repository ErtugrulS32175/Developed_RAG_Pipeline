import copy
import json

import pytest

from eval import quality_gate


SENTINELS = {
    "question": "PRIVATE_QUESTION_SENTINEL",
    "answer": "PRIVATE_ANSWER_SENTINEL",
    "context": "PRIVATE_CONTEXT_SENTINEL",
    "miss": "PRIVATE_MISS_SENTINEL",
}


def _world(tmp_path, *, answer="555 birim Sayfa 42"):
    run_dir = tmp_path / "run"
    question_dir = tmp_path / "questions"
    run_dir.mkdir()
    question_dir.mkdir()
    questions = [{"q": SENTINELS["question"], "key": "555 birim",
                  "answer": "555 birim", "pages": [42], "type": "metin"}]
    (question_dir / "invented.json").write_text(
        json.dumps(questions), encoding="utf-8")
    (run_dir / "rag_answers_invented.json").write_text(json.dumps({
        "ozet": {"cevap_dogrulugu": 0.0},
        "sorular": [{"soru": SENTINELS["question"], "cevap": answer,
                     "baglam": f"{SENTINELS['context']} 555 birim"}],
    }), encoding="utf-8")
    (run_dir / "rag_eval_invented.json").write_text(json.dumps({
        "profile": {"n": 1, "ctx_recall": 1.0, "hit@5": 1.0,
                    "mrr": 1.0, "ctx_kacan": [SENTINELS["miss"]]},
    }), encoding="utf-8")
    policy = {
        "version": 1,
        "sets": {"invented": {
            "n": 1,
            "retrieval_profile": "profile",
            "retrieval_minimum": {"ctx_recall": 1.0, "hit@5": 1.0,
                                  "mrr": 1.0},
            "answer_minimum": {"ctx_recall": 1.0,
                               "cevap_dogrulugu": 1.0,
                               "ust_sinir": 1.0,
                               "sayfa_dogrulugu": 1.0},
            "answer_maximum": {"incele_orani": 0.0},
        }},
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return run_dir, question_dir, policy_path, policy


def test_a_green_saved_run_passes_without_trusting_its_old_summary(tmp_path):
    run_dir, question_dir, policy_path, _ = _world(tmp_path)
    result = quality_gate.evaluate_quality(run_dir, question_dir, policy_path)
    assert result.passed is True
    assert result.report["sets"]["invented"]["passed"] is True


def test_the_gate_never_calls_the_embedding_service(tmp_path, monkeypatch):
    run_dir, question_dir, policy_path, policy = _world(
        tmp_path, answer="an unsettled paraphrase Sayfa 42")
    policy["sets"]["invented"]["answer_minimum"]["cevap_dogrulugu"] = 0.0
    policy["sets"]["invented"]["answer_maximum"]["incele_orani"] = 1.0
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("embedding service must stay offline")

    monkeypatch.setattr("eval.answer.judge.similarity", forbidden)
    result = quality_gate.evaluate_quality(run_dir, question_dir, policy_path)
    assert result.passed is True
    checks = result.report["sets"]["invented"]["checks"]
    assert next(c for c in checks if c["metric"] == "incele_orani")["actual"] == 1.0


def test_the_report_contains_no_question_answer_context_or_miss_text(tmp_path):
    run_dir, question_dir, policy_path, _ = _world(tmp_path)
    report = quality_gate.evaluate_quality(
        run_dir, question_dir, policy_path).report
    rendered = json.dumps(dict(report), sort_keys=True)
    assert all(sentinel not in rendered for sentinel in SENTINELS.values())
    assert set(report) == {"quality_gate_version", "passed", "sets", "failures"}


def test_an_answer_regression_is_a_closed_red_verdict(tmp_path):
    run_dir, question_dir, policy_path, _ = _world(
        tmp_path, answer="unrelated answer Sayfa 42")
    result = quality_gate.evaluate_quality(run_dir, question_dir, policy_path)
    assert result.passed is False
    assert result.report["failures"] == [
        {"set": "invented", "family": "answer",
         "metric": "cevap_dogrulugu"},
        {"set": "invented", "family": "answer",
         "metric": "incele_orani"},
    ]


def test_a_retrieval_regression_is_red(tmp_path):
    run_dir, question_dir, policy_path, _ = _world(tmp_path)
    path = run_dir / "rag_eval_invented.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["profile"]["hit@5"] = 0.0
    path.write_text(json.dumps(report), encoding="utf-8")
    result = quality_gate.evaluate_quality(run_dir, question_dir, policy_path)
    assert result.passed is False
    assert {failure["metric"] for failure in result.report["failures"]} == {"hit@5"}


@pytest.mark.parametrize("mutation", [
    lambda policy: policy.update(extra=True),
    lambda policy: policy.update(version=2),
    lambda policy: policy["sets"]["invented"].update(extra=True),
    lambda policy: policy["sets"]["invented"]["retrieval_minimum"].pop("mrr"),
    lambda policy: policy["sets"]["invented"]["answer_maximum"].update(
        incele_orani=float("nan")),
])
def test_the_policy_is_closed_and_finite(tmp_path, mutation):
    _, _, policy_path, policy = _world(tmp_path)
    mutation(policy)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(quality_gate.QualityGateError):
        quality_gate.load_policy(policy_path)


@pytest.mark.parametrize("breakage", [
    "missing_answer", "missing_retrieval", "duplicate_question",
    "missing_answer_row", "wrong_profile", "wrong_n",
])
def test_incomplete_or_ambiguous_evidence_fails_closed(tmp_path, breakage):
    run_dir, question_dir, policy_path, _ = _world(tmp_path)
    answer_path = run_dir / "rag_answers_invented.json"
    retrieval_path = run_dir / "rag_eval_invented.json"
    if breakage == "missing_answer":
        answer_path.unlink()
    elif breakage == "missing_retrieval":
        retrieval_path.unlink()
    elif breakage == "duplicate_question":
        questions = json.loads((question_dir / "invented.json").read_text())
        questions.append(copy.deepcopy(questions[0]))
        (question_dir / "invented.json").write_text(json.dumps(questions))
    elif breakage == "missing_answer_row":
        answer_path.write_text(json.dumps({"ozet": {}, "sorular": []}))
    elif breakage == "wrong_profile":
        retrieval_path.write_text(json.dumps({"other": {}}))
    else:
        report = json.loads(retrieval_path.read_text())
        report["profile"]["n"] = 2
        retrieval_path.write_text(json.dumps(report))
    with pytest.raises(quality_gate.QualityGateError):
        quality_gate.evaluate_quality(run_dir, question_dir, policy_path)


def test_cli_exit_codes_distinguish_green_red_and_invalid(tmp_path, capsys):
    run_dir, question_dir, policy_path, policy = _world(tmp_path)
    args = ["--run-dir", str(run_dir), "--question-dir", str(question_dir),
            "--policy", str(policy_path)]
    assert quality_gate.main(args) == 0
    policy["sets"]["invented"]["answer_minimum"]["cevap_dogrulugu"] = 1.0
    (run_dir / "rag_answers_invented.json").write_text(json.dumps({
        "ozet": {}, "sorular": [{"soru": SENTINELS["question"],
                                  "cevap": "wrong",
                                  "baglam": SENTINELS["context"]}]}))
    assert quality_gate.main(args) == 1
    policy_path.write_text("not-json", encoding="utf-8")
    assert quality_gate.main(args) == 2
    output = capsys.readouterr().out
    assert all(sentinel not in output for sentinel in SENTINELS.values())
