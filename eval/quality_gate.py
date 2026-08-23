"""Offline, content-free release gate for saved RAG evaluations.

The retrieval harness already writes aggregate reports.  Answer reports also
carry the raw answers needed to re-score a run after the scorer changes.  This
module deliberately trusts neither an old answer summary nor an online
embedding service: deterministic tiers form the lower bound and every
unsettled answer remains in the upper bound.

The emitted report contains only set names, counts, metric names and numbers.
Questions, answers, contexts, filenames and review reasons never cross this
boundary.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from eval.answer.judge import SIM_THRESHOLD
from eval.answer.rag_answer_eval import score_one, summarize


POLICY_VERSION = 1
RETRIEVAL_METRICS = ("ctx_recall", "hit@5", "mrr")
ANSWER_MINIMUM_METRICS = (
    "ctx_recall", "cevap_dogrulugu", "ust_sinir", "sayfa_dogrulugu",
)
ANSWER_MAXIMUM_METRICS = ("incele_orani",)


class QualityGateError(ValueError):
    """A malformed or incomplete input; the gate must fail closed."""


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    report: Mapping[str, object]


def _object(value, label):
    if type(value) is not dict:
        raise QualityGateError(f"{label}: nesne olmali")
    return value


def _closed(value, expected, label):
    actual = set(value)
    if actual != set(expected):
        raise QualityGateError(
            f"{label}: alanlar kapali olmali; eksik={sorted(set(expected)-actual)}, "
            f"fazla={sorted(actual-set(expected))}"
        )


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise QualityGateError(f"{label}: en az {minimum} olan tamsayi olmali")
    return value


def _metric(value, label):
    if type(value) not in (int, float):
        raise QualityGateError(f"{label}: sayi olmali")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise QualityGateError(f"{label}: 0 ile 1 arasinda olmali")
    return number


def load_policy(path):
    """Read the closed quality policy; unknown fields are an error."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityGateError("kalite politikasi okunamadi") from exc
    policy = _object(raw, "policy")
    _closed(policy, ("version", "sets"), "policy")
    if policy["version"] != POLICY_VERSION:
        raise QualityGateError("policy.version desteklenmiyor")
    sets = _object(policy["sets"], "policy.sets")
    if not sets:
        raise QualityGateError("policy.sets bos olamaz")
    parsed = {}
    for name, entry_raw in sets.items():
        if type(name) is not str or not name or Path(name).name != name:
            raise QualityGateError("set adi kapali bir dosya parcasi olmali")
        entry = _object(entry_raw, f"policy.sets.{name}")
        _closed(entry, ("n", "retrieval_profile", "retrieval_minimum",
                        "answer_minimum", "answer_maximum"),
                f"policy.sets.{name}")
        n = _integer(entry["n"], f"policy.sets.{name}.n")
        profile = entry["retrieval_profile"]
        if type(profile) is not str or not profile:
            raise QualityGateError(f"policy.sets.{name}.retrieval_profile gecersiz")
        retrieval = _object(entry["retrieval_minimum"],
                            f"policy.sets.{name}.retrieval_minimum")
        answer_min = _object(entry["answer_minimum"],
                             f"policy.sets.{name}.answer_minimum")
        answer_max = _object(entry["answer_maximum"],
                             f"policy.sets.{name}.answer_maximum")
        _closed(retrieval, RETRIEVAL_METRICS,
                f"policy.sets.{name}.retrieval_minimum")
        _closed(answer_min, ANSWER_MINIMUM_METRICS,
                f"policy.sets.{name}.answer_minimum")
        _closed(answer_max, ANSWER_MAXIMUM_METRICS,
                f"policy.sets.{name}.answer_maximum")
        parsed[name] = MappingProxyType({
            "n": n,
            "retrieval_profile": profile,
            "retrieval_minimum": MappingProxyType({
                key: _metric(retrieval[key], f"{name}.retrieval.{key}")
                for key in RETRIEVAL_METRICS
            }),
            "answer_minimum": MappingProxyType({
                key: _metric(answer_min[key], f"{name}.answer.{key}")
                for key in ANSWER_MINIMUM_METRICS
            }),
            "answer_maximum": MappingProxyType({
                key: _metric(answer_max[key], f"{name}.answer.{key}")
                for key in ANSWER_MAXIMUM_METRICS
            }),
        })
    return MappingProxyType(parsed)


def _load_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualityGateError(f"{label} okunamadi") from exc


def _retrieval_metrics(run_dir, name, policy):
    report = _object(
        _load_json(Path(run_dir) / f"rag_eval_{name}.json", "retrieval raporu"),
        "retrieval raporu",
    )
    profile = policy["retrieval_profile"]
    if set(report) != {profile}:
        raise QualityGateError(f"{name}: retrieval profili policy ile eslesmiyor")
    row = _object(report[profile], f"{name}.retrieval")
    if _integer(row.get("n"), f"{name}.retrieval.n") != policy["n"]:
        raise QualityGateError(f"{name}: retrieval ornek sayisi degisti")
    return {key: _metric(row.get(key), f"{name}.retrieval.{key}")
            for key in RETRIEVAL_METRICS}


def _answer_metrics(run_dir, question_dir, name, policy):
    questions = _load_json(Path(question_dir) / f"{name}.json", "soru seti")
    if type(questions) is not list:
        raise QualityGateError(f"{name}: soru seti liste olmali")
    by_question = {}
    for row in questions:
        row = _object(row, f"{name}.soru")
        question = row.get("q")
        if type(question) is not str or not question or question in by_question:
            raise QualityGateError(f"{name}: soru kimlikleri tekil metin olmali")
        by_question[question] = row
    if len(by_question) != policy["n"]:
        raise QualityGateError(f"{name}: policy ile soru sayisi eslesmiyor")

    saved = _object(
        _load_json(Path(run_dir) / f"rag_answers_{name}.json", "cevap raporu"),
        f"{name}.cevap_raporu",
    )
    rows = saved.get("sorular")
    if type(rows) is not list:
        raise QualityGateError(f"{name}: sorular listesi yok")
    saved_by_question = {}
    for row in rows:
        row = _object(row, f"{name}.cevap")
        question = row.get("soru")
        if (type(question) is not str or question not in by_question
                or question in saved_by_question):
            raise QualityGateError(f"{name}: cevap soru baglamasi gecersiz")
        if type(row.get("cevap")) is not str or type(row.get("baglam")) is not str:
            raise QualityGateError(f"{name}: cevap ve baglam metin olmali")
        saved_by_question[question] = row
    if set(saved_by_question) != set(by_question):
        raise QualityGateError(f"{name}: cevaplar soru setini tam kapsamiyor")

    # Passing the exact threshold makes every non-deterministic match REVIEW,
    # never WRONG or CORRECT.  No embedding service is imported or called.
    scored = [
        score_one(
            by_question[question],
            saved_by_question[question]["cevap"],
            saved_by_question[question]["baglam"],
            sim=SIM_THRESHOLD,
        )
        for question in sorted(by_question)
    ]
    summary = summarize(scored)
    if summary.get("n") != policy["n"]:
        raise QualityGateError(f"{name}: yeniden puanlama sayisi degisti")
    keys = ANSWER_MINIMUM_METRICS + ANSWER_MAXIMUM_METRICS
    return {key: _metric(summary.get(key), f"{name}.answer.{key}")
            for key in keys}


def evaluate_quality(run_dir, question_dir, policy_path):
    """Evaluate every required set and return a content-free closed report."""
    policy = load_policy(policy_path)
    sets_report, failures = {}, []
    epsilon = 0.00005
    for name in sorted(policy):
        entry = policy[name]
        retrieval = _retrieval_metrics(run_dir, name, entry)
        answer = _answer_metrics(run_dir, question_dir, name, entry)
        checks = []
        for metric in RETRIEVAL_METRICS:
            threshold = entry["retrieval_minimum"][metric]
            passed = retrieval[metric] + epsilon >= threshold
            checks.append({"family": "retrieval", "metric": metric,
                           "actual": retrieval[metric], "threshold": threshold,
                           "operator": "minimum", "passed": passed})
        for metric in ANSWER_MINIMUM_METRICS:
            threshold = entry["answer_minimum"][metric]
            passed = answer[metric] + epsilon >= threshold
            checks.append({"family": "answer", "metric": metric,
                           "actual": answer[metric], "threshold": threshold,
                           "operator": "minimum", "passed": passed})
        for metric in ANSWER_MAXIMUM_METRICS:
            threshold = entry["answer_maximum"][metric]
            passed = answer[metric] <= threshold + epsilon
            checks.append({"family": "answer", "metric": metric,
                           "actual": answer[metric], "threshold": threshold,
                           "operator": "maximum", "passed": passed})
        failed = [check for check in checks if not check["passed"]]
        failures.extend({"set": name, "family": check["family"],
                         "metric": check["metric"]} for check in failed)
        sets_report[name] = {
            "n": entry["n"], "passed": not failed, "checks": checks,
        }
    report = {
        "quality_gate_version": POLICY_VERSION,
        "passed": not failures,
        "sets": sets_report,
        "failures": failures,
    }
    return QualityGateResult(report["passed"], MappingProxyType(report))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--question-dir", default="data/rag_eval")
    parser.add_argument("--policy", default="eval/quality_policy.json")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        result = evaluate_quality(args.run_dir, args.question_dir, args.policy)
    except QualityGateError as exc:
        print(json.dumps({"quality_gate_version": POLICY_VERSION,
                          "passed": False, "error": type(exc).__name__},
                         sort_keys=True))
        return 2
    text = json.dumps(dict(result.report), sort_keys=True, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
