"""Replay one guard policy consistently across saved answer runs.

Only aggregate counts are printed. Questions, answers, quotations, figures and
page values remain in ignored run files.

Saved runs can span a code edit when a batch launches one subprocess per set.
Their recorded guard statuses therefore cannot safely be compared as if they
came from one policy. This module re-scores the saved answer and re-runs the
selected guard against the exact saved context, without another model call.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from eval.answer.guard_floor import legacy_context
from eval.answer.judge import DOGRU, INCELE, YANLIS
from eval.answer.rag_answer_eval import score_one
from eval.retrieval.rag_eval import QUESTION_DIR
from pipeline.validation.rag.answer_guard import (
    check,
    check_structured,
    cited_pages,
    is_abstention,
)
from pipeline.validation.rag.binding_guard import check_binding

PLAIN = "plain"
STRUCTURED_DERIVED = "structured_derived"
STRUCTURED_EXPLICIT = "structured_explicit_page"
# Measurement-only variants: the base policy's flags UNIONED with the
# wrong-binding check. They price "what would wiring the binding validator in
# cost and catch" on any saved run, without touching the publication path.
PLAIN_BINDING = "plain_binding"
STRUCTURED_DERIVED_BINDING = "structured_derived_binding"
POLICIES = (PLAIN, STRUCTURED_DERIVED, STRUCTURED_EXPLICIT,
            PLAIN_BINDING, STRUCTURED_DERIVED_BINDING)
_SWEEP_SUFFIX = re.compile(r"_k\d+$")
_MISSING_PAGE = "eksik_sayfa"


def _reply(row):
    raw = row.get("ham_yanit")
    if isinstance(raw, str) and raw:
        return raw
    return {
        "dayanak": row.get("dayanak") or [],
        "cevap": row.get("cevap") or "",
    }


def _binding_flags(row, context):
    claims = row.get("dayanak") or []
    handles = [c.get("pasaj") for c in claims
               if isinstance(c, dict) and isinstance(c.get("pasaj"), int)]
    return list(check_binding(row.get("soru") or "", row.get("cevap") or "",
                              context, handles or None))


def replay_flags(row, context, policy):
    """Flags under one policy, independent of what the run saved."""
    if policy == PLAIN:
        return list(check(row.get("cevap") or "", context))
    if policy == PLAIN_BINDING:
        return (list(check(row.get("cevap") or "", context))
                + _binding_flags(row, context))
    if policy == STRUCTURED_DERIVED_BINDING:
        return (list(check_structured(_reply(row), context))
                + _binding_flags(row, context))
    if policy not in {STRUCTURED_DERIVED, STRUCTURED_EXPLICIT}:
        raise ValueError(f"unknown guard policy: {policy}")

    flags = list(check_structured(_reply(row), context))
    answer = row.get("cevap") or ""
    if (
        policy == STRUCTURED_EXPLICIT
        and not is_abstention(answer)
        and not cited_pages(answer)
        and not any(code == _MISSING_PAGE for code, _ in flags)
    ):
        flags.append((_MISSING_PAGE, []))
    return flags


def _new_counter():
    return Counter({
        "n": 0,
        DOGRU: 0,
        INCELE: 0,
        YANLIS: 0,
        "explicit_page": 0,
        "page_correct": 0,
        "abstained": 0,
        "guard_review": 0,
        "correct_withheld": 0,
        "noncorrect_caught": 0,
        "settled_wrong_caught": 0,
        "published_answer": 0,
        "noncorrect_published": 0,
        "settled_wrong_published": 0,
    })


def _record(summary, flag_tally, scored, answer, flags):
    verdict = scored["durum"]
    codes = {code for code, _ in flags}
    abstained = is_abstention(answer)

    summary["n"] += 1
    summary[verdict] += 1
    summary["explicit_page"] += bool(cited_pages(answer))
    summary["page_correct"] += bool(scored["sayfa_dogru"])
    summary["abstained"] += abstained

    if codes:
        summary["guard_review"] += 1
        if verdict == DOGRU:
            summary["correct_withheld"] += 1
        else:
            summary["noncorrect_caught"] += 1
        if verdict == YANLIS:
            summary["settled_wrong_caught"] += 1
    elif not abstained:
        summary["published_answer"] += 1
        if verdict != DOGRU:
            summary["noncorrect_published"] += 1
        if verdict == YANLIS:
            summary["settled_wrong_published"] += 1

    for code in codes:
        flag_tally[(code, "raised")] += 1
        flag_tally[(code, verdict)] += 1
        if len(codes) == 1:
            flag_tally[(code, "exclusive")] += 1
            flag_tally[(code, f"exclusive_{verdict}")] += 1


def _flag_report(tally):
    fields = (
        "raised",
        DOGRU,
        INCELE,
        YANLIS,
        "exclusive",
        f"exclusive_{DOGRU}",
        f"exclusive_{INCELE}",
        f"exclusive_{YANLIS}",
    )
    codes = sorted({code for code, _ in tally})
    return {
        code: {field: tally[(code, field)] for field in fields}
        for code in codes
    }


def _summary_report(counter):
    report = dict(counter)
    explicit = counter["explicit_page"]
    report["conditional_page_accuracy"] = (
        round(counter["page_correct"] / explicit, 4)
        if explicit else None
    )
    return report


def measure_run(run_dir, policy):
    """Per-set and aggregate cost/benefit ledger for one saved run."""
    if policy not in POLICIES:
        raise ValueError(f"unknown guard policy: {policy}")
    run_dir = Path(run_dir)
    total = _new_counter()
    total_flags = Counter()
    per_set = {}
    seconds = 0.0
    result_files = 0

    for path in sorted(run_dir.glob("rag_answers_*.json")):
        set_name = path.stem.removeprefix("rag_answers_")
        if _SWEEP_SUFFIX.search(set_name):
            continue
        question_path = QUESTION_DIR / f"{set_name}.json"
        if not question_path.exists():
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = {
            item["q"]: item
            for item in json.loads(question_path.read_text(encoding="utf-8"))
        }
        set_summary = _new_counter()
        set_flags = Counter()

        for row in payload.get("sorular") or []:
            question = questions.get(row.get("soru"))
            if question is None:
                continue
            answer = row.get("cevap") or ""
            context_text = row.get("baglam") or ""
            scored = score_one(question, answer, context_text)
            context = legacy_context(context_text)
            flags = replay_flags(row, context, policy)
            _record(set_summary, set_flags, scored, answer, flags)
            _record(total, total_flags, scored, answer, flags)

        seconds += float(payload.get("ozet", {}).get("saniye") or 0)
        result_files += 1
        per_set[set_name] = {
            "summary": _summary_report(set_summary),
            "flags": _flag_report(set_flags),
        }

    return {
        "policy": policy,
        "result_files": result_files,
        "recorded_seconds": round(seconds, 1),
        "total": {
            "summary": _summary_report(total),
            "flags": _flag_report(total_flags),
        },
        "sets": dict(sorted(per_set.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    args = parser.parse_args()
    print(json.dumps(
        measure_run(args.run_dir, args.policy),
        ensure_ascii=True,
        indent=2,
    ))


if __name__ == "__main__":
    main()
