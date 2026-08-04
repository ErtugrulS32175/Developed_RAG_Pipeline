"""Measure page-citation relaxations on saved structured-answer runs.

The report contains counts only. Saved questions, answers, quotations, figures
and page values are private document content and are never printed.

This is an eval-only compatibility path. Current production validation receives
the original ``RagContext`` built from retrieval records. Historical run files
store only the exact rendered context shown to the model, so ``legacy_context``
rebuilds the closest available snapshot rather than retrieving a different
context after the fact.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from eval.answer.guard_floor import legacy_context
from eval.answer.judge import DOGRU, INCELE, YANLIS
from pipeline.lang.tr_notation import normalize, number_forms
from pipeline.validation.rag.answer_guard import (
    check_structured,
    cited_pages,
    is_abstention,
    unsupported_figures,
)

_BLOCK = "\n\n---\n\n"
_MISSING_PAGE = "eksik_sayfa"
_SWEEP_SUFFIX = re.compile(r"_k\d+$")

ONE_PASSAGE = "one_passage"
ONE_PAGE = "one_passage_or_same_page"
FIGURE_PAGE = "plus_unique_figure_page"
POLICIES = (ONE_PASSAGE, ONE_PAGE, FIGURE_PAGE)


def _reply(row):
    raw = row.get("ham_yanit")
    if isinstance(raw, str) and raw:
        return raw
    return {
        "dayanak": row.get("dayanak") or [],
        "cevap": row.get("cevap") or "",
    }


def _other_flags(row, context):
    """Replay the current real checks; saved flags may span code versions."""
    return [
        item
        for item in check_structured(_reply(row), context)
        if item[0] != _MISSING_PAGE
    ]


def _missing_explicit_page(row):
    answer = row.get("cevap") or ""
    return not is_abstention(answer) and not cited_pages(answer)


def _evidence_passages(row, context):
    known = context.by_handle()
    handles = []
    for item in row.get("dayanak") or []:
        if not isinstance(item, dict) or type(item.get("pasaj")) is not int:
            return None, "malformed_evidence"
        handles.append(item["pasaj"])
    handles = list(dict.fromkeys(handles))
    if not handles:
        return None, "no_evidence"
    if any(handle not in known for handle in handles):
        return None, "unknown_handle"
    passages = [known[handle] for handle in handles]
    if any(passage.page is None for passage in passages):
        return None, "missing_page_metadata"
    return passages, None


def _has_figures(answer):
    return any(
        forms
        for forms in number_forms(normalize(answer or ""))
    )


def candidate_page(row, context, policy):
    """Return a page only when the selected candidate rule is unambiguous."""
    if policy not in POLICIES:
        raise ValueError(f"unknown page policy: {policy}")
    passages, problem = _evidence_passages(row, context)
    if problem:
        return None, problem
    if len(passages) == 1:
        return passages[0].page, ONE_PASSAGE

    pages = {passage.page for passage in passages}
    if len(pages) == 1:
        if policy in {ONE_PAGE, FIGURE_PAGE}:
            return next(iter(pages)), "same_page"
        return None, "same_page_not_enabled"

    if policy == FIGURE_PAGE and len(pages) > 1:
        if not _has_figures(row.get("cevap")):
            return None, "multi_page_without_figures"
        supported_pages = []
        for page in pages:
            scope = _BLOCK.join(
                passage.text
                for passage in passages
                if passage.page == page
            )
            if not unsupported_figures(row.get("cevap"), scope):
                supported_pages.append(page)
        if len(supported_pages) == 1:
            return supported_pages[0], "unique_figure_page"
        return None, "multi_page_no_unique_figure_page"

    return None, "multi_page"


def measure_rows(rows):
    """Counter-metrics for each cumulative candidate policy."""
    missing = [row for row in rows if _missing_explicit_page(row)]
    contexts = [
        (row, legacy_context(row.get("baglam") or ""))
        for row in missing
    ]

    shape = Counter()
    for row, context in contexts:
        passages, problem = _evidence_passages(row, context)
        if problem:
            shape[problem] += 1
            continue
        pages = {passage.page for passage in passages}
        if len(passages) == 1:
            shape[ONE_PASSAGE] += 1
        elif len(pages) == 1:
            shape["same_page"] += 1
        else:
            shape["multiple_pages"] += 1

    policies = {}
    for policy in POLICIES:
        result = Counter()
        reasons = Counter()
        for row, context in contexts:
            page, reason = candidate_page(row, context, policy)
            reasons[reason] += 1
            remaining = list(_other_flags(row, context))
            if page is None:
                remaining.append((_MISSING_PAGE, []))
                result["unresolved_missing_page"] += 1
            else:
                result["derived_cases"] += 1

            if remaining:
                result["still_review_required"] += 1
            elif row.get("durum") == DOGRU or (
                row.get("durum") is None
                and row.get("cevap_dogru") is True
            ):
                result["correct_answers_rescued"] += 1
            else:
                result["noncorrect_answers_newly_publishable"] += 1
                if row.get("durum") == INCELE:
                    result["review_answers_newly_publishable"] += 1
                elif row.get("durum") == YANLIS:
                    result["settled_wrong_answers_newly_publishable"] += 1

        policies[policy] = {
            "derived_cases": result["derived_cases"],
            "correct_answers_rescued": result["correct_answers_rescued"],
            "noncorrect_answers_newly_publishable":
                result["noncorrect_answers_newly_publishable"],
            "review_answers_newly_publishable":
                result["review_answers_newly_publishable"],
            "settled_wrong_answers_newly_publishable":
                result["settled_wrong_answers_newly_publishable"],
            "unresolved_missing_page": result["unresolved_missing_page"],
            "still_review_required": result["still_review_required"],
            "reasons": dict(sorted(reasons.items())),
        }

    return {
        "rows": len(rows),
        "missing_page_cases": len(missing),
        "evidence_shapes": dict(sorted(shape.items())),
        "policies": policies,
        "genuinely_ambiguous_after_all_candidates":
            sum(
                count
                for reason, count in policies[FIGURE_PAGE]["reasons"].items()
                if reason.startswith("multi_page")
            ),
    }


def saved_sets(run_dirs):
    """Distinct non-sweep result rows grouped by question set."""
    sets = {}
    files = 0
    for run_dir in run_dirs:
        for path in sorted(Path(run_dir).glob("rag_answers_*.json")):
            set_name = path.stem.removeprefix("rag_answers_")
            if _SWEEP_SUFFIX.search(set_name):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            sets.setdefault(set_name, []).extend(payload.get("sorular") or [])
            files += 1
    return files, dict(sorted(sets.items()))


def saved_rows(run_dirs):
    """Distinct non-sweep result rows from the requested directories."""
    files, sets = saved_sets(run_dirs)
    return files, [row for rows in sets.values() for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    files, sets = saved_sets(args.run_dirs)
    rows = [row for set_rows in sets.values() for row in set_rows]
    report = {
        "result_files": files,
        **measure_rows(rows),
        "sets": {
            name: measure_rows(set_rows)
            for name, set_rows in sets.items()
        },
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
