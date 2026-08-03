"""How often would the answer checks cry wolf?

    python -m eval.answer.guard_eval output/RAG_Outputs/run1/native

A guard that flags answers nobody would have questioned is worse than no
guard: people stop reading the warnings, and then the one real warning is
invisible too. So before any check is wired into the product, it gets measured
against answers whose correctness is already settled -- the false-flag rate is
the number that decides whether a check can ever become a block.

Run on saved answers, so it costs no GPU time. Counts only: the figures a check
objects to are document content and do not belong in a report.
"""
import argparse
import json
from pathlib import Path

from eval.answer.guard_floor import legacy_context
from eval.answer.judge import DOGRU, INCELE, YANLIS
from eval.answer.rag_answer_eval import score_one
from eval.retrieval.rag_eval import QUESTION_DIR
from pipeline.validation.rag.answer_guard import check


def rows(run_dir):
    for f in sorted(Path(run_dir).glob("rag_answers_*.json")):
        name = f.stem.replace("rag_answers_", "")
        if "_k" in name:
            continue
        qfile = QUESTION_DIR / f"{name}.json"
        if not qfile.exists():
            continue
        gt = {g["q"]: g for g in json.loads(qfile.read_text(encoding="utf-8"))}
        for r in json.loads(f.read_text(encoding="utf-8"))["sorular"]:
            g = gt.get(r["soru"])
            if g:
                yield name, g, r["cevap"], r.get("baglam") or ""


def measure(run_dirs, minimum, derive):
    """Per verdict: how many answers the checks flag, and on what grounds."""
    tally = {}
    for run_dir in run_dirs:
        for _, g, answer, context in rows(run_dir):
            durum = score_one(g, answer, context)["durum"]
            flags = check(
                answer,
                legacy_context(context),
                minimum,
                derive,
            )
            seen = {name for name, _ in flags}
            t = tally.setdefault(durum, {"n": 0, "flagged": 0,
                                         "kaynaksiz_sayi": 0, "kaynaksiz_sayfa": 0})
            t["n"] += 1
            t["flagged"] += bool(seen)
            for name in seen:
                t[name] += 1
    return tally


def report(tally, minimum, derive):
    print(f"\n  en kucuk sayi esigi: {minimum}   "
          f"oran turetme: {'acik' if derive else 'kapali'}")
    print(f"  {'karar':<10}{'n':>5}{'isaretlenen':>13}{'oran':>9}"
          f"{'sayi':>7}{'sayfa':>7}")
    print("  " + "-" * 51)
    for durum in (DOGRU, INCELE, YANLIS):
        t = tally.get(durum)
        if not t:
            continue
        rate = t["flagged"] / t["n"]
        print(f"  {durum:<10}{t['n']:>5}{t['flagged']:>13}{rate:>9.1%}"
              f"{t['kaynaksiz_sayi']:>7}{t['kaynaksiz_sayfa']:>7}")
    d = tally.get(DOGRU)
    if d:
        # the only number that matters yet: of the answers already settled as
        # correct, how many would a user have been warned about for nothing
        print(f"\n  YANLIS ALARM ORANI: {d['flagged']}/{d['n']} = "
              f"{d['flagged'] / d['n']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--minimum", type=float, nargs="+", default=[0],
                    help="kucuk sayilari yoksay; birden fazla verilirse taranir")
    args = ap.parse_args()

    for derive in (False, True):
        for minimum in args.minimum:
            report(measure(args.run_dirs, minimum, derive), minimum, derive)


if __name__ == "__main__":
    main()
