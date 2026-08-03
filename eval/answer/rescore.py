"""Re-score a saved answer run with the CURRENT scorer, and queue what still fails.

    python -m eval.answer.rescore output/RAG_Outputs/run1/native

Answers cost GPU time; scoring does not. Keeping the raw answers means a fix to
the scorer can be applied to every past run for free, which is the only way to
find out whether a reported failure was the system's or the metric's. It has
already paid for itself: a run once reported 0.588 on one set and re-scored to
0.882 after two parsing defects were fixed -- the pipeline had been right and
the measurement wrong, and the difference was five correct answers.

What still fails is written to a review file rather than printed, because
judging it needs the question, the expected answer and the model's answer side
by side, and those are document content.
"""
import argparse
import json
from pathlib import Path

from eval.answer.judge import INCELE
from eval.answer.rag_answer_eval import score_one, summarize
from eval.retrieval.rag_eval import QUESTION_DIR


def rescore(run_dir, sets=None):
    run_dir = Path(run_dir)
    rows_by_set, pending = {}, []
    files = sorted(run_dir.glob("rag_answers_*.json"))
    for f in files:
        name = f.stem.replace("rag_answers_", "")
        if "_k" in name:                      # settings-tagged archive copy
            continue
        if sets and name not in sets:
            continue
        qfile = QUESTION_DIR / f"{name}.json"
        if not qfile.exists():
            continue
        saved = json.loads(f.read_text(encoding="utf-8"))
        gt = {g["q"]: g for g in json.loads(qfile.read_text(encoding="utf-8"))}

        scored, before = [], saved["ozet"].get("cevap_dogrulugu")
        for r in saved["sorular"]:
            g = gt.get(r["soru"])
            if not g:
                continue
            row = score_one(g, r["cevap"], r.get("baglam") or "")
            scored.append(row)
            # only what the scorer could not settle goes to a person; an answer
            # it ruled wrong has already been judged, and queueing those too is
            # what made the review list mostly noise
            if row["durum"] == INCELE:
                pending.append((name, g, r["cevap"], row["gerekce"]))
        rows_by_set[name] = (before, summarize(scored))
    return rows_by_set, pending


def write_review(pending, path):
    """The unsettled answers, for a human to judge. A metric cannot decide
    whether a differently-phrased answer is right; only a person can, and the
    verdicts belong back in the question set."""
    lines = ["# Inceleme kuyrugu", "",
             f"{len(pending)} cevap karara baglanamadi. Her biri icin: "
             "gercekten yanlis mi, yoksa ayni seyi baska turlu mu soyluyor?", ""]
    for i, (setname, g, answer, why) in enumerate(pending, 1):
        lines += [f"## {i}. [{setname}] {g['q']}", "",
                  f"- **beklenen** : `{g.get('key')}`",
                  f"- **referans** : {g.get('answer')}",
                  f"- **model**    : {answer}",
                  f"- **gerekce**  : {why}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--review", default=None,
                    help="inceleme dosyasi; verilmezse <run_dir>/review.md")
    args = ap.parse_args()

    by_set, pending = rescore(args.run_dir, args.sets)
    if not by_set:
        raise SystemExit(f"kayitli cevap bulunamadi: {args.run_dir}")

    print(f"{'set':<12}{'n':>4}{'onceki':>9}{'alt':>7}{'ust':>7}{'incele':>8}   hata")
    print("-" * 66)
    lower = upper = count = 0
    for name, (before, m) in sorted(by_set.items()):
        lower += m["cevap_dogrulugu"] * m["n"]
        upper += m["ust_sinir"] * m["n"]
        count += m["n"]
        was = f"{before:.3f}" if before is not None else "  -  "
        print(f"{name:<12}{m['n']:>4}{was:>9}{m['cevap_dogrulugu']:>7.3f}"
              f"{m['ust_sinir']:>7.3f}{m['incele_orani']:>8.3f}   "
              f"{m.get('hata_dagilimi') or ''}")
    print("-" * 66)
    print(f"{'AGIRLIKLI':<12}{count:>4}{'':>9}{lower / count:>7.3f}{upper / count:>7.3f}")

    out = Path(args.review) if args.review else Path(args.run_dir) / "review.md"
    write_review(pending, out)
    print(f"\ninceleme kuyrugu ({len(pending)} cevap): {out}")


if __name__ == "__main__":
    main()
