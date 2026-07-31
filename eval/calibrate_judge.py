"""Measure the judge before trusting it, and pick its threshold from data.

    python -m eval.calibrate_judge output/RAG_Outputs/run1/native \
                                   output/RAG_Outputs/run1/llamaindex

Three things get measured, because a scorer that is only checked against the
cases it was written for proves nothing:

  1. **Adjudicated cases.** The answers a human has ruled on. Tiers 1-2 must
     accept every one ruled correct and none ruled wrong.
  2. **Mismatched pairs.** Every expected answer against a DIFFERENT question's
     answer. These are wrong by construction and cost nothing to make, so they
     are the only large sample of true negatives available. Any acceptance here
     is a false one, and that rate is what says whether tier 2's looseness --
     stems, number words, excused connectives -- is safe.
  3. **A similarity sweep.** For each candidate threshold, how many adjudicated
     cases land on the wrong side, and how many mismatched pairs survive as
     review. The threshold to choose is the highest one that sends NO case a
     human called correct-or-arguable down to "wrong": being sent to review
     costs a minute of someone's attention, being called wrong corrupts the
     number the whole project is judged on.

Needs the embedding service for part 3 only; parts 1 and 2 run offline.
"""
import argparse
import json
import random
from pathlib import Path

from eval.judge import DOGRU, INCELE, YANLIS, notation_match, similarity
from eval.rag_eval import QUESTION_DIR, contains_key

ADJUDICATED = QUESTION_DIR / "adjudicated.json"

# How much of the clearly-unrelated has to be rejected for the semantic tier to
# be earning its place. Below this it is mostly forwarding work to a human, and
# a review queue nobody can finish is the same as no review at all.
MIN_FILTERED = 0.95


def tiers_accept(answer, key):
    return bool(contains_key(answer, key)) or notation_match(answer, key)


def load_answers(run_dirs):
    """(key, reference, answer) for every saved answer, across runs."""
    out = []
    for run_dir in run_dirs:
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
                if g and g.get("key"):
                    out.append((g["key"], g.get("answer", ""), r["cevap"]))
    return out


def check_adjudicated(cases):
    print("1) INSAN KARARLI VAKALAR")
    wrong = []
    for c in cases:
        accepted = tiers_accept(c["answer"], c["key"])
        expected = c["karar"] == DOGRU
        if accepted != expected:
            wrong.append(c)
            print(f"   HATA {c['id']:3} bekl={c['karar']:7} kabul={accepted} [{c['sinif']}]")
    print(f"   {len(cases) - len(wrong)}/{len(cases)} dogru siniflandi\n")
    return wrong


def check_mismatched(pairs, sample, seed):
    """False-accept rate of tiers 1-2 on pairs that cannot be correct."""
    print("2) ESLESMEYEN CIFTLER (kurgu geregi yanlis)")
    rng = random.Random(seed)
    tried = accepted = 0
    examples = []
    for _ in range(sample):
        (key, _, _), (_, _, answer) = rng.sample(pairs, 2)
        tried += 1
        if tiers_accept(answer, key):
            accepted += 1
            if len(examples) < 5:
                examples.append(key)
    rate = accepted / tried if tried else 0.0
    print(f"   {accepted}/{tried} yanlis kabul  ({rate:.3%})")
    # the LENGTH of what was falsely accepted, never the text: a short expected
    # answer has fewer words to disagree on, so a false accept is far likelier
    # there, and seeing them cluster at two or three words says the rules are
    # fine and the question set needs a more distinctive key
    if examples:
        print(f"     yanlis kabul edilen anahtarlarin uzunlugu (kelime): "
              f"{', '.join(str(len(k.split())) for k in examples)}")
    print()
    return rate


def sweep(cases, pairs, sample, seed):
    print("3) BENZERLIK ESIGI")
    unresolved = [c for c in cases if not tiers_accept(c["answer"], c["key"])]
    scored = []
    for c in unresolved:
        s = similarity(c["answer"], c["reference"])
        if s is None:
            print("   gomu servisi kapali -- esik taranamadi "
                  "(kapali kaldiginda hepsi 'incele' olur, guvenli taraf)\n")
            return None
        scored.append((c, s))
        print(f"   {c['id']:3} {c['karar']:7} benzerlik={s:.3f}  [{c['sinif']}]")

    rng = random.Random(seed)
    negatives = []
    for _ in range(sample):
        (_, reference, _), (_, _, answer) = rng.sample(pairs, 2)
        if reference:
            s = similarity(answer, reference)
            if s is not None:
                negatives.append(s)

    print(f"\n   {'esik':>6}{'yanlis diyip hata':>20}{'incelemede kalan':>19}"
          f"{'alakasiz eleme':>17}{'marj':>8}")
    best, best_margin = None, -1.0
    for threshold in [x / 100 for x in range(30, 96, 5)]:
        misjudged = sum(1 for c, s in scored if c["karar"] != YANLIS and s < threshold)
        in_review = sum(1 for c, s in scored if s >= threshold)
        filtered = sum(1 for s in negatives if s < threshold) / len(negatives) \
            if negatives else 0.0
        # distance to the nearest case the threshold has to separate: a cut that
        # only just clears an anchor is one rephrasing away from misjudging it
        margin = min(abs(s - threshold) for _, s in scored) if scored else 0.0
        print(f"   {threshold:>6.2f}{misjudged:>20}{in_review:>19}"
              f"{filtered:>16.1%}{margin:>8.3f}")
        if misjudged == 0 and filtered >= MIN_FILTERED and margin > best_margin:
            best, best_margin = threshold, margin
    # Two requirements, and neither alone picks a sane cut. Margin alone chose a
    # threshold so low that most unrelated answers went to review instead of
    # being rejected -- the tier stops doing its job. Filtering alone chose one
    # sitting six thousandths from a case it must not misjudge. So: reject at
    # least MIN_FILTERED of what is unrelated, and among the settings that do,
    # stand as far as possible from any real case.
    print(f"\n   secilen esik: {best}  (alakasizin >={MIN_FILTERED:.0%}'ini eleyenler "
          f"arasinda marji en genis olan; en yakin vakaya {best_margin:.3f})\n")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sweep", action="store_true",
                    help="gomu servisi gerektiren 3. adimi atla")
    args = ap.parse_args()

    cases = json.loads(ADJUDICATED.read_text(encoding="utf-8"))
    pairs = load_answers(args.run_dirs)
    print(f"{len(cases)} insan kararli vaka, {len(pairs)} kayitli cevap\n")

    check_adjudicated(cases)
    check_mismatched(pairs, args.sample, args.seed)
    if not args.skip_sweep:
        sweep(cases, pairs, min(args.sample, 120), args.seed)


if __name__ == "__main__":
    main()
