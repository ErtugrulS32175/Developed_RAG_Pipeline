"""Measure the judge before trusting it, and pick its threshold from data.

    python -m eval.answer.calibrate_judge output/RAG_Outputs/run1/native \
                                          output/RAG_Outputs/run1/llamaindex

Three things get measured, because a scorer that is only checked against the
cases it was written for proves nothing:

  1. **Adjudicated cases.** The answers a human has ruled on. Tiers 1-2 must
     accept every one ruled correct and none ruled wrong.
  2. **Mismatched pairs.** Every expected answer against a DIFFERENT question's
     answer. These are free proxy negatives, not guaranteed true negatives: a
     short or generic key can occur in both questions by coincidence. Their
     acceptance rate is still a useful regression signal, especially when the
     exact sampled pairs are compared before and after a matcher change.
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

from eval.answer.judge import (
    DOGRU, INCELE, YANLIS, accepts_without_similarity, similarity)
from eval.retrieval.rag_eval import QUESTION_DIR, contains_key

ADJUDICATED = QUESTION_DIR / "adjudicated.json"

# How much of the clearly-unrelated has to be rejected for the semantic tier to
# be earning its place. Below this it is mostly forwarding work to a human, and
# a review queue nobody can finish is the same as no review at all.
MIN_FILTERED = 0.95


def tiers_accept(answer, key):
    return accepts_without_similarity(key, answer)


def selective_stats(expected, predicted):
    """Coverage and error rate among decisions not sent to review."""
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    total = len(expected)
    decided = [
        (wanted, got)
        for wanted, got in zip(expected, predicted)
        if got != INCELE
    ]
    errors = sum(wanted != got for wanted, got in decided)
    false_accepts = sum(
        wanted != DOGRU and got == DOGRU for wanted, got in decided
    )
    false_rejects = sum(
        wanted != YANLIS and got == YANLIS for wanted, got in decided
    )
    return {
        "n": total,
        "decided": len(decided),
        "review": total - len(decided),
        "coverage": len(decided) / total if total else 0.0,
        "selective_risk": errors / len(decided) if decided else 0.0,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


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
    stats = selective_stats(
        [case["karar"] for case in cases],
        [DOGRU if tiers_accept(case["answer"], case["key"]) else INCELE
         for case in cases],
    )
    print("   deterministik kapsam "
          f"{stats['decided']}/{stats['n']} = {stats['coverage']:.1%}; "
          f"selective risk={stats['selective_risk']:.1%}; "
          f"yanlis kabul={stats['false_accepts']}; "
          f"yanlis ret={stats['false_rejects']}\n")
    return wrong


def check_mismatched(pairs, sample, seed):
    """Acceptance regression on different-question proxy negatives."""
    print("2) ESLESMEYEN CIFTLER (proxy negatif)")
    rng = random.Random(seed)
    tried = accepted = introduced = retired = 0
    examples = []
    for _ in range(sample):
        (key, _, _), (_, _, answer) = rng.sample(pairs, 2)
        tried += 1
        current = tiers_accept(answer, key)
        legacy = bool(contains_key(answer, key))
        introduced += current and not legacy
        retired += legacy and not current
        if current:
            accepted += 1
            if len(examples) < 5:
                examples.append(key)
    rate = accepted / tried if tried else 0.0
    print(f"   {accepted}/{tried} yanlis kabul  ({rate:.3%})")
    print(f"   legacy ustune yeni kabul: {introduced}; "
          f"artik kabul edilmeyen legacy: {retired}")
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

    print(f"\n   {'esik':>6}{'yanlis diyip hata':>20}{'kapsam':>10}"
          f"{'risk':>10}{'alakasiz eleme':>17}{'marj':>8}")
    best, best_margin = None, -1.0
    for threshold in [x / 100 for x in range(30, 96, 5)]:
        misjudged = sum(1 for c, s in scored if c["karar"] != YANLIS and s < threshold)
        filtered = sum(1 for s in negatives if s < threshold) / len(negatives) \
            if negatives else 0.0
        score_by_id = {case["id"]: score for case, score in scored}
        predicted = []
        for case in cases:
            if tiers_accept(case["answer"], case["key"]):
                predicted.append(DOGRU)
                continue
            score = score_by_id.get(case["id"])
            predicted.append(
                YANLIS if score is not None and score < threshold else INCELE
            )
        stats = selective_stats(
            [case["karar"] for case in cases],
            predicted,
        )
        # distance to the nearest case the threshold has to separate: a cut that
        # only just clears an anchor is one rephrasing away from misjudging it
        margin = min(abs(s - threshold) for _, s in scored) if scored else 0.0
        print(f"   {threshold:>6.2f}{misjudged:>20}"
              f"{stats['coverage']:>9.1%}{stats['selective_risk']:>10.1%}"
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
