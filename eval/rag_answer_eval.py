"""Answer eval: given the context retrieval produced, does the model answer
correctly and cite the right page?

    python -m eval.rag_answer_eval --set human
    python -m eval.rag_answer_eval --set human --top-k 50 --rerank

Runs the REAL production pieces -- query.build_context and query.generate -- so
the prompt and the citation format under test are the ones that ship.

No LLM judge: the question set carries a `key` (the distinctive part of the
right answer, usually a figure), so correctness is a deterministic match. What
that buys is an attribution, which is the whole point of running this:

    key not in context            -> retrieval's fault, the model never had it
    key in context, not in answer -> the generator's fault
    key in both                   -> correct

Without the split, a wrong answer is just a wrong answer and you cannot tell
which half of the system to fix.

Every raw answer is written to output/eval/rag_answers.json. Scoring can then be
corrected offline instead of paying for another GPU hour to regenerate them.
"""
import argparse
import json
import re
import time
from pathlib import Path

from eval.judge import DOGRU, INCELE, YANLIS, judge
from eval.rag_eval import QUESTION_DIR, OUT_DIR, contains_key, retrieve_chunks

# "Sayfa 13'e göre" / "sayfa 58" -- the citation shape the answer prompt asks for
_PAGE_CITE = re.compile(r"sayfa\s*(\d+)", re.IGNORECASE)
# the exact refusal the prompt instructs the model to use when it cannot answer
_ABSTAIN = "bulunamadı"


def cited_pages(answer):
    return {int(m) for m in _PAGE_CITE.findall(answer or "")}


def abstained(answer):
    return _ABSTAIN in (answer or "").lower()


def score_one(question, answer, context, sim=None):
    """Grade one answer and say WHERE it went wrong.

    The verdict has three states, not two -- see eval/judge. An answer the
    scorer cannot confirm is not therefore wrong, and counting it as wrong is
    how one run's table score was reported at 0.588 when it was nearer 0.99.
    `sim` lets a caller supply the similarity itself, which is what keeps this
    callable with no embedding service running.
    """
    key = question.get("key")
    in_ctx = contains_key(context, key)
    durum, gerekce = judge(key, answer, question.get("answer"), sim=sim)
    cites = cited_pages(answer)

    if in_ctx is False:
        fault = "retrieval"          # the model was never given the answer
    elif durum == DOGRU:
        fault = None
    elif abstained(answer):
        fault = "uretim_cekimser"    # had it, still refused
    elif durum == YANLIS:
        fault = "uretim_yanlis"      # had it, answered something else
    else:
        fault = None                 # under review: no fault attributed yet

    return {
        "soru": question["q"],
        "tip": question.get("type", "?"),
        "cevap": answer,
        # kept so the run can be re-scored later without re-retrieving: the
        # index changes between runs, so a context fetched afterwards would not
        # be the one this answer was actually produced from
        "baglam": context,
        "ctx_var": in_ctx,
        "durum": durum,
        "gerekce": gerekce,
        "cevap_dogru": durum == DOGRU,
        "sayfa_dogru": bool(cites & set(question["pages"])),
        "sayfa_verdi": bool(cites),
        "atif_edilen": sorted(cites),
        "beklenen_sayfa": question["pages"],
        "hata": fault,
    }


def summarize(rows, split_by_type=True):
    n = len(rows)
    if not n:
        return {"n": 0}
    faults = [r["hata"] for r in rows if r["hata"]]
    review = sum(1 for r in rows if r.get("durum") == INCELE)
    correct = sum(1 for r in rows if r["cevap_dogru"])
    out = {
        "n": n,
        "ctx_recall": round(sum(1 for r in rows if r["ctx_var"]) / n, 4),
        # Accuracy is a BAND, not a number. The lower bound is what the scorer
        # could confirm; the upper bound adds what it could not settle either
        # way. Reporting only the lower bound is what made a run look far worse
        # than it was, and reporting only the upper one would flatter it.
        "cevap_dogrulugu": round(correct / n, 4),
        "incele_orani": round(review / n, 4),
        "ust_sinir": round((correct + review) / n, 4),
        # of the answers that were RIGHT, how many pointed the user at a page
        # they could actually verify -- this is what the page-number fix bought
        "sayfa_dogrulugu": round(sum(1 for r in rows if r["sayfa_dogru"]) / n, 4),
        "sayfa_verdi": round(sum(1 for r in rows if r["sayfa_verdi"]) / n, 4),
        "hata_dagilimi": {f: faults.count(f) for f in sorted(set(faults))},
    }
    # Figures and prose fail differently: a figure is copied or it is not, while
    # a prose answer can be fluent and still miss the point. Reporting one number
    # over both hides whichever is the weaker half.
    types = {r.get("tip", "?") for r in rows}
    if split_by_type and len(types) > 1:
        out["tipe_gore"] = {
            t: summarize([r for r in rows if r.get("tip") == t], split_by_type=False)
            for t in sorted(types)
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="human")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--rrf-k", type=int, default=1)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-to", type=int, default=10)
    ap.add_argument("--backend", default="native",
                    help="cevaplari hangi erisim motoru beslesin (native/llamaindex)")
    ap.add_argument("--out-dir", default=None,
                    help="ciktinin yazilacagi klasor; verilmezse output/eval")
    args = ap.parse_args()

    from pipeline import db
    from pipeline.query import build_context, generate

    questions = json.loads((QUESTION_DIR / f"{args.set}.json").read_text(encoding="utf-8"))
    conn = db.get_conn()
    rerank_to = args.rerank_to if args.rerank else None

    rows, t0 = [], time.time()
    for i, q in enumerate(questions, start=1):
        chunks = retrieve_chunks(conn, q["q"], top_k=args.top_k,
                                 rrf_k=args.rrf_k, rerank_to=rerank_to,
                                 backend=args.backend)
        context = build_context(chunks)
        try:
            answer = generate(q["q"], context)
        except Exception as e:                      # keep the run alive, record it
            answer = f"[URETIM HATASI: {type(e).__name__}: {e}]"
        rows.append(score_one(q, answer, context))
        print(f"  {i}/{len(questions)} {'OK ' if rows[-1]['cevap_dogru'] else 'HATA'} "
              f"{q['q'][:60]}")

    m = summarize(rows)
    m["saniye"] = round(time.time() - t0, 1)

    # --out-dir keeps a run's results in their own folder, so a later run of the
    # same set cannot destroy answers that cost GPU time to produce. Falls back
    # to the flat layout when nobody asks for one.
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    # named per question set: a second run used to overwrite the first, and
    # those answers cost GPU time to produce -- losing them means renting again
    out = out_dir / f"rag_answers_{args.set}.json"
    out.write_text(json.dumps({"ozet": m, "sorular": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print()
    print(f"  n                : {m['n']}")
    print(f"  ctx_recall       : {m['ctx_recall']:.3f}   (cevap modele ulasti mi)")
    print(f"  cevap dogrulugu  : {m['cevap_dogrulugu']:.3f} - {m['ust_sinir']:.3f}"
          f"   (incelenecek: {m['incele_orani']:.3f})")
    print(f"  sayfa dogrulugu  : {m['sayfa_dogrulugu']:.3f}   (sayfa verdi: {m['sayfa_verdi']:.3f})")
    print(f"  hata dagilimi    : {m['hata_dagilimi']}")
    for tip, tm in m.get("tipe_gore", {}).items():
        print(f"    {tip:8s} n={tm['n']:3d}  ctx={tm['ctx_recall']:.3f}  "
              f"cevap={tm['cevap_dogrulugu']:.3f}  sayfa={tm['sayfa_dogrulugu']:.3f}")
    print(f"\nham cevaplar: {out}")


if __name__ == "__main__":
    main()
