"""Score a saved answer run with Ragas, and check the LLM judge against ground truth.

    ragas_env\\Scripts\\python -m eval.ragas_check --set human

Two separate purposes, and the first matters more:

1. VALIDATE THE JUDGE. Ragas scores with an LLM. Whether an LLM judges Turkish
   reliably is unknown to us, and adopting a judge on faith would repeat the
   mistake this project keeps finding: trusting a metric without checking it.
   We are unusually well placed to check -- there are hand-written reference
   answers and a deterministic verdict for every question. `FactualCorrectness`
   is scored against those same references, so where it disagrees with the
   deterministic verdict, the disagreement is about the JUDGE, not the system.

2. MEASURE WHAT WE CANNOT. `Faithfulness` asks whether every claim in an answer
   is supported by the context -- the model can produce the right figure and
   fabricate the sentence around it, and key-matching scores that as correct.
   `ContextPrecision` asks how much of the retrieved context was useless, which
   we have never measured despite passing 15 chunks every time.

Runs entirely on our own services: the judge is whatever `LLM_API_URL` points
at, which is the same self-hosted model the pipeline answers with. Nothing
leaves the machines it already runs on. The three metrics below need no
embedding model.

Responses are cached on disk, so re-running after a scoring change costs
nothing -- the same reason the raw answers are kept.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IN_DIR = Path("output/eval")
CACHE_DIR = Path(os.getenv("RAGAS_CACHE_DIR", "output/eval/.ragas_cache"))


def _clients():
    """Point Ragas at our own vLLM and embedding service.

    Both speak the OpenAI wire format, which is a message shape rather than a
    vendor: the servers are the ones this project already runs. `api_key` is a
    placeholder because they require no authentication.
    """
    from openai import AsyncOpenAI, OpenAI

    from pipeline.index import embeddings as emb
    from pipeline.retrieval import query

    # ASYNC on purpose: the metrics are awaited through `ascore`, and ragas
    # refuses to drive an async call with a synchronous client -- it fails every
    # metric with "Cannot use agenerate() with a synchronous client" rather than
    # falling back, so the whole run scores None.
    llm_client = AsyncOpenAI(
        base_url=query.LLM_API_URL.rsplit("/v1/", 1)[0] + "/v1",
        api_key=os.getenv("LLM_API_KEY") or "not-needed",
    )
    embed_client = OpenAI(
        base_url=emb.EMBED_API_URL.rsplit("/v1/", 1)[0] + "/v1",
        api_key=os.getenv("EMBED_API_KEY", "not-needed"),
    )
    return llm_client, query.LLM_MODEL_NAME, embed_client, emb.EMBED_MODEL_NAME


def build_metrics():
    """The three metrics chosen here are LLM-only, so no embedding model is
    needed. `AnswerRelevancy` is left out on purpose: it also requires
    embeddings, and it measures whether an answer addresses the question --
    worth having, but not what this first run is for, which is finding out
    whether the judge can be trusted at all.
    """
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        ContextPrecisionWithoutReference,
        FactualCorrectness,
        Faithfulness,
    )

    llm_client, llm_model, _, _ = _clients()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = DiskCacheBackend(cache_dir=str(CACHE_DIR))
    llm = llm_factory(llm_model, provider="openai", client=llm_client, cache=cache)

    return {
        # judged against OUR reference answers -- this is the judge check
        "factual_correctness": FactualCorrectness(llm=llm),
        # reference-free, and the dimension key-matching cannot see
        "faithfulness": Faithfulness(llm=llm),
        # the expensive one: it asks the LLM about each retrieved chunk in turn
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
    }


async def score_row(metrics, row, reference):
    """Ragas scores for one saved answer. A metric that fails is recorded as
    None rather than aborting the run -- a single unparseable judgement should
    not cost the whole evaluation."""
    contexts = [c for c in (row.get("baglam") or "").split("\n\n---\n\n") if c.strip()]
    out = {}
    for name, metric in metrics.items():
        kwargs = {
            "user_input": row["soru"],
            "response": row["cevap"],
            "retrieved_contexts": contexts,
        }
        if name == "factual_correctness":
            kwargs = {"response": row["cevap"], "reference": reference}
        try:
            result = await metric.ascore(**kwargs)
            out[name] = round(float(result.value), 4)
        except Exception as e:
            out[name] = None
            out[f"{name}_hata"] = f"{type(e).__name__}: {e}"[:120]
    return out


def agreement(rows):
    """How often the LLM judge and the deterministic verdict reach the same
    conclusion -- and, when they differ, in which direction."""
    both = [r for r in rows if r.get("factual_correctness") is not None]
    if not both:
        return {"karsilastirilabilir": 0}
    agree = sum(1 for r in both
                if (r["factual_correctness"] >= 0.5) == bool(r["bizim_dogru"]))
    judge_stricter = sum(1 for r in both
                         if r["bizim_dogru"] and r["factual_correctness"] < 0.5)
    judge_looser = sum(1 for r in both
                       if not r["bizim_dogru"] and r["factual_correctness"] >= 0.5)
    return {
        "karsilastirilabilir": len(both),
        "uyum": round(agree / len(both), 4),
        "hakem_daha_katı": judge_stricter,
        "hakem_daha_gevşek": judge_looser,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="human")
    ap.add_argument("--limit", type=int, help="ilk N soru (deneme icin)")
    # A run's answers and its judgements live in their own folders, so scoring
    # one run never reads or overwrites another's.
    ap.add_argument("--in-dir", default=None, help="kayitli cevaplarin klasoru")
    ap.add_argument("--out-dir", default=None, help="hakem sonuclarinin klasoru")
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else IN_DIR
    out_dir = Path(args.out_dir) if args.out_dir else IN_DIR
    answers_path = in_dir / f"rag_answers_{args.set}.json"
    if not answers_path.exists():
        raise SystemExit(
            f"kayitli cevap yok: {answers_path}\n"
            f"once 'python -m eval.rag_answer_eval --set {args.set}' calistir")

    saved = json.loads(answers_path.read_text(encoding="utf-8"))["sorular"]
    gt = {g["q"]: g for g in json.loads(
        (Path("data/rag_eval") / f"{args.set}.json").read_text(encoding="utf-8"))}
    if args.limit:
        saved = saved[:args.limit]

    if not any(r.get("baglam") for r in saved):
        raise SystemExit(
            "kayitli cevaplarda baglam yok -- bu dosya baglam kaydedilmeden once\n"
            f"uretilmis. 'python -m eval.rag_answer_eval --set {args.set}' ile yenile.")

    metrics = build_metrics()
    rows = []
    for i, row in enumerate(saved, start=1):
        g = gt.get(row["soru"])
        if not g:
            continue
        t0 = time.time()
        scored = asyncio.run(score_row(metrics, row, g["answer"]))
        secs = round(time.time() - t0, 1)
        scored.update({"soru": row["soru"], "bizim_dogru": bool(row["cevap_dogru"]),
                       "saniye": secs})
        rows.append(scored)
        fc = scored.get("factual_correctness")
        # Per-question timing is what prices the full run: ~19 LLM calls each,
        # so five questions tell you whether 38 is worth it.
        print(f"  {i}/{len(saved)} fc={fc} faith={scored.get('faithfulness')} "
              f"{secs}s bizim={'dogru' if row['cevap_dogru'] else 'yanlis'}  "
              f"{row['soru'][:42]}")

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n": len(rows),
        "faithfulness": mean("faithfulness"),
        "context_precision": mean("context_precision"),
        "factual_correctness": mean("factual_correctness"),
        "hakem_uyumu": agreement(rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ragas_{args.set}.json"
    out.write_text(json.dumps({"ozet": summary, "sorular": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"  faithfulness       : {summary['faithfulness']}   (cevap baglamda destekli mi)")
    print(f"  context_precision  : {summary['context_precision']}   (baglamin ne kadari ise yaradi)")
    print(f"  factual_correctness: {summary['factual_correctness']}")
    print(f"  hakem uyumu        : {summary['hakem_uyumu']}")
    print(f"\nyazildi: {out}")


if __name__ == "__main__":
    main()
