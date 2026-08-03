"""Diagnostic second-opinion scores for a saved answer run.

    ragas_env\\Scripts\\python -m eval.answer.ragas_check --set human

Only two deliberately narrow metrics are run:

* Faithfulness: are the answer's claims supported by the supplied context?
* Answer similarity: is the response semantically close to the reference?

Both are DIAGNOSTIC, never release gates. Faithfulness uses an LLM judge whose
agreement with Turkish domain experts has not been calibrated. Answer
similarity is intentionally threshold-free: five adjudicated cases reaching
the semantic tier cannot justify a correct/incorrect cut.

By default answer similarity reuses the production embedding service. That is
cheap but correlated with retrieval and is labelled as such in every output.
Set both EVAL_EMBED_API_URL and EVAL_EMBED_MODEL_NAME to run an explicitly
configured audit model; a different endpoint alone does not prove statistical
independence, so the exact model name is always recorded.

Runs entirely on our own services. Nothing leaves the machines it already runs
on. Context precision was retired because its per-chunk LLM calls dominated
cost, while factual correctness was retired after poor agreement with the
deterministic and human-adjudicated scorer.

Responses are cached on disk, so re-running after a scoring change costs
nothing -- the same reason the raw answers are kept.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IN_DIR = Path("output/eval")
CACHE_DIR = Path(os.getenv("RAGAS_CACHE_DIR", "output/eval/.ragas_cache"))
OUTPUT_SCHEMA_VERSION = 2
METRIC_SET = ("faithfulness", "answer_similarity")
_RUN_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def evaluation_config_id(metadata):
    """Stable identity for results that may be compared like-for-like."""
    comparable = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_set": list(METRIC_SET),
        "ragas_version": metadata["ragas_version"],
        "metric_classes": metadata["metric_classes"],
        "llm_model": metadata["llm_model"],
        "embedding_model": metadata["embedding_model"],
        "embedding_mode": metadata["embedding_mode"],
        "similarity_threshold": metadata["similarity_threshold"],
        "diagnostic_only": metadata["diagnostic_only"],
    }
    canonical = json.dumps(
        comparable,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def result_path(out_dir, set_name, metadata, *, limit=None, run_tag=None):
    """A filename that keeps evaluator configurations and samples separate."""
    if run_tag is not None and not _RUN_TAG.fullmatch(run_tag):
        raise ValueError(
            "run tag 1-64 karakter olmali; yalnizca harf, rakam, nokta, "
            "alt cizgi ve tire kullanilabilir"
        )
    parts = [
        f"ragas_diagnostic_v{OUTPUT_SCHEMA_VERSION}",
        set_name,
        metadata["embedding_mode"],
        metadata["evaluation_config_id"],
    ]
    if limit is not None:
        parts.append(f"limit{limit}")
    if run_tag:
        parts.append(run_tag)
    return Path(out_dir) / ("_".join(parts) + ".json")


def refuse_existing_result(path):
    """Fail before any metric calls rather than overwrite a paid run."""
    if path.exists():
        raise SystemExit(
            f"sonuc zaten var, uzerine yazilmadi: {path}\n"
            "ayni ayari yeniden kosmak icin benzersiz bir --run-tag ver"
        )


def write_new_result(path, payload):
    """Create a result once; a concurrent run cannot silently replace it."""
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    except FileExistsError:
        raise SystemExit(
            f"sonuc bu sirada olusturuldu, uzerine yazilmadi: {path}"
        ) from None


def _embedding_settings():
    """Embedding endpoint/model and whether it is the production default."""
    from pipeline.index import embeddings as emb

    url_override = os.getenv("EVAL_EMBED_API_URL", "").strip()
    model_override = os.getenv("EVAL_EMBED_MODEL_NAME", "").strip()
    if bool(url_override) != bool(model_override):
        raise ValueError(
            "EVAL_EMBED_API_URL ve EVAL_EMBED_MODEL_NAME birlikte ayarlanmali"
        )
    if url_override:
        return {
            "url": url_override,
            "model": model_override,
            "api_key": os.getenv("EVAL_EMBED_API_KEY") or "not-needed",
            "mode": "explicit_eval_model",
        }
    return {
        "url": emb.EMBED_API_URL,
        "model": emb.EMBED_MODEL_NAME,
        "api_key": os.getenv("EMBED_API_KEY") or "not-needed",
        "mode": "production_embedding_correlated",
    }


def _clients():
    """Point Ragas at our own vLLM and embedding service.

    Both speak the OpenAI wire format, which is a message shape rather than a
    vendor: the servers are the ones this project already runs. `api_key` is a
    placeholder because they require no authentication.
    """
    from openai import AsyncOpenAI

    from pipeline.generation import answer as generation

    # ASYNC on purpose: the metrics are awaited through `ascore`, and ragas
    # refuses to drive an async call with a synchronous client -- it fails every
    # metric with "Cannot use agenerate() with a synchronous client" rather than
    # falling back, so the whole run scores None.
    embedding = _embedding_settings()
    llm_client = AsyncOpenAI(
        base_url=generation.LLM_API_URL.rsplit("/v1/", 1)[0] + "/v1",
        api_key=generation.LLM_API_KEY or "not-needed",
    )
    embed_client = AsyncOpenAI(
        base_url=embedding["url"].rsplit("/v1/", 1)[0] + "/v1",
        api_key=embedding["api_key"],
    )
    return {
        "llm_client": llm_client,
        "llm_model": generation.LLM_MODEL_NAME,
        "embedding_client": embed_client,
        "embedding_model": embedding["model"],
        "embedding_mode": embedding["mode"],
    }


def build_metrics():
    """The two diagnostic metrics and non-secret configuration metadata."""
    import ragas
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        Faithfulness,
        SemanticSimilarity,
    )

    clients = _clients()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = DiskCacheBackend(cache_dir=str(CACHE_DIR))
    llm = llm_factory(
        clients["llm_model"],
        provider="openai",
        client=clients["llm_client"],
        cache=cache,
    )
    embeddings = embedding_factory(
        "openai",
        model=clients["embedding_model"],
        client=clients["embedding_client"],
        interface="modern",
        cache=cache,
    )

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        # Ragas 0.4.x calls the AnswerSimilarity concept SemanticSimilarity.
        # No threshold: this is a continuous, uncalibrated diagnostic.
        "answer_similarity": SemanticSimilarity(embeddings=embeddings),
    }
    metadata = {
        "ragas_version": getattr(ragas, "__version__", "unknown"),
        "metric_classes": {
            "faithfulness": "Faithfulness",
            "answer_similarity": "SemanticSimilarity",
        },
        "llm_model": clients["llm_model"],
        "embedding_model": clients["embedding_model"],
        "embedding_mode": clients["embedding_mode"],
        "similarity_threshold": None,
        "diagnostic_only": True,
    }
    metadata["evaluation_config_id"] = evaluation_config_id(metadata)
    return metrics, metadata


async def score_row(metrics, row, reference):
    """Ragas scores for one saved answer. A metric that fails is recorded as
    None rather than aborting the run -- a single unparseable judgement should
    not cost the whole evaluation."""
    contexts = [c for c in (row.get("baglam") or "").split("\n\n---\n\n") if c.strip()]
    out = {}
    for name, metric in metrics.items():
        if name == "answer_similarity":
            kwargs = {"response": row["cevap"], "reference": reference}
        else:
            kwargs = {
                "user_input": row["soru"],
                "response": row["cevap"],
                "retrieved_contexts": contexts,
            }
        started = time.perf_counter()
        try:
            result = await metric.ascore(**kwargs)
            out[name] = round(float(result.value), 4)
        except Exception as e:
            out[name] = None
            out[f"{name}_hata"] = f"{type(e).__name__}: {e}"[:120]
        finally:
            out[f"{name}_saniye"] = round(time.perf_counter() - started, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="human")
    ap.add_argument("--limit", type=int, help="ilk N soru (deneme icin)")
    ap.add_argument(
        "--run-tag",
        default=None,
        help="ayni yapilandirmanin ayri kosusu icin benzersiz etiket",
    )
    # A run's answers and its judgements live in their own folders, so scoring
    # one run never reads or overwrites another's.
    ap.add_argument("--in-dir", default=None, help="kayitli cevaplarin klasoru")
    ap.add_argument("--out-dir", default=None, help="hakem sonuclarinin klasoru")
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit pozitif bir tam sayi olmali")

    in_dir = Path(args.in_dir) if args.in_dir else IN_DIR
    out_dir = Path(args.out_dir) if args.out_dir else IN_DIR
    answers_path = in_dir / f"rag_answers_{args.set}.json"
    if not answers_path.exists():
        raise SystemExit(
            f"kayitli cevap yok: {answers_path}\n"
            f"once 'python -m eval.answer.rag_answer_eval "
            f"--set {args.set}' calistir")

    saved = json.loads(answers_path.read_text(encoding="utf-8"))["sorular"]
    gt = {g["q"]: g for g in json.loads(
        (Path("data/rag_eval") / f"{args.set}.json").read_text(encoding="utf-8"))}
    if args.limit:
        saved = saved[:args.limit]

    if not any(r.get("baglam") for r in saved):
        raise SystemExit(
            "kayitli cevaplarda baglam yok -- bu dosya baglam kaydedilmeden once\n"
            f"uretilmis. 'python -m eval.answer.rag_answer_eval "
            f"--set {args.set}' ile yenile.")

    metrics, metric_metadata = build_metrics()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out = result_path(
            out_dir,
            args.set,
            metric_metadata,
            limit=args.limit,
            run_tag=args.run_tag,
        )
    except ValueError as error:
        ap.error(str(error))
    refuse_existing_result(out)

    rows = []
    run_started = time.perf_counter()
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
        print(f"  {i}/{len(saved)} sim={scored.get('answer_similarity')} "
              f"faith={scored.get('faithfulness')} "
              f"{secs}s bizim={'dogru' if row['cevap_dogru'] else 'yanlis'}  "
              f"{row['soru'][:42]}")

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def timing(name):
        vals = [r[f"{name}_saniye"] for r in rows
                if r.get(f"{name}_saniye") is not None]
        return {
            "toplam": round(sum(vals), 3),
            "ortalama": round(sum(vals) / len(vals), 3) if vals else None,
        }

    summary = {
        "n": len(rows),
        "faithfulness": mean("faithfulness"),
        "answer_similarity": mean("answer_similarity"),
        "metrik_sureleri": {name: timing(name) for name in METRIC_SET},
        "toplam_saniye": round(time.perf_counter() - run_started, 3),
    }
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "metric_set": list(METRIC_SET),
        "evaluation_config_id": metric_metadata["evaluation_config_id"],
        "sample_limit": args.limit,
        "run_tag": args.run_tag,
        "metadata": metric_metadata,
        "ozet": summary,
        "sorular": rows,
    }
    write_new_result(out, payload)

    print()
    print(f"  faithfulness       : {summary['faithfulness']}   (cevap baglamda destekli mi)")
    print(f"  answer similarity  : {summary['answer_similarity']}   "
          f"(esiksiz, kalibre edilmemis)")
    print(f"  embedding modu     : {metric_metadata['embedding_mode']}   "
          f"({metric_metadata['embedding_model']})")
    print(f"  metrik sureleri    : {summary['metrik_sureleri']}")
    print(f"  toplam saniye      : {summary['toplam_saniye']}")
    print(f"\nyazildi: {out}")


if __name__ == "__main__":
    main()
