"""Run every question set against the pod LLM in one unattended batch.

    python -m eval.run_answer_batch                 # her set, uretim ayari
    python -m eval.run_answer_batch --context-ab    # + ayni setler 10 chunk ile
    python -m eval.run_answer_batch --sets X Y      # yalnizca secilenler

Written for rented-GPU time, which is why it does two things a plain loop would
not:

1. PREFLIGHT FIRST. Every dependency is checked before a single token is
   generated. Discovering a missing PG_DSN or a dead tunnel after the GPU meter
   has started is the expensive failure mode, and it is entirely avoidable.

2. ARCHIVE EVERY RUN. `rag_answer_eval` names its output per question set, so a
   second run at different settings silently overwrites the first -- and those
   answers cost GPU time. Each run is copied to a settings-tagged filename, so
   re-running is never destructive and comparisons can be made offline later.

Retrieval runs locally (pgvector + bge-m3 + reranker in Docker); only answer
generation crosses the tunnel. No document is sent to the pod beyond the
context of the question being asked.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

QUESTION_DIR = Path("data/rag_eval")
OUT_DIR = Path("output/eval")
# Each measured run gets its own folder, one subfolder per answering engine:
#
#     output/RAG_Outputs/run1/native/
#     output/RAG_Outputs/run1/llamaindex/
#     output/RAG_Outputs/run1/ragas/
#
# The flat layout let a re-run silently replace answers that cost GPU time, and
# it had nowhere to put a second engine's results at all.
RUNS_DIR = Path("output/RAG_Outputs")


def discover_sets():
    """Question sets found in the question directory, sorted.

    Discovered rather than listed, so adding a set needs no code change.
    """
    return sorted(p.stem for p in QUESTION_DIR.glob("*.json"))


# Production defaults: query.py ranks with the reranker but never SHRINKS the
# list (TOP_RERANK defaults to TOP_K). Measuring at a smaller rerank_to is the
# --context-ab experiment, not the baseline.
TOP_K = 15
RERANK_TO = 15


def preflight(sets):
    """Everything that can fail cheaply, checked before the GPU meter runs."""
    problems, notes = [], []

    for s in sets:
        p = QUESTION_DIR / f"{s}.json"
        if not p.exists():
            problems.append(f"soru seti yok: {p}")
        else:
            notes.append(f"{s}: {len(json.loads(p.read_text(encoding='utf-8')))} soru")

    try:
        import psycopg
        from pipeline import db
        # Bounded on purpose. A wrong password or a stopped container otherwise
        # spends ~130s retrying before it admits defeat, which on a rented GPU
        # reads as a hang and bills like one.
        with psycopg.connect(db.PG_DSN, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*), count(DISTINCT page) FROM chunks")
                n, pages = cur.fetchone()
        notes.append(f"veritabani: {n} chunk / {pages} sayfa")
        if n == 0:
            problems.append("veritabani bos -- once ingest calistir")
    except Exception as e:
        problems.append(f"veritabanina baglanilamadi ({type(e).__name__}) "
                        f"-- PG_DSN ayarli mi, ragtest-pgvector ayakta mi?")

    import requests
    from pipeline import embeddings as emb
    from pipeline import query

    base = query.LLM_API_URL.rsplit("/v1/", 1)[0] + "/v1"
    try:
        r = requests.get(f"{base}/models", headers=query.llm_headers(), timeout=10)
        if r.status_code == 401:
            # Worth its own message: a wrong key looks exactly like a dead
            # endpoint if you only read "the LLM check failed".
            problems.append("LLM 401 -- LLM_API_KEY yanlis ya da eksik "
                            "(sunucu --api-key ile korunuyor)")
        else:
            ids = [m["id"] for m in r.json().get("data", [])]
            notes.append(f"LLM: {', '.join(ids) or 'model listesi bos'}"
                         f"{'  [key ile]' if query.llm_headers() else ''}")
            if query.LLM_MODEL_NAME not in ids:
                problems.append(f"LLM_MODEL_NAME='{query.LLM_MODEL_NAME}' sunucuda yok: {ids}")
    except Exception as e:
        problems.append(f"LLM'e ulasilamadi ({base}) -- adres/tunel dogru mu? "
                        f"[{type(e).__name__}]")

    for name, url in (("embed", emb.EMBED_API_URL), ("rerank", query.RERANK_API_URL)):
        try:
            requests.post(url, json={}, timeout=10)
            notes.append(f"{name}: ayakta")
        except Exception:
            problems.append(f"{name} servisi kapali ({url}) -- docker start?")

    return problems, notes


def run_set(name, rerank_to, top_k=TOP_K, backend="native", out_dir=None):
    """One question set. Returns its summary dict, or None if the run failed."""
    out_dir = out_dir or OUT_DIR
    cmd = [sys.executable, "-m", "eval.rag_answer_eval", "--set", name,
           "--top-k", str(top_k), "--rerank", "--rerank-to", str(rerank_to),
           "--backend", backend, "--out-dir", str(out_dir)]
    print(f"\n{'=' * 68}\n{name}  ({backend}, top_k={top_k}, rerank_to={rerank_to})"
          f"\n{'=' * 68}")
    t0 = time.time()
    if subprocess.run(cmd).returncode != 0:
        print(f"  !! {name} basarisiz -- batch devam ediyor")
        return None

    live = out_dir / f"rag_answers_{name}.json"
    # tagged copy: the canonical filename is what ragas_check reads, but a later
    # run at other settings would otherwise destroy answers that cost GPU time
    archive = out_dir / f"rag_answers_{name}_k{rerank_to}.json"
    shutil.copy2(live, archive)
    data = json.loads(live.read_text(encoding="utf-8"))
    data["ozet"]["_dakika"] = round((time.time() - t0) / 60, 1)
    data["ozet"]["_arsiv"] = archive.name
    return data["ozet"]


def table(results, title=""):
    print(f"\n{'=' * 78}")
    if title:
        print(title)
    print(f"{'set':<22}{'n':>4}{'ctx_recall':>12}{'cevap':>9}{'sayfa':>8}{'dk':>7}")
    print("-" * 78)
    for label, m in results:
        if m is None:
            print(f"{label:<22}{'-':>4}{'BASARISIZ':>12}")
            continue
        print(f"{label:<22}{m['n']:>4}{m['ctx_recall']:>12.3f}"
              f"{m['cevap_dogrulugu']:>9.3f}{m['sayfa_dogrulugu']:>8.3f}"
              f"{m.get('_dakika', 0):>7.1f}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=None,
                    help="olculecek soru setleri; vermezsen klasordeki hepsi")
    ap.add_argument("--rerank-to", type=int, default=RERANK_TO)
    ap.add_argument("--backends", nargs="+", default=["native"],
                    help="olculecek erisim motorlari (native llamaindex)")
    ap.add_argument("--run", default=None,
                    help="koşu adi (or. run1); verilirse sonuclar "
                         "output/RAG_Outputs/<run>/<backend>/ altina yazilir")
    ap.add_argument("--context-ab", action="store_true",
                    help="ayni setleri 10 chunk ile de kosarak baglam boyutunu fiyatlandir")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    sets = args.sets or discover_sets()
    if not sets:
        raise SystemExit(f"soru seti bulunamadi: {QUESTION_DIR}/*.json")

    if not args.skip_preflight:
        print("On kontrol...")
        problems, notes = preflight(sets)
        for n in notes:
            print(f"  ok   {n}")
        for p in problems:
            print(f"  HATA {p}")
        if problems:
            print("\nGPU saati yakilmadan durduruldu. Yukaridakileri duzelt.")
            raise SystemExit(1)
        print("  -> hazir\n")

    for backend in args.backends:
        out_dir = (RUNS_DIR / args.run / backend) if args.run else None
        results = []
        for s in sets:
            results.append((s, run_set(s, args.rerank_to, backend=backend,
                                       out_dir=out_dir)))

        if args.context_ab:
            # Worth running on the sets that FAIL, not on one already at 1.000:
            # the context is 17-26k characters and the model has to pick one
            # figure out of many similar ones. Fewer chunks means fewer
            # distractors -- but also less recall. Only accuracy prices that.
            for s in sets:
                results.append((f"{s} (10 chunk)",
                                run_set(s, 10, backend=backend, out_dir=out_dir)))

        table(results, title=f"motor: {backend}"
                             + (f"   klasor: {out_dir}" if out_dir else ""))
    print("\nSonraki adim: hakemi kucuk baslat --")
    print("  ragas_env\\Scripts\\python -m eval.ragas_check --set human --limit 5")


if __name__ == "__main__":
    main()
