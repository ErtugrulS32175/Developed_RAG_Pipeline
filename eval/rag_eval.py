"""Retrieval eval: does the right passage come back, and at what rank?

    python -m eval.rag_eval --bootstrap 60     # build a known-item question set
    python -m eval.rag_eval                    # score with production settings
    python -m eval.rag_eval --top-k 100 --rerank
    python -m eval.rag_eval --sweep-top-k 15 50 100 --rerank

Scored at PAGE level: a retrieved chunk counts as a hit when its page is one of
the expected pages. That survives re-chunking, is what a human can actually
label, and needs NO LLM -- retrieval quality is measurable without a generator,
which is why this can run entirely on local services.

Question sets live in a GITIGNORED directory (questions and page numbers are
document content and never belong in the repo).

Two kinds of question set, and they measure different things:
  * BOOTSTRAP (--bootstrap) -- generated from the index itself: for a sampled
    chunk, the query is a bag of that chunk's rarest terms and the expected page
    is the chunk's own page. Free, unbiased about WHICH pages get asked about,
    and good for tuning breadth (top-k) and fusion. But it shares surface forms
    with the source text, so it FLATTERS lexical search and is NOT a fair test
    of stemming.
  * HUMAN -- real questions with the pages that answer them. The only set that
    measures what a user actually experiences, because a person writes
    "kitaplari ne kadar" where the document says "kitaplarindaki", which is
    exactly the morphology that stemming has to bridge.
"""
import argparse
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

QUESTION_DIR = Path(os.getenv("RAG_EVAL_DIR", "data/rag_eval"))
OUT_DIR = Path("output/eval")

# a content word: letters only (so numbers, which BM25 handles separately and
# which appear in nearly every chunk of a financial document, don't dominate)
_WORD = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}")


# --- metrics (pure, unit-tested) ---------------------------------------------

def first_hit_rank(ranked_pages, expected_pages):
    """1-based rank of the first retrieved page that is an expected one, or None."""
    expected = set(expected_pages)
    for i, page in enumerate(ranked_pages, start=1):
        if page in expected:
            return i
    return None


def contains_key(text, key):
    """Is the expected answer actually present in this text?

    Page-level hit rates flatter the system: a page becomes 3-10 chunks and only
    one of them holds the figure, so retrieving SOME chunk of the right page
    scores 1.0 even when the answer never reached the model. This asks the
    stricter question, and it is the real ceiling on answer quality -- if the
    key is absent from the context, no generator can answer correctly.

    Falls back to digit-only comparison against numeric TOKENS (not against the
    whole string, which would splice adjacent numbers together and match things
    that are not there) so 8.765 / 8765 / 8,765 count as the same figure.
    """
    if not key:
        return None
    hay = " ".join(str(text).lower().split())
    if " ".join(str(key).lower().split()) in hay:
        return True
    digits = re.sub(r"\D", "", str(key))
    if len(digits) >= 3:
        from pipeline.number_verify import numeric_token_set
        return digits in numeric_token_set(text)
    return False


def summarize(ranks, ks=(1, 3, 5, 10, 20)):
    """Aggregate per-question first-hit ranks into the usual retrieval metrics.

    `ranks` holds one entry per question: an int rank, or None for a miss.
    MRR counts a miss as 0, so it is comparable across runs with different k.
    """
    total = len(ranks)
    if not total:
        return {"n": 0}
    hits = [r for r in ranks if r is not None]
    out = {
        "n": total,
        "miss": total - len(hits),
        "mrr": round(sum(1.0 / r for r in hits) / total, 4),
    }
    for k in ks:
        out[f"hit@{k}"] = round(sum(1 for r in hits if r <= k) / total, 4)
    out["median_rank"] = statistics.median(hits) if hits else None
    return out


def format_summary(label, m):
    if not m.get("n"):
        return f"{label}: soru yok"
    ks = " ".join(f"{k}={m[k]:.3f}" for k in m if k.startswith("hit@"))
    ctx = f"  ctx_recall={m['ctx_recall']:.3f}" if "ctx_recall" in m else ""
    return (f"{label:28s} n={m['n']:3d}  kacan={m['miss']:3d}  "
            f"MRR={m['mrr']:.4f}  {ks}  medyan_sira={m['median_rank']}{ctx}")


# --- question sets ------------------------------------------------------------

def load_questions(name):
    path = QUESTION_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"soru dosyasi yok: {path}\n"
            f"once 'python -m eval.rag_eval --bootstrap 60' calistir, "
            f"ya da elle {path} olustur: [{{\"q\": \"...\", \"pages\": [12]}}, ...]")
    return json.loads(path.read_text(encoding="utf-8"))


def rarest_terms(text, doc_freq, n):
    """The n least common content words of `text`, rarest first. Terms that occur
    on every page identify nothing, so a query built from them would be
    unanswerable for reasons unrelated to retrieval quality."""
    words = {w.lower() for w in _WORD.findall(text)}
    return sorted(words, key=lambda w: (doc_freq[w], w))[:n]


def bootstrap_questions(conn, n, seed=0, terms_per_query=6):
    """Build a known-item set from the index: query = a chunk's rarest terms,
    expected page = that chunk's page. Rarest-first because terms that occur on
    every page identify nothing, so a query built from them would be unanswerable
    for reasons that have nothing to do with retrieval quality."""
    from psycopg.rows import dict_row

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT page, text FROM chunks WHERE page > 0 AND length(text) > 300")
        rows = cur.fetchall()
    if not rows:
        raise SystemExit("indekste uygun chunk yok -- once ingest calistir")

    doc_freq = Counter()
    for r in rows:
        doc_freq.update({w.lower() for w in _WORD.findall(r["text"])})

    rnd = random.Random(seed)
    sample = rnd.sample(rows, min(n, len(rows)))
    questions = []
    for r in sample:
        rarest = rarest_terms(r["text"], doc_freq, terms_per_query)
        if len(rarest) < 3:
            continue
        questions.append({"q": " ".join(rarest), "pages": [r["page"]]})
    return questions


# --- retrieval ----------------------------------------------------------------

def retrieve_chunks(conn, question, *, top_k, rrf_k, rerank_to=None):
    """The chunks production would put in front of the model, best first."""
    from pipeline import db
    from pipeline.embeddings import embed_dense, embed_sparse

    dense = embed_dense(question)
    idx, val = embed_sparse(question)
    chunks = db.hybrid_search(conn, dense, idx, val, top_k=top_k, rrf_k=rrf_k)
    if rerank_to:
        chunks = rerank(question, chunks, rerank_to)
    return chunks


RERANK_BATCH = int(os.getenv("RERANK_BATCH", "32"))


def rerank(question, chunks, top_n):
    """Cross-encoder rerank, scoring candidates in batched requests.

    query.rerank posts one request PER chunk, which is tolerable at 15 candidates
    and unusable at 100 -- and 100 is the breadth a reranker actually needs to be
    worth having. Batched rather than one giant request because the reranker runs
    on a 0.15 GPU-memory share and a 100-pair batch can exceed it. Kept here
    rather than in query.py until the sweep shows the wider net is worth adopting.
    """
    import requests

    url = os.getenv("RERANK_API_URL", "http://localhost:8002/v1/score")
    model = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

    scores = []
    for i in range(0, len(chunks), RERANK_BATCH):
        batch = chunks[i:i + RERANK_BATCH]
        r = requests.post(url, json={"model": model, "text_1": question,
                                     "text_2": [c["text"] for c in batch]}, timeout=180)
        r.raise_for_status()
        # the endpoint may return results out of order; "index" is the position
        # within the batch that was sent
        scores += [d["score"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]

    ranked = sorted(zip(scores, range(len(chunks))), key=lambda p: p[0], reverse=True)
    return [chunks[i] for _, i in ranked[:top_n]]


def run(conn, questions, *, top_k, rrf_k, rerank_to=None):
    ranks, key_hits, misses, t0 = [], [], [], time.time()
    for q in questions:
        chunks = retrieve_chunks(conn, q["q"], top_k=top_k, rrf_k=rrf_k, rerank_to=rerank_to)
        ranks.append(first_hit_rank([c["page"] for c in chunks], q["pages"]))
        found = contains_key(" ".join(c.get("text", "") for c in chunks), q.get("key"))
        if found is not None:
            key_hits.append(found)
            if not found:
                misses.append(q["q"])
    m = summarize(ranks)
    if key_hits:
        # the ceiling: share of questions whose answer actually reached the model
        m["ctx_recall"] = round(sum(key_hits) / len(key_hits), 4)
        m["ctx_kacan"] = misses
    m["saniye"] = round(time.time() - t0, 1)
    return m, ranks


# --- cli ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="questions", help="soru dosyasinin adi (uzantisiz)")
    ap.add_argument("--bootstrap", type=int, metavar="N",
                    help="indeksten N soruluk known-item seti uret ve kaydet")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--rrf-k", type=int, default=1)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-to", type=int, default=10)
    ap.add_argument("--sweep-top-k", type=int, nargs="+", metavar="K")
    ap.add_argument("--sweep-rrf-k", type=int, nargs="+", metavar="K")
    args = ap.parse_args()

    from pipeline import db
    conn = db.get_conn()

    if args.bootstrap:
        qs = bootstrap_questions(conn, args.bootstrap)
        QUESTION_DIR.mkdir(parents=True, exist_ok=True)
        path = QUESTION_DIR / f"{args.set}.json"
        path.write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(qs)} soru uretildi -> {path}")
        return

    questions = load_questions(args.set)
    rerank_to = args.rerank_to if args.rerank else None
    results = {}

    top_ks = args.sweep_top_k or [args.top_k]
    rrf_ks = args.sweep_rrf_k or [args.rrf_k]
    for tk in top_ks:
        for rk in rrf_ks:
            label = f"top_k={tk} rrf_k={rk}" + (f" rerank->{rerank_to}" if rerank_to else "")
            m, _ = run(conn, questions, top_k=tk, rrf_k=rk, rerank_to=rerank_to)
            results[label] = m
            print(format_summary(label, m))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rag_eval.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nyazildi: {out}")


if __name__ == "__main__":
    main()
