import os

import requests
from dotenv import load_dotenv

from pipeline.generation import answer as gen
from pipeline.index import db
from pipeline.index.embeddings import embed_dense, embed_sparse

load_dotenv()

# --- Config ---
RERANK_API_URL     = os.getenv("RERANK_API_URL", "http://localhost:8002/v1/score")
RERANK_MODEL_NAME  = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

TOP_K      = int(os.getenv("RAG_TOP_K", "15"))
# The reranker's job is to ORDER the retrieved passages, not to throw any away.
# Measured on the eval set: keeping only 10 of them dropped a passage that held
# the answer -- reproducibly, in both a 15- and a 50-candidate pool -- which
# makes that question unanswerable no matter which model generates from it.
# Page-level scoring hid this completely, because a different passage from the
# same page kept coming back. Ordering still earns its keep: it is what puts the
# right passage first. Defaults to TOP_K so the ranking stage cannot silently
# shrink the context again.
TOP_RERANK = int(os.getenv("RAG_TOP_RERANK", str(TOP_K)))

# --- Init ---
_conn = None


def get_conn():
    """Connect on first use rather than at import, so importing this module (for
    build_context, for a test, for anything that isn't a query) doesn't require a
    running database."""
    global _conn
    if _conn is None:
        _conn = db.get_conn()
    return _conn


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid search in Postgres (pgvector) combining dense and sparse vectors via RRF."""
    dense_vector = embed_dense(query)
    sparse_indices, sparse_values = embed_sparse(query)
    return db.hybrid_search(get_conn(), dense_vector, sparse_indices, sparse_values, top_k=top_k)


def rerank(query: str, chunks: list[dict], top_n: int = TOP_RERANK) -> list[dict]:
    """Rerank retrieved chunks using the vLLM cross-encoder score endpoint."""
    scored = []
    for chunk in chunks:
        response = requests.post(
            RERANK_API_URL,
            json={
                "model": RERANK_MODEL_NAME,
                "text_1": query,
                "text_2": chunk["text"],
            },
            timeout=60,
        )
        response.raise_for_status()
        score = response.json()["data"][0]["score"]
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]


def citation(chunk: dict) -> str:
    """Source line for one passage, built from its stored metadata.

    Every passage gets one, whatever produced it -- the old version only added a
    citation for non-table chunks, on the assumption that table text already
    carried its own header. That held for tables from the verified image
    pipeline but NOT for tables parsed out of a native PDF, which reached the
    model with no source at all.

    The section path is included because it is the strongest disambiguator in a
    long report, where similar tables and figures repeat across sections.
    """
    parts = [str(chunk.get("filename") or "?"), f"Sayfa {chunk.get('page', 0)}"]

    headings = chunk.get("headings") or []
    if headings:
        parts.append(" > ".join(str(h) for h in headings))

    table_data = chunk.get("table_data") or {}
    confidence = table_data.get("confidence")
    if confidence is not None:
        # a verified table: tell the model how much to trust the numbers
        parts.append(f"tablo, guven {confidence:.2f}")
    elif chunk.get("type") == "table":
        parts.append("tablo")

    return "[" + " | ".join(parts) + "]"


def build_context(chunks: list[dict], numbered: bool = False) -> str:
    """The passages as the model sees them.

    `numbered` puts a [P1], [P2] handle on each one so an answer can say which
    passage it used. Without a handle the only thing checkable afterwards is
    whether a figure appears SOMEWHERE in fifteen passages, and measurement
    showed that set is large enough to cover values the answer got wrong.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        head = f"[P{i}] {citation(chunk)}" if numbered else citation(chunk)
        blocks.append(f"{head}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


def ask(question: str, structured: bool = False) -> str:
    """The whole path for this engine: retrieve, rank, assemble, answer.

    The composition lives with the retrieval it belongs to rather than in the
    generation module, because each engine composes differently -- this one
    ranks with a cross-encoder first, the LlamaIndex engine does not.
    """
    chunks  = retrieve(question)
    chunks  = rerank(question, chunks)
    context = build_context(chunks, numbered=structured)
    generate = gen.generate_structured if structured else gen.generate
    return generate(question, context)


# --- Interactive loop ---
if __name__ == "__main__":
    print("RAG Pipeline hazır. Çıkmak için 'quit' yaz.\n")
    while True:
        question = input("Soru: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print("\nAranıyor ve rerank ediliyor...")
        answer = ask(question)
        print(f"\nCevap:\n{answer}\n")
        print("-" * 60)

        