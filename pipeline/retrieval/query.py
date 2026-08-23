import os
from dataclasses import replace

import requests
from dotenv import load_dotenv

from pipeline.generation import answer as gen
from pipeline.index import db
from pipeline.index.embeddings import embed_dense, embed_sparse
from pipeline.retrieval.context import Passage, RagContext
from pipeline.retrieval.trace import TraceStage, clock, elapsed_ms, new_trace

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

def retrieve(query: str, top_k: int = TOP_K, *,
             document_ids=None) -> list[dict]:
    """Hybrid search in Postgres (pgvector) combining dense and sparse vectors via RRF.

    Borrows a pooled connection per query instead of caching one at module
    level: the cached connection had no rollback and no reconnect, so a single
    failed statement (or a server-side kill) took down every later chat request
    until the process restarted. The pool revalidates on checkout and nothing
    connects until the first query, so importing this module still needs no
    database.

    `document_ids` narrows the search to a named set of documents. It is
    keyword-only with a default, so the existing positional call sites --
    including the evaluation harness -- keep their exact meaning. The scope
    is handed to the query seam rather than applied to its result: both
    rankings are drawn from the scope, so the fusion never sees a candidate
    from outside it."""
    dense_vector = embed_dense(query)
    sparse_indices, sparse_values = embed_sparse(query)
    with db.get_pool().connection() as conn:
        return db.hybrid_search(conn, dense_vector, sparse_indices,
                                sparse_values, top_k=top_k,
                                document_ids=document_ids)


def rerank(query: str, chunks: list[dict], top_n: int = TOP_RERANK) -> list[dict]:
    """Rerank retrieved chunks using the vLLM cross-encoder score endpoint.

    One request for all candidates: vLLM's /score accepts ``text_2`` as a
    list and returns explicitly indexed scores. The old loop paid fifteen
    HTTP round-trips per question on every chat request.

    Two independent comparisons against the live service, ten saved contexts
    each, agreed on the shape and differed on the magnitude: top-1 identical
    10/10 both times, full ordering 9/10 both times (one adjacent pair
    swapped), largest score difference 0.0013 in one run and 0.0020 in the
    other -- batched-kernel numerics, visible only at near-ties.

    What that does and does not establish: with the shipped default nothing
    is cut, because TOP_RERANK defaults to TOP_K, so a near-tie swap moves a
    passage within the prompt and the SET reaching the model is unchanged.
    Set RAG_TOP_RERANK below the candidate count and this function does cut,
    and there a swap across the cut-off changes membership. Even with the
    set fixed, prompt order is an input to generation: this is measured
    equivalence of ranking, not of answers."""
    if not chunks:
        return []
    response = requests.post(
        RERANK_API_URL,
        json={
            "model": RERANK_MODEL_NAME,
            "text_1": query,
            "text_2": [chunk["text"] for chunk in chunks],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()["data"]
    # Scores are bound through the service's OWN index field, never through
    # response order: nothing in the API contract promises the list comes
    # back in request order, and binding by position would silently attach
    # scores to the wrong chunks the day it does not.
    indices = [item.get("index") for item in data]
    if sorted(indices) != list(range(len(chunks))):
        raise ValueError("rerank service response is not a permutation "
                         "of the request")
    scored = sorted(
        ((item["score"], item["index"]) for item in data),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [chunks[index] for _, index in scored[:top_n]]


def _trusted_page(chunk: dict) -> int | None:
    """A page number fit for provenance, not merely for display."""
    value = chunk.get("page")
    return value if type(value) is int and value > 0 else None


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
    page = _trusted_page(chunk)
    parts = [str(chunk.get("filename") or "?"), f"Sayfa {page or 0}"]

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


def build_rag_context(chunks: list[dict], numbered: bool = False) -> RagContext:
    """Build model-visible text and trusted provenance from the same chunks.

    `numbered` puts a [P1], [P2] handle on each one so an answer can say which
    passage it used. Without a handle the only thing checkable afterwards is
    whether a figure appears SOMEWHERE in fifteen passages, and measurement
    showed that set is large enough to cover values the answer got wrong.

    The rendered text is deliberately never parsed back into provenance.
    Document content may contain the separator, a fake handle or a citation
    line; none of those can alter the tuple built directly from ``chunks``.
    """
    passages = []
    for i, chunk in enumerate(chunks, start=1):
        raw_text = chunk.get("text")
        text = "" if raw_text is None else str(raw_text)
        passages.append(Passage(
            handle=i,
            page=_trusted_page(chunk),
            text=text,
            citation=citation(chunk),
        ))
    return RagContext(
        passages=tuple(passages),
        numbered=numbered,
    )


def build_context(chunks: list[dict], numbered: bool = False) -> str:
    """Backward-compatible model-visible text.

    Validation callers must keep the full ``RagContext`` returned by
    ``build_rag_context`` instead of passing this string back to a parser.
    Numbered text is refused here because discarding its paired provenance is
    precisely the unsafe legacy shape this boundary removes.
    """
    if numbered:
        raise ValueError("numbered context requires build_rag_context")
    return build_rag_context(chunks).model_text


def ask(question: str, structured: bool = False) -> str:
    """The whole path for this engine: retrieve, rank, assemble, answer.

    The composition lives with the retrieval it belongs to rather than in the
    generation module, because each engine composes differently -- this one
    ranks with a cross-encoder first, the LlamaIndex engine does not.
    """
    chunks  = retrieve(question)
    chunks  = rerank(question, chunks)
    context = build_rag_context(chunks, numbered=structured)
    generate = gen.generate_structured if structured else gen.generate
    return generate(question, context.model_text)


def ask_checked(question: str, *, document_ids=None):
    """Generate a structured answer and return its publication decision.

    This is the product path. The plain ``ask`` function remains available for
    historical comparisons, but a public caller must not receive its unchecked
    string. Keeping the ``RagContext`` beside the exact model-visible text also
    prevents validation against different retrieval results.

    `document_ids` scopes the retrieval this answer is built from, and it is
    the ONLY thing it scopes: reranking, context assembly and provenance all
    read the chunks retrieval returned, so a scope applied at the query is
    already a scope on the reranker's candidates, on the assembled context
    and on every citation the answer can carry.
    """
    from pipeline.validation.rag.answer_guard import validate_structured

    # Forwarded ONLY when a scope was asked for. An absent scope must reach
    # `retrieve` as the single-argument call it has always been -- callers
    # that replace this seam (the evaluation harness, the structured-answer
    # tests) declare the signature they have always been handed.
    scope = {} if document_ids is None else {"document_ids": document_ids}
    started = clock()
    chunks = retrieve(question, **scope)
    stages = [TraceStage("retrieve", elapsed_ms(started))]
    retrieved_count = len(chunks)
    started = clock()
    chunks = rerank(question, chunks)
    stages.append(TraceStage("rerank", elapsed_ms(started)))
    reranked_count = len(chunks)
    started = clock()
    context = build_rag_context(chunks, numbered=True)
    stages.append(TraceStage("context", elapsed_ms(started)))
    started = clock()
    reply = gen.generate_structured(question, context.model_text)
    stages.append(TraceStage("generate", elapsed_ms(started)))
    started = clock()
    result = validate_structured(reply, context)
    stages.append(TraceStage("validate", elapsed_ms(started)))
    trace = new_trace(
        backend="native",
        scope_document_count=(None if document_ids is None
                              else len(document_ids)),
        retrieved_count=retrieved_count,
        reranked_count=reranked_count,
        context_passage_count=len(context.passages),
        stages=stages,
    )
    return replace(result, trace=trace)


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
