import os
from dataclasses import replace
import hashlib
import json
from uuid import UUID

import requests
from dotenv import load_dotenv

from pipeline.generation import answer as gen
from pipeline.index import db
from pipeline.index.embeddings import embed_dense, embed_sparse
from pipeline.retrieval.context import Passage, RagContext
from pipeline.retrieval import planner
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


class _PlannedChunks(list):
    """List-compatible retrieval result with private settled-plan evidence."""

    def __init__(self, chunks, evidence):
        super().__init__(chunks)
        self.evidence = evidence

def _scope_kind(document_ids, collection_ids, tags, resolved):
    if resolved == ():
        return "empty"
    dimensions = sum(value is not None for value in (
        document_ids, collection_ids, tags))
    if dimensions == 0:
        return "all_visible"
    if dimensions > 1:
        return "intersection"
    if document_ids is not None:
        return "explicit_documents"
    return "metadata_filters"


def _scope_digest(document_ids):
    raw = json.dumps(
        (None if document_ids is None else list(document_ids)),
        ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _candidate_utf8_bytes(chunks):
    total = 0
    try:
        for chunk in chunks:
            value = chunk.get("text")
            text = "" if value is None else str(value)
            total += len(text.encode("utf-8", errors="strict"))
    except (AttributeError, UnicodeEncodeError):
        raise planner.PlannerError("planner_candidate_invalid") from None
    return total


def _resolve_scope(conn, *, document_ids, collection_ids, tags):
    """Resolve one immutable scope inside the retrieval transaction."""
    dimensions = (document_ids, collection_ids, tags)
    if any(value is not None and len(value) == 0 for value in dimensions):
        resolved = ()
    elif all(value is None for value in dimensions):
        # Native PostgreSQL retrieval inherits tenant/actor RLS directly.
        # Preserve its historical unscoped statement while still binding it
        # to this request's repeatable-read authority window.
        resolved = None
    else:
        try:
            resolved = db.resolve_document_scope(
                conn, document_ids=document_ids,
                collection_ids=collection_ids, tags=tags)
        except ValueError:
            raise planner.PlannerError("planner_scope_invalid") from None
    if resolved is not None:
        resolved = tuple(sorted({str(value) for value in resolved}))
    return resolved, _scope_kind(
        document_ids, collection_ids, tags, resolved)


def _authorized_retrieve(query, *, top_k, document_ids, collection_ids,
                         tags, planned):
    """Bind policy, scope and both rankings to one database snapshot."""
    if planned:
        # The caller-owned query is bounded before a connection, embedding or
        # retrieval service can be touched.  The settled plan below rechecks
        # and digest-binds the same bytes once policy/scope facts are known.
        planner.query_class(query)
    plan_started = clock()
    with db.get_pool().connection() as conn:
        tenant_id, service = db.current_execution_tenant()
        if service:
            raise RuntimeError(
                "servis baglami kullanici retrieval kapsami olamaz")
        actor_id = db.current_execution_actor()
        identity = {} if actor_id is None else {"actor_id": actor_id}
        db.set_tenant_context(conn, tenant_id, service=False, **identity)
        try:
            db.begin_retrieval_snapshot(conn)
            policy_epoch = db.retrieval_policy_epoch(conn)
            resolved, scope_kind = _resolve_scope(
                conn, document_ids=document_ids,
                collection_ids=collection_ids, tags=tags)
            plan = None
            if planned:
                plan = planner.build_plan(
                    query, backend="native", scope_kind=scope_kind,
                    policy_epoch=policy_epoch,
                    scope_digest=_scope_digest(resolved))
                planner.verify_query(plan, query)
                top_k = plan.budget.top_k
            plan_ms = elapsed_ms(plan_started)
            retrieve_started = clock()
            if resolved == ():
                chunks = []
            else:
                dense_vector = embed_dense(query)
                sparse_indices, sparse_values = embed_sparse(query)
                chunks = db.hybrid_search(
                    conn, dense_vector, sparse_indices, sparse_values,
                    top_k=top_k, document_ids=resolved)
            retrieve_ms = elapsed_ms(retrieve_started)
            scope_count = None if resolved is None else len(resolved)
            return (chunks, plan, scope_count, scope_kind,
                    policy_epoch, plan_ms, retrieve_ms)
        finally:
            conn.rollback()
            db.clear_tenant_context(conn)


def retrieve(query: str, top_k: int = TOP_K, *, document_ids=None,
             collection_ids=None, tags=None) -> list[dict]:
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
    evidence = _authorized_retrieve(
        query, top_k=top_k, document_ids=document_ids,
        collection_ids=collection_ids, tags=tags,
        planned=(top_k == planner.TOP_K))
    return _PlannedChunks(evidence[0], evidence)


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
    # One item is already totally ordered. Avoid a network round-trip which
    # cannot change either membership or order, and keep the original mapping
    # object rather than manufacturing a second representation of the chunk.
    if len(chunks) == 1:
        return list(chunks[:top_n])
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
        raw_chunk_id = chunk.get("id")
        if raw_chunk_id is None:
            chunk_id = None
        elif isinstance(raw_chunk_id, UUID):
            chunk_id = str(raw_chunk_id)
        elif type(raw_chunk_id) is str:
            # Provenance is an internal database identity.  It is deliberately
            # not rendered into model_text and is never itself authorization.
            # Requiring the canonical UUID shape here prevents a vector-store
            # metadata string from becoming a public evidence capability.
            try:
                parsed_chunk_id = str(UUID(raw_chunk_id))
            except ValueError:
                parsed_chunk_id = None
            chunk_id = (
                raw_chunk_id if parsed_chunk_id == raw_chunk_id else None
            )
        else:
            chunk_id = None
        raw_document_name = chunk.get("filename")
        if raw_document_name is not None and type(raw_document_name) is not str:
            raise TypeError("document name must be text or None")
        document_name = raw_document_name
        passages.append(Passage(
            handle=i,
            page=_trusted_page(chunk),
            text=text,
            citation=citation(chunk),
            chunk_id=chunk_id,
            document_name=document_name,
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


def ask_checked(question: str, *, document_ids=None, collection_ids=None,
                tags=None):
    """Generate a structured answer and return its publication decision.

    This is the product path. The plain ``ask`` function remains available for
    historical comparisons, but a public caller must not receive its unchecked
    string. Keeping the ``RagContext`` beside the exact model-visible text also
    prevents validation against different retrieval results.

    Direct document ids, collection ids and tags scope the retrieval this
    answer is built from.  Reranking, context assembly and provenance all read
    the chunks retrieval returned, so the resolved query scope also governs
    the reranker's candidates, assembled context and every citation.
    """
    from pipeline.validation.rag.answer_guard import validate_structured

    if (TOP_K != planner.TOP_K
            or TOP_RERANK != planner.RERANK_LIMIT):
        raise planner.PlannerError("planner_runtime_policy_mismatch")
    # Forwarded ONLY when a scope was asked for. An absent scope must reach
    # `retrieve` as the single-argument call it has always been -- callers
    # that replace this seam (the evaluation harness, the structured-answer
    # tests) declare the signature they have always been handed.
    planner.query_class(question)
    retrieve_started = clock()
    scope = {
        name: value for name, value in (
            ("document_ids", document_ids),
            ("collection_ids", collection_ids),
            ("tags", tags),
        ) if value is not None
    }
    chunks = retrieve(question, **scope)
    if isinstance(chunks, _PlannedChunks) and chunks.evidence[1] is not None:
        (_same_chunks, plan, scope_count, scope_kind, policy_epoch,
         plan_ms, retrieve_ms) = chunks.evidence
    else:
        # Test/extension replacements of the historical `retrieve(question)`
        # seam remain usable for an unscoped or direct-id call.  Metadata
        # scopes cannot be reconstructed from returned chunks and therefore
        # fail closed rather than receiving invented planner evidence.
        if collection_ids is not None or tags is not None:
            raise RuntimeError("planned retrieval evidence is missing")
        if document_ids is None:
            scope_kind, scope_count, digest = "all_visible", None, None
        elif len(document_ids) == 0:
            scope_kind, scope_count, digest = "empty", 0, _scope_digest(())
        else:
            scope_kind = "explicit_documents"
            scope_count = len(set(document_ids))
            digest = _scope_digest(sorted({str(v) for v in document_ids}))
        policy_epoch = 1
        plan = planner.build_plan(
            question, backend="native", scope_kind=scope_kind,
            policy_epoch=policy_epoch, scope_digest=digest)
        plan_ms = 0
        retrieve_ms = elapsed_ms(retrieve_started)
    stages = [
        TraceStage("plan", plan_ms),
        TraceStage("retrieve", retrieve_ms),
    ]
    retrieved_count = len(chunks)
    if _candidate_utf8_bytes(chunks) > plan.budget.candidate_utf8_max:
        raise planner.PlannerError("planner_candidate_limit")
    started = clock()
    # V1 preserves the historical two-argument seam.  The code-owned default
    # above is first proved equal to the settled plan, so omitting the keyword
    # cannot hand control back to an environment knob.
    chunks = rerank(question, chunks)
    stages.append(TraceStage("rerank", elapsed_ms(started)))
    reranked_count = len(chunks)
    started = clock()
    context = build_rag_context(chunks, numbered=True)
    stages.append(TraceStage("context", elapsed_ms(started)))
    context_utf8_bytes = len(context.model_text.encode("utf-8"))
    if context_utf8_bytes > plan.budget.context_utf8_max:
        raise planner.PlannerError("planner_context_limit")
    started = clock()
    reply = gen.generate_structured(question, context.model_text)
    stages.append(TraceStage("generate", elapsed_ms(started)))
    started = clock()
    result = validate_structured(reply, context)
    stages.append(TraceStage("validate", elapsed_ms(started)))
    trace = new_trace(
        backend="native",
        planner_policy_version=plan.policy_version,
        query_class=plan.query_class,
        retrieval_mode=plan.mode,
        fallback=plan.fallback,
        scope_kind=scope_kind,
        policy_epoch=policy_epoch,
        top_k=plan.budget.top_k,
        candidate_limit=plan.budget.candidate_limit,
        scope_document_count=scope_count,
        retrieved_count=retrieved_count,
        reranked_count=reranked_count,
        context_passage_count=len(context.passages),
        context_utf8_bytes=context_utf8_bytes,
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
