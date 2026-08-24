"""LlamaIndex as an alternative question-answering engine.

Runs beside the pipeline built here rather than replacing it -- selected with
`RAG_BACKEND=llamaindex`, or per-conversation through its own model id in
OpenWebUI. Both engines answer the same contract, so `eval/retrieval/rag_eval.py` measures
them against the same question sets and the comparison is a number rather than
an opinion.

What is held constant on purpose: the source chunks, the embedding model and
the LLM are the same for both engines. Only the retrieval and answering
strategy differs. Change more than one thing and a difference in the result
tells you nothing about which change caused it.

LlamaIndex keeps its own table, because its store expects a schema of its own
design and writing our rows into that shape is what makes its retriever usable
at all. `build_index()` copies the chunks across; it re-embeds them, which is
the cost of getting a genuinely independent second opinion.

The package is optional and deliberately not in requirements.txt: nobody using
the default engine should have to install it. Set it up with

    pip install -r requirements-llamaindex.txt
    python -m pipeline.retrieval.rag_llamaindex build
"""
import os
import hashlib
import json

from dotenv import load_dotenv

load_dotenv()

# lowercased on purpose: SQLAlchemy folds unquoted identifiers to lower
# case while our own reset/swap statements quote them exactly -- a
# mixed-case value here would make the store write one table and the
# maintenance statements manage another
TABLE = os.getenv("LLAMAINDEX_TABLE", "llamaindex_chunks").strip().lower()
TOP_K = int(os.getenv("LLAMAINDEX_TOP_K", "15"))


class _PlannedChunks(list):
    """List-compatible result carrying private planner evidence."""

    def __init__(self, chunks, evidence):
        super().__init__(chunks)
        self.evidence = evidence

# The vector table and its coverage manifest are one snapshot.  Retrieval
# holds the shared half from the live-authority read through node projection;
# the build takes the exclusive half only for the final two-table swap.  This
# closes the otherwise possible split where coverage is proved against one
# snapshot and PGVectorStore queries another.
_SNAPSHOT_LOCK = "ragtest-llamaindex-snapshot"
_SNAPSHOT_STALE = (
    "LlamaIndex anlik goruntusu etkin belge surumunu kapsamiyor; sorgu "
    "reddedildi. Eski surumden sonuc dondurulmez."
)


def _scope_table(table=None):
    """Closed companion name for the snapshot's exact authority keys."""
    chosen = TABLE if table is None else table
    return f"data_{chosen}_kapsam"


_MISSING = (
    "LlamaIndex kurulu degil. Bu motoru kullanmak icin:\n"
    "    pip install -r requirements-llamaindex.txt\n"
    "    python -m pipeline.retrieval.rag_llamaindex build"
)


def _require():
    """Imported here rather than at module level so the backend registry stays
    importable -- and the default engine keeps working -- without the package."""
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.vector_stores.types import VectorStoreQueryMode
        from llama_index.embeddings.openai_like import OpenAILikeEmbedding
        from llama_index.llms.openai_like import OpenAILike
        from llama_index.vector_stores.postgres import PGVectorStore
    except ImportError as e:
        raise RuntimeError(f"{_MISSING}\n({e})") from e
    return (Settings, VectorStoreIndex, VectorStoreQueryMode,
            OpenAILikeEmbedding, OpenAILike, PGVectorStore)


# A scoped question this engine CANNOT scope is refused, never widened.
_SCOPE_UNSUPPORTED = (
    "LlamaIndex metadata filtresi bu kurulumda kullanilamiyor; belge "
    "kapsamli sorgu reddedildi. Kapsamsiz bir sonuc kapsamliymis gibi "
    "dondurulmez."
)


def _require_filters():
    """The metadata-filter API, or a refusal -- never an unscoped fallback.

    Imported at the same seam and for the same reason as `_require`: the
    package is optional, so nothing here may be needed by someone using the
    default engine. The difference is what a failure MEANS. A missing
    package means "this engine is unavailable"; a missing FILTER API means
    "this engine cannot answer the question that was asked", and the one
    answer it must never give is the wider one.
    """
    try:
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
    except ImportError as e:
        raise RuntimeError(f"{_SCOPE_UNSUPPORTED}\n({e})") from e
    return FilterOperator, MetadataFilter, MetadataFilters


def _scope_kind(document_ids, collection_ids, tags, resolved):
    if not resolved:
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
        list(document_ids), ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _candidate_utf8_bytes(chunks):
    from pipeline.retrieval.planner import PlannerError

    total = 0
    try:
        for chunk in chunks:
            value = chunk.get("text")
            text = "" if value is None else str(value)
            total += len(text.encode("utf-8", errors="strict"))
    except (AttributeError, UnicodeEncodeError):
        raise PlannerError("planner_candidate_invalid") from None
    return total


def _lifecycle_scope_keys(document_ids, collection_ids=None, tags=None):
    """Hold lifecycle and snapshot authority through node projection.

    A metadata filter cannot distinguish "no matching passage" from "this
    active version was never copied into the snapshot".  The companion
    manifest does: every live four-part key must be present before a retriever
    is constructed.  Missing table, unreadable table and missing key are the
    same closed outcome -- the snapshot cannot prove it covers the request.
    """
    from contextlib import contextmanager

    from psycopg import sql as _sql

    from pipeline.index import db

    @contextmanager
    def locked():
        with db.get_pool().connection() as conn:
            tenant_id, service = db.current_execution_tenant()
            if service:
                raise RuntimeError(
                    "servis baglami kullanici retrieval kapsami olamaz")
            actor_id = db.current_execution_actor()
            identity = {} if actor_id is None else {"actor_id": actor_id}
            db.set_tenant_context(
                conn, tenant_id, service=False, **identity)
            try:
                db.begin_retrieval_snapshot(conn)
                policy_epoch = db.retrieval_policy_epoch(conn)
                dimensions = (document_ids, collection_ids, tags)
                if any(value is not None and len(value) == 0
                       for value in dimensions):
                    resolved_ids = ()
                elif all(value is None for value in dimensions):
                    resolved_ids = db.active_document_ids(conn)
                else:
                    try:
                        resolved_ids = db.resolve_document_scope(
                            conn, document_ids=document_ids,
                            collection_ids=collection_ids, tags=tags)
                    except ValueError:
                        from pipeline.retrieval.planner import PlannerError
                        raise PlannerError(
                            "planner_scope_invalid") from None
                resolved_ids = tuple(sorted(
                    {str(value) for value in resolved_ids}))
                scope_kind = _scope_kind(
                    document_ids, collection_ids, tags, resolved_ids)
                if not resolved_ids:
                    yield ([], resolved_ids, policy_epoch, scope_kind)
                    return
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock_shared(hashtext(%s))",
                        (_SNAPSHOT_LOCK,))
                scope_keys = db.lock_retrieval_scope_keys(conn, resolved_ids)
                if scope_keys:
                    try:
                        with conn.cursor() as cur:
                            cur.execute(
                                _sql.SQL(
                                    "SELECT scope_key FROM {} "
                                    "WHERE scope_key = ANY(%s::text[])").format(
                                        _sql.Identifier(_scope_table())),
                                (list(scope_keys),))
                            covered = {str(row[0]) for row in cur.fetchall()}
                    except Exception:
                        conn.rollback()
                        raise RuntimeError(_SNAPSHOT_STALE) from None
                    if covered != set(scope_keys):
                        conn.rollback()
                        raise RuntimeError(_SNAPSHOT_STALE)
                yield (scope_keys, resolved_ids, policy_epoch, scope_kind)
            finally:
                conn.rollback()
                db.clear_tenant_context(conn)

    return locked()


def _scope_filters(scope_keys):
    """An exact metadata filter over tenant-qualified document authority."""
    FilterOperator, MetadataFilter, MetadataFilters = _require_filters()
    return MetadataFilters(filters=[
        MetadataFilter(key="scope_key", value=list(scope_keys),
                       operator=FilterOperator.IN),
    ])


def _scope_reached(retriever, filters):
    """Did the constructed retriever actually TAKE the filter?

    A kwarg a retriever silently ignores is the exact failure this engine
    must not survive: the query would run unscoped and its result would be
    published as though it had been scoped. Checked against the object that
    was handed over, so "the filter reached construction" is evidence
    rather than an assumption about a version of an optional package.
    """
    for attribute in ("filters", "_filters"):
        carried = getattr(retriever, attribute, None)
        if carried is filters or (carried is not None and carried == filters):
            return True
    return False


def _dsn_parts():
    """Split PG_DSN, which the store wants as separate fields."""
    from urllib.parse import urlparse

    from pipeline.index import db

    u = urlparse(db.PG_DSN)
    return {
        "host": u.hostname or "localhost",
        "port": u.port or 5432,
        "user": u.username or "rag",
        "password": u.password or "",
        "database": (u.path or "/ragdb").lstrip("/"),
    }


def _configure_models():
    """Point LlamaIndex at the same embedding model and LLM the other engine uses.

    Holding these constant is what makes the comparison mean anything: if the
    two engines answered through different models, a difference in the result
    would say nothing about the retrieval strategy being compared.
    """
    (Settings, _, _, OpenAILikeEmbedding, OpenAILike, _) = _require()

    from pipeline.generation import answer as gen
    from pipeline.index import embeddings

    Settings.embed_model = OpenAILikeEmbedding(
        model_name=embeddings.EMBED_MODEL_NAME,
        api_base=embeddings.EMBED_API_URL.rsplit("/v1/", 1)[0] + "/v1",
        api_key=os.getenv("EMBED_API_KEY", "not-needed"),
    )
    Settings.llm = OpenAILike(
        model=gen.LLM_MODEL_NAME,
        api_base=gen.LLM_API_URL.rsplit("/v1/", 1)[0] + "/v1",
        api_key=os.getenv("LLM_API_KEY", "not-needed"),
        is_chat_model=True,
        temperature=0.1,
    )


def _store(table_name: str | None = None):
    """LlamaIndex's own table. It expects a schema of its own design, so the
    chunks are copied into that shape rather than read from ours in place."""
    (_, _, _, _, _, PGVectorStore) = _require()
    return PGVectorStore.from_params(
        **_dsn_parts(),
        table_name=table_name or TABLE,
        # bge-m3; must match what the embedding service returns
        embed_dim=int(os.getenv("EMBED_DIM", "1024")),
        hybrid_search=True,
        # Postgres ships no Turkish text-search configuration, and "simple" does
        # no stemming at all rather than applying another language's rules
        text_search_config="simple",
    )


def _index():
    (_, VectorStoreIndex, VectorStoreQueryMode, _, _, _) = _require()
    _configure_models()
    return VectorStoreIndex.from_vector_store(_store()), VectorStoreQueryMode


def build_index():
    """Copy the chunks this pipeline produced into LlamaIndex's own table.

    Reads from `chunks` rather than re-parsing the PDFs so both engines answer
    from identical text -- the comparison is about retrieval, not about who
    parses a document better. That is worth measuring too, but separately.

    The build is SHADOW-FIRST with an atomic swap: an earlier version
    dropped the live table before building, so a build that failed midway
    left NO comparison index at all -- the healthy old snapshot died for
    a new one that never arrived. Now the copy lands in a shadow table;
    only after the write succeeds and the shadow verifies non-empty does
    one transaction drop the old table and rename the shadow into place.
    A failed build leaves the previous snapshot serving, untouched.
    The vector table and its scope manifest swap under one exclusive advisory
    lock.  Queries hold the shared half from live-scope resolution through
    node projection, so a query cannot prove coverage against one snapshot and
    then read another.
    """
    from llama_index.core import Document, StorageContext, VectorStoreIndex

    from psycopg import sql as _sql

    from pipeline.index import db

    _configure_models()
    # PGVectorStore prefixes its table with "data_"; the shadow gets the
    # same treatment, so the swap below renames data_<shadow> onto
    # data_<table>.
    shadow = f"{TABLE}_kurulum"
    store = _store(shadow)
    shadow_scope = _scope_table(shadow)
    live_scope = _scope_table()

    # A snapshot contains every tenant and is never itself an authorization
    # boundary.  Query-time RLS-derived scope keys are.  Only the internal
    # service connection may perform this cross-tenant copy.
    conn = db.get_conn(service=True)
    try:
        with conn.cursor() as cur:
            # a dead previous attempt's shadow must not pollute this build
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(
                _sql.Identifier(f"data_{shadow}")))
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(
                _sql.Identifier(shadow_scope)))
        conn.commit()
        with conn.cursor() as cur:
            # Every retained READY version is copied, not merely the version
            # active at build time.  That is what lets a later rollback change
            # only the live four-part filter instead of requiring an index
            # rebuild.  The second arm preserves the one safe legacy shape:
            # its active generation has no version identity because the old
            # system never recorded one.  Staging/partial generations match
            # neither arm.
            cur.execute(
                "SELECT c.text, c.page, c.type, d.filename, c.id::text, "
                "d.tenant_id::text || ':' || d.id::text || ':' || "
                "c.version_id::text || ':' || c.generation::text "
                "AS scope_key "
                "FROM document_version_builds b "
                "JOIN chunks c ON c.tenant_id = b.tenant_id "
                "AND c.document_id = b.document_id "
                "AND c.version_id = b.version_id "
                "AND c.generation = b.generation "
                "JOIN documents d ON d.tenant_id = b.tenant_id "
                "AND d.id = b.document_id "
                "UNION ALL "
                "SELECT c.text, c.page, c.type, d.filename, c.id::text, "
                "d.tenant_id::text || ':' || d.id::text || ':legacy:' || "
                "c.generation::text AS scope_key "
                "FROM chunks c JOIN documents d "
                "ON c.tenant_id = d.tenant_id AND c.document_id = d.id "
                "WHERE c.version_id IS NULL "
                "AND d.active_version_id IS NULL "
                "AND c.generation = d.active_generation"
            )
            rows = cur.fetchall()

        docs = [
            Document(text=text,
                     metadata={"page": page, "type": ctype,
                               "filename": filename,
                               "chunk_id": chunk_id,
                               "scope_key": scope_key})
            for text, page, ctype, filename, chunk_id, scope_key in rows
        ]
        print(f"[LLAMAINDEX] {len(docs)} chunk golge tabloya aktariliyor "
              f"({len(rows)} satir); eski indeks takasa kadar hizmette")
        VectorStoreIndex.from_documents(
            docs,
            storage_context=StorageContext.from_defaults(vector_store=store),
            show_progress=True,
        )

        with conn.cursor() as cur:
            cur.execute(_sql.SQL("SELECT count(*) FROM {}").format(
                _sql.Identifier(f"data_{shadow}")))
            built = int(cur.fetchone()[0])
        if docs and built == 0:
            raise RuntimeError(
                "golge tablo bos kaldi; takas yapilmadi, eski indeks "
                "hizmette")
        scope_keys = sorted({str(row[5]) for row in rows})
        with conn.cursor() as cur:
            cur.execute(_sql.SQL(
                "CREATE TABLE {} (scope_key text PRIMARY KEY)").format(
                    _sql.Identifier(shadow_scope)))
            cur.executemany(
                _sql.SQL("INSERT INTO {} (scope_key) VALUES (%s)").format(
                    _sql.Identifier(shadow_scope)),
                [(scope_key,) for scope_key in scope_keys])
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (_SNAPSHOT_LOCK,))
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(
                _sql.Identifier(f"data_{TABLE}")))
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(
                _sql.Identifier(live_scope)))
            cur.execute(_sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                _sql.Identifier(f"data_{shadow}"),
                _sql.Identifier(f"data_{TABLE}")))
            cur.execute(_sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                _sql.Identifier(shadow_scope),
                _sql.Identifier(live_scope)))
        conn.commit()
        print("[LLAMAINDEX] tamam: golge tablo atomik takasla hizmete girdi")
    finally:
        conn.close()


def _as_chunks(nodes, allowed_scope_keys):
    """Map LlamaIndex nodes back to the shape the rest of the project uses, so
    the same context builder and the same eval harness work unchanged."""
    out = []
    allowed = frozenset(allowed_scope_keys)
    for n in nodes:
        meta = n.metadata or {}
        scope_key = meta.get("scope_key")
        if type(scope_key) is not str or scope_key not in allowed:
            raise RuntimeError(
                "LlamaIndex yetki kapsami disinda dugum dondurdu")
        out.append({
            "id": meta.get("chunk_id"),
            "text": n.get_content(),
            "page": meta.get("page", 0),
            "type": meta.get("type", "text"),
            "filename": meta.get("filename"),
            "headings": [],
            "table_data": None,
        })
    return out


def _authorized_retrieve(question, *, top_k, document_ids, collection_ids,
                         tags, planned):
    from pipeline.retrieval import planner
    from pipeline.retrieval.trace import clock, elapsed_ms

    if planned:
        planner.query_class(question)
    plan_started = clock()
    with _lifecycle_scope_keys(
            document_ids, collection_ids, tags) as authority:
        scope_keys, resolved_ids, policy_epoch, scope_kind = authority
        plan = None
        if planned:
            plan = planner.build_plan(
                question, backend="llamaindex", scope_kind=scope_kind,
                policy_epoch=policy_epoch,
                scope_digest=_scope_digest(resolved_ids))
            planner.verify_query(plan, question)
            top_k = plan.budget.top_k
        plan_ms = elapsed_ms(plan_started)
        retrieve_started = clock()
        if not scope_keys:
            chunks = []
        else:
            index, VectorStoreQueryMode = _index()
            _require_filters()
            scope = _scope_filters(scope_keys)
            retriever = index.as_retriever(
                similarity_top_k=top_k,
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                filters=scope,
            )
            if not _scope_reached(retriever, scope):
                raise RuntimeError(_SCOPE_UNSUPPORTED)
            chunks = _as_chunks(retriever.retrieve(question), scope_keys)
        retrieve_ms = elapsed_ms(retrieve_started)
        return (chunks, plan, len(resolved_ids), scope_kind,
                policy_epoch, plan_ms, retrieve_ms)


def retrieve(question, top_k=TOP_K, *, document_ids=None,
             collection_ids=None, tags=None):
    """Retrieve active documents, optionally scoped to a requested set.

    THE SCOPE IS PART OF THE QUERY. It is resolved to tenant-qualified keys, turned
    into a metadata filter and handed to the retriever AT CONSTRUCTION, so
    the store answers the scoped question itself. Nothing is filtered out
    of the returned node list: that would answer a scoped question with
    whatever survived an UNSCOPED top-k, and a document that really holds
    the answer would come back empty whenever other documents filled the
    pool first.

    The vector table is a snapshot but archive/restore is live metadata, so
    an otherwise unscoped query still resolves the current active scope keys
    and filters at construction. Legacy snapshot nodes without a document
    authority are excluded fail-closed on this comparison engine.

    Four fail-closed outcomes, all before a node is fetched: the filter API
    is unavailable, the active-version key is absent from the snapshot, the
    retriever did not take the filter, or the authority resolves to no active
    document. Only the last returns an empty result; an uncovered active key
    is a stale snapshot and therefore a refusal, not a false empty answer.
    """
    from pipeline.retrieval import planner

    evidence = _authorized_retrieve(
        question, top_k=top_k, document_ids=document_ids,
        collection_ids=collection_ids, tags=tags,
        planned=(top_k == planner.TOP_K))
    return _PlannedChunks(evidence[0], evidence)


def answer(question):
    """Retrieve with LlamaIndex, then answer with THIS project's prompt.

    Reusing this project's own context assembly and prompt keeps the citation
    format and the grounding instructions identical across engines. Otherwise a
    difference in answer quality could just as easily be a difference in
    prompting, and we would not be able to tell.
    """
    from pipeline.generation.answer import generate
    from pipeline.retrieval.query import build_context

    return generate(question, build_context(retrieve(question)))


def answer_checked(question, *, document_ids=None, collection_ids=None,
                   tags=None):
    """Structured, provenance-checked answer for the public API.

    Direct document ids, collection ids and tags scope retrieval and therefore
    everything built from it: assembled context and every citation the answer
    may publish.  Dimensions are forwarded only when supplied, so an unscoped
    question reaches ``retrieve`` as its historical single-argument call."""
    from dataclasses import replace

    from pipeline.generation.answer import generate_structured
    from pipeline.retrieval.query import build_rag_context
    from pipeline.retrieval.trace import (
        TraceStage, clock, elapsed_ms, new_trace,
    )
    from pipeline.validation.rag.answer_guard import validate_structured

    from pipeline.retrieval import planner

    if TOP_K != planner.TOP_K:
        raise planner.PlannerError("planner_runtime_policy_mismatch")
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
            question, backend="llamaindex", scope_kind=scope_kind,
            policy_epoch=policy_epoch, scope_digest=digest)
        plan_ms = 0
        retrieve_ms = elapsed_ms(retrieve_started)
    stages = [
        TraceStage("plan", plan_ms),
        TraceStage("retrieve", retrieve_ms),
    ]
    if _candidate_utf8_bytes(chunks) > plan.budget.candidate_utf8_max:
        raise planner.PlannerError("planner_candidate_limit")
    started = clock()
    context = build_rag_context(chunks, numbered=True)
    stages.append(TraceStage("context", elapsed_ms(started)))
    context_utf8_bytes = len(context.model_text.encode("utf-8"))
    if context_utf8_bytes > plan.budget.context_utf8_max:
        raise planner.PlannerError("planner_context_limit")
    started = clock()
    reply = generate_structured(question, context.model_text)
    stages.append(TraceStage("generate", elapsed_ms(started)))
    started = clock()
    result = validate_structured(reply, context)
    stages.append(TraceStage("validate", elapsed_ms(started)))
    return replace(result, trace=new_trace(
        backend="llamaindex",
        planner_policy_version=plan.policy_version,
        query_class=plan.query_class,
        retrieval_mode=plan.mode,
        fallback=plan.fallback,
        scope_kind=scope_kind,
        policy_epoch=policy_epoch,
        top_k=plan.budget.top_k,
        candidate_limit=plan.budget.candidate_limit,
        scope_document_count=scope_count,
        retrieved_count=len(chunks),
        reranked_count=None,
        context_passage_count=len(context.passages),
        context_utf8_bytes=context_utf8_bytes,
        stages=stages,
    ))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        print(__doc__)
