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

from dotenv import load_dotenv

load_dotenv()

# lowercased on purpose: SQLAlchemy folds unquoted identifiers to lower
# case while our own reset/swap statements quote them exactly -- a
# mixed-case value here would make the store write one table and the
# maintenance statements manage another
TABLE = os.getenv("LLAMAINDEX_TABLE", "llamaindex_chunks").strip().lower()
TOP_K = int(os.getenv("LLAMAINDEX_TOP_K", "15"))

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
    Queries issued DURING the swap transaction wait; a comparison run
    still should not race a rebuild -- written down rather than
    pretended away.
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

    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            # a dead previous attempt's shadow must not pollute this build
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(
                _sql.Identifier(f"data_{shadow}")))
        conn.commit()
        with conn.cursor() as cur:
            # The SAME active-generation filter as the native engine's
            # hybrid_search, or the two engines stop answering from the
            # same chunk set: without it this copy swept staging, partial
            # and superseded rows into LlamaIndex while native retrieval
            # saw only the served generation -- and the A/B comparison
            # silently stopped measuring retrieval strategy.
            cur.execute(
                "SELECT c.text, c.page, c.type, d.filename "
                "FROM chunks c LEFT JOIN documents d ON c.document_id = d.id "
                "AND c.generation = d.active_generation "
                "WHERE c.document_id IS NULL OR d.id IS NOT NULL"
            )
            rows = cur.fetchall()

        docs = [
            Document(text=text,
                     metadata={"page": page, "type": ctype,
                               "filename": filename})
            for text, page, ctype, filename in rows
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
        with conn.cursor() as cur:
            cur.execute(_sql.SQL("DROP TABLE IF EXISTS {}").format(
                _sql.Identifier(f"data_{TABLE}")))
            cur.execute(_sql.SQL("ALTER TABLE {} RENAME TO {}").format(
                _sql.Identifier(f"data_{shadow}"),
                _sql.Identifier(f"data_{TABLE}")))
        conn.commit()
        print("[LLAMAINDEX] tamam: golge tablo atomik takasla hizmete girdi")
    finally:
        conn.close()


def _as_chunks(nodes):
    """Map LlamaIndex nodes back to the shape the rest of the project uses, so
    the same context builder and the same eval harness work unchanged."""
    out = []
    for n in nodes:
        meta = n.metadata or {}
        out.append({
            "text": n.get_content(),
            "page": meta.get("page", 0),
            "type": meta.get("type", "text"),
            "filename": meta.get("filename"),
            "headings": [],
            "table_data": None,
        })
    return out


def retrieve(question, top_k=TOP_K):
    index, VectorStoreQueryMode = _index()
    retriever = index.as_retriever(
        similarity_top_k=top_k,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    )
    return _as_chunks(retriever.retrieve(question))


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


def answer_checked(question):
    """Structured, provenance-checked answer for the public API."""
    from pipeline.generation.answer import generate_structured
    from pipeline.retrieval.query import build_rag_context
    from pipeline.validation.rag.answer_guard import validate_structured

    context = build_rag_context(retrieve(question), numbered=True)
    reply = generate_structured(question, context.model_text)
    return validate_structured(reply, context)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        print(__doc__)
