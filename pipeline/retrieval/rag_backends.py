"""Pluggable question-answering engines, the same idea the table side already uses.

`router.TABLE_BACKENDS` lets a table engine be swapped by setting one variable,
because every engine answers the same contract. Retrieval had no such seam: one
implementation was wired straight into `query.py`, so trying a second approach
meant editing the one that works.

A backend is two callables:

    retrieve(question, top_k) -> [chunk dicts]   what would go in front of the model
    answer(question)          -> str             the full question-to-answer path

A chunk dict carries at least `text`, `page` and `filename`, which is what
`query.build_context` and the eval harness read. Keeping that shape means one
harness measures every backend against the same questions -- the point of having
alternatives at all is being able to compare them on evidence rather than taste.
"""
import os

RAG_BACKEND = os.getenv("RAG_BACKEND", "native").lower()


def _native():
    """The pipeline built here: pgvector hybrid search, RRF, cross-encoder."""
    from pipeline.retrieval import query
    return query.retrieve, query.ask


def _llamaindex():
    """LlamaIndex over the same chunks, copied into a table of its own shape, so
    a comparison isolates the strategy rather than also changing the text."""
    from pipeline.retrieval import rag_llamaindex
    return rag_llamaindex.retrieve, rag_llamaindex.answer


# Imported lazily: a backend's dependencies should not be needed by anyone who
# is not using it, which is what keeps an optional engine genuinely optional.
BACKENDS = {
    "native": _native,
    "llamaindex": _llamaindex,
}


def get(name=None):
    """Return (retrieve, answer) for a backend."""
    name = (name or RAG_BACKEND).lower()
    if name not in BACKENDS:
        raise ValueError(f"bilinmeyen RAG_BACKEND '{name}' (secenekler: {sorted(BACKENDS)})")
    return BACKENDS[name]()


def retrieve(question, top_k=None, backend=None):
    fn, _ = get(backend)
    return fn(question) if top_k is None else fn(question, top_k)


def answer(question, backend=None):
    _, fn = get(backend)
    return fn(question)
