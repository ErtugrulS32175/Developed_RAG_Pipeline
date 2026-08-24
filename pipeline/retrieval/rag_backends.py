"""Pluggable question-answering engines, the same idea the table side already uses.

`router.TABLE_BACKENDS` lets a table engine be swapped by setting one variable,
because every engine answers the same contract. Retrieval had no such seam: one
implementation was wired straight into `query.py`, so trying a second approach
meant editing the one that works.

Built-in backends expose three callables internally:

    retrieve(question, top_k) -> [chunk dicts]   what would go in front of the model
    answer(question)          -> str             the full question-to-answer path
    answer_checked(question)  -> GuardResult     the public, publication-safe path

The checked callable may additionally accept keyword-only document,
collection and tag scope dimensions.  Each is passed only when a caller asks
for it, so an engine that never learned about scoping still answers every
unscoped question exactly as before.

The historical two-callable shape remains valid for retrieval/evaluation
extensions. Such a backend cannot serve the public API until it adds the third
checked callable; failing closed there must not break its existing plain path.

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
    return query.retrieve, query.ask, query.ask_checked


def _llamaindex():
    """LlamaIndex over the same chunks, copied into a table of its own shape, so
    a comparison isolates the strategy rather than also changing the text."""
    from pipeline.retrieval import rag_llamaindex
    return (
        rag_llamaindex.retrieve,
        rag_llamaindex.answer,
        rag_llamaindex.answer_checked,
    )


# Imported lazily: a backend's dependencies should not be needed by anyone who
# is not using it, which is what keeps an optional engine genuinely optional.
BACKENDS = {
    "native": _native,
    "llamaindex": _llamaindex,
}


def _get_all(name=None):
    """Return and validate every callable supplied by one backend."""
    name = (name or RAG_BACKEND).lower()
    if name not in BACKENDS:
        raise ValueError(f"bilinmeyen RAG_BACKEND '{name}' (secenekler: {sorted(BACKENDS)})")
    functions = tuple(BACKENDS[name]())
    if len(functions) not in {2, 3} or any(
            not callable(function) for function in functions):
        raise TypeError(
            f"RAG_BACKEND '{name}' iki veya uc callable dondurmeli"
        )
    return functions


def get(name=None):
    """Return the historical (retrieve, plain-answer) pair for a backend."""
    return _get_all(name)[:2]


def retrieve(question, top_k=None, backend=None):
    fn, _ = get(backend)
    return fn(question) if top_k is None else fn(question, top_k)


def answer(question, backend=None):
    _, fn = get(backend)
    return fn(question)


def answer_checked(question, backend=None, *, document_ids=None,
                   collection_ids=None, tags=None):
    """Return only the status-bearing answer suitable for publication.

    Document ids, collection ids and tags narrow the question.  Each is
    keyword-only and FORWARDED ONLY WHEN SUPPLIED: an unscoped question reaches
    the engine as the one-argument call every backend has always been asked,
    so a two- or three-callable backend written before these parameters existed
    keeps working untouched.  The arity check validates how many callables a
    backend supplies, never their signatures.
    """
    functions = _get_all(backend)
    if len(functions) != 3:
        raise RuntimeError("secilen RAG backend checked answer desteklemiyor")
    fn = functions[2]
    scope = {
        name: value for name, value in (
            ("document_ids", document_ids),
            ("collection_ids", collection_ids),
            ("tags", tags),
        ) if value is not None
    }
    return fn(question, **scope)
