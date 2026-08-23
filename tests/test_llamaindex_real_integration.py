"""The optional engine's real metadata-filter API, when installed."""
import importlib.util
import os

import pytest


GATE = os.getenv("RAGTEST_LLAMA_GATE", "").strip() == "1"
LLAMA_PACKAGE = importlib.util.find_spec("llama_index")
AVAILABLE = (
    LLAMA_PACKAGE is not None
    and importlib.util.find_spec("llama_index.core") is not None
)

if GATE and not AVAILABLE:
    raise RuntimeError(
        "RAGTEST_LLAMA_GATE=1 but the real llama_index.core is unavailable")

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="real llama_index.core is not installed in this environment",
)


def test_the_real_llamaindex_retriever_enforces_the_filename_scope():
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    from pipeline.retrieval import rag_llamaindex

    documents = [
        Document(text="inside answer", metadata={"filename": "inside.pdf"}),
        Document(text="outside answer", metadata={"filename": "outside.pdf"}),
    ]
    index = VectorStoreIndex.from_documents(
        documents, embed_model=MockEmbedding(embed_dim=8))
    filters = rag_llamaindex._scope_filters(["inside.pdf"])
    retriever = index.as_retriever(similarity_top_k=10, filters=filters)

    assert rag_llamaindex._scope_reached(retriever, filters)
    nodes = retriever.retrieve("answer")
    assert [node.metadata["filename"] for node in nodes] == ["inside.pdf"]
    assert type(filters).__module__ == "llama_index.core.vector_stores.types"
