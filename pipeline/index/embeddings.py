import os

import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding

load_dotenv()

EMBED_API_URL    = os.getenv("EMBED_API_URL", "http://localhost:8011/v1/embeddings")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

# fastembed's BM25 defaults to language="english" -- an English stemmer and
# English stopword list. Turkish is agglutinative, so on Turkish text that
# default leaves inflected forms of the same word unmatched ("kitaplarindaki"
# vs "kitaplari") and guts the sparse half of hybrid search. Changing this
# changes the tokens that get indexed, so the corpus must be re-ingested.
BM25_LANGUAGE = os.getenv("BM25_LANGUAGE", "turkish")

_sparse_model = None


def get_sparse_model():
    """Loaded on first use, not at import: importing this module (or anything
    that imports it, like query.py) should not cost a model load."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25", language=BM25_LANGUAGE)
    return _sparse_model


def embed_dense(text: str) -> list[float]:
    """Call the vLLM embedding server for a dense vector."""
    r = requests.post(EMBED_API_URL, json={"model": EMBED_MODEL_NAME, "input": text[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def embed_sparse(text: str):
    """Compute a BM25 sparse vector locally via FastEmbed. Returns (indices, values)."""
    result = list(get_sparse_model().embed([text]))[0]
    return result.indices.tolist(), result.values.tolist()
