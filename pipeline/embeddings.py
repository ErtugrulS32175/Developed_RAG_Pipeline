import os

import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding

load_dotenv()

EMBED_API_URL    = os.getenv("EMBED_API_URL", "http://localhost:8011/v1/embeddings")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_dense(text: str) -> list[float]:
    """Call the vLLM embedding server for a dense vector."""
    r = requests.post(EMBED_API_URL, json={"model": EMBED_MODEL_NAME, "input": text[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def embed_sparse(text: str):
    """Compute a BM25 sparse vector locally via FastEmbed. Returns (indices, values)."""
    result = list(sparse_model.embed([text]))[0]
    return result.indices.tolist(), result.values.tolist()
