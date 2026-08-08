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
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME,
                                            language=BM25_LANGUAGE)
    return _sparse_model


# A guard against blowing the embedding model's window, NOT a routine step.
# The old cap was 2000 characters and SILENT: every chunk longer than that --
# table markdown especially, which bypasses the chunk merger because a table
# is a unit -- had its tail invisible to dense retrieval while BM25 saw all
# of it, and nothing anywhere recorded that it happened. bge-m3's window is
# 8192 TOKENS; Turkish runs well over two characters per token, so the
# default leaves real margin while being four times the old ceiling.
# Characters are a PROXY for tokens, stated as such: the margin is what
# makes the proxy safe, not any equivalence.
_EMBED_CAP_DEFAULT = 8000


def _resolve_embed_cap(raw) -> int:
    """A misconfigured cap must not become a zero cap. EMBED_MAX_CHARS=0
    once truncated every text to EMPTY and shipped blank inputs to the
    embedding service -- a broken knob silently disabling the feature it
    tunes. Anything that is not a positive integer falls back loudly."""
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        cap = -1
    if cap <= 0:
        print(f"[EMBED] uyari: EMBED_MAX_CHARS={raw!r} gecersiz; "
              f"varsayilan {_EMBED_CAP_DEFAULT} kullanilacak")
        return _EMBED_CAP_DEFAULT
    return cap


EMBED_MAX_CHARS = _resolve_embed_cap(os.getenv("EMBED_MAX_CHARS", "8000"))

# The sparse model id is pinned where the fingerprint can see it: it is
# passed to SparseTextEmbedding below and named here once, not twice.
SPARSE_MODEL_NAME = "Qdrant/bm25"


def embedding_fingerprint() -> str:
    """The identity of the VECTORS this configuration produces.

    Reusing a stored vector is only sound when everything that shaped it
    is unchanged -- an audit probe flipped EMBED_MODEL_NAME, BM25_LANGUAGE
    and EMBED_MAX_CHARS in turn and the content key (which speaks only
    about the TEXT) happily copied stale vectors into the new generation.
    Dense model, truncation cap, sparse model and its language all feed
    the fingerprint; rows are reused only on an EXACT match, and legacy
    rows without one are re-embedded rather than trusted."""
    import hashlib

    parts = "|".join((
        "dense", EMBED_MODEL_NAME, str(EMBED_MAX_CHARS),
        "sparse", SPARSE_MODEL_NAME, BM25_LANGUAGE,
    ))
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def embed_dense(text: str) -> list[float]:
    """Call the vLLM embedding server for a dense vector.

    Truncation still exists as a last resort, but it is LOUD: a chunk whose
    tail cannot be embedded is a retrieval blind spot someone should know
    about, not a slicing expression someone has to find."""
    if len(text) > EMBED_MAX_CHARS:
        print(f"[EMBED] uyari: {len(text)} karakter {EMBED_MAX_CHARS} "
              f"karaktere kirpildi; kuyruk yogun vektore girmeyecek")
        text = text[:EMBED_MAX_CHARS]
    r = requests.post(EMBED_API_URL, json={"model": EMBED_MODEL_NAME, "input": text}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def embed_sparse(text: str):
    """Compute a BM25 sparse vector locally via FastEmbed. Returns (indices, values)."""
    result = list(get_sparse_model().embed([text]))[0]
    return result.indices.tolist(), result.values.tolist()
