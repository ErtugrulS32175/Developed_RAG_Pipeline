CREATE EXTENSION IF NOT EXISTS vector;

-- fastembed's Qdrant/bm25 has no fixed vocabulary: it hashes tokens with
-- MurmurHash3 (abs() of a signed 32-bit hash), giving raw indices up to
-- ~2.15 billion -- past pgvector's sparsevec dimension cap (1e9). Every
-- sparse index is remapped via `(raw_id % SPARSE_DIM) + 1` before it
-- reaches this table (see embeddings.py). SPARSE_DIM must match exactly
-- between this column and that remap function.
CREATE TABLE IF NOT EXISTS chunks (
    id          uuid PRIMARY KEY,
    type        text NOT NULL,
    text        text NOT NULL,
    source_tag  text NOT NULL,
    page        integer NOT NULL DEFAULT 0,
    headings    jsonb NOT NULL DEFAULT '[]'::jsonb,
    table_data  jsonb,
    dense       vector(1024) NOT NULL,
    sparse      sparsevec(999999937) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- No ANN index yet: this is a small, single-document demo dataset, and a
-- sequential scan over `dense <=> query` / `sparse <#> query` is effectively
-- instant at this scale. Add an HNSW index (e.g. `USING hnsw (dense
-- vector_cosine_ops)`) once the corpus grows large enough for that to matter.
