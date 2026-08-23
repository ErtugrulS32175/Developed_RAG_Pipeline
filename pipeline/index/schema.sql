CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested source file. Looked up by filename so re-ingesting
-- the same file reuses its id instead of creating a duplicate document.
CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY,
    filename    text NOT NULL UNIQUE,
    file_type   text NOT NULL,
    uploaded_at timestamptz NOT NULL DEFAULT now(),
    status      text NOT NULL DEFAULT 'pending'
);

-- fastembed's Qdrant/bm25 has no fixed vocabulary: it hashes tokens with
-- MurmurHash3 (abs() of a signed 32-bit hash), giving raw indices up to
-- ~2.15 billion -- past pgvector's sparsevec dimension cap (1e9). Every
-- sparse index is remapped via `(raw_id % SPARSE_DIM) + 1` before it
-- reaches this table (see embeddings.py). SPARSE_DIM must match exactly
-- between this column and that remap function.
CREATE TABLE IF NOT EXISTS chunks (
    id          uuid PRIMARY KEY,
    document_id uuid REFERENCES documents(id),
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

-- Safety net for the pre-existing dev table: CREATE TABLE IF NOT EXISTS
-- above won't add this column to an already-created chunks table.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_id uuid REFERENCES documents(id);

-- Document-scoped retrieval filters this column before ranking. Measured on a
-- real PostgreSQL 17 + pgvector server with 10,002 chunks and a 1% scope: the
-- unindexed plan scanned all 10,002 rows (estimated cost 322), while this
-- index selected the 101 scoped rows directly (estimated cost 29). Unscoped
-- retrieval kept its sequential plan, so the index narrows only the new path.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

-- What the filename cannot say: WHICH bytes this row represents. Documents
-- are keyed by filename, so two different files sharing a basename used to
-- merge silently -- and the second ingest then deleted the first file's
-- chunks as "stale". The hash lets the upsert refuse that collision instead.
-- NULL means a legacy row ingested before the column existed. A legacy row
-- that still SERVES chunks is fail-closed in the upsert gate: different
-- content needs explicit replace authority, because "I cannot compare"
-- must never read as "therefore overwrite".
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_sha256 text;

-- Why a status says what it says: for 'partial', WHICH pages/tables were
-- lost and to what error. A console line scrolls away; this row does not.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status_note text;

-- Versioned ingest, immutable generations. Every ingest ATTEMPT allocates
-- its own generation number from last_generation (atomic increment), stages
-- rows under it, and only a COMPLETE run promotes -- a compare-and-swap on
-- active_generation plus the stale delete in one transaction. Rows are
-- never MOVED between generations (an early adopt-by-UPDATE stole the
-- active version's own rows into staging and blinded retrieval to them);
-- reuse copies. active_content_sha records WHICH bytes the served
-- generation came from; content_sha256 is merely the last candidate.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS active_generation integer NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_generation integer NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS active_content_sha text;
-- The IMMUTABLE identity of the recorded candidate. A hash alone is not an
-- identity: an audited race had an old ingest re-recording the OLD bytes'
-- hash over a newly authorised upload's candidate and then promoting -- the
-- disk, the candidate column and the served index ended on three different
-- stories, every step reporting success. Each accepted knock that CHANGES
-- the candidate bytes mints a new id; promotion and status stamps
-- compare-and-swap on it, so a run bound to a superseded candidate fails
-- loudly instead of quietly reverting someone else's authority.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS candidate_id uuid;
-- WHERE that candidate is in its publication: 'staged' means the row
-- knows about bytes that are not on disk yet, 'published' means row and
-- disk agree. A process request in the gap between the two used to read
-- a candidate whose bytes were not published, refuse (correctly), and
-- mark the document error -- while the upload returned 200 pending. Two
-- truthful answers, one contradictory record. NULL is a legacy row and
-- is fail-closed: what cannot be verified is not processable.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS candidate_state text;
-- The LEASE. One attempt at a time indexes a document, and attempt_id is
-- the fencing token every write of that run compare-and-swaps on. The
-- expiry is compared against the DATABASE clock -- a worker's own clock
-- has no authority over a lease it holds. attempt_owner names the worker
-- for an operator reading the row; it is never authority.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS attempt_id uuid;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS attempt_owner text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS attempt_expires_at timestamptz;

-- One row per indexing RUN. The system had an identity for a document,
-- for a name and for a content version, and none for "this attempt to
-- index that content" -- so a losing run's verdict could land on a
-- healthy served document, and two runs of one candidate were
-- indistinguishable to every guard. status is NULL while the attempt is
-- running and terminal afterwards (done/error/partial/superseded);
-- SUPERSEDED is written by the SYSTEM at takeover or fencing, never by
-- the displaced worker, which from that moment may write nothing.
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id      uuid PRIMARY KEY,
    document_id     uuid NOT NULL REFERENCES documents(id),
    candidate_id    uuid NOT NULL,
    candidate_sha   text NOT NULL,
    observed_active integer NOT NULL,
    owner           text,
    status          text,
    note            text,
    started_at      timestamptz NOT NULL DEFAULT now(),
    ended_at        timestamptz
);

-- The same rule the code carries, carried by the database: a guard that
-- lives only in Python is one forgotten caller away from being no guard,
-- and an unknown terminal value would make every later reader guess.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'attempts'::regclass
          AND conname = 'attempts_status_gecerli'
    ) THEN
        ALTER TABLE attempts ADD CONSTRAINT attempts_status_gecerli
            CHECK (status IS NULL
                   OR status IN ('done', 'error', 'partial', 'superseded'));
    END IF;
END
$$;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS generation integer NOT NULL DEFAULT 0;
-- content_key identifies WHAT a chunk is (doc|tag|index|text) independent of
-- WHICH generation carries it: embedding reuse looks rows up by it, and the
-- promotion manifest is stated in it.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_key uuid;
-- ...and embedding_fingerprint identifies WHICH CONFIGURATION produced the
-- row's vectors (dense model + truncation cap + sparse model + language).
-- The content key speaks only about the text, and a probe that changed the
-- embedding model saw stale vectors copied into the new generation on a
-- text match alone. Reuse requires an EXACT fingerprint match; NULL means
-- a legacy row whose vectors are re-embedded, never trusted.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_fingerprint text;

-- No ANN index yet: this is a small, single-document demo dataset, and a
-- sequential scan over `dense <=> query` / `sparse <#> query` is effectively
-- instant at this scale. Add an HNSW index (e.g. `USING hnsw (dense
-- vector_cosine_ops)`) once the corpus grows large enough for that to matter.
