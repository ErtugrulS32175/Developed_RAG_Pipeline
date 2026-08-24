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

-- Human organisation is independent of ingest identity. Display names keep
-- their spelling; name_key is the one case-folded authority supplied by the
-- application, so case variants cannot create two logical collections/tags.
CREATE TABLE IF NOT EXISTS collections (
    id         uuid PRIMARY KEY,
    name       text NOT NULL,
    name_key   text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tags (
    id         uuid PRIMARY KEY,
    name       text NOT NULL,
    name_key   text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS collection_documents (
    collection_id uuid NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    document_id   uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, document_id)
);

CREATE INDEX IF NOT EXISTS collection_documents_document_idx
    ON collection_documents(document_id);

CREATE TABLE IF NOT EXISTS document_tags (
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id      uuid NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, tag_id)
);

CREATE INDEX IF NOT EXISTS document_tags_tag_idx ON document_tags(tag_id);

-- Durable ingest requests are separate from attempt leases. A job survives an
-- API or worker restart; an attempt remains the fenced authority for writes to
-- one candidate. Only the digest of the caller's idempotency key is persisted.
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id                     uuid PRIMARY KEY,
    document_id            uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    candidate_id           uuid NOT NULL,
    candidate_sha          text NOT NULL CHECK (length(candidate_sha) = 64),
    idempotency_key_sha256 text NOT NULL
                           CHECK (length(idempotency_key_sha256) = 64),
    status                 text NOT NULL DEFAULT 'queued'
                           CHECK (status IN ('queued', 'running', 'succeeded',
                                             'partial', 'failed', 'cancelled')),
    attempt_count          integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    worker_id              text,
    lease_expires_at       timestamptz,
    created_at             timestamptz NOT NULL DEFAULT now(),
    started_at             timestamptz,
    finished_at            timestamptz,
    outcome_note           text CHECK (outcome_note IS NULL OR
                                       length(outcome_note) BETWEEN 1 AND 100),
    CHECK ((status = 'running') =
           (worker_id IS NOT NULL AND lease_expires_at IS NOT NULL)),
    CHECK ((status IN ('succeeded', 'partial', 'failed', 'cancelled')) =
           (finished_at IS NOT NULL)),
    UNIQUE (document_id, idempotency_key_sha256)
);

CREATE UNIQUE INDEX IF NOT EXISTS ingest_jobs_one_active_document_idx
    ON ingest_jobs(document_id)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS ingest_jobs_claim_idx
    ON ingest_jobs(status, created_at, id);

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

-- Reversible document lifecycle. NULL is active; a timestamp is archived.
-- Chunks stay intact so restore is metadata-only, but every retrieval path
-- joins through this authority and excludes archived rows before ranking.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS archived_at timestamptz;

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

-- Tenant isolation is enforced by PostgreSQL, not only by remembering a
-- predicate in each caller. The legacy single-key installation is tenant 1;
-- a request connection sets rag.tenant_id, while the internal queue worker
-- sets rag.service=1 for cross-tenant claiming and then binds the claimed
-- tenant before running ingest.
CREATE OR REPLACE FUNCTION rag_effective_tenant() RETURNS uuid
LANGUAGE sql STABLE AS $tenant$
    SELECT COALESCE(
        NULLIF(current_setting('rag.tenant_id', true), '')::uuid,
        '00000000-0000-0000-0000-000000000001'::uuid
    )
$tenant$;

CREATE OR REPLACE FUNCTION rag_service_access() RETURNS boolean
LANGUAGE sql STABLE AS $service$
    SELECT COALESCE(current_setting('rag.service', true), '') = '1'
$service$;

ALTER TABLE documents ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE collections ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE tags ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE collection_documents ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE document_tags ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();
ALTER TABLE attempts ADD COLUMN IF NOT EXISTS tenant_id uuid NOT NULL
    DEFAULT rag_effective_tenant();

-- Names are identities only inside one tenant. The old global constraints are
-- removed after every legacy row has received the default tenant.
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_filename_key;
ALTER TABLE collections DROP CONSTRAINT IF EXISTS collections_name_key_key;
ALTER TABLE tags DROP CONSTRAINT IF EXISTS tags_name_key_key;
CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_filename_key
    ON documents(tenant_id, filename);
CREATE UNIQUE INDEX IF NOT EXISTS collections_tenant_name_key
    ON collections(tenant_id, name_key);
CREATE UNIQUE INDEX IF NOT EXISTS tags_tenant_name_key
    ON tags(tenant_id, name_key);
CREATE INDEX IF NOT EXISTS documents_tenant_inventory_idx
    ON documents(tenant_id, uploaded_at DESC, id DESC);

-- The association rows carry tenant_id for RLS, and these composite keys make
-- that value part of referential integrity too. Without them a service-role
-- statement could manufacture a tenant-A membership pointing at a tenant-B
-- document while still satisfying both single-column foreign keys.
CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_id_key
    ON documents(tenant_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS collections_tenant_id_key
    ON collections(tenant_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS tags_tenant_id_key
    ON tags(tenant_id, id);

-- Durable document-version catalogue. candidate_id IS the version id; two
-- identities for the same source bytes would let attempts/jobs/chunks agree
-- on one while the activation pointer names the other. A version is immutable
-- source identity; a build below proves which retained chunk generation is
-- ready to serve that identity.
CREATE TABLE IF NOT EXISTS document_versions (
    id              uuid PRIMARY KEY,
    tenant_id       uuid NOT NULL,
    document_id     uuid NOT NULL,
    version_number  bigint NOT NULL CHECK (version_number > 0),
    content_sha256  text NOT NULL
                    CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, document_id, id),
    UNIQUE (tenant_id, document_id, version_number),
    FOREIGN KEY (tenant_id, document_id)
        REFERENCES documents(tenant_id, id) ON DELETE RESTRICT
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS active_version_id uuid;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS revision bigint NOT NULL
    DEFAULT 0 CHECK (revision >= 0);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_version_number bigint
    NOT NULL DEFAULT 0 CHECK (last_version_number >= 0);

-- Safe legacy state: every candidate identity whose bytes are known becomes
-- an immutable version, including candidates retained only by an old job or
-- attempt. We intentionally do NOT infer active_version_id: candidate_id may
-- already name newer staged bytes while active_generation still serves the
-- prior version. NULL active_version_id + NULL chunk.version_id is the closed
-- legacy pair until a new promotion proves both together.
DO $$
BEGIN
    IF EXISTS (
        SELECT tenant_id, document_id, version_id
        FROM (
            SELECT tenant_id, id AS document_id,
                   candidate_id AS version_id, content_sha256 AS sha
            FROM documents
            WHERE candidate_id IS NOT NULL AND content_sha256 IS NOT NULL
            UNION ALL
            SELECT tenant_id, document_id, candidate_id, candidate_sha
            FROM attempts
            UNION ALL
            SELECT tenant_id, document_id, candidate_id, candidate_sha
            FROM ingest_jobs
        ) candidates
        GROUP BY tenant_id, document_id, version_id
        HAVING count(DISTINCT sha) > 1
    ) THEN
        RAISE EXCEPTION 'one version id names different source digests'
            USING ERRCODE = '55000';
    END IF;
END
$$;

WITH identities AS (
    SELECT tenant_id, id AS document_id, candidate_id AS version_id,
           content_sha256 AS content_sha256
    FROM documents
    WHERE candidate_id IS NOT NULL AND content_sha256 IS NOT NULL
    UNION
    SELECT tenant_id, document_id, candidate_id, candidate_sha
    FROM attempts
    UNION
    SELECT tenant_id, document_id, candidate_id, candidate_sha
    FROM ingest_jobs
), numbered AS (
    SELECT tenant_id, document_id, version_id, content_sha256,
           row_number() OVER (
               PARTITION BY tenant_id, document_id ORDER BY version_id
           ) AS version_number
    FROM identities
)
INSERT INTO document_versions
    (id, tenant_id, document_id, version_number, content_sha256)
SELECT version_id, tenant_id, document_id, version_number, content_sha256
FROM numbered
ON CONFLICT (id) DO NOTHING;

UPDATE documents d SET last_version_number = GREATEST(
    d.last_version_number,
    COALESCE((SELECT max(v.version_number) FROM document_versions v
              WHERE v.tenant_id = d.tenant_id AND v.document_id = d.id), 0));

ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS version_id uuid;
UPDATE ingest_jobs SET version_id = candidate_id WHERE version_id IS NULL;
ALTER TABLE ingest_jobs ALTER COLUMN version_id SET NOT NULL;

ALTER TABLE attempts ADD COLUMN IF NOT EXISTS version_id uuid;
UPDATE attempts SET version_id = candidate_id WHERE version_id IS NULL;
ALTER TABLE attempts ALTER COLUMN version_id SET NOT NULL;

-- Legacy chunk rows remain NULL because no historical row can prove WHICH
-- candidate produced them. New writes always carry the attempt's version id.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS version_id uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'documents'::regclass
          AND conname = 'documents_active_version_fk'
    ) THEN
        ALTER TABLE documents ADD CONSTRAINT documents_active_version_fk
            FOREIGN KEY (tenant_id, id, active_version_id)
            REFERENCES document_versions(tenant_id, document_id, id)
            ON DELETE RESTRICT;
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS document_versions_tenant_document_order_idx
    ON document_versions(tenant_id, document_id, version_number DESC);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'ingest_jobs_version_fk') THEN
        ALTER TABLE ingest_jobs ADD CONSTRAINT ingest_jobs_version_fk
            FOREIGN KEY (tenant_id, document_id, version_id)
            REFERENCES document_versions(tenant_id, document_id, id)
            ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'attempts_version_fk') THEN
        ALTER TABLE attempts ADD CONSTRAINT attempts_version_fk
            FOREIGN KEY (tenant_id, document_id, version_id)
            REFERENCES document_versions(tenant_id, document_id, id)
            ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'chunks_version_fk') THEN
        ALTER TABLE chunks ADD CONSTRAINT chunks_version_fk
            FOREIGN KEY (tenant_id, document_id, version_id)
            REFERENCES document_versions(tenant_id, document_id, id)
            ON DELETE RESTRICT;
    END IF;
END
$$;

-- Version history is append-only.  UPDATE would rewrite what a version id
-- meant after an audit/event had named it; DELETE would make an activation
-- event point at evidence that no longer exists.  Reject both at the owner
-- role too -- RLS alone cannot protect against the table owner.
CREATE OR REPLACE FUNCTION reject_document_version_mutation()
RETURNS trigger LANGUAGE plpgsql AS $immutable_version$
BEGIN
    RAISE EXCEPTION 'document version history is immutable'
        USING ERRCODE = '55000';
END
$immutable_version$;

DROP TRIGGER IF EXISTS document_versions_immutable ON document_versions;
CREATE TRIGGER document_versions_immutable
BEFORE UPDATE OR DELETE ON document_versions
FOR EACH ROW EXECUTE FUNCTION reject_document_version_mutation();

CREATE TABLE IF NOT EXISTS document_version_events (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id          uuid NOT NULL,
    document_id        uuid NOT NULL,
    event_type         text NOT NULL
                       CHECK (event_type IN ('registered', 'activated')),
    from_version_id    uuid,
    to_version_id      uuid NOT NULL,
    expected_revision  bigint NOT NULL CHECK (expected_revision >= 0),
    resulting_revision bigint NOT NULL CHECK (resulting_revision >= 0),
    created_at         timestamptz NOT NULL DEFAULT now(),
    CHECK ((event_type = 'registered' AND from_version_id IS NULL AND
            resulting_revision = expected_revision) OR
           (event_type = 'activated' AND
            resulting_revision = expected_revision + 1)),
    FOREIGN KEY (tenant_id, document_id)
        REFERENCES documents(tenant_id, id) ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, document_id, from_version_id)
        REFERENCES document_versions(tenant_id, document_id, id)
        ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id, document_id, to_version_id)
        REFERENCES document_versions(tenant_id, document_id, id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS document_version_events_tenant_document_time_idx
    ON document_version_events(tenant_id, document_id, created_at DESC, id);

-- A build row exists only after a complete manifest is promoted. Retaining
-- these rows and their chunks is what makes a later activation a rollback,
-- not merely a pointer aimed at data the stale sweep already deleted.
CREATE TABLE IF NOT EXISTS document_version_builds (
    tenant_id       uuid NOT NULL,
    document_id     uuid NOT NULL,
    version_id      uuid NOT NULL,
    generation      integer NOT NULL CHECK (generation > 0),
    content_sha256  text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    attempt_id      uuid NOT NULL REFERENCES attempts(attempt_id)
                    ON DELETE RESTRICT,
    chunk_count     integer NOT NULL CHECK (chunk_count > 0),
    ready_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, document_id, version_id, generation),
    FOREIGN KEY (tenant_id, document_id, version_id)
        REFERENCES document_versions(tenant_id, document_id, id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS document_version_builds_ready_order_idx
    ON document_version_builds(
        tenant_id, document_id, version_id, generation DESC);

CREATE OR REPLACE FUNCTION record_document_version_registration()
RETURNS trigger LANGUAGE plpgsql AS $registered_version$
DECLARE current_revision bigint;
BEGIN
    SELECT revision INTO STRICT current_revision
    FROM documents WHERE tenant_id = NEW.tenant_id AND id = NEW.document_id;
    INSERT INTO document_version_events
        (tenant_id, document_id, event_type, from_version_id, to_version_id,
         expected_revision, resulting_revision)
    VALUES (NEW.tenant_id, NEW.document_id, 'registered', NULL, NEW.id,
            current_revision, current_revision);
    RETURN NEW;
END
$registered_version$;

DROP TRIGGER IF EXISTS document_versions_record_registration
    ON document_versions;
CREATE TRIGGER document_versions_record_registration
AFTER INSERT ON document_versions
FOR EACH ROW EXECUTE FUNCTION record_document_version_registration();

CREATE OR REPLACE FUNCTION record_document_version_activation()
RETURNS trigger LANGUAGE plpgsql AS $activated_version$
DECLARE ready_count integer;
DECLARE version_sha text;
BEGIN
    IF NEW.active_version_id IS NOT DISTINCT FROM OLD.active_version_id
       AND NEW.active_generation IS NOT DISTINCT FROM OLD.active_generation
    THEN
        RETURN NEW;
    END IF;
    IF NEW.active_version_id IS NULL THEN
        RAISE EXCEPTION 'active version cannot be cleared'
            USING ERRCODE = '55000';
    END IF;
    SELECT content_sha256 INTO STRICT version_sha
    FROM document_versions
    WHERE tenant_id = NEW.tenant_id AND document_id = NEW.id
      AND id = NEW.active_version_id;
    SELECT count(*) INTO ready_count FROM chunks
    WHERE tenant_id = NEW.tenant_id AND document_id = NEW.id
      AND version_id = NEW.active_version_id
      AND generation = NEW.active_generation;
    IF ready_count < 1 OR version_sha IS DISTINCT FROM NEW.active_content_sha
    THEN
        RAISE EXCEPTION 'active version build is not ready'
            USING ERRCODE = '55000';
    END IF;
    IF OLD.attempt_id IS NOT NULL THEN
        INSERT INTO document_version_builds
            (tenant_id, document_id, version_id, generation, content_sha256,
             attempt_id, chunk_count)
        VALUES (NEW.tenant_id, NEW.id, NEW.active_version_id,
                NEW.active_generation, NEW.active_content_sha,
                OLD.attempt_id, ready_count)
        ON CONFLICT DO NOTHING;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM document_version_builds b
        WHERE b.tenant_id = NEW.tenant_id AND b.document_id = NEW.id
          AND b.version_id = NEW.active_version_id
          AND b.generation = NEW.active_generation
          AND b.content_sha256 = NEW.active_content_sha
          AND (OLD.attempt_id IS NULL OR b.attempt_id = OLD.attempt_id)
          AND b.chunk_count = ready_count
    ) THEN
        RAISE EXCEPTION 'active version build receipt is missing'
            USING ERRCODE = '55000';
    END IF;
    INSERT INTO document_version_events
        (tenant_id, document_id, event_type, from_version_id, to_version_id,
         expected_revision, resulting_revision)
    VALUES (NEW.tenant_id, NEW.id, 'activated', OLD.active_version_id,
            NEW.active_version_id, OLD.revision, NEW.revision);
    RETURN NEW;
END
$activated_version$;

DROP TRIGGER IF EXISTS documents_record_version_activation ON documents;
CREATE TRIGGER documents_record_version_activation
AFTER UPDATE OF active_version_id, active_generation ON documents
FOR EACH ROW EXECUTE FUNCTION record_document_version_activation();

DROP TRIGGER IF EXISTS document_version_events_immutable
    ON document_version_events;
CREATE TRIGGER document_version_events_immutable
BEFORE UPDATE OR DELETE ON document_version_events
FOR EACH ROW EXECUTE FUNCTION reject_document_version_mutation();

DROP TRIGGER IF EXISTS document_version_builds_immutable
    ON document_version_builds;
CREATE TRIGGER document_version_builds_immutable
BEFORE UPDATE OR DELETE ON document_version_builds
FOR EACH ROW EXECUTE FUNCTION reject_document_version_mutation();

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'collection_documents_tenant_collection_fk') THEN
        ALTER TABLE collection_documents ADD CONSTRAINT
            collection_documents_tenant_collection_fk
            FOREIGN KEY (tenant_id, collection_id)
            REFERENCES collections(tenant_id, id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'collection_documents_tenant_document_fk') THEN
        ALTER TABLE collection_documents ADD CONSTRAINT
            collection_documents_tenant_document_fk
            FOREIGN KEY (tenant_id, document_id)
            REFERENCES documents(tenant_id, id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'document_tags_tenant_document_fk') THEN
        ALTER TABLE document_tags ADD CONSTRAINT document_tags_tenant_document_fk
            FOREIGN KEY (tenant_id, document_id)
            REFERENCES documents(tenant_id, id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'document_tags_tenant_tag_fk') THEN
        ALTER TABLE document_tags ADD CONSTRAINT document_tags_tenant_tag_fk
            FOREIGN KEY (tenant_id, tag_id)
            REFERENCES tags(tenant_id, id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'ingest_jobs_tenant_document_fk') THEN
        ALTER TABLE ingest_jobs ADD CONSTRAINT ingest_jobs_tenant_document_fk
            FOREIGN KEY (tenant_id, document_id)
            REFERENCES documents(tenant_id, id) ON DELETE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'chunks_tenant_document_fk') THEN
        ALTER TABLE chunks ADD CONSTRAINT chunks_tenant_document_fk
            FOREIGN KEY (tenant_id, document_id)
            REFERENCES documents(tenant_id, id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'attempts_tenant_document_fk') THEN
        ALTER TABLE attempts ADD CONSTRAINT attempts_tenant_document_fk
            FOREIGN KEY (tenant_id, document_id)
            REFERENCES documents(tenant_id, id);
    END IF;
END
$$;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON documents;
CREATE POLICY tenant_isolation ON documents
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE collections FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON collections;
CREATE POLICY tenant_isolation ON collections
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE tags FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON tags;
CREATE POLICY tenant_isolation ON tags
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE collection_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE collection_documents FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON collection_documents;
CREATE POLICY tenant_isolation ON collection_documents
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE document_tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_tags FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON document_tags;
CREATE POLICY tenant_isolation ON document_tags
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE ingest_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingest_jobs FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON ingest_jobs;
CREATE POLICY tenant_isolation ON ingest_jobs
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON chunks;
CREATE POLICY tenant_isolation ON chunks
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE attempts FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON attempts;
CREATE POLICY tenant_isolation ON attempts
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_versions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON document_versions;
CREATE POLICY tenant_isolation ON document_versions
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE document_version_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_version_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON document_version_events;
CREATE POLICY tenant_isolation ON document_version_events
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE document_version_builds ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_version_builds FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON document_version_builds;
CREATE POLICY tenant_isolation ON document_version_builds
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

-- Organisation control plane. Tenant is the customer/company boundary; the
-- tree below expresses directional visibility inside that boundary. App roles
-- remain action permissions and never substitute for a tree relationship.
CREATE TABLE IF NOT EXISTS org_tenants (
    id                   uuid PRIMARY KEY,
    name                 text NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
    architecture_version bigint NOT NULL DEFAULT 1
                         CHECK (architecture_version > 0),
    policy_epoch         bigint NOT NULL DEFAULT 1 CHECK (policy_epoch > 0),
    created_at           timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS org_identities (
    id         uuid PRIMARY KEY,
    issuer     text NOT NULL CHECK (length(issuer) BETWEEN 1 AND 100),
    subject    text NOT NULL CHECK (length(subject) BETWEEN 1 AND 200),
    state      text NOT NULL DEFAULT 'active'
               CHECK (state IN ('active', 'pending', 'suspended')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (issuer, subject)
);

CREATE TABLE IF NOT EXISTS org_positions (
    id                          uuid PRIMARY KEY,
    tenant_id                   uuid NOT NULL REFERENCES org_tenants(id)
                                ON DELETE CASCADE,
    parent_id                   uuid,
    title                       text NOT NULL
                                CHECK (length(title) BETWEEN 1 AND 200),
    kind                        text NOT NULL DEFAULT 'member'
                                CHECK (kind IN ('root', 'manager', 'member')),
    can_monitor_descendants     boolean NOT NULL DEFAULT false,
    protected_from_monitoring   boolean NOT NULL DEFAULT false,
    created_at                  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, id),
    FOREIGN KEY (tenant_id, parent_id)
        REFERENCES org_positions(tenant_id, id) ON DELETE RESTRICT,
    CHECK ((parent_id IS NULL) = (kind = 'root')),
    CHECK (kind <> 'root' OR
           (can_monitor_descendants AND protected_from_monitoring)),
    CHECK (kind <> 'member' OR NOT can_monitor_descendants)
);

CREATE UNIQUE INDEX IF NOT EXISTS org_positions_one_root_idx
    ON org_positions(tenant_id) WHERE parent_id IS NULL;
CREATE INDEX IF NOT EXISTS org_positions_parent_idx
    ON org_positions(tenant_id, parent_id);

CREATE TABLE IF NOT EXISTS org_closure (
    tenant_id    uuid NOT NULL REFERENCES org_tenants(id) ON DELETE CASCADE,
    ancestor_id  uuid NOT NULL,
    descendant_id uuid NOT NULL,
    depth        integer NOT NULL CHECK (depth >= 0),
    PRIMARY KEY (tenant_id, ancestor_id, descendant_id),
    FOREIGN KEY (tenant_id, ancestor_id)
        REFERENCES org_positions(tenant_id, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, descendant_id)
        REFERENCES org_positions(tenant_id, id) ON DELETE CASCADE,
    CHECK ((ancestor_id = descendant_id) = (depth = 0))
);

CREATE TABLE IF NOT EXISTS org_memberships (
    tenant_id  uuid NOT NULL REFERENCES org_tenants(id) ON DELETE CASCADE,
    identity_id uuid NOT NULL REFERENCES org_identities(id) ON DELETE CASCADE,
    position_id uuid,
    display_label text NOT NULL CHECK (length(display_label) BETWEEN 1 AND 200),
    app_role    text NOT NULL DEFAULT 'reader'
                CHECK (app_role IN ('reader', 'editor', 'admin')),
    state       text NOT NULL DEFAULT 'pending'
                CHECK (state IN ('active', 'pending', 'suspended')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, identity_id),
    FOREIGN KEY (tenant_id, position_id)
        REFERENCES org_positions(tenant_id, id) ON DELETE RESTRICT,
    CHECK (state <> 'active' OR position_id IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS org_memberships_active_position_idx
    ON org_memberships(tenant_id, position_id)
    WHERE state = 'active';
CREATE INDEX IF NOT EXISTS org_memberships_identity_idx
    ON org_memberships(identity_id, state);

-- Architecture authority is deliberately separate from business membership.
-- It can manage topology metadata but grants no document/conversation read.
CREATE TABLE IF NOT EXISTS org_architects (
    tenant_id   uuid NOT NULL REFERENCES org_tenants(id) ON DELETE CASCADE,
    identity_id uuid NOT NULL REFERENCES org_identities(id) ON DELETE CASCADE,
    active      boolean NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, identity_id)
);

CREATE TABLE IF NOT EXISTS org_audit_events (
    id           uuid PRIMARY KEY,
    tenant_id    uuid NOT NULL REFERENCES org_tenants(id) ON DELETE CASCADE,
    actor_id     uuid NOT NULL REFERENCES org_identities(id),
    subject_id   uuid REFERENCES org_identities(id),
    action       text NOT NULL CHECK (action IN (
                   'monitor_view', 'topology_read', 'topology_change',
                   'access_preview')),
    reason_code  text NOT NULL CHECK (reason_code IN (
                   'management_duty', 'security_review', 'system_operation',
                   'policy_preview')),
    decision     text NOT NULL CHECK (decision IN ('allowed', 'denied')),
    request_id   text NOT NULL CHECK (length(request_id) BETWEEN 8 AND 64),
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS org_audit_events_tenant_time_idx
    ON org_audit_events(tenant_id, created_at DESC, id);

CREATE OR REPLACE FUNCTION rag_rebuild_org_closure(target_tenant uuid)
RETURNS void LANGUAGE plpgsql AS $closure$
BEGIN
    DELETE FROM org_closure WHERE tenant_id = target_tenant;
    INSERT INTO org_closure (tenant_id, ancestor_id, descendant_id, depth)
    WITH RECURSIVE reach AS (
        SELECT tenant_id, id AS ancestor_id, id AS descendant_id, 0 AS depth
        FROM org_positions WHERE tenant_id = target_tenant
        UNION ALL
        SELECT reach.tenant_id, reach.ancestor_id, child.id,
               reach.depth + 1
        FROM reach JOIN org_positions child
          ON child.tenant_id = reach.tenant_id
         AND child.parent_id = reach.descendant_id
    )
    SELECT tenant_id, ancestor_id, descendant_id, depth FROM reach;
END
$closure$;

CREATE OR REPLACE FUNCTION rag_guard_org_position()
RETURNS trigger LANGUAGE plpgsql AS $guard$
BEGIN
    IF NEW.parent_id = NEW.id THEN
        RAISE EXCEPTION 'org cycle refused';
    END IF;
    IF NEW.parent_id IS NOT NULL AND EXISTS (
        WITH RECURSIVE lineage AS (
            SELECT id, parent_id FROM org_positions
            WHERE tenant_id = NEW.tenant_id AND id = NEW.parent_id
            UNION ALL
            SELECT parent.id, parent.parent_id
            FROM org_positions parent JOIN lineage child
              ON parent.tenant_id = NEW.tenant_id
             AND parent.id = child.parent_id
        )
        SELECT 1 FROM lineage WHERE id = NEW.id
    ) THEN
        RAISE EXCEPTION 'org cycle refused';
    END IF;
    RETURN NEW;
END
$guard$;

DROP TRIGGER IF EXISTS org_positions_cycle_guard ON org_positions;
CREATE TRIGGER org_positions_cycle_guard
BEFORE INSERT OR UPDATE OF tenant_id, parent_id ON org_positions
FOR EACH ROW EXECUTE FUNCTION rag_guard_org_position();

CREATE OR REPLACE FUNCTION rag_refresh_org_closure()
RETURNS trigger LANGUAGE plpgsql AS $refresh$
BEGIN
    PERFORM rag_rebuild_org_closure(COALESCE(NEW.tenant_id, OLD.tenant_id));
    RETURN COALESCE(NEW, OLD);
END
$refresh$;

DROP TRIGGER IF EXISTS org_positions_closure_refresh ON org_positions;
CREATE TRIGGER org_positions_closure_refresh
AFTER INSERT OR UPDATE OR DELETE ON org_positions
FOR EACH ROW EXECUTE FUNCTION rag_refresh_org_closure();

ALTER TABLE org_tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_tenants FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON org_tenants;
CREATE POLICY tenant_isolation ON org_tenants
    USING (rag_service_access() OR id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR id = rag_effective_tenant());

ALTER TABLE org_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_positions FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON org_positions;
CREATE POLICY tenant_isolation ON org_positions
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE org_closure ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_closure FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON org_closure;
CREATE POLICY tenant_isolation ON org_closure
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE org_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_memberships FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON org_memberships;
CREATE POLICY tenant_isolation ON org_memberships
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE org_architects ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_architects FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON org_architects;
CREATE POLICY tenant_isolation ON org_architects
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

ALTER TABLE org_audit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE org_audit_events FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON org_audit_events;
CREATE POLICY tenant_isolation ON org_audit_events
    USING (rag_service_access() OR tenant_id = rag_effective_tenant())
    WITH CHECK (rag_service_access() OR tenant_id = rag_effective_tenant());

-- No ANN index yet: this is a small, single-document demo dataset, and a
-- sequential scan over `dense <=> query` / `sparse <#> query` is effectively
-- instant at this scale. Add an HNSW index (e.g. `USING hnsw (dense
-- vector_cosine_ops)`) once the corpus grows large enough for that to matter.
