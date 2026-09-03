CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;
DO $assert_pgcrypto_namespace$
DECLARE
    extension_oid oid;
    extension_namespace oid;
BEGIN
    SELECT oid, extnamespace INTO extension_oid, extension_namespace
    FROM pg_catalog.pg_extension WHERE extname = 'pgcrypto';
    IF extension_oid IS NULL
       OR extension_namespace <> 'public'::pg_catalog.regnamespace
       OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_depend AS dependency
            WHERE dependency.refobjid = extension_oid
              AND dependency.refclassid =
                  'pg_catalog.pg_extension'::regclass
              AND dependency.classid = 'pg_catalog.pg_proc'::regclass
              AND dependency.objid = pg_catalog.to_regprocedure(
                  'public.hmac(bytea,bytea,text)')
              AND dependency.deptype = 'e')
       OR NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_depend AS dependency
            WHERE dependency.refobjid = extension_oid
              AND dependency.refclassid =
                  'pg_catalog.pg_extension'::regclass
              AND dependency.classid = 'pg_catalog.pg_proc'::regclass
              AND dependency.objid = pg_catalog.to_regprocedure(
                  'public.gen_random_bytes(integer)')
              AND dependency.deptype = 'e')
    THEN
        RAISE EXCEPTION USING ERRCODE = '55000',
            MESSAGE = 'pgcrypto_namespace_refused';
    END IF;
END
$assert_pgcrypto_namespace$;
CREATE SCHEMA IF NOT EXISTS rag_control;

CREATE TABLE IF NOT EXISTS rag_control.control_schema_history (
    schema_version integer PRIMARY KEY CHECK (schema_version > 0),
    schema_sha256 text NOT NULL CHECK (schema_sha256 ~ '^[0-9a-f]{64}$'),
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_control.control_regions (
    region_code text PRIMARY KEY
        CHECK (region_code ~ '^[a-z][a-z0-9-]{1,31}$'),
    state text NOT NULL CHECK (state IN ('active', 'draining', 'disabled')),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0)
);

CREATE TABLE IF NOT EXISTS rag_control.control_tenants (
    tenant_id uuid PRIMARY KEY,
    lifecycle text NOT NULL CHECK (lifecycle IN (
        'provisioning', 'active', 'suspended', 'decommissioning',
        'decommissioned')),
    deployment_profile text NOT NULL CHECK (
        deployment_profile IN ('local', 'team', 'enterprise')),
    policy_revision bigint NOT NULL DEFAULT 1 CHECK (policy_revision > 0),
    configuration_revision bigint NOT NULL DEFAULT 1
        CHECK (configuration_revision > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_control.control_tenant_routes (
    tenant_id uuid PRIMARY KEY REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    route_kind text NOT NULL CHECK (
        route_kind IN ('shared_rls', 'dedicated_postgres')),
    region_code text NOT NULL REFERENCES rag_control.control_regions(region_code)
        ON DELETE RESTRICT,
    connection_ref text NOT NULL UNIQUE
        CHECK (connection_ref ~ '^[a-z][a-z0-9_-]{1,31}:[A-Za-z0-9._/-]{1,96}$')
        CHECK (position('://' in connection_ref) = 0)
        CHECK (position('@' in connection_ref) = 0),
    state text NOT NULL CHECK (
        state IN ('pending', 'active', 'draining', 'retired')),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0)
);

CREATE TABLE IF NOT EXISTS rag_control.control_feature_catalog (
    feature_code text PRIMARY KEY
        CHECK (feature_code ~ '^[a-z][a-z0-9_]{1,63}$'),
    state text NOT NULL CHECK (state IN ('active', 'retired')),
    introduced_schema_version integer NOT NULL CHECK (
        introduced_schema_version > 0)
);

CREATE TABLE IF NOT EXISTS rag_control.control_tenant_features (
    tenant_id uuid NOT NULL REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    feature_code text NOT NULL
        REFERENCES rag_control.control_feature_catalog(feature_code)
        ON DELETE RESTRICT,
    enabled boolean NOT NULL,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    PRIMARY KEY (tenant_id, feature_code)
);

CREATE TABLE IF NOT EXISTS rag_control.control_tenant_quotas (
    tenant_id uuid PRIMARY KEY REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    request_per_minute bigint NOT NULL CHECK (request_per_minute >= 0),
    concurrent_requests bigint NOT NULL CHECK (concurrent_requests >= 0),
    daily_ingest_jobs bigint NOT NULL CHECK (daily_ingest_jobs >= 0),
    storage_bytes bigint NOT NULL CHECK (storage_bytes >= 0),
    document_count bigint NOT NULL CHECK (document_count >= 0),
    evaluation_runs bigint NOT NULL CHECK (evaluation_runs >= 0),
    export_jobs bigint NOT NULL CHECK (export_jobs >= 0),
    model_tokens_per_day bigint NOT NULL CHECK (model_tokens_per_day >= 0),
    quota_enforcement text NOT NULL DEFAULT 'declared'
        CHECK (quota_enforcement = 'declared'),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0)
);

CREATE TABLE IF NOT EXISTS rag_control.control_identity_routes (
    identity_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    state text NOT NULL CHECK (state IN ('active', 'disabled')),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0)
);

CREATE TABLE IF NOT EXISTS rag_control.control_identity_route_digests (
    identity_id uuid NOT NULL
        REFERENCES rag_control.control_identity_routes(identity_id)
        ON DELETE RESTRICT,
    key_version integer NOT NULL CHECK (key_version > 0),
    digest bytea NOT NULL CHECK (octet_length(digest) = 32),
    state text NOT NULL CHECK (state IN ('active', 'retired')),
    PRIMARY KEY (identity_id, key_version),
    UNIQUE (key_version, digest)
);

CREATE TABLE IF NOT EXISTS rag_control.control_platform_operators (
    operator_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    role text NOT NULL CHECK (role IN (
        'platform_reader', 'platform_operator', 'platform_security')),
    state text NOT NULL CHECK (state IN ('active', 'suspended', 'revoked')),
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0)
);

CREATE TABLE IF NOT EXISTS rag_control.control_platform_operator_digests (
    operator_id uuid NOT NULL
        REFERENCES rag_control.control_platform_operators(operator_id)
        ON DELETE RESTRICT,
    key_version integer NOT NULL CHECK (key_version > 0),
    digest bytea NOT NULL CHECK (octet_length(digest) = 32),
    state text NOT NULL CHECK (state IN ('active', 'retired')),
    PRIMARY KEY (operator_id, key_version),
    UNIQUE (key_version, digest)
);

CREATE TABLE IF NOT EXISTS rag_control.control_service_accounts (
    service_account_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    state text NOT NULL CHECK (state IN ('active', 'revoked')),
    expires_at timestamptz NOT NULL,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (expires_at > created_at)
);

-- Version 2 already shipped this table without the composite key. The index
-- therefore lives outside CREATE TABLE so a v2 -> v3 re-run adds the
-- referenced authority before the lifecycle event foreign key is parsed.
CREATE UNIQUE INDEX IF NOT EXISTS control_service_account_tenant_identity
ON rag_control.control_service_accounts (tenant_id, service_account_id);

CREATE TABLE IF NOT EXISTS rag_control.control_service_account_scopes (
    service_account_id uuid NOT NULL
        REFERENCES rag_control.control_service_accounts(service_account_id)
        ON DELETE RESTRICT,
    scope_code text NOT NULL CHECK (scope_code IN (
        'rag.query', 'documents.read', 'documents.write',
        'documents.lifecycle', 'collections.manage', 'tables.extract')),
    PRIMARY KEY (service_account_id, scope_code)
);

CREATE TABLE IF NOT EXISTS rag_control.control_service_account_credentials (
    service_account_id uuid NOT NULL
        REFERENCES rag_control.control_service_accounts(service_account_id)
        ON DELETE RESTRICT,
    credential_version integer NOT NULL CHECK (credential_version > 0),
    digest bytea NOT NULL CHECK (octet_length(digest) = 32),
    state text NOT NULL CHECK (state IN ('active', 'retired', 'revoked')),
    not_before timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (expires_at > not_before),
    PRIMARY KEY (service_account_id, credential_version),
    UNIQUE (digest)
);

CREATE UNIQUE INDEX IF NOT EXISTS control_one_active_service_credential
ON rag_control.control_service_account_credentials (service_account_id)
WHERE state = 'active';

CREATE TABLE IF NOT EXISTS rag_control.control_service_account_events (
    sequence_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id uuid NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    occurred_at timestamptz NOT NULL DEFAULT now(),
    operator_id uuid NOT NULL
        REFERENCES rag_control.control_platform_operators(operator_id)
        ON DELETE RESTRICT,
    actor_kind text NOT NULL DEFAULT 'platform_security' CHECK (
        actor_kind IN ('platform_security', 'tenant_org_admin')),
    tenant_actor_digest bytea CHECK (
        tenant_actor_digest IS NULL
        OR octet_length(tenant_actor_digest) = 32),
    org_policy_epoch bigint CHECK (org_policy_epoch > 0),
    target_tenant_id uuid NOT NULL,
    service_account_id uuid NOT NULL,
    action text NOT NULL CHECK (action IN (
        'service_account_issue', 'service_account_rotate',
        'service_account_revoke')),
    reason_code text NOT NULL CHECK (reason_code IN (
        'security_provisioning', 'incident_response',
        'scheduled_rotation', 'suspected_compromise',
        'tenant_suspension', 'security_response', 'access_removed')),
    expected_revision bigint CHECK (expected_revision > 0),
    resulting_revision bigint NOT NULL CHECK (resulting_revision > 0),
    request_digest bytea NOT NULL CHECK (octet_length(request_digest) = 32),
    resulting_fact_digest bytea NOT NULL
        CHECK (octet_length(resulting_fact_digest) = 32),
    FOREIGN KEY (target_tenant_id, service_account_id)
        REFERENCES rag_control.control_service_accounts(
            tenant_id, service_account_id) ON DELETE RESTRICT,
    CHECK (
        (actor_kind = 'platform_security'
         AND tenant_actor_digest IS NULL AND org_policy_epoch IS NULL)
        OR
        (actor_kind = 'tenant_org_admin'
         AND tenant_actor_digest IS NOT NULL AND org_policy_epoch IS NOT NULL))
);

ALTER TABLE rag_control.control_service_account_events
    ADD COLUMN IF NOT EXISTS actor_kind text NOT NULL
        DEFAULT 'platform_security';
ALTER TABLE rag_control.control_service_account_events
    ADD COLUMN IF NOT EXISTS tenant_actor_digest bytea;
ALTER TABLE rag_control.control_service_account_events
    ADD COLUMN IF NOT EXISTS org_policy_epoch bigint;
ALTER TABLE rag_control.control_service_account_events
    DROP CONSTRAINT IF EXISTS control_service_account_events_actor_kind_check;
ALTER TABLE rag_control.control_service_account_events
    ADD CONSTRAINT control_service_account_events_actor_kind_check CHECK (
        actor_kind IN ('platform_security', 'tenant_org_admin'));
ALTER TABLE rag_control.control_service_account_events
    DROP CONSTRAINT IF EXISTS control_service_account_events_actor_digest_check;
ALTER TABLE rag_control.control_service_account_events
    ADD CONSTRAINT control_service_account_events_actor_digest_check CHECK (
        tenant_actor_digest IS NULL
        OR octet_length(tenant_actor_digest) = 32);
ALTER TABLE rag_control.control_service_account_events
    DROP CONSTRAINT IF EXISTS control_service_account_events_epoch_check;
ALTER TABLE rag_control.control_service_account_events
    ADD CONSTRAINT control_service_account_events_epoch_check CHECK (
        org_policy_epoch IS NULL OR org_policy_epoch > 0);
ALTER TABLE rag_control.control_service_account_events
    DROP CONSTRAINT IF EXISTS control_service_account_events_actor_shape_check;
ALTER TABLE rag_control.control_service_account_events
    ADD CONSTRAINT control_service_account_events_actor_shape_check CHECK (
        (actor_kind = 'platform_security'
         AND tenant_actor_digest IS NULL AND org_policy_epoch IS NULL)
        OR
        (actor_kind = 'tenant_org_admin'
         AND tenant_actor_digest IS NOT NULL AND org_policy_epoch IS NOT NULL));

CREATE TABLE IF NOT EXISTS rag_control.control_service_account_approvals (
    approval_id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    service_account_id uuid NOT NULL,
    action text NOT NULL CHECK (action IN ('issue', 'rotate')),
    state text NOT NULL CHECK (state IN (
        'approved', 'redeemed', 'cancelled')),
    approval_revision bigint NOT NULL DEFAULT 1 CHECK (approval_revision > 0),
    platform_operator_id uuid NOT NULL
        REFERENCES rag_control.control_platform_operators(operator_id)
        ON DELETE RESTRICT,
    reason_code text NOT NULL CHECK (reason_code IN (
        'security_provisioning', 'incident_response',
        'scheduled_rotation', 'suspected_compromise')),
    scopes text[],
    account_expires_at timestamptz,
    credential_expires_at timestamptz NOT NULL,
    expected_account_revision bigint CHECK (expected_account_revision > 0),
    control_policy_revision bigint NOT NULL CHECK (control_policy_revision > 0),
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
    redeemed_at timestamptz,
    request_digest bytea NOT NULL CHECK (octet_length(request_digest) = 32),
    resulting_fact_digest bytea NOT NULL
        CHECK (octet_length(resulting_fact_digest) = 32),
    UNIQUE (approval_id, tenant_id, service_account_id),
    CHECK (credential_expires_at > created_at),
    CHECK (expires_at > created_at
           AND expires_at <= created_at + interval '15 minutes'),
    CHECK (
        (action = 'issue'
         AND scopes IS NOT NULL AND cardinality(scopes) BETWEEN 1 AND 6
         AND account_expires_at IS NOT NULL
         AND expected_account_revision IS NULL)
        OR
        (action = 'rotate'
         AND scopes IS NULL AND account_expires_at IS NULL
         AND expected_account_revision IS NOT NULL)),
    CHECK (
        (state = 'approved' AND approval_revision = 1
         AND redeemed_at IS NULL)
        OR
        (state = 'redeemed' AND approval_revision = 2
         AND redeemed_at IS NOT NULL)
        OR
        (state = 'cancelled' AND approval_revision = 2
         AND redeemed_at IS NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS control_one_pending_account_approval
ON rag_control.control_service_account_approvals (service_account_id)
WHERE state = 'approved';

-- The online control processes never receive these bytes. Migration ownership
-- loads
-- the same versioned key material into the tenant and control databases; only
-- SECURITY DEFINER proof functions may read it.  This is deliberately a
-- separate domain from identity, audit, service-token and RLS-context keys.
CREATE TABLE IF NOT EXISTS
rag_control.control_service_account_assertion_keys (
    key_version integer PRIMARY KEY CHECK (key_version > 0),
    secret bytea NOT NULL CHECK (octet_length(secret) = 32),
    state text NOT NULL CHECK (state IN (
        'staged', 'active', 'verify_only', 'retired')),
    not_before timestamptz NOT NULL,
    verify_started_at timestamptz,
    verify_until timestamptz,
    rotation_id uuid,
    created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
    CHECK (verify_until IS NULL OR verify_until > not_before),
    CHECK ((state = 'verify_only') = (verify_until IS NOT NULL))
);
ALTER TABLE rag_control.control_service_account_assertion_keys
    ADD COLUMN IF NOT EXISTS verify_started_at timestamptz;
ALTER TABLE rag_control.control_service_account_assertion_keys
    ADD COLUMN IF NOT EXISTS rotation_id uuid;
DO $assertion_key_constraints$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM rag_control.control_service_account_assertion_keys
        WHERE state = 'verify_only' AND verify_started_at IS NULL)
    THEN
        RAISE EXCEPTION USING ERRCODE = '55000',
            MESSAGE = 'assertion_key_overlap_unknown';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'rag_control.control_service_account_assertion_keys'::regclass
          AND conname = 'control_assertion_key_state_shape')
    THEN
        ALTER TABLE rag_control.control_service_account_assertion_keys
            ADD CONSTRAINT control_assertion_key_state_shape CHECK (
                (state = 'verify_only'
                 AND verify_started_at IS NOT NULL
                 AND verify_until IS NOT NULL
                 AND rotation_id IS NOT NULL)
                OR
                (state <> 'verify_only'
                 AND verify_started_at IS NULL
                 AND verify_until IS NULL
                 AND (state <> 'staged' OR rotation_id IS NOT NULL)))
                NOT VALID;
        ALTER TABLE rag_control.control_service_account_assertion_keys
            VALIDATE CONSTRAINT control_assertion_key_state_shape;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'rag_control.control_service_account_assertion_keys'::regclass
          AND conname = 'control_assertion_key_overlap_bounded')
    THEN
        ALTER TABLE rag_control.control_service_account_assertion_keys
            ADD CONSTRAINT control_assertion_key_overlap_bounded CHECK (
                verify_until IS NULL OR (
                    verify_until > verify_started_at
                    AND verify_until <= verify_started_at
                        + interval '300 seconds')) NOT VALID;
        ALTER TABLE rag_control.control_service_account_assertion_keys
            VALIDATE CONSTRAINT control_assertion_key_overlap_bounded;
    END IF;
END
$assertion_key_constraints$;
CREATE UNIQUE INDEX IF NOT EXISTS control_one_active_assertion_key
    ON rag_control.control_service_account_assertion_keys ((state))
    WHERE state = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS control_one_staged_assertion_key
    ON rag_control.control_service_account_assertion_keys ((state))
    WHERE state = 'staged';
CREATE UNIQUE INDEX IF NOT EXISTS control_one_verify_only_assertion_key
    ON rag_control.control_service_account_assertion_keys ((state))
    WHERE state = 'verify_only';

CREATE TABLE IF NOT EXISTS
rag_control.control_service_account_assertion_rotations (
    rotation_id uuid PRIMARY KEY,
    previous_key_version integer NOT NULL CHECK (previous_key_version > 0),
    target_key_version integer NOT NULL UNIQUE CHECK (target_key_version > 0),
    target_key_fingerprint bytea NOT NULL
        CHECK (octet_length(target_key_fingerprint) = 32),
    verify_started_at timestamptz NOT NULL,
    verify_until timestamptz NOT NULL,
    phase text NOT NULL CHECK (phase IN (
        'staged', 'admitted', 'activated', 'completed', 'aborted', 'retired')),
    created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
    completed_at timestamptz,
    retired_at timestamptz,
    CHECK (previous_key_version <> target_key_version),
    CHECK (verify_until > verify_started_at),
    CHECK (verify_until <= verify_started_at + interval '300 seconds'),
    CHECK ((phase IN ('completed', 'retired')) = (completed_at IS NOT NULL)),
    CHECK ((phase = 'retired') = (retired_at IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS control_one_live_assertion_rotation
    ON rag_control.control_service_account_assertion_rotations ((true))
    WHERE phase IN ('staged', 'admitted', 'activated');
DO $assertion_rotation_fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint
        WHERE conrelid =
              'rag_control.control_service_account_assertion_keys'::regclass
          AND conname = 'control_assertion_key_rotation_fk')
    THEN
        ALTER TABLE rag_control.control_service_account_assertion_keys
            ADD CONSTRAINT control_assertion_key_rotation_fk
            FOREIGN KEY (rotation_id) REFERENCES
                rag_control.control_service_account_assertion_rotations(
                    rotation_id) ON DELETE RESTRICT;
    END IF;
END
$assertion_rotation_fk$;

CREATE OR REPLACE FUNCTION
rag_control.control_assertion_rotation_keys_bound()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $assertion_rotation_keys_bound$
DECLARE
    previous_bound boolean;
    target_bound boolean;
    bound_count integer;
    members_allowed boolean;
BEGIN
    SELECT bool_or(key_version = NEW.previous_key_version),
           bool_or(key_version = NEW.target_key_version), count(*),
           bool_and(key_version IN (
               NEW.previous_key_version, NEW.target_key_version))
    INTO previous_bound, target_bound, bound_count, members_allowed
    FROM rag_control.control_service_account_assertion_keys
    WHERE rotation_id = NEW.rotation_id;
    -- A retired tombstone, like an aborted one, carries its own versions
    -- and may keep zero, one or both members: its target has to be free
    -- to become the previous member of the NEXT rotation, or the ledger
    -- could record exactly one rotation per database.
    IF (bound_count > 0 AND members_allowed IS DISTINCT FROM true)
       OR bound_count > 2
       OR (NEW.phase NOT IN ('aborted', 'retired')
           AND (previous_bound IS DISTINCT FROM true OR bound_count <> 2))
    THEN
        RAISE EXCEPTION USING ERRCODE = '23514',
            MESSAGE = 'assertion_rotation_keys_unbound';
    END IF;
    RETURN NULL;
END
$assertion_rotation_keys_bound$;
DROP TRIGGER IF EXISTS control_assertion_rotation_keys_bound
    ON rag_control.control_service_account_assertion_rotations;
CREATE CONSTRAINT TRIGGER control_assertion_rotation_keys_bound
AFTER INSERT OR UPDATE ON
    rag_control.control_service_account_assertion_rotations
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    rag_control.control_assertion_rotation_keys_bound();
REVOKE ALL ON FUNCTION
    rag_control.control_assertion_rotation_keys_bound() FROM PUBLIC;

CREATE OR REPLACE FUNCTION
rag_control.control_assertion_key_rotation_bound()
RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $assertion_key_rotation_bound$
DECLARE
    bound_rotation_ids uuid[];
    bound_rotation_id uuid;
    checked_rotation_id uuid;
    rotation_phase text;
    previous_bound boolean;
    target_bound boolean;
    bound_count integer;
    members_allowed boolean;
BEGIN
    IF TG_OP = 'INSERT' THEN
        bound_rotation_ids := ARRAY[NEW.rotation_id];
    ELSIF TG_OP = 'DELETE' THEN
        bound_rotation_ids := ARRAY[OLD.rotation_id];
    ELSE
        bound_rotation_ids := ARRAY[OLD.rotation_id, NEW.rotation_id];
    END IF;
    FOREACH bound_rotation_id IN ARRAY bound_rotation_ids LOOP
        IF bound_rotation_id IS NULL
           OR bound_rotation_id IS NOT DISTINCT FROM checked_rotation_id
        THEN
            CONTINUE;
        END IF;
        SELECT rotation.phase,
               bool_or(key.key_version = rotation.previous_key_version),
               bool_or(key.key_version = rotation.target_key_version),
               count(key.key_version),
               bool_and(key.key_version IN (
                   rotation.previous_key_version,
                   rotation.target_key_version))
        INTO rotation_phase, previous_bound, target_bound, bound_count,
             members_allowed
        FROM rag_control.control_service_account_assertion_rotations AS rotation
        LEFT JOIN rag_control.control_service_account_assertion_keys AS key
          ON key.rotation_id = rotation.rotation_id
        WHERE rotation.rotation_id = bound_rotation_id
        GROUP BY rotation.phase, rotation.previous_key_version,
                 rotation.target_key_version;
        IF rotation_phase IS NULL
           OR (bound_count > 0 AND members_allowed IS DISTINCT FROM true)
           OR bound_count > 2
           OR (rotation_phase NOT IN ('aborted', 'retired')
               AND (previous_bound IS DISTINCT FROM true OR bound_count <> 2))
        THEN
            RAISE EXCEPTION USING ERRCODE = '23514',
                MESSAGE = 'assertion_rotation_keys_unbound';
        END IF;
        checked_rotation_id := bound_rotation_id;
    END LOOP;
    RETURN NULL;
END
$assertion_key_rotation_bound$;
DROP TRIGGER IF EXISTS control_assertion_key_rotation_bound
    ON rag_control.control_service_account_assertion_keys;
CREATE CONSTRAINT TRIGGER control_assertion_key_rotation_bound
AFTER INSERT OR UPDATE OR DELETE ON
    rag_control.control_service_account_assertion_keys
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    rag_control.control_assertion_key_rotation_bound();
REVOKE ALL ON FUNCTION
    rag_control.control_assertion_key_rotation_bound() FROM PUBLIC;

CREATE OR REPLACE FUNCTION
rag_control.control_assertion_rotation_lifecycle_guard()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $assertion_rotation_lifecycle_guard$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION USING ERRCODE = '55000',
            MESSAGE = 'assertion_rotation_tombstone_immutable';
    END IF;
    IF NEW.rotation_id IS DISTINCT FROM OLD.rotation_id
       OR NEW.previous_key_version IS DISTINCT FROM OLD.previous_key_version
       OR NEW.target_key_version IS DISTINCT FROM OLD.target_key_version
       OR NEW.target_key_fingerprint IS DISTINCT FROM OLD.target_key_fingerprint
       OR NEW.verify_started_at IS DISTINCT FROM OLD.verify_started_at
       OR NEW.verify_until IS DISTINCT FROM OLD.verify_until
       OR NEW.created_at IS DISTINCT FROM OLD.created_at
       OR (OLD.phase, NEW.phase) NOT IN (
            ('staged', 'admitted'), ('staged', 'activated'),
            ('staged', 'aborted'), ('admitted', 'completed'),
            ('admitted', 'aborted'), ('activated', 'completed'),
            ('completed', 'retired'))
       OR (OLD.phase = 'completed'
           AND NEW.completed_at IS DISTINCT FROM OLD.completed_at)
    THEN
        RAISE EXCEPTION USING ERRCODE = '55000',
            MESSAGE = 'assertion_rotation_transition_refused';
    END IF;
    RETURN NEW;
END
$assertion_rotation_lifecycle_guard$;
DROP TRIGGER IF EXISTS control_assertion_rotation_lifecycle_guard
    ON rag_control.control_service_account_assertion_rotations;
CREATE TRIGGER control_assertion_rotation_lifecycle_guard
BEFORE UPDATE OR DELETE ON
    rag_control.control_service_account_assertion_rotations
FOR EACH ROW EXECUTE FUNCTION
    rag_control.control_assertion_rotation_lifecycle_guard();
REVOKE ALL ON FUNCTION
    rag_control.control_assertion_rotation_lifecycle_guard() FROM PUBLIC;

CREATE TABLE IF NOT EXISTS
rag_control.control_service_account_assertion_nonces (
    key_version integer NOT NULL REFERENCES
        rag_control.control_service_account_assertion_keys(key_version)
        ON DELETE RESTRICT,
    purpose text NOT NULL CHECK (purpose IN (
        'approval_list', 'approval_get',
        'approval_redeem_issue', 'approval_redeem_rotate')),
    nonce bytea NOT NULL CHECK (octet_length(nonce) = 16),
    tenant_id uuid NOT NULL,
    approval_id uuid,
    expires_at timestamptz NOT NULL,
    consumed_at timestamptz NOT NULL DEFAULT statement_timestamp(),
    PRIMARY KEY (key_version, nonce),
    CHECK ((purpose = 'approval_list' AND approval_id IS NULL)
        OR (purpose IN ('approval_get', 'approval_redeem_issue',
                       'approval_redeem_rotate')
            AND approval_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS control_assertion_nonces_expiry
    ON rag_control.control_service_account_assertion_nonces
       (expires_at, key_version, nonce);

REVOKE ALL ON rag_control.control_service_account_assertion_keys FROM PUBLIC;
REVOKE ALL ON rag_control.control_service_account_assertion_rotations
    FROM PUBLIC;
REVOKE ALL ON rag_control.control_service_account_assertion_nonces FROM PUBLIC;

CREATE TABLE IF NOT EXISTS rag_control.control_service_account_approval_events (
    sequence_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id uuid NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    occurred_at timestamptz NOT NULL DEFAULT statement_timestamp(),
    approval_id uuid NOT NULL,
    target_tenant_id uuid NOT NULL,
    service_account_id uuid NOT NULL,
    action text NOT NULL CHECK (action IN (
        'approval_created', 'approval_redeemed', 'approval_cancelled')),
    reason_code text NOT NULL CHECK (reason_code IN (
        'security_provisioning', 'incident_response',
        'scheduled_rotation', 'suspected_compromise',
        'approval_redeemed', 'approval_expired', 'approval_cancelled',
        'service_account_revoked', 'security_response',
        'tenant_suspension', 'access_removed')),
    actor_kind text NOT NULL CHECK (actor_kind IN (
        'platform_security', 'tenant_org_admin', 'system')),
    platform_operator_id uuid
        REFERENCES rag_control.control_platform_operators(operator_id)
        ON DELETE RESTRICT,
    tenant_actor_digest bytea
        CHECK (tenant_actor_digest IS NULL
               OR octet_length(tenant_actor_digest) = 32),
    org_policy_epoch bigint CHECK (org_policy_epoch > 0),
    prior_state text CHECK (prior_state IS NULL OR prior_state = 'approved'),
    resulting_state text NOT NULL CHECK (resulting_state IN (
        'approved', 'redeemed', 'cancelled')),
    prior_revision bigint CHECK (prior_revision > 0),
    approval_revision bigint NOT NULL CHECK (approval_revision > 0),
    approval_created_at timestamptz NOT NULL,
    approval_expires_at timestamptz NOT NULL,
    request_digest bytea NOT NULL CHECK (octet_length(request_digest) = 32),
    resulting_fact_digest bytea NOT NULL
        CHECK (octet_length(resulting_fact_digest) = 32),
    FOREIGN KEY (approval_id, target_tenant_id, service_account_id)
        REFERENCES rag_control.control_service_account_approvals(
            approval_id, tenant_id, service_account_id) ON DELETE RESTRICT,
    CHECK (
        (action = 'approval_created'
         AND prior_state IS NULL AND prior_revision IS NULL
         AND resulting_state = 'approved' AND approval_revision = 1)
        OR
        (action IN ('approval_redeemed', 'approval_cancelled')
         AND prior_state = 'approved' AND prior_revision = 1
         AND resulting_state IN ('redeemed', 'cancelled')
         AND approval_revision = 2)),
    CHECK (
        (actor_kind = 'platform_security'
         AND platform_operator_id IS NOT NULL
         AND tenant_actor_digest IS NULL AND org_policy_epoch IS NULL)
        OR
        (actor_kind = 'tenant_org_admin'
         AND platform_operator_id IS NULL
         AND tenant_actor_digest IS NOT NULL AND org_policy_epoch IS NOT NULL)
        OR
        (actor_kind = 'system'
         AND platform_operator_id IS NULL
         AND tenant_actor_digest IS NULL AND org_policy_epoch IS NULL))
);

CREATE TABLE IF NOT EXISTS rag_control.control_admin_events (
    sequence_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id uuid NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    occurred_at timestamptz NOT NULL DEFAULT now(),
    operator_id uuid NOT NULL
        REFERENCES rag_control.control_platform_operators(operator_id)
        ON DELETE RESTRICT,
    target_tenant_id uuid REFERENCES rag_control.control_tenants(tenant_id)
        ON DELETE RESTRICT,
    action text NOT NULL CHECK (action IN (
        'tenant_create', 'tenant_configure', 'tenant_activate',
        'tenant_suspend', 'tenant_decommission', 'route_change',
        'feature_change', 'quota_declare', 'identity_bind',
        'identity_disable', 'operator_change', 'service_account_issue',
        'service_account_rotate', 'service_account_revoke', 'scim_apply',
        'break_glass_request', 'break_glass_approve', 'break_glass_use',
        'break_glass_expire', 'break_glass_revoke')),
    target_kind text NOT NULL CHECK (target_kind IN (
        'tenant', 'route', 'feature', 'quota', 'identity', 'operator',
        'service_account', 'break_glass')),
    reason_code text NOT NULL CHECK (
        reason_code ~ '^[a-z][a-z0-9_]{1,63}$'),
    decision text NOT NULL CHECK (decision IN ('accepted', 'refused')),
    expected_revision bigint CHECK (expected_revision > 0),
    resulting_revision bigint CHECK (resulting_revision > 0),
    request_digest bytea NOT NULL CHECK (octet_length(request_digest) = 32),
    resulting_fact_digest bytea
        CHECK (octet_length(resulting_fact_digest) = 32)
);

CREATE OR REPLACE FUNCTION rag_control.control_events_immutable()
RETURNS trigger LANGUAGE plpgsql
SET search_path = pg_catalog, rag_control AS $events_immutable$
BEGIN
    RAISE EXCEPTION USING ERRCODE = '55000',
        MESSAGE = 'control_event_immutable';
END
$events_immutable$;

CREATE OR REPLACE FUNCTION
rag_control.control_seal_service_account_approval_event()
RETURNS trigger LANGUAGE plpgsql
SET search_path = pg_catalog, rag_control AS $seal_approval_event$
DECLARE
    actor_digest text;
    request_facts jsonb;
    result_facts jsonb;
BEGIN
    actor_digest := CASE
        WHEN NEW.tenant_actor_digest IS NULL THEN NULL
        ELSE encode(NEW.tenant_actor_digest, 'hex')
    END;
    request_facts := jsonb_build_array(
        'approval_event_request_v1', NEW.approval_id::text,
        NEW.target_tenant_id::text, NEW.service_account_id::text,
        NEW.action, NEW.reason_code, NEW.actor_kind,
        NEW.platform_operator_id::text, actor_digest,
        NEW.org_policy_epoch, NEW.prior_state, NEW.prior_revision);
    result_facts := jsonb_build_array(
        'approval_event_result_v1', NEW.approval_id::text,
        NEW.target_tenant_id::text, NEW.service_account_id::text,
        NEW.action, NEW.reason_code, NEW.actor_kind,
        NEW.platform_operator_id::text, actor_digest,
        NEW.org_policy_epoch, NEW.prior_state, NEW.resulting_state,
        NEW.prior_revision, NEW.approval_revision,
        extract(epoch FROM NEW.approval_created_at),
        extract(epoch FROM NEW.approval_expires_at),
        extract(epoch FROM NEW.occurred_at));
    NEW.request_digest := sha256(convert_to(request_facts::text, 'UTF8'));
    NEW.resulting_fact_digest := sha256(
        convert_to(result_facts::text, 'UTF8'));
    RETURN NEW;
END
$seal_approval_event$;

DROP TRIGGER IF EXISTS control_service_account_approval_events_seal
    ON rag_control.control_service_account_approval_events;
CREATE TRIGGER control_service_account_approval_events_seal
BEFORE INSERT ON rag_control.control_service_account_approval_events
FOR EACH ROW EXECUTE FUNCTION
    rag_control.control_seal_service_account_approval_event();

DROP TRIGGER IF EXISTS control_events_immutable_write
    ON rag_control.control_admin_events;
CREATE TRIGGER control_events_immutable_write
BEFORE UPDATE OR DELETE ON rag_control.control_admin_events
FOR EACH ROW EXECUTE FUNCTION rag_control.control_events_immutable();

DROP TRIGGER IF EXISTS control_service_account_events_immutable_write
    ON rag_control.control_service_account_events;
CREATE TRIGGER control_service_account_events_immutable_write
BEFORE UPDATE OR DELETE ON rag_control.control_service_account_events
FOR EACH ROW EXECUTE FUNCTION rag_control.control_events_immutable();

DROP TRIGGER IF EXISTS control_service_account_approval_events_immutable_write
    ON rag_control.control_service_account_approval_events;
CREATE TRIGGER control_service_account_approval_events_immutable_write
BEFORE UPDATE OR DELETE ON rag_control.control_service_account_approval_events
FOR EACH ROW EXECUTE FUNCTION rag_control.control_events_immutable();

CREATE OR REPLACE FUNCTION rag_control.control_tenant_facts(
    requested_tenant uuid)
RETURNS TABLE (
    tenant_id uuid,
    deployment_profile text,
    region_code text,
    route_kind text,
    configuration_revision bigint,
    policy_revision bigint,
    features jsonb,
    quotas jsonb,
    quota_enforcement text
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $tenant_facts$
    SELECT t.tenant_id, t.deployment_profile, r.region_code, r.route_kind,
           t.configuration_revision, t.policy_revision,
           COALESCE((
               SELECT jsonb_object_agg(f.feature_code, f.enabled)
               FROM rag_control.control_tenant_features AS f
               JOIN rag_control.control_feature_catalog AS c
                 ON c.feature_code = f.feature_code
               WHERE f.tenant_id = t.tenant_id AND c.state = 'active'
           ), '{}'::jsonb),
           jsonb_build_object(
               'request_per_minute', q.request_per_minute,
               'concurrent_requests', q.concurrent_requests,
               'daily_ingest_jobs', q.daily_ingest_jobs,
               'storage_bytes', q.storage_bytes,
               'document_count', q.document_count,
               'evaluation_runs', q.evaluation_runs,
               'export_jobs', q.export_jobs,
               'model_tokens_per_day', q.model_tokens_per_day),
           q.quota_enforcement
    FROM rag_control.control_tenants AS t
    JOIN rag_control.control_tenant_routes AS r ON r.tenant_id = t.tenant_id
    JOIN rag_control.control_regions AS g ON g.region_code = r.region_code
    JOIN rag_control.control_tenant_quotas AS q ON q.tenant_id = t.tenant_id
    WHERE t.tenant_id = requested_tenant
      AND t.lifecycle = 'active'
      AND r.state = 'active'
      AND g.state = 'active'
$tenant_facts$;

CREATE OR REPLACE FUNCTION rag_control.control_resolve_identity(
    requested_key_version integer,
    requested_digest bytea)
RETURNS TABLE (
    tenant_id uuid,
    deployment_profile text,
    region_code text,
    route_kind text,
    connection_ref text,
    configuration_revision bigint,
    policy_revision bigint,
    features jsonb,
    quotas jsonb,
    quota_enforcement text
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $resolve_identity$
    SELECT facts.tenant_id, facts.deployment_profile, facts.region_code,
           facts.route_kind, route.connection_ref,
           facts.configuration_revision, facts.policy_revision,
           facts.features, facts.quotas, facts.quota_enforcement
    FROM rag_control.control_identity_route_digests AS digest
    JOIN rag_control.control_identity_routes AS identity_route
      ON identity_route.identity_id = digest.identity_id
    JOIN rag_control.control_tenant_routes AS route
      ON route.tenant_id = identity_route.tenant_id
    CROSS JOIN LATERAL rag_control.control_tenant_facts(
        identity_route.tenant_id) AS facts
    WHERE digest.key_version = requested_key_version
      AND digest.digest = requested_digest
      AND digest.state = 'active'
      AND identity_route.state = 'active'
      AND octet_length(requested_digest) = 32
$resolve_identity$;

CREATE OR REPLACE FUNCTION rag_control.control_resolve_platform_operator(
    requested_key_version integer,
    requested_digest bytea)
RETURNS TABLE (
    operator_id uuid,
    role text,
    revision bigint
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $resolve_platform_operator$
    SELECT platform_actor.operator_id, platform_actor.role,
           platform_actor.revision
    FROM rag_control.control_platform_operator_digests AS digest
    JOIN rag_control.control_platform_operators AS platform_actor
      ON platform_actor.operator_id = digest.operator_id
    WHERE digest.key_version = requested_key_version
      AND digest.digest = requested_digest
      AND digest.state = 'active'
      AND platform_actor.state = 'active'
      AND octet_length(requested_digest) = 32
$resolve_platform_operator$;

CREATE OR REPLACE FUNCTION rag_control.control_resolve_service_account(
    requested_account_id uuid,
    requested_credential_version integer,
    requested_digest bytea)
RETURNS TABLE (
    service_account_id uuid,
    tenant_id uuid,
    scopes text[],
    deployment_profile text,
    region_code text,
    route_kind text,
    connection_ref text,
    configuration_revision bigint,
    policy_revision bigint,
    features jsonb,
    quotas jsonb,
    quota_enforcement text
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $resolve_service_account$
    SELECT account.service_account_id, facts.tenant_id, scope_set.scopes,
           facts.deployment_profile, facts.region_code,
           facts.route_kind, route.connection_ref,
           facts.configuration_revision, facts.policy_revision,
           facts.features, facts.quotas, facts.quota_enforcement
    FROM rag_control.control_service_accounts AS account
    JOIN rag_control.control_service_account_credentials AS credential
      ON credential.service_account_id = account.service_account_id
    JOIN rag_control.control_tenant_routes AS route
      ON route.tenant_id = account.tenant_id
    CROSS JOIN LATERAL (
        SELECT array_agg(scope.scope_code ORDER BY scope.scope_code) AS scopes
        FROM rag_control.control_service_account_scopes AS scope
        WHERE scope.service_account_id = account.service_account_id
    ) AS scope_set
    CROSS JOIN LATERAL rag_control.control_tenant_facts(
        account.tenant_id) AS facts
    WHERE account.service_account_id = requested_account_id
      AND credential.credential_version = requested_credential_version
      AND credential.digest = requested_digest
      AND account.state = 'active'
      AND credential.state = 'active'
      AND statement_timestamp() >= credential.not_before
      AND statement_timestamp() < credential.expires_at
      AND statement_timestamp() < account.expires_at
      AND octet_length(requested_digest) = 32
      AND cardinality(scope_set.scopes) > 0
$resolve_service_account$;

CREATE OR REPLACE FUNCTION rag_control.control_require_platform_security(
    requested_key_version integer,
    requested_operator_digest bytea)
RETURNS uuid
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $require_platform_security$
DECLARE
    resolved_operator_id uuid;
BEGIN
    SELECT platform_actor.operator_id INTO STRICT resolved_operator_id
        FROM rag_control.control_platform_operators AS platform_actor
        JOIN rag_control.control_platform_operator_digests AS digest
          ON digest.operator_id = platform_actor.operator_id
        WHERE digest.key_version = requested_key_version
          AND digest.digest = requested_operator_digest
          AND digest.state = 'active'
          AND platform_actor.role = 'platform_security'
          AND platform_actor.state = 'active'
          AND octet_length(requested_operator_digest) = 32;
    RETURN resolved_operator_id;
EXCEPTION
    WHEN NO_DATA_FOUND OR TOO_MANY_ROWS THEN
        RAISE EXCEPTION USING ERRCODE = '42501',
            MESSAGE = 'service_account_management_refused';
END
$require_platform_security$;

CREATE OR REPLACE FUNCTION rag_control.control_lock_service_account(
    requested_account_id uuid)
RETURNS void
LANGUAGE sql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $lock_service_account$
    SELECT pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(requested_account_id::text, 0))
$lock_service_account$;

CREATE OR REPLACE FUNCTION rag_control.control_expire_service_account_approval(
    requested_account_id uuid)
RETURNS void
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $expire_service_account_approval$
BEGIN
    WITH expired AS (
        UPDATE rag_control.control_service_account_approvals
        SET state = 'cancelled', approval_revision = 2
        WHERE service_account_id = requested_account_id
          AND state = 'approved'
          AND expires_at <= statement_timestamp()
        RETURNING *
    )
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, prior_state, resulting_state,
        prior_revision, approval_revision, approval_created_at,
        approval_expires_at, request_digest, resulting_fact_digest)
    SELECT approval_id, tenant_id, service_account_id,
           'approval_cancelled', 'approval_expired', 'system',
           'approved', 'cancelled', 1, approval_revision, created_at,
           expires_at, request_digest, resulting_fact_digest
    FROM expired;
END
$expire_service_account_approval$;

CREATE OR REPLACE FUNCTION rag_control.control_approve_service_account_issue(
    requested_operator_key_version integer,
    requested_operator_digest bytea,
    requested_approval_id uuid,
    requested_tenant_id uuid,
    requested_account_id uuid,
    requested_scopes text[],
    requested_account_expires_at timestamptz,
    requested_credential_expires_at timestamptz,
    requested_control_policy_revision bigint,
    requested_reason_code text,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS TABLE (
    approval_revision bigint,
    control_policy_revision bigint,
    created_at timestamptz,
    expires_at timestamptz)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $approve_service_account_issue$
DECLARE
    canonical_scopes text[];
    resolved_operator_id uuid;
    tenant_policy_revision bigint;
BEGIN
    resolved_operator_id := rag_control.control_require_platform_security(
        requested_operator_key_version, requested_operator_digest);
    SELECT array_agg(scope ORDER BY scope) INTO canonical_scopes
    FROM (SELECT DISTINCT unnest(requested_scopes) AS scope) AS normalized;
    SELECT tenant.policy_revision INTO tenant_policy_revision
    FROM rag_control.control_tenants AS tenant
    WHERE tenant.tenant_id = requested_tenant_id
      AND tenant.lifecycle = 'active'
    FOR SHARE;
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    PERFORM rag_control.control_expire_service_account_approval(
        requested_account_id);
    IF canonical_scopes IS NULL
       OR requested_scopes <> canonical_scopes
       OR NOT requested_scopes <@ ARRAY[
           'rag.query', 'documents.read', 'documents.write',
           'documents.lifecycle', 'collections.manage',
           'tables.extract']::text[]
       OR tenant_policy_revision IS NULL
       OR tenant_policy_revision <> requested_control_policy_revision
       OR NOT EXISTS (
           SELECT 1 FROM rag_control.control_tenant_facts(
               requested_tenant_id))
       OR requested_account_expires_at <= statement_timestamp()
       OR requested_credential_expires_at <= statement_timestamp()
       OR requested_account_expires_at < requested_credential_expires_at
       OR requested_account_expires_at >
          statement_timestamp() + interval '366 days'
       OR requested_reason_code NOT IN (
           'security_provisioning', 'incident_response')
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
       OR EXISTS (
           SELECT 1 FROM rag_control.control_service_accounts
           WHERE service_account_id = requested_account_id)
    THEN
        RAISE EXCEPTION USING ERRCODE = '22023',
            MESSAGE = 'service_account_approval_invalid';
    END IF;
    INSERT INTO rag_control.control_service_account_approvals (
        approval_id, tenant_id, service_account_id, action, state,
        platform_operator_id, reason_code, scopes, account_expires_at,
        credential_expires_at, control_policy_revision, expires_at,
        request_digest, resulting_fact_digest)
    VALUES (
        requested_approval_id, requested_tenant_id, requested_account_id,
        'issue', 'approved', resolved_operator_id, requested_reason_code,
        requested_scopes, requested_account_expires_at,
        requested_credential_expires_at, tenant_policy_revision,
        statement_timestamp() + interval '15 minutes',
        requested_request_digest, requested_resulting_fact_digest);
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, platform_operator_id, prior_state,
        resulting_state, prior_revision, approval_revision,
        approval_created_at, approval_expires_at, request_digest,
        resulting_fact_digest)
    VALUES (
        requested_approval_id, requested_tenant_id, requested_account_id,
        'approval_created', requested_reason_code, 'platform_security',
        resolved_operator_id, NULL, 'approved', NULL, 1,
        statement_timestamp(), statement_timestamp() + interval '15 minutes',
        requested_request_digest, requested_resulting_fact_digest);
    RETURN QUERY
    SELECT approval.approval_revision, approval.control_policy_revision,
           approval.created_at, approval.expires_at
    FROM rag_control.control_service_account_approvals AS approval
    WHERE approval.approval_id = requested_approval_id;
END
$approve_service_account_issue$;

CREATE OR REPLACE FUNCTION rag_control.control_approve_service_account_rotation(
    requested_operator_key_version integer,
    requested_operator_digest bytea,
    requested_approval_id uuid,
    requested_tenant_id uuid,
    requested_account_id uuid,
    requested_expected_account_revision bigint,
    requested_credential_expires_at timestamptz,
    requested_control_policy_revision bigint,
    requested_reason_code text,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS TABLE (
    approval_revision bigint,
    control_policy_revision bigint,
    created_at timestamptz,
    expires_at timestamptz)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $approve_service_account_rotation$
DECLARE
    account_expiry timestamptz;
    account_revision bigint;
    resolved_operator_id uuid;
    tenant_policy_revision bigint;
BEGIN
    resolved_operator_id := rag_control.control_require_platform_security(
        requested_operator_key_version, requested_operator_digest);
    SELECT tenant.policy_revision INTO tenant_policy_revision
    FROM rag_control.control_tenants AS tenant
    WHERE tenant.tenant_id = requested_tenant_id
      AND tenant.lifecycle = 'active'
    FOR SHARE;
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    PERFORM rag_control.control_expire_service_account_approval(
        requested_account_id);
    SELECT account.expires_at, account.revision
    INTO account_expiry, account_revision
    FROM rag_control.control_service_accounts AS account
    WHERE account.service_account_id = requested_account_id
      AND account.tenant_id = requested_tenant_id
      AND account.state = 'active'
    FOR UPDATE;
    IF account_revision IS NULL
       OR account_revision <> requested_expected_account_revision
       OR tenant_policy_revision IS NULL
       OR tenant_policy_revision <> requested_control_policy_revision
       OR NOT EXISTS (
           SELECT 1 FROM rag_control.control_tenant_facts(
               requested_tenant_id))
       OR requested_credential_expires_at <= statement_timestamp()
       OR account_expiry < requested_credential_expires_at
       OR requested_reason_code NOT IN (
           'scheduled_rotation', 'suspected_compromise')
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
    THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_approval_conflict';
    END IF;
    INSERT INTO rag_control.control_service_account_approvals (
        approval_id, tenant_id, service_account_id, action, state,
        platform_operator_id, reason_code, credential_expires_at,
        expected_account_revision, control_policy_revision, expires_at,
        request_digest, resulting_fact_digest)
    VALUES (
        requested_approval_id, requested_tenant_id, requested_account_id,
        'rotate', 'approved', resolved_operator_id, requested_reason_code,
        requested_credential_expires_at,
        requested_expected_account_revision, tenant_policy_revision,
        statement_timestamp() + interval '15 minutes',
        requested_request_digest, requested_resulting_fact_digest);
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, platform_operator_id, prior_state,
        resulting_state, prior_revision, approval_revision,
        approval_created_at, approval_expires_at, request_digest,
        resulting_fact_digest)
    VALUES (
        requested_approval_id, requested_tenant_id, requested_account_id,
        'approval_created', requested_reason_code, 'platform_security',
        resolved_operator_id, NULL, 'approved', NULL, 1,
        statement_timestamp(), statement_timestamp() + interval '15 minutes',
        requested_request_digest, requested_resulting_fact_digest);
    RETURN QUERY
    SELECT approval.approval_revision, approval.control_policy_revision,
           approval.created_at, approval.expires_at
    FROM rag_control.control_service_account_approvals AS approval
    WHERE approval.approval_id = requested_approval_id;
END
$approve_service_account_rotation$;

CREATE OR REPLACE FUNCTION rag_control.control_cancel_service_account_approval(
    requested_operator_key_version integer,
    requested_operator_digest bytea,
    requested_approval_id uuid,
    requested_tenant_id uuid,
    requested_account_id uuid,
    requested_expected_approval_revision bigint,
    requested_reason_code text)
RETURNS bigint
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $cancel_service_account_approval$
DECLARE
    cancelled rag_control.control_service_account_approvals%ROWTYPE;
    resolved_operator_id uuid;
BEGIN
    resolved_operator_id := rag_control.control_require_platform_security(
        requested_operator_key_version, requested_operator_digest);
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    UPDATE rag_control.control_service_account_approvals
    SET state = 'cancelled', approval_revision = 2
    WHERE approval_id = requested_approval_id
      AND tenant_id = requested_tenant_id
      AND service_account_id = requested_account_id
      AND state = 'approved'
      AND approval_revision = requested_expected_approval_revision
      AND requested_expected_approval_revision = 1
      AND requested_reason_code IN (
          'approval_cancelled', 'security_response',
          'tenant_suspension', 'access_removed')
    RETURNING * INTO cancelled;
    IF cancelled.approval_id IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_approval_conflict';
    END IF;
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, platform_operator_id, prior_state,
        resulting_state, prior_revision, approval_revision,
        approval_created_at, approval_expires_at, request_digest,
        resulting_fact_digest)
    VALUES (
        cancelled.approval_id, cancelled.tenant_id,
        cancelled.service_account_id, 'approval_cancelled',
        requested_reason_code, 'platform_security', resolved_operator_id,
        'approved', 'cancelled', 1, cancelled.approval_revision,
        cancelled.created_at, cancelled.expires_at,
        cancelled.request_digest, cancelled.resulting_fact_digest);
    RETURN cancelled.approval_revision;
END
$cancel_service_account_approval$;

CREATE OR REPLACE FUNCTION rag_control.control_service_account_assertion_payload(
    requested_purpose text,
    requested_key_version integer,
    requested_tenant_id uuid,
    requested_tenant_actor_digest bytea,
    requested_org_policy_epoch bigint,
    requested_approval_id uuid,
    requested_approval_revision bigint,
    requested_service_account_id uuid,
    requested_credential_digest bytea,
    requested_limit integer,
    requested_issued_at bigint,
    requested_expires_at bigint,
    requested_nonce bytea)
RETURNS bytea
LANGUAGE plpgsql IMMUTABLE
SET search_path = pg_catalog, rag_control
AS $service_account_assertion_payload$
DECLARE
    purpose_bytes bytea;
BEGIN
    IF requested_purpose IS NULL
       OR requested_purpose NOT IN (
           'approval_list', 'approval_get',
           'approval_redeem_issue', 'approval_redeem_rotate')
       OR requested_key_version IS NULL OR requested_key_version < 1
       OR requested_tenant_id IS NULL
       OR requested_tenant_actor_digest IS NULL
       OR octet_length(requested_tenant_actor_digest) <> 32
       OR requested_org_policy_epoch IS NULL
       OR requested_org_policy_epoch < 1
       OR requested_issued_at IS NULL OR requested_expires_at IS NULL
       OR requested_expires_at - requested_issued_at <> 30
       OR requested_nonce IS NULL OR octet_length(requested_nonce) <> 16
       OR (requested_purpose = 'approval_list'
           AND (requested_approval_id IS NOT NULL
                OR requested_approval_revision IS NOT NULL
                OR requested_service_account_id IS NOT NULL
                OR requested_credential_digest IS NOT NULL
                OR requested_limit IS NULL
                OR requested_limit < 1 OR requested_limit > 100))
       OR (requested_purpose = 'approval_get'
           AND (requested_approval_id IS NULL
                OR requested_approval_revision IS NULL
                OR requested_approval_revision < 1
                OR requested_service_account_id IS NULL
                OR requested_credential_digest IS NOT NULL
                OR requested_limit IS NOT NULL))
       OR (requested_purpose IN (
               'approval_redeem_issue', 'approval_redeem_rotate')
           AND (requested_approval_id IS NULL
                OR requested_approval_revision IS NULL
                OR requested_approval_revision < 1
                OR requested_service_account_id IS NULL
                OR requested_credential_digest IS NULL
                OR octet_length(requested_credential_digest) <> 32
                OR requested_limit IS NOT NULL))
    THEN
        RAISE EXCEPTION USING ERRCODE = '22023',
            MESSAGE = 'service_account_assertion_invalid';
    END IF;
    purpose_bytes := convert_to(requested_purpose, 'UTF8');
    RETURN convert_to('ragtest.service-account.assertion.v1', 'UTF8')
        || int4send(octet_length(purpose_bytes)) || purpose_bytes
        || int4send(requested_key_version)
        || uuid_send(requested_tenant_id)
        || requested_tenant_actor_digest
        || int8send(requested_org_policy_epoch)
        || CASE WHEN requested_approval_id IS NULL
                THEN decode('00', 'hex')
                ELSE decode('01', 'hex')
                     || uuid_send(requested_approval_id) END
        || CASE WHEN requested_approval_revision IS NULL
                THEN decode('00', 'hex')
                ELSE decode('01', 'hex')
                     || int8send(requested_approval_revision) END
        || CASE WHEN requested_service_account_id IS NULL
                THEN decode('00', 'hex')
                ELSE decode('01', 'hex')
                     || uuid_send(requested_service_account_id) END
        || CASE WHEN requested_credential_digest IS NULL
                THEN decode('00', 'hex')
                ELSE decode('01', 'hex')
                     || requested_credential_digest END
        || CASE WHEN requested_limit IS NULL
                THEN decode('00', 'hex')
                ELSE decode('01', 'hex') || int4send(requested_limit) END
        || int8send(requested_issued_at)
        || int8send(requested_expires_at)
        || requested_nonce;
END
$service_account_assertion_payload$;

CREATE OR REPLACE FUNCTION rag_control.control_secure_bytea_equal(
    left_value bytea, right_value bytea)
RETURNS boolean
LANGUAGE plpgsql IMMUTABLE
SET search_path = pg_catalog
AS $secure_bytea_equal$
DECLARE
    difference integer := 0;
    byte_index integer;
BEGIN
    IF left_value IS NULL OR right_value IS NULL
       OR octet_length(left_value) <> 32
       OR octet_length(right_value) <> 32 THEN
        RETURN false;
    END IF;
    FOR byte_index IN 0..31 LOOP
        difference := difference
            | (get_byte(left_value, byte_index)
               # get_byte(right_value, byte_index));
    END LOOP;
    RETURN difference = 0;
END
$secure_bytea_equal$;

CREATE OR REPLACE FUNCTION rag_control.control_consume_service_account_assertion(
    requested_assertion_version smallint,
    requested_purpose text,
    requested_key_version integer,
    requested_tenant_id uuid,
    requested_tenant_actor_digest bytea,
    requested_org_policy_epoch bigint,
    requested_approval_id uuid,
    requested_approval_revision bigint,
    requested_service_account_id uuid,
    requested_credential_digest bytea,
    requested_limit integer,
    requested_issued_at bigint,
    requested_expires_at bigint,
    requested_nonce bytea,
    requested_mac bytea)
RETURNS bytea
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $consume_service_account_assertion$
DECLARE
    verification_key control_service_account_assertion_keys%ROWTYPE;
    payload bytea;
    expected_mac bytea;
    now_epoch bigint;
    inserted integer;
BEGIN
    now_epoch := floor(extract(epoch FROM statement_timestamp()))::bigint;
    IF requested_assertion_version IS DISTINCT FROM 1
       OR requested_mac IS NULL OR octet_length(requested_mac) <> 32
       OR requested_issued_at > now_epoch + 5
       OR requested_expires_at <= now_epoch
    THEN
        RAISE EXCEPTION USING ERRCODE = '42501',
            MESSAGE = 'service_account_assertion_invalid';
    END IF;
    payload := rag_control.control_service_account_assertion_payload(
        requested_purpose, requested_key_version, requested_tenant_id,
        requested_tenant_actor_digest, requested_org_policy_epoch,
        requested_approval_id, requested_approval_revision,
        requested_service_account_id, requested_credential_digest,
        requested_limit,
        requested_issued_at, requested_expires_at, requested_nonce);
    SELECT key.* INTO verification_key
    FROM rag_control.control_service_account_assertion_keys AS key
    WHERE key.key_version = requested_key_version
      AND key.state IN ('active', 'verify_only')
      AND key.not_before <= statement_timestamp()
      AND (key.verify_started_at IS NULL
           OR key.verify_started_at <= statement_timestamp())
      AND (key.verify_until IS NULL
           OR key.verify_until > statement_timestamp())
    FOR SHARE;
    IF verification_key.key_version IS NULL THEN
        RAISE EXCEPTION USING ERRCODE = '42501',
            MESSAGE = 'service_account_assertion_invalid';
    END IF;
    expected_mac := public.hmac(payload, verification_key.secret, 'sha256');
    IF NOT rag_control.control_secure_bytea_equal(
            expected_mac, requested_mac) THEN
        RAISE EXCEPTION USING ERRCODE = '42501',
            MESSAGE = 'service_account_assertion_invalid';
    END IF;
    DELETE FROM rag_control.control_service_account_assertion_nonces AS nonce
    WHERE (nonce.key_version, nonce.nonce) IN (
        SELECT expired.key_version, expired.nonce
        FROM rag_control.control_service_account_assertion_nonces AS expired
        WHERE expired.expires_at <= statement_timestamp()
        ORDER BY expired.expires_at, expired.key_version, expired.nonce
        LIMIT 128
        FOR UPDATE SKIP LOCKED);
    INSERT INTO rag_control.control_service_account_assertion_nonces (
        key_version, purpose, nonce, tenant_id, approval_id, expires_at)
    VALUES (requested_key_version, requested_purpose, requested_nonce,
            requested_tenant_id, requested_approval_id,
            to_timestamp(requested_expires_at))
    ON CONFLICT (key_version, nonce) DO NOTHING;
    GET DIAGNOSTICS inserted = ROW_COUNT;
    IF inserted <> 1 THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_assertion_replayed';
    END IF;
    RETURN requested_tenant_actor_digest;
END
$consume_service_account_assertion$;

CREATE OR REPLACE FUNCTION
rag_control.control_asserted_list_redeemable_service_account_approvals(
    requested_assertion_version smallint,
    requested_key_version integer,
    requested_tenant_id uuid,
    requested_tenant_actor_digest bytea,
    requested_org_policy_epoch bigint,
    requested_limit integer,
    requested_issued_at bigint,
    requested_expires_at bigint,
    requested_nonce bytea,
    requested_mac bytea)
RETURNS TABLE (
    approval_id uuid,
    tenant_id uuid,
    service_account_id uuid,
    action text,
    state text,
    approval_revision bigint,
    reason_code text,
    scopes text[],
    account_expires_at timestamptz,
    credential_expires_at timestamptz,
    expected_account_revision bigint,
    control_policy_revision bigint,
    expires_at timestamptz,
    created_at timestamptz)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $asserted_list_service_account_approvals$
BEGIN
    PERFORM rag_control.control_consume_service_account_assertion(
        requested_assertion_version, 'approval_list', requested_key_version,
        requested_tenant_id, requested_tenant_actor_digest,
        requested_org_policy_epoch, NULL, NULL, NULL, NULL,
        requested_limit, requested_issued_at, requested_expires_at,
        requested_nonce, requested_mac);
    RETURN QUERY SELECT approval.approval_id, approval.tenant_id,
           approval.service_account_id, approval.action, approval.state,
           approval.approval_revision, approval.reason_code, approval.scopes,
           approval.account_expires_at, approval.credential_expires_at,
           approval.expected_account_revision,
           approval.control_policy_revision, approval.expires_at,
           approval.created_at
    FROM rag_control.control_service_account_approvals AS approval
    JOIN LATERAL rag_control.control_tenant_facts(
        approval.tenant_id) AS facts ON true
    LEFT JOIN rag_control.control_service_accounts AS account
      ON account.service_account_id = approval.service_account_id
    WHERE approval.tenant_id = requested_tenant_id
      AND approval.state = 'approved'
      AND approval.approval_revision = 1
      AND statement_timestamp() < approval.expires_at
      AND approval.control_policy_revision = facts.policy_revision
      AND (
          (approval.action = 'issue'
           AND account.service_account_id IS NULL)
          OR
          (approval.action = 'rotate'
           AND account.tenant_id = approval.tenant_id
           AND account.state = 'active'
           AND account.revision = approval.expected_account_revision
           AND statement_timestamp() < account.expires_at
           AND approval.credential_expires_at <= account.expires_at))
      AND requested_limit BETWEEN 1 AND 100
    ORDER BY approval.created_at, approval.approval_id
    LIMIT requested_limit;
END
$asserted_list_service_account_approvals$;

CREATE OR REPLACE FUNCTION
rag_control.control_asserted_get_redeemable_service_account_approval(
    requested_assertion_version smallint,
    requested_key_version integer,
    requested_tenant_id uuid,
    requested_tenant_actor_digest bytea,
    requested_org_policy_epoch bigint,
    requested_approval_id uuid,
    requested_approval_revision bigint,
    requested_service_account_id uuid,
    requested_issued_at bigint,
    requested_expires_at bigint,
    requested_nonce bytea,
    requested_mac bytea)
RETURNS TABLE (
    approval_id uuid,
    tenant_id uuid,
    service_account_id uuid,
    action text,
    state text,
    approval_revision bigint,
    reason_code text,
    scopes text[],
    account_expires_at timestamptz,
    credential_expires_at timestamptz,
    expected_account_revision bigint,
    control_policy_revision bigint,
    expires_at timestamptz,
    created_at timestamptz)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $asserted_get_service_account_approval$
BEGIN
    PERFORM rag_control.control_consume_service_account_assertion(
        requested_assertion_version, 'approval_get', requested_key_version,
        requested_tenant_id, requested_tenant_actor_digest,
        requested_org_policy_epoch, requested_approval_id,
        requested_approval_revision, requested_service_account_id,
        NULL, NULL, requested_issued_at, requested_expires_at,
        requested_nonce, requested_mac);
    RETURN QUERY SELECT approval.approval_id, approval.tenant_id,
           approval.service_account_id, approval.action, approval.state,
           approval.approval_revision, approval.reason_code, approval.scopes,
           approval.account_expires_at, approval.credential_expires_at,
           approval.expected_account_revision,
           approval.control_policy_revision, approval.expires_at,
           approval.created_at
    FROM rag_control.control_service_account_approvals AS approval
    JOIN LATERAL rag_control.control_tenant_facts(
        approval.tenant_id) AS facts ON true
    LEFT JOIN rag_control.control_service_accounts AS account
      ON account.service_account_id = approval.service_account_id
    WHERE approval.approval_id = requested_approval_id
      AND approval.tenant_id = requested_tenant_id
      AND approval.service_account_id = requested_service_account_id
      AND approval.state = 'approved'
      AND approval.approval_revision = requested_approval_revision
      AND approval.approval_revision = 1
      AND statement_timestamp() < approval.expires_at
      AND approval.control_policy_revision = facts.policy_revision
      AND (
          (approval.action = 'issue'
           AND account.service_account_id IS NULL)
          OR
          (approval.action = 'rotate'
           AND account.tenant_id = approval.tenant_id
           AND account.state = 'active'
           AND account.revision = approval.expected_account_revision
           AND statement_timestamp() < account.expires_at
           AND approval.credential_expires_at <= account.expires_at))
    FOR SHARE OF approval;
END
$asserted_get_service_account_approval$;

-- These redemption functions remain an offline atomic authority until the
-- asserted redeemer role is installed by a later migration.
-- PUBLIC is revoked below and no deployable role receives EXECUTE until the
-- data-plane architect+admin proof can be cryptographically bound here.

CREATE OR REPLACE FUNCTION
rag_control.control_asserted_redeem_service_account_issue(
    requested_assertion_version smallint,
    requested_key_version integer,
    requested_tenant_id uuid,
    requested_tenant_actor_digest bytea,
    requested_org_policy_epoch bigint,
    requested_approval_id uuid,
    requested_expected_approval_revision bigint,
    requested_account_id uuid,
    requested_credential_digest bytea,
    requested_issued_at bigint,
    requested_expires_at bigint,
    requested_nonce bytea,
    requested_mac bytea,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS TABLE (
    account_revision bigint,
    credential_version integer,
    account_expires_at timestamptz,
    credential_expires_at timestamptz)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $redeem_service_account_issue$
DECLARE
    approved rag_control.control_service_account_approvals%ROWTYPE;
    tenant_policy_revision bigint;
BEGIN
    PERFORM rag_control.control_consume_service_account_assertion(
        requested_assertion_version, 'approval_redeem_issue',
        requested_key_version, requested_tenant_id,
        requested_tenant_actor_digest, requested_org_policy_epoch,
        requested_approval_id, requested_expected_approval_revision,
        requested_account_id, requested_credential_digest, NULL,
        requested_issued_at, requested_expires_at, requested_nonce,
        requested_mac);
    SELECT tenant.policy_revision INTO tenant_policy_revision
    FROM rag_control.control_tenants AS tenant
    WHERE tenant.tenant_id = requested_tenant_id
      AND tenant.lifecycle = 'active'
    FOR SHARE;
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    SELECT approval.* INTO approved
    FROM rag_control.control_service_account_approvals AS approval
    WHERE approval.approval_id = requested_approval_id
      AND approval.tenant_id = requested_tenant_id
      AND approval.service_account_id = requested_account_id
    FOR UPDATE;
    IF approved.approval_id IS NULL
       OR approved.action <> 'issue'
       OR approved.state <> 'approved'
       OR approved.approval_revision <> requested_expected_approval_revision
       OR requested_expected_approval_revision <> 1
       OR approved.expires_at <= statement_timestamp()
       OR approved.control_policy_revision <> tenant_policy_revision
       OR requested_org_policy_epoch < 1
       OR octet_length(requested_tenant_actor_digest) <> 32
       OR octet_length(requested_credential_digest) <> 32
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
       OR approved.account_expires_at <= statement_timestamp()
       OR approved.credential_expires_at <= statement_timestamp()
       OR approved.account_expires_at < approved.credential_expires_at
       OR NOT EXISTS (
           SELECT 1 FROM rag_control.control_tenant_facts(
               requested_tenant_id))
       OR EXISTS (
           SELECT 1 FROM rag_control.control_service_accounts
           WHERE service_account_id = requested_account_id)
    THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_approval_conflict';
    END IF;
    INSERT INTO rag_control.control_service_accounts (
        service_account_id, tenant_id, state, expires_at)
    VALUES (requested_account_id, requested_tenant_id, 'active',
            approved.account_expires_at);
    INSERT INTO rag_control.control_service_account_scopes (
        service_account_id, scope_code)
    SELECT requested_account_id, scope FROM unnest(approved.scopes) AS scope;
    INSERT INTO rag_control.control_service_account_credentials (
        service_account_id, credential_version, digest, state,
        not_before, expires_at)
    VALUES (requested_account_id, 1, requested_credential_digest, 'active',
            statement_timestamp(), approved.credential_expires_at);
    UPDATE rag_control.control_service_account_approvals
    SET state = 'redeemed', approval_revision = 2,
        redeemed_at = statement_timestamp()
    WHERE approval_id = requested_approval_id;
    INSERT INTO rag_control.control_service_account_events (
        operator_id, actor_kind, tenant_actor_digest, org_policy_epoch,
        target_tenant_id, service_account_id, action, reason_code,
        expected_revision, resulting_revision, request_digest,
        resulting_fact_digest)
    VALUES (approved.platform_operator_id, 'tenant_org_admin',
            requested_tenant_actor_digest, requested_org_policy_epoch,
            requested_tenant_id, requested_account_id, 'service_account_issue',
            approved.reason_code, NULL, 1, requested_request_digest,
            requested_resulting_fact_digest);
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, tenant_actor_digest, org_policy_epoch,
        prior_state, resulting_state, prior_revision, approval_revision,
        approval_created_at, approval_expires_at, request_digest,
        resulting_fact_digest)
    VALUES (approved.approval_id, approved.tenant_id,
            approved.service_account_id, 'approval_redeemed',
            'approval_redeemed', 'tenant_org_admin',
            requested_tenant_actor_digest, requested_org_policy_epoch,
            'approved', 'redeemed', 1, 2, approved.created_at,
            approved.expires_at, approved.request_digest,
            approved.resulting_fact_digest);
    RETURN QUERY SELECT 1::bigint, 1, approved.account_expires_at,
                        approved.credential_expires_at;
END
$redeem_service_account_issue$;

CREATE OR REPLACE FUNCTION
rag_control.control_asserted_redeem_service_account_rotation(
    requested_assertion_version smallint,
    requested_key_version integer,
    requested_tenant_id uuid,
    requested_tenant_actor_digest bytea,
    requested_org_policy_epoch bigint,
    requested_approval_id uuid,
    requested_expected_approval_revision bigint,
    requested_account_id uuid,
    requested_credential_digest bytea,
    requested_issued_at bigint,
    requested_expires_at bigint,
    requested_nonce bytea,
    requested_mac bytea,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS TABLE (
    account_revision bigint,
    credential_version integer,
    account_expires_at timestamptz,
    credential_expires_at timestamptz)
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $redeem_service_account_rotation$
DECLARE
    account rag_control.control_service_accounts%ROWTYPE;
    approved rag_control.control_service_account_approvals%ROWTYPE;
    next_revision bigint;
    tenant_policy_revision bigint;
BEGIN
    PERFORM rag_control.control_consume_service_account_assertion(
        requested_assertion_version, 'approval_redeem_rotate',
        requested_key_version, requested_tenant_id,
        requested_tenant_actor_digest, requested_org_policy_epoch,
        requested_approval_id, requested_expected_approval_revision,
        requested_account_id, requested_credential_digest, NULL,
        requested_issued_at, requested_expires_at, requested_nonce,
        requested_mac);
    SELECT tenant.policy_revision INTO tenant_policy_revision
    FROM rag_control.control_tenants AS tenant
    WHERE tenant.tenant_id = requested_tenant_id
      AND tenant.lifecycle = 'active'
    FOR SHARE;
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    SELECT approval.* INTO approved
    FROM rag_control.control_service_account_approvals AS approval
    WHERE approval.approval_id = requested_approval_id
      AND approval.tenant_id = requested_tenant_id
      AND approval.service_account_id = requested_account_id
    FOR UPDATE;
    SELECT service_account.* INTO account
    FROM rag_control.control_service_accounts AS service_account
    WHERE service_account.service_account_id = requested_account_id
      AND service_account.tenant_id = requested_tenant_id
      AND service_account.state = 'active'
    FOR UPDATE;
    IF approved.approval_id IS NULL
       OR approved.action <> 'rotate'
       OR approved.state <> 'approved'
       OR approved.approval_revision <> requested_expected_approval_revision
       OR requested_expected_approval_revision <> 1
       OR approved.expires_at <= statement_timestamp()
       OR approved.control_policy_revision <> tenant_policy_revision
       OR account.service_account_id IS NULL
       OR account.revision <> approved.expected_account_revision
       OR account.expires_at <= statement_timestamp()
       OR account.expires_at < approved.credential_expires_at
       OR requested_org_policy_epoch < 1
       OR octet_length(requested_tenant_actor_digest) <> 32
       OR octet_length(requested_credential_digest) <> 32
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
       OR NOT EXISTS (
           SELECT 1 FROM rag_control.control_tenant_facts(
               requested_tenant_id))
    THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_approval_conflict';
    END IF;
    next_revision := account.revision + 1;
    UPDATE rag_control.control_service_account_credentials
    SET state = 'retired'
    WHERE service_account_id = requested_account_id AND state = 'active';
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_approval_conflict';
    END IF;
    INSERT INTO rag_control.control_service_account_credentials (
        service_account_id, credential_version, digest, state,
        not_before, expires_at)
    VALUES (requested_account_id, next_revision, requested_credential_digest,
            'active', statement_timestamp(), approved.credential_expires_at);
    UPDATE rag_control.control_service_accounts
    SET revision = next_revision
    WHERE service_account_id = requested_account_id;
    UPDATE rag_control.control_service_account_approvals
    SET state = 'redeemed', approval_revision = 2,
        redeemed_at = statement_timestamp()
    WHERE approval_id = requested_approval_id;
    INSERT INTO rag_control.control_service_account_events (
        operator_id, actor_kind, tenant_actor_digest, org_policy_epoch,
        target_tenant_id, service_account_id, action, reason_code,
        expected_revision, resulting_revision, request_digest,
        resulting_fact_digest)
    VALUES (approved.platform_operator_id, 'tenant_org_admin',
            requested_tenant_actor_digest, requested_org_policy_epoch,
            requested_tenant_id, requested_account_id,
            'service_account_rotate',
            approved.reason_code, account.revision, next_revision,
            requested_request_digest, requested_resulting_fact_digest);
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, tenant_actor_digest, org_policy_epoch,
        prior_state, resulting_state, prior_revision, approval_revision,
        approval_created_at, approval_expires_at, request_digest,
        resulting_fact_digest)
    VALUES (approved.approval_id, approved.tenant_id,
            approved.service_account_id, 'approval_redeemed',
            'approval_redeemed', 'tenant_org_admin',
            requested_tenant_actor_digest, requested_org_policy_epoch,
            'approved', 'redeemed', 1, 2, approved.created_at,
            approved.expires_at, approved.request_digest,
            approved.resulting_fact_digest);
    RETURN QUERY SELECT next_revision, next_revision::integer,
                        account.expires_at, approved.credential_expires_at;
END
$redeem_service_account_rotation$;

CREATE OR REPLACE FUNCTION rag_control.control_issue_service_account(
    requested_operator_key_version integer,
    requested_operator_digest bytea,
    requested_tenant_id uuid,
    requested_account_id uuid,
    requested_digest bytea,
    requested_scopes text[],
    requested_account_expires_at timestamptz,
    requested_credential_expires_at timestamptz,
    requested_reason_code text,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS bigint
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $issue_service_account$
DECLARE
    canonical_scopes text[];
    resolved_operator_id uuid;
BEGIN
    resolved_operator_id := rag_control.control_require_platform_security(
        requested_operator_key_version, requested_operator_digest);
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    SELECT array_agg(scope ORDER BY scope) INTO canonical_scopes
    FROM (SELECT DISTINCT unnest(requested_scopes) AS scope) AS normalized;
    IF canonical_scopes IS NULL
       OR requested_scopes <> canonical_scopes
       OR NOT requested_scopes <@ ARRAY[
           'rag.query', 'documents.read', 'documents.write',
           'documents.lifecycle', 'collections.manage',
           'tables.extract']::text[]
       OR octet_length(requested_digest) <> 32
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
       OR requested_reason_code NOT IN (
           'security_provisioning', 'incident_response')
       OR requested_credential_expires_at <= statement_timestamp()
       OR requested_account_expires_at < requested_credential_expires_at
       OR requested_account_expires_at >
          statement_timestamp() + interval '366 days'
       OR NOT EXISTS (
           SELECT 1 FROM rag_control.control_tenant_facts(
               requested_tenant_id))
    THEN
        RAISE EXCEPTION USING ERRCODE = '22023',
            MESSAGE = 'service_account_request_invalid';
    END IF;

    INSERT INTO rag_control.control_service_accounts (
        service_account_id, tenant_id, state, expires_at)
    VALUES (requested_account_id, requested_tenant_id, 'active',
            requested_account_expires_at);
    INSERT INTO rag_control.control_service_account_scopes (
        service_account_id, scope_code)
    SELECT requested_account_id, scope FROM unnest(requested_scopes) AS scope;
    INSERT INTO rag_control.control_service_account_credentials (
        service_account_id, credential_version, digest, state,
        not_before, expires_at)
    VALUES (requested_account_id, 1, requested_digest, 'active',
            statement_timestamp(), requested_credential_expires_at);
    INSERT INTO rag_control.control_service_account_events (
        operator_id, target_tenant_id, service_account_id, action,
        reason_code, expected_revision, resulting_revision, request_digest,
        resulting_fact_digest)
    VALUES (resolved_operator_id, requested_tenant_id,
            requested_account_id, 'service_account_issue',
            requested_reason_code, NULL, 1, requested_request_digest,
            requested_resulting_fact_digest);
    RETURN 1;
END
$issue_service_account$;

CREATE OR REPLACE FUNCTION rag_control.control_rotate_service_account(
    requested_operator_key_version integer,
    requested_operator_digest bytea,
    requested_tenant_id uuid,
    requested_account_id uuid,
    requested_expected_revision bigint,
    requested_digest bytea,
    requested_credential_expires_at timestamptz,
    requested_reason_code text,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS bigint
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $rotate_service_account$
DECLARE
    account_revision bigint;
    account_expiry timestamptz;
    next_revision bigint;
    resolved_operator_id uuid;
BEGIN
    resolved_operator_id := rag_control.control_require_platform_security(
        requested_operator_key_version, requested_operator_digest);
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    SELECT revision, expires_at INTO account_revision, account_expiry
    FROM rag_control.control_service_accounts
    WHERE service_account_id = requested_account_id
      AND tenant_id = requested_tenant_id
      AND state = 'active'
    FOR UPDATE;
    IF account_revision IS NULL
       OR account_revision <> requested_expected_revision
       OR octet_length(requested_digest) <> 32
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
       OR requested_reason_code NOT IN (
           'scheduled_rotation', 'suspected_compromise')
       OR requested_credential_expires_at <= statement_timestamp()
       OR account_expiry < requested_credential_expires_at
    THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_revision_conflict';
    END IF;
    next_revision := account_revision + 1;
    UPDATE rag_control.control_service_account_credentials
    SET state = 'retired'
    WHERE service_account_id = requested_account_id AND state = 'active';
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_revision_conflict';
    END IF;
    INSERT INTO rag_control.control_service_account_credentials (
        service_account_id, credential_version, digest, state,
        not_before, expires_at)
    VALUES (requested_account_id, next_revision, requested_digest, 'active',
            statement_timestamp(), requested_credential_expires_at);
    UPDATE rag_control.control_service_accounts
    SET revision = next_revision
    WHERE service_account_id = requested_account_id;
    INSERT INTO rag_control.control_service_account_events (
        operator_id, target_tenant_id, service_account_id, action,
        reason_code, expected_revision, resulting_revision, request_digest,
        resulting_fact_digest)
    VALUES (resolved_operator_id, requested_tenant_id,
            requested_account_id, 'service_account_rotate',
            requested_reason_code, requested_expected_revision,
            next_revision, requested_request_digest,
            requested_resulting_fact_digest);
    RETURN next_revision;
END
$rotate_service_account$;

CREATE OR REPLACE FUNCTION rag_control.control_revoke_service_account(
    requested_operator_key_version integer,
    requested_operator_digest bytea,
    requested_tenant_id uuid,
    requested_account_id uuid,
    requested_expected_revision bigint,
    requested_reason_code text,
    requested_request_digest bytea,
    requested_resulting_fact_digest bytea)
RETURNS bigint
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, rag_control
AS $revoke_service_account$
DECLARE
    account_revision bigint;
    next_revision bigint;
    resolved_operator_id uuid;
BEGIN
    resolved_operator_id := rag_control.control_require_platform_security(
        requested_operator_key_version, requested_operator_digest);
    PERFORM rag_control.control_lock_service_account(requested_account_id);
    SELECT revision INTO account_revision
    FROM rag_control.control_service_accounts
    WHERE service_account_id = requested_account_id
      AND tenant_id = requested_tenant_id
      AND state = 'active'
    FOR UPDATE;
    IF account_revision IS NULL
       OR account_revision <> requested_expected_revision
       OR octet_length(requested_request_digest) <> 32
       OR octet_length(requested_resulting_fact_digest) <> 32
       OR requested_reason_code NOT IN (
           'suspected_compromise', 'tenant_suspension',
           'security_response', 'access_removed')
    THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_revision_conflict';
    END IF;
    next_revision := account_revision + 1;
    UPDATE rag_control.control_service_account_credentials
    SET state = 'revoked'
    WHERE service_account_id = requested_account_id AND state = 'active';
    IF NOT FOUND THEN
        RAISE EXCEPTION USING ERRCODE = '40001',
            MESSAGE = 'service_account_revision_conflict';
    END IF;
    UPDATE rag_control.control_service_accounts
    SET state = 'revoked', revision = next_revision
    WHERE service_account_id = requested_account_id;
    WITH cancelled AS (
        UPDATE rag_control.control_service_account_approvals
        SET state = 'cancelled', approval_revision = 2
        WHERE tenant_id = requested_tenant_id
          AND service_account_id = requested_account_id
          AND state = 'approved'
        RETURNING *
    )
    INSERT INTO rag_control.control_service_account_approval_events (
        approval_id, target_tenant_id, service_account_id, action,
        reason_code, actor_kind, platform_operator_id, prior_state,
        resulting_state, prior_revision, approval_revision,
        approval_created_at, approval_expires_at, request_digest,
        resulting_fact_digest)
    SELECT approval_id, tenant_id, service_account_id,
           'approval_cancelled', 'service_account_revoked',
           'platform_security', resolved_operator_id,
           'approved', 'cancelled', 1, approval_revision, created_at,
           expires_at, request_digest, resulting_fact_digest
    FROM cancelled;
    INSERT INTO rag_control.control_service_account_events (
        operator_id, target_tenant_id, service_account_id, action,
        reason_code, expected_revision, resulting_revision, request_digest,
        resulting_fact_digest)
    VALUES (resolved_operator_id, requested_tenant_id,
            requested_account_id, 'service_account_revoke',
            requested_reason_code, requested_expected_revision,
            next_revision, requested_request_digest,
            requested_resulting_fact_digest);
    RETURN next_revision;
END
$revoke_service_account$;

-- Version 6 removes every assertion-less tenant redemption authority.  The
-- online role introduced with this schema can reach only the asserted outer
-- functions below; an older application cannot retain a callable bypass.
DROP FUNCTION IF EXISTS
    rag_control.control_list_redeemable_service_account_approvals(
        uuid, integer);
DROP FUNCTION IF EXISTS rag_control.control_redeem_service_account_issue(
    uuid, uuid, uuid, bigint, bytea, bigint, bytea, bytea, bytea);
DROP FUNCTION IF EXISTS rag_control.control_redeem_service_account_rotation(
    uuid, uuid, uuid, bigint, bytea, bigint, bytea, bytea, bytea);

REVOKE ALL ON FUNCTION rag_control.control_events_immutable() FROM PUBLIC;
REVOKE ALL ON FUNCTION
    rag_control.control_seal_service_account_approval_event() FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_service_account_assertion_payload(
    text, integer, uuid, bytea, bigint, uuid, bigint, uuid, bytea, integer,
    bigint, bigint, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_secure_bytea_equal(bytea, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_consume_service_account_assertion(
    smallint, text, integer, uuid, bytea, bigint, uuid, bigint, uuid, bytea,
    integer, bigint, bigint, bytea, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_tenant_facts(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_resolve_identity(integer, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_resolve_platform_operator(
    integer, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_resolve_service_account(
    uuid, integer, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_require_platform_security(
    integer, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_lock_service_account(uuid)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_expire_service_account_approval(uuid)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_approve_service_account_issue(
    integer, bytea, uuid, uuid, uuid, text[], timestamptz, timestamptz,
    bigint, text, bytea, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_approve_service_account_rotation(
    integer, bytea, uuid, uuid, uuid, bigint, timestamptz, bigint, text,
    bytea, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_cancel_service_account_approval(
    integer, bytea, uuid, uuid, uuid, bigint, text)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION
    rag_control.control_asserted_list_redeemable_service_account_approvals(
        smallint, integer, uuid, bytea, bigint, integer, bigint, bigint,
        bytea, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION
    rag_control.control_asserted_get_redeemable_service_account_approval(
        smallint, integer, uuid, bytea, bigint, uuid, bigint, uuid, bigint,
        bigint, bytea, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION
    rag_control.control_asserted_redeem_service_account_issue(
        smallint, integer, uuid, bytea, bigint, uuid, bigint, uuid, bytea,
        bigint, bigint, bytea, bytea, bytea, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION
    rag_control.control_asserted_redeem_service_account_rotation(
        smallint, integer, uuid, bytea, bigint, uuid, bigint, uuid, bytea,
        bigint, bigint, bytea, bytea, bytea, bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_issue_service_account(
    integer, bytea, uuid, uuid, bytea, text[], timestamptz, timestamptz,
    text, bytea, bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_rotate_service_account(
    integer, bytea, uuid, uuid, bigint, bytea, timestamptz, text, bytea,
    bytea)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_revoke_service_account(
    integer, bytea, uuid, uuid, bigint, text, bytea, bytea) FROM PUBLIC;
