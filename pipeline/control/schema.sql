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

DROP TRIGGER IF EXISTS control_events_immutable_write
    ON rag_control.control_admin_events;
CREATE TRIGGER control_events_immutable_write
BEFORE UPDATE OR DELETE ON rag_control.control_admin_events
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

REVOKE ALL ON FUNCTION rag_control.control_events_immutable() FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_tenant_facts(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION rag_control.control_resolve_identity(integer, bytea)
    FROM PUBLIC;
