#!/bin/bash
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${CONTROL_RUNTIME_PASSWORD:?CONTROL_RUNTIME_PASSWORD is required}"

psql --set=ON_ERROR_STOP=1 \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set=database_name="${POSTGRES_DB}" <<'SQL'
\getenv runtime_password CONTROL_RUNTIME_PASSWORD
SELECT format('CREATE ROLE rag_control_runtime LOGIN PASSWORD %L',
              :'runtime_password')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'rag_control_runtime')
\gexec

ALTER ROLE rag_control_runtime LOGIN PASSWORD :'runtime_password'
    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
GRANT CONNECT ON DATABASE :"database_name" TO rag_control_runtime;
REVOKE CREATE ON DATABASE :"database_name" FROM rag_control_runtime;
REVOKE ALL ON SCHEMA rag_control FROM rag_control_runtime;
GRANT USAGE ON SCHEMA rag_control TO rag_control_runtime;
REVOKE ALL ON ALL TABLES IN SCHEMA rag_control FROM rag_control_runtime;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control FROM rag_control_runtime;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA rag_control FROM rag_control_runtime;
GRANT EXECUTE ON FUNCTION rag_control.control_tenant_facts(uuid)
    TO rag_control_runtime;
GRANT EXECUTE ON FUNCTION rag_control.control_resolve_identity(integer, bytea)
    TO rag_control_runtime;
GRANT EXECUTE ON FUNCTION rag_control.control_resolve_platform_operator(
    integer, bytea) TO rag_control_runtime;
GRANT EXECUTE ON FUNCTION rag_control.control_resolve_service_account(
    uuid, integer, bytea) TO rag_control_runtime;
GRANT SELECT ON rag_control.control_schema_state TO rag_control_runtime;
SQL
