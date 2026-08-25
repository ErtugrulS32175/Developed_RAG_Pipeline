#!/bin/bash
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${CONTROL_ADMIN_PASSWORD:?CONTROL_ADMIN_PASSWORD is required}"

psql --set=ON_ERROR_STOP=1 \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set=database_name="${POSTGRES_DB}" <<'SQL'
\getenv admin_password CONTROL_ADMIN_PASSWORD
SELECT format('CREATE ROLE rag_control_admin LOGIN PASSWORD %L',
              :'admin_password')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'rag_control_admin')
\gexec

ALTER ROLE rag_control_admin LOGIN PASSWORD :'admin_password'
    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
GRANT CONNECT ON DATABASE :"database_name" TO rag_control_admin;
REVOKE CREATE ON DATABASE :"database_name" FROM rag_control_admin;
REVOKE ALL ON SCHEMA rag_control FROM rag_control_admin;
GRANT USAGE ON SCHEMA rag_control TO rag_control_admin;
REVOKE ALL ON ALL TABLES IN SCHEMA rag_control FROM rag_control_admin;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control FROM rag_control_admin;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA rag_control FROM rag_control_admin;
GRANT EXECUTE ON FUNCTION rag_control.control_issue_service_account(
    integer, bytea, uuid, uuid, bytea, text[], timestamptz, timestamptz,
    text, bytea, bytea) TO rag_control_admin;
GRANT EXECUTE ON FUNCTION rag_control.control_rotate_service_account(
    integer, bytea, uuid, uuid, bigint, bytea, timestamptz, text, bytea,
    bytea)
    TO rag_control_admin;
GRANT EXECUTE ON FUNCTION rag_control.control_revoke_service_account(
    integer, bytea, uuid, uuid, bigint, text, bytea, bytea)
    TO rag_control_admin;
GRANT SELECT ON rag_control.control_schema_state TO rag_control_admin;
SQL
