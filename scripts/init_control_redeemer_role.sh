#!/bin/bash
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${CONTROL_REDEEMER_PASSWORD:?CONTROL_REDEEMER_PASSWORD is required}"

psql --set=ON_ERROR_STOP=1 \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set=database_name="${POSTGRES_DB}" <<'SQL'
\getenv redeemer_password CONTROL_REDEEMER_PASSWORD
SELECT format('CREATE ROLE rag_control_redeemer LOGIN PASSWORD %L',
              :'redeemer_password')
WHERE NOT EXISTS (
    SELECT 1 FROM pg_roles WHERE rolname = 'rag_control_redeemer')
\gexec

ALTER ROLE rag_control_redeemer LOGIN PASSWORD :'redeemer_password'
    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
GRANT CONNECT ON DATABASE :"database_name" TO rag_control_redeemer;
REVOKE CREATE ON DATABASE :"database_name" FROM rag_control_redeemer;
REVOKE ALL ON SCHEMA rag_control FROM rag_control_redeemer;
GRANT USAGE ON SCHEMA rag_control TO rag_control_redeemer;
REVOKE ALL ON ALL TABLES IN SCHEMA rag_control FROM rag_control_redeemer;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA rag_control FROM rag_control_redeemer;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA rag_control FROM rag_control_redeemer;
GRANT EXECUTE ON FUNCTION
    rag_control.control_asserted_list_redeemable_service_account_approvals(
        smallint, integer, uuid, bytea, bigint, integer, bigint, bigint,
        bytea, bytea) TO rag_control_redeemer;
GRANT EXECUTE ON FUNCTION
    rag_control.control_asserted_get_redeemable_service_account_approval(
        smallint, integer, uuid, bytea, bigint, uuid, bigint, uuid, bigint,
        bigint, bytea, bytea) TO rag_control_redeemer;
GRANT EXECUTE ON FUNCTION
    rag_control.control_asserted_redeem_service_account_issue(
        smallint, integer, uuid, bytea, bigint, uuid, bigint, uuid, bytea,
        bigint, bigint, bytea, bytea, bytea, bytea)
    TO rag_control_redeemer;
GRANT EXECUTE ON FUNCTION
    rag_control.control_asserted_redeem_service_account_rotation(
        smallint, integer, uuid, bytea, bigint, uuid, bigint, uuid, bytea,
        bigint, bigint, bytea, bytea, bytea, bytea)
    TO rag_control_redeemer;
GRANT SELECT ON rag_control.control_schema_state TO rag_control_redeemer;
SQL
