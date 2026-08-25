#!/bin/bash
# Provision the non-owner runtime role after schema.sql on an empty database.
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${DB_RUNTIME_PASSWORD:?DB_RUNTIME_PASSWORD is required}"
: "${RAG_DB_CONTEXT_SECRET:?RAG_DB_CONTEXT_SECRET is required}"

psql --set=ON_ERROR_STOP=1 \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set=database_name="${POSTGRES_DB}" \
  --set=runtime_password="${DB_RUNTIME_PASSWORD}" \
  --set=context_secret="${RAG_DB_CONTEXT_SECRET}" <<'SQL'
SELECT format('CREATE ROLE rag_runtime LOGIN PASSWORD %L', :'runtime_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_runtime')
\gexec

ALTER ROLE rag_runtime NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOINHERIT;
GRANT CONNECT ON DATABASE :"database_name" TO rag_runtime;
GRANT USAGE ON SCHEMA public TO rag_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO rag_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO rag_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO rag_runtime;
REVOKE ALL ON rag_context_secrets FROM rag_runtime;
REVOKE ALL ON rag_service_account_assertion_keys FROM rag_runtime;
REVOKE ALL ON org_identity_tenant_bindings FROM rag_runtime;
SELECT 'REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER '
       'ON rag_schema_state FROM rag_runtime'
WHERE to_regclass('rag_schema_state') IS NOT NULL
\gexec
SELECT 'GRANT SELECT ON rag_schema_state TO rag_runtime'
WHERE to_regclass('rag_schema_state') IS NOT NULL
\gexec
SELECT 'REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER '
       'ON rag_schema_history FROM rag_runtime'
WHERE to_regclass('rag_schema_history') IS NOT NULL
\gexec
INSERT INTO rag_context_secrets (singleton, secret)
VALUES (true, convert_to(:'context_secret', 'UTF8'))
ON CONFLICT (singleton) DO UPDATE SET secret = EXCLUDED.secret;
SQL
