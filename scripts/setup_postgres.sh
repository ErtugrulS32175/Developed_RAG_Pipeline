#!/bin/bash
# Self-hosted PostgreSQL + pgvector via Docker (replaces Qdrant Cloud).
# No Visual Studio toolchain needed -- pgvector has no official Windows build,
# so this runs Postgres in the official pgvector/pgvector Linux image instead
# of the native Windows Postgres service (which stays untouched on port 5432).
# Run from the repo root: ./scripts/setup_postgres.sh
set -e

CONTAINER_NAME="ragtest-pgvector"
VOLUME_NAME="ragtest_pgvector_data"
HOST_PORT=5433
IMAGE="pgvector/pgvector:pg17"

# Separate DDL and runtime credentials plus the context signing key are
# required at run time and never committed.
DB_MIGRATION_PASSWORD="${DB_MIGRATION_PASSWORD:?set DB_MIGRATION_PASSWORD}"
DB_RUNTIME_PASSWORD="${DB_RUNTIME_PASSWORD:?set DB_RUNTIME_PASSWORD}"
RAG_DB_CONTEXT_SECRET="${RAG_DB_CONTEXT_SECRET:?set RAG_DB_CONTEXT_SECRET}"

echo "[1/4] Pulling ${IMAGE}"
docker pull "${IMAGE}"

echo "[2/4] Creating named volume for persistence: ${VOLUME_NAME}"
docker volume create "${VOLUME_NAME}" >/dev/null 2>&1 || true

echo "[3/4] Starting container ${CONTAINER_NAME} on host port ${HOST_PORT}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e POSTGRES_USER=rag_migrator \
  -e POSTGRES_PASSWORD="${DB_MIGRATION_PASSWORD}" \
  -e POSTGRES_DB=ragdb \
  -e DB_RUNTIME_PASSWORD="${DB_RUNTIME_PASSWORD}" \
  -e RAG_DB_CONTEXT_SECRET="${RAG_DB_CONTEXT_SECRET}" \
  -p "127.0.0.1:${HOST_PORT}:5432" \
  -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
  "${IMAGE}"

echo "[4/4] Waiting for Postgres to accept connections..."
until docker exec "${CONTAINER_NAME}" pg_isready -U rag_migrator >/dev/null 2>&1; do
  sleep 1
done

echo "Enabling pgvector extension and creating schema..."
docker exec "${CONTAINER_NAME}" psql -U rag_migrator -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec -i "${CONTAINER_NAME}" psql -U rag_migrator -d ragdb < pipeline/index/schema.sql
docker exec -i "${CONTAINER_NAME}" bash -s < scripts/init_runtime_role.sh

echo "Done."
echo "Runtime DSN:   postgresql://rag_runtime:<DB_RUNTIME_PASSWORD>@localhost:${HOST_PORT}/ragdb"
echo "Migration DSN: postgresql://rag_migrator:<DB_MIGRATION_PASSWORD>@localhost:${HOST_PORT}/ragdb"
echo "To stop:        docker stop ${CONTAINER_NAME}"
echo "To start again: docker start ${CONTAINER_NAME}"
