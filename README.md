# Developed RAG Pipeline — Docling + vLLM + PostgreSQL/pgvector

A self-hosted RAG pipeline for Turkish PDF and image documents, built entirely on open-source components. All models run locally via vLLM, and the vector store is a self-hosted PostgreSQL + pgvector instance (via Docker) — no third-party managed service is required anywhere in the pipeline.

## Architecture

Ingestion: input -> format router -> parser -> chunker -> embeddings -> PostgreSQL (pgvector)

Query: question -> hybrid search (RRF) -> reranker -> LLM -> answer with source pages

Models are served via vLLM as OpenAI-compatible APIs on separate ports:

| Service | Model | Port |
|---|---|---|
| LLM | Qwen/Qwen3-14B | 8000 |
| Reranker | BAAI/bge-reranker-v2-m3 | 8002 |
| Embedding | BAAI/bge-m3 | 8011 |
| OCR (isolated service) | PaddleOCR PP-OCRv5 | 8100 |
| Table extraction (isolated service) | google/gemma-4-E4B-it | 8101 |

## Router

A format-aware router directs each input to the appropriate parser:

- Images go to the OCR pipeline for plain text; table detection on the same image is handled by the Gemma table service, since PaddleOCR's table-structure model misaligns cells on spreadsheet-style screenshots (extra phantom rows/columns, drifting column assignment).
- PDFs are analyzed per page: pages with a text layer (native) go to Docling with TableFormer for deterministic, high-fidelity table extraction; scanned pages go to the OCR pipeline for text plus the Gemma table service for tables.

Every input is normalized into a unified document representation, so the downstream chunking, embedding, and retrieval layers are identical regardless of source format. New input formats only require a new branch in the router.

## Layout

    pipeline/   core RAG code: router, ingest, query, db, embeddings, table export, schema
    services/   isolated microservices (own venvs): PaddleOCR, Gemma table extraction
    scripts/    setup scripts

## Retrieval

Hybrid search combines dense embeddings (bge-m3, pgvector `vector` column) for semantic matching with sparse BM25 (fastembed `Qdrant/bm25`, pgvector `sparsevec` column) for exact terms (proper nouns, codes, tickers), fused with Reciprocal Rank Fusion computed in Python (`db.hybrid_search`). A reranker (bge-reranker-v2-m3) reorders candidates before the LLM answers in Turkish and cites source pages.

BM25's token hashing produces indices beyond pgvector's `sparsevec` dimension cap, so every sparse index is remapped modulo a fixed `SPARSE_DIM` before storage/query (see `db.py`) — collisions are statistically negligible at this corpus size.

## Setup

Run from the repo root:

    ./scripts/setup.sh
    ./scripts/setup_paddle.sh
    ./scripts/setup_gemma.sh
    ./scripts/setup_postgres.sh
    cp .env.example .env

The OCR and table-extraction services each run in their own isolated environment, exposed over localhost, keeping their dependencies (and Transformers version) separate from the main pipeline and from each other. PostgreSQL + pgvector runs in Docker (official `pgvector/pgvector` image) rather than the native Windows PostgreSQL service, since pgvector has no official Windows build.

## Usage

Run from the repo root:

    nohup vllm serve BAAI/bge-m3 --task embed --gpu-memory-utilization 0.1 --port 8011 > embed.log 2>&1 &
    python3 -m pipeline.ingest_router ./data/yourfile.pdf
    python3 -m pipeline.query

## Stack

Docling (parsing, TableFormer), PaddleOCR (OCR), Gemma 4 E4B (table extraction on scanned/image tables), bge-m3 (embeddings), BM25 (sparse), PostgreSQL + pgvector (vector store, self-hosted via Docker), bge-reranker-v2-m3 (reranking), Qwen3-14B (LLM), all served with vLLM. Every component is open-source and runs fully on-premise.

## Future Work

- Validate the Gemma table service against PaddleOCR's table output on real GPU hardware (untested as of writing — built without GPU access to avoid rental cost).
- Evaluation harness (retrieval recall, answer accuracy, faithfulness).
- Add an HNSW index on `chunks.dense`/`chunks.sparse` once the corpus grows past a small demo dataset.
- Optional LLM upgrade.
