# Cloud RAG Pipeline — Docling + vLLM + Qdrant

A self-hosted RAG pipeline for Turkish PDF and image documents, built entirely on open-source components. All models run locally via vLLM. The current setup uses Qdrant Cloud as the vector store, but every component — including Qdrant — can be run fully on-premise, so the architecture has no hard dependency on external services.

## Architecture

Ingestion: input -> format router -> parser -> chunker -> embeddings -> Qdrant

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

## Retrieval

Hybrid search combines dense embeddings (bge-m3) for semantic matching with sparse BM25 for exact terms (proper nouns, codes, tickers), fused with Reciprocal Rank Fusion. A reranker (bge-reranker-v2-m3) reorders candidates before the LLM answers in Turkish and cites source pages.

## Setup

    ./setup.sh
    ./setup_paddle.sh
    ./setup_gemma.sh
    cp .env.example .env

The OCR and table-extraction services each run in their own isolated environment, exposed over localhost, keeping their dependencies (and Transformers version) separate from the main pipeline and from each other.

## Usage

    nohup vllm serve BAAI/bge-m3 --task embed --gpu-memory-utilization 0.1 --port 8011 > embed.log 2>&1 &
    python3 ingest_router.py ./data/yourfile.pdf
    python3 query.py

## Stack

Docling (parsing, TableFormer), PaddleOCR (OCR), Gemma 4 E4B (table extraction on scanned/image tables), bge-m3 (embeddings), BM25 (sparse), Qdrant (vector store), bge-reranker-v2-m3 (reranking), Qwen3-14B (LLM), all served with vLLM. Every component is open-source and can run fully on-premise.

## Future Work

- Validate the Gemma table service against PaddleOCR's table output on real GPU hardware (untested as of writing — built without GPU access to avoid rental cost).
- Evaluation harness (retrieval recall, answer accuracy, faithfulness).
- Move from Qdrant Cloud to a self-hosted Qdrant instance for a fully on-prem deployment.
- Optional LLM upgrade.
