# Developed RAG Pipeline

A self-hosted pipeline for Turkish PDF and image documents, built entirely on
open-source components and running fully on-premise. Its main focus is **faithful
table extraction** — turning scanned tables into editable Excel — with a document
Q&A (RAG) layer alongside.

## Table extraction

Each scanned/photographed table is read by **two independent vision-language models
and cross-checked**:

- **PaddleOCR-VL (0.9B)** — primary reader (structure + content)
- **HunyuanOCR (1B)** — independent second reader
- **PaddleOCR (PP-OCRv5)** — deterministic OCR used to verify every number

How it works:

- **Two-model consensus** — cells both models agree on are kept; cells they
  disagree on are highlighted for a human to review.
- **Number verification** — each number is checked against the deterministic OCR
  reading, catching silently altered digits.
- **Turkish normalization** — repairs Turkish characters and number formatting.
- **Templates** — for known forms with multi-row (grouped) headers, a canonical
  header can be defined once and applied automatically.
- **Human-in-the-loop** — nothing uncertain is accepted silently; only the flagged
  cells need a human check.
- **Output** — a faithful Excel file with disagreements marked.

The table engine is pluggable (PaddleOCR-VL, HunyuanOCR, Gemma, Docling, TATR) and
swappable with one setting; every downstream step is engine-agnostic.

## Setup

Each model runs in its own isolated environment. Table extraction needs a GPU
(CUDA 12.6):

    ./scripts/setup_paddle.sh        # PaddleOCR OCR service      -> port 8100
    ./scripts/setup_paddleocrvl.sh   # PaddleOCR-VL table service -> port 8104
    ./scripts/setup_hunyuan.sh       # HunyuanOCR table service   -> port 8105
    cp .env.example .env

Start the services (each downloads its model weights on the first request):

    ./scripts/serve_ab.sh            # starts PaddleOCR-VL (8104) + HunyuanOCR (8105)
    nohup paddle_env/bin/uvicorn paddle_service:app --app-dir services --port 8100 &

Extract a table to Excel using two-model consensus:

    TABLE_XLSX=out.xlsx python -m pipeline.table_pipeline path/to/image.png consensus

## Document Q&A (RAG)

Alongside table extraction, the pipeline can ingest documents and answer questions
in Turkish with source-page citations: inputs are normalized into chunks stored in
PostgreSQL + pgvector, and queries use hybrid search (dense + BM25) with a reranker
before an LLM answers.

    ./scripts/setup_postgres.sh
    python -m pipeline.ingest_router path/to/file.pdf
    python -m pipeline.query

## Stack

PaddleOCR-VL and HunyuanOCR (table extraction), PaddleOCR PP-OCRv5 (OCR),
Docling (PDF parsing), bge-m3 (embeddings), BM25 (sparse), PostgreSQL + pgvector
(vector store), a reranker, and an LLM — served with vLLM. Fully open-source and
on-premise.
