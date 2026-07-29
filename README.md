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

The Python pipeline is not containerised: it runs on the host and reaches every
model over HTTP. That is deliberate — the models can sit on a different machine,
which is how a workstation without a GPU runs the whole system.

### Containers

Set the database password in your shell first (it is never stored in the repo),
then bring up the containers:

    export DB_PASSWORD=...              # PowerShell: $env:DB_PASSWORD = "..."
    docker compose up -d                # PostgreSQL + Open WebUI
    docker compose --profile gpu up -d  # ...plus embeddings (8011) + reranker (8002)

The database creates its schema on first start. Leave out `--profile gpu` on a
machine with no GPU and point `EMBED_API_URL` / `RERANK_API_URL` at the host that
does have one.

### Model services

Each model runs in its own isolated environment. Table extraction needs a GPU
(CUDA 12.6) and a Linux host — the CUDA stack these use is broken on Windows:

    ./scripts/setup_paddle.sh        # PaddleOCR OCR service      -> port 8100
    ./scripts/setup_paddleocrvl.sh   # PaddleOCR-VL table service -> port 8104
    ./scripts/setup_hunyuan.sh       # HunyuanOCR table service   -> port 8105
    cp .env.example .env

### Authentication

`pipeline/api.py` is protected by a shared secret. Set `API_KEY` in `.env` and
enter the same value as the API key on the OpenWebUI connection. Every endpoint
that reads or adds documents then requires `Authorization: Bearer <key>`.

Left empty, the API runs unauthenticated and warns about it at startup — fine
bound to localhost, never acceptable beyond it. Two endpoints stay open by
design: `/health` and `/ready`, so monitoring can reach them without a
credential, and the generated-xlsx link, whose filename is a content hash so
that knowing the URL is the authorisation.

`/health` answers "is the process alive", `/ready` answers "can it serve" by
checking the database and the embedding service, returning 503 when either is
down.

### Split across two machines

Every service address is an environment variable, so nothing in the code changes.
On the machine running the pipeline, point each URL at the GPU host in `.env`:
`EMBED_API_URL`, `RERANK_API_URL`, `LLM_API_URL`, `PADDLE_OCR_URL` and the
`*_TABLE_URL` entries. Reaching them over an SSH tunnel keeps the model ports off
the network:

    ssh -N -L 8011:127.0.0.1:8011 -L 8002:127.0.0.1:8002 user@gpu-host

Note that `data/` is deliberately untracked, so documents, ground truth and
evaluation question sets do not travel with a clone — copy them separately.

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

    python -m pipeline.ingest_router path/to/file.pdf
    python -m pipeline.query

Retrieval and answer quality are measured by `eval/rag_eval.py` (does the answer
reach the context, and at what rank) and `eval/rag_answer_eval.py` (is the answer
right, is the cited page right, and which stage is at fault when it is not).

### Two engines, one measurement

The answering engine is pluggable, like the table engine. `native` is the
pipeline described above; `llamaindex` is LlamaIndex retrieving over the same
chunks, kept as a second opinion rather than a replacement.

    pip install -r requirements-llamaindex.txt
    python -m pipeline.rag_llamaindex build       # copy chunks into its own table
    python -m eval.rag_eval --set human --backend llamaindex

Pick one with `RAG_BACKEND`, or per conversation in OpenWebUI by choosing the
`ragtest-rag-llamaindex` model. The source chunks, the embedding model, the LLM
and the answer prompt are identical for both — only the retrieval strategy
differs, so a difference in the numbers is attributable to the thing being
compared.

## Stack

PaddleOCR-VL and HunyuanOCR (table extraction), PaddleOCR PP-OCRv5 (OCR),
Docling (PDF parsing), bge-m3 (embeddings), BM25 (sparse), PostgreSQL + pgvector
(vector store), a reranker, and an LLM — served with vLLM. Fully open-source and
on-premise.
