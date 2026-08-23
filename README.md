# Developed RAG Pipeline

A self-hosted pipeline for Turkish PDF and image documents. Everything runs on
your own machines, on open-source components, with nothing sent to an outside
service.

Two things live here. The main one is table extraction: turning scanned tables
into Excel files you can actually trust. Next to it is a document Q&A layer that
answers questions about your documents and cites the page it used.

## Table extraction

Scanned tables are hard because a single misread digit looks exactly like a
correct one. So every table is read twice, by two different vision-language
models, and the readings are compared:

* **PaddleOCR-VL (0.9B)** reads structure and content.
* **HunyuanOCR (1B)** reads the same table independently.
* **PaddleOCR (PP-OCRv5)** does deterministic OCR, used to check every number.

Cells the two models agree on are kept. Cells they disagree on are marked in the
output so a person can look at them. Numbers get a third opinion from the
deterministic OCR, which is what catches a digit that quietly changed. Turkish
characters and number formatting are repaired along the way, and for forms you
see often you can define the header layout once and reuse it.

Nothing uncertain is accepted silently. You get an Excel file with the
disagreements flagged, and usually only a handful of cells need your attention.

The table engine is swappable with one setting (PaddleOCR-VL, HunyuanOCR, Gemma,
Docling, TATR). Everything downstream works the same whichever you pick.

## Setup

The Python pipeline is not containerised. It runs on the host and talks to every
model over HTTP, which is deliberate: the models can sit on another machine, so a
laptop without a GPU can still run the whole thing.

Set the database password in your shell first. It is never stored in the repo.

```bash
export DB_PASSWORD=...              # PowerShell: $env:DB_PASSWORD = "..."
docker compose up -d                # PostgreSQL + Open WebUI
docker compose --profile gpu up -d  # plus embeddings (8011) and reranker (8002)
```

The container bootstraps an empty database. Every deploy must then run the
versioned migration command below; it serialises concurrent deploys and records
the exact `schema.sql` digest that `/ready` later checks. On a machine with no
GPU, skip `--profile gpu` and point `EMBED_API_URL` and `RERANK_API_URL` at the
host that has one.

Each model service gets its own environment. Table extraction needs a GPU
(CUDA 12.6) and a Linux host, because the CUDA stack these use does not work on
Windows.

```bash
./scripts/setup_paddle.sh        # OCR service            -> 8100
./scripts/setup_paddleocrvl.sh   # PaddleOCR-VL tables    -> 8104
./scripts/setup_hunyuan.sh       # HunyuanOCR tables      -> 8105
cp .env.example .env
```

Then start them and pull a table out:

```bash
./scripts/serve_ab.sh
TABLE_XLSX=out.xlsx python -m pipeline.extraction.table_pipeline path/to/image.png consensus
```

### Authentication

Set `API_KEY` in `.env` and use the same value as the API key on the OpenWebUI
connection. Every endpoint that reads or adds documents then wants
`Authorization: Bearer <key>`.

For more than one tenant, set `API_KEYS_JSON` to a JSON list whose entries are
exactly `key`, `tenant_id` (UUID), and `role`. Roles are cumulative:
`reader` can query/list, `editor` can also upload, process, tag and queue work,
and `admin` can also archive/restore or delete organisation metadata. Raw keys
are hashed during startup and are never written to PostgreSQL or logs.
`API_KEY`, when retained beside that registry, remains a default-tenant admin
for backward compatibility.

Tenant separation is enforced in two places: every request binds its tenant to
the PostgreSQL connection and the core tables use forced row-level-security;
uploaded source files for non-default tenants live in separate UUID-named
directories. The ingest worker uses a service context only to claim queued work,
then binds the claimed tenant while indexing it. An unscoped multi-tenant RAG
request is converted to the active document ids visible to that tenant before
either retrieval backend runs, so an external index cannot widen PostgreSQL's
tenant boundary.

Leave it empty and the API runs open, warning you about it at startup. That is
fine on localhost and never fine anywhere else. Three endpoints stay open on
purpose so monitoring can reach them: `/health` says the process is alive;
`/ready` checks the database, exact schema version/digest, and embedding
service; `/metrics` publishes only bounded route templates, methods, status
classes, counts and summed duration. It never records a query, document,
tenant or raw URL. Every handled HTTP response carries an `X-Request-ID` for
correlation without copying request content into logs.

### Production operations

Apply migrations before sending traffic to a new application version:

```bash
python -m scripts.migrate_db
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/metrics
```

The migration takes a transaction-scoped PostgreSQL advisory lock, applies the
DDL and its schema receipt in one transaction, and prints closed JSON.
Readiness remains false if either the version or digest differs, so a process
cannot silently serve against a partially or incorrectly migrated database.

Backups use PostgreSQL custom format and a SHA-256 sidecar. Credentials reach
`pg_dump` and `pg_restore` through the process environment, never argv or JSON
output. Restore is limited to an empty target and needs explicit confirmation:

```bash
python -m scripts.db_snapshot backup --output backups/rag.dump
python -m scripts.db_snapshot verify --archive backups/rag.dump

# Point this at a newly created, empty database; never the live database.
export PG_RESTORE_DSN=postgresql://...
python -m scripts.db_snapshot restore \
  --archive backups/rag.dump --confirm EMPTY_DATABASE
```

Rehearse restoration regularly: create an empty temporary database, restore,
run the migration, check readiness, then destroy the temporary database. A
backup that has never survived a restore rehearsal is only a hopeful file.

The rollout gate joins offline quality with live readiness. Save the aggregate
quality report and start the migrated candidate before switching traffic:

```bash
python -m eval.quality_gate \
  --run-dir output/eval --out output/quality-gate.json
python -m scripts.rollout_gate \
  --quality-report output/quality-gate.json \
  --ready-url http://127.0.0.1:8000/ready
```

When the gate fails, route traffic back to the previous application image. Do
not replay old DDL against the live database: restore a verified snapshot into
a new empty database, migrate it forward, validate both gates, and switch the
DSN under a human change gate.

### Running across two machines

Every service address is an environment variable, so no code changes. Point the
URLs in `.env` at the GPU host (`EMBED_API_URL`, `RERANK_API_URL`, `LLM_API_URL`,
`PADDLE_OCR_URL`, and the `*_TABLE_URL` entries). An SSH tunnel keeps the model
ports off the network entirely:

```bash
ssh -N -L 8011:127.0.0.1:8011 -L 8002:127.0.0.1:8002 user@gpu-host
```

One thing to know: `data/` is deliberately untracked. Documents, ground truth and
evaluation sets do not come with a clone, so copy them over separately.

## Document Q&A

Documents are normalised into chunks and stored in PostgreSQL with pgvector.
Queries use hybrid search (dense plus BM25), a reranker, and then an LLM writes
the answer with source-page citations.

```bash
python -m pipeline.index.ingest path/to/file.pdf
python -m pipeline.retrieval.query
```

Answers are checked against the exact passage they claim to quote. The result is
`answered`, `abstained` or `review_required`, and a review-required result never
carries publishable answer text. The public API only exposes this checked path,
and both JSON and streaming replies carry `rag_status`.

Set `include_trace: true` on a chat request when an operator needs to explain
the retrieval path. Both JSON and streaming responses then carry the same
`rag_trace`: a random trace id, backend name, requested-scope size, counts after
retrieval/reranking/context assembly, and per-stage milliseconds. It is a
closed, content-free record: no question, passage text, score, filename or
document id is present. The field is omitted by default, so existing
OpenAI-compatible clients keep their prior response shape.

Quality is measured rather than assumed. `eval/retrieval/rag_eval.py` asks
whether the answer even reached the context and at what rank;
`eval/answer/rag_answer_eval.py` asks whether the answer and the cited page are
right, and which stage is at fault when they are not. Reports show coverage and
risk together, because suppressing every answer would otherwise look perfectly
safe.

Saved runs can be checked against the repository's closed quality policy
without calling a model or embedding service:

```bash
python -m eval.quality_gate --run-dir output/eval
```

The gate re-scores raw saved answers with the current deterministic scorer,
rather than trusting the old summary embedded in each file. Unsettled answers
remain review cases, so they can raise the upper bound but never the confirmed
lower bound. Missing sets, changed sample counts, ambiguous retrieval profiles,
malformed policy fields and metric regressions all fail closed. Its output is
aggregate-only: set names, counts, metric names and numbers, with no question,
answer, context or missed-query text.

Hybrid retrieval takes a bounded candidate pool from both dense and sparse
rankings before reciprocal-rank fusion: four times the requested result count,
up to 200 candidates per ranking. The final `top_k` is unchanged. This lets a
passage supported by both rankings beat modality-specific leaders just outside
the old cut, while keeping database work bounded. Distance ties and fused-score
ties use chunk identity as a deterministic final key, so pagination-free
retrieval does not depend on database return order.

Ragas adds two optional diagnostic signals, neither of them a release gate:
faithfulness against the supplied context, and similarity to a hand-written
reference. The judge is whatever `LLM_API_URL` points at, so nothing leaves your
machines. Results are cached, so re-scoring after a change is free.

```bash
python -m venv ragas_env
ragas_env\Scripts\pip install -r requirements-ragas.txt
ragas_env\Scripts\python -m eval.answer.ragas_check --set human --limit 5
```

The answering engine is swappable too. `native` is the pipeline above;
`llamaindex` retrieves over the same chunks as a second opinion. Same chunks,
same embeddings, same LLM, same prompt, so a difference in the numbers really is
about retrieval.

```bash
python -m pipeline.retrieval.rag_llamaindex build
python -m eval.retrieval.rag_eval --set human --backend llamaindex
```

### Document lifecycle

`POST /documents/{document_id}/archive` removes a document from the normal
inventory and from both retrieval engines without deleting its chunks.
`POST /documents/{document_id}/restore` makes the same stored generation
retrievable again. Both operations are idempotent, require the same API key as
the other document routes, and return 409 while an ingest lease is active.

`GET /documents` lists active documents by default; use `archived=true` to list
only archived ones. An archived document cannot start processing. Lifecycle
authority is checked again while the ingest lease is taken and is held through
snapshot-backed LlamaIndex retrieval, so a concurrent request cannot publish a
document halfway through its archive transition.

### Collections and tags

Collections and tags organise documents without copying or owning them. Create
and list collections with `POST /collections` and `GET /collections`; attach or
detach a document with `PUT` or `DELETE` on
`/collections/{collection_id}/documents/{document_id}`. Delete a collection
with `DELETE /collections/{collection_id}`. None of these operations deletes
the documents or their chunks. Replace a document's complete tag set with
`PUT /documents/{document_id}/tags` and a JSON body such as
`{"tags":["finance","urgent"]}`; an empty list clears it.
`GET /tags` lists the shared vocabulary and `DELETE /tags/{tag_id}` removes one
tag and its memberships without removing a document.

`GET /documents` accepts `collection_id` and `tag` filters. Chat requests may
carry `collection_ids` and `tags` beside `document_ids`: collection ids use ANY
semantics, tags use ALL semantics, and the three dimensions intersect. Names
are case-insensitive identities while preserving their first display spelling.
Archived documents never resolve into a chat scope, and an empty resolved scope
stays empty rather than widening to the whole corpus.

### Durable ingest jobs

For work that must survive an API or worker restart, enqueue the current
published candidate with `POST /documents/{document_id}/ingest-jobs` and a
bounded `Idempotency-Key` header. Repeating that key returns the same job rather
than starting duplicate work. Read it with `GET /ingest-jobs/{job_id}` or cancel
a still-queued job with `DELETE /ingest-jobs/{job_id}`. Responses expose closed
job state and counters, never the idempotency key or candidate digest.

Run workers separately from the API:

```bash
python -m pipeline.index.job_worker
```

Workers claim rows with a database lease, bind each job to the candidate that
was current when it was queued, and use the existing fenced ingest-attempt
authority for every chunk write and publication. Expired work is retried up to
the configured attempt budget; an expired owner cannot finish or requeue it.
Only one queued/running job or synchronous `/process` call may own a document at
a time, and archive/restore refuses while that work is active. The synchronous
`POST /documents/{document_id}/process` route remains available for callers that
intentionally want request-bound processing.

## Controlled task runs (agent-loop)

`tools/agent_loop` runs a coding task through an implementer model and an
evaluator model under a written contract. The task file says what may be edited,
which commands decide the outcome, and what the run may spend. None of that is
guessed at runtime: a run refuses rather than widening its own permissions.

You write a manifest naming acceptance commands by id (`pytest_full`,
`pytest_selected`, `p0_gate`, `leak_scan`). It never spells out a command line,
because the actual argv lives in a registry in the code. A task may not list
`tools/agent_loop/` as editable, for the obvious reason.

Binaries are always explicit. Nothing is discovered from PATH:

```python
from tools.agent_loop import runner

binaries = {"implementer": "/usr/local/bin/claude",
            "evaluator": "/usr/local/bin/codex"}

result = runner.preflight("tasks/normalise.json", repo=".", binaries=binaries)
if result.stop_reason == "completed":
    result = runner.run("tasks/normalise.json", repo=".", binaries=binaries)
```

Preflight runs every gate that has to pass before a model could be called, and it
costs nothing, so read its answer first. During the run the implementer works in
a flat copy of the tree and can only read and edit files. It has no shell, so it
cannot install, fetch or commit anything, and the acceptance commands are run by
the runner rather than by the model. That split is what makes the gates mean
something.

While `runner.run` or `runner.resume` is active, treat the main checkout as
read-only evidence. Do not run pytest (including `--collect-only`), formatters,
IDE tasks or any other command that can write a cache, bytecode or generated
file there: even `.pytest_cache` is deliberately part of the before/after
evidence, and a write makes the runner reject an otherwise valid candidate.
Measure in a disposable clone pinned to the run's baseline instead, outside the
candidate workspace and `.agent-loop/` holder. Reading the closed `RunResult`
after the call returns is safe; inspecting or changing the live run is not.

A run gets exactly one repair attempt. It can be spent either by the evaluator
asking for changes, or by the first acceptance run failing on ordinary test
failures the runner can identify. Anything it cannot identify, such as a
collection error, a timeout or a truncated output, still stops the run for a
person to look at. After a repair, acceptance and the evaluator both run again
before anything is applied.

`RunResult` gives you identities, counts and closed codes, never raw model output
or a patch. The durable record sits in `.agent-loop/`: `state.json` for where the
run got to, `events.jsonl` for the transitions, `findings.json` for what the
evaluator said.

An accepted candidate is moved into your working tree behind a write-ahead
journal, and only while each target still matches the baseline. Line endings from
`core.autocrlf` are handled properly and not mistaken for edits. If a file has
genuinely drifted, the run refuses rather than throwing your work away.

**The loop never stages, commits, pushes or touches history.** Those, along with
dependency installs and contract changes, are on a frozen human-approval list
that no task file can shorten. The changes land in your working tree and the rest
is yours:

```bash
git diff
git add -p && git commit
```

When a run is done, `runner.finalize` archives its evidence byte for byte and
resets the repository in one call. It starts no process and makes no network
request. Nothing is deleted until every file has been copied, read back off the
disk and matched by digest, and if anything cannot be proven the call refuses and
removes nothing.

## Stack

PaddleOCR-VL and HunyuanOCR for tables, PaddleOCR PP-OCRv5 for OCR, Docling for
PDF parsing, bge-m3 for embeddings, BM25 for sparse retrieval, PostgreSQL with
pgvector as the store, a reranker, and an LLM served with vLLM. All open source,
all on-premise.
