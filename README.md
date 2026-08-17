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

`pipeline/api/app.py` is protected by a shared secret. Set `API_KEY` in `.env` and
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

    TABLE_XLSX=out.xlsx python -m pipeline.extraction.table_pipeline path/to/image.png consensus

## Document Q&A (RAG)

Alongside table extraction, the pipeline can ingest documents and answer questions
in Turkish with source-page citations: inputs are normalized into chunks stored in
PostgreSQL + pgvector, and queries use hybrid search (dense + BM25) with a reranker
before an LLM answers.

    python -m pipeline.index.ingest path/to/file.pdf
    python -m pipeline.retrieval.query

Retrieval and answer quality are measured by `eval/retrieval/rag_eval.py` (does the answer
reach the context, and at what rank) and `eval/answer/rag_answer_eval.py` (is the answer
right, is the cited page right, and which stage is at fault when it is not).

Structured answers are checked against the exact passage and literal quote they
claim to use. The checked result is `answered`, `abstained` or
`review_required`; the last state carries no publishable answer text. Rate
derivation is off by default because it previously absorbed a measured
arithmetic restatement error. Structured evaluation reports publication
coverage, review rate, false review among confirmed-correct answers, and
selective risk among settled published answers. Coverage and risk must be read
together: suppressing every answer would otherwise look perfectly safe.

The public RAG API uses only that checked path. Both regular JSON replies and
streaming events expose `rag_status`; a review-required result displays a fixed
review notice instead of any unchecked model text. Unknown model ids are
rejected before a retrieval backend is called. The table-extraction model keeps
its separate service route and does not use the RAG status contract.

### Checking the checker

Scoring here is deterministic: an expected answer is present in the text or it
is not. That is cheap and repeatable, but blind to whether the claims *around*
a correct figure are supported by the context.

Ragas adds two diagnostic signals, neither of which is a release gate:
`Faithfulness` checks claims against the supplied context, and
`SemanticSimilarity` compares the answer with the hand-written reference.
Similarity is reported as a continuous score with NO correctness threshold;
the adjudicated set is not large enough to calibrate one. `FactualCorrectness`
and per-chunk `ContextPrecision` were retired after the former disagreed too
often with the validated scorer and the latter dominated evaluation cost.

    python -m venv ragas_env
    ragas_env\Scripts\pip install -r requirements-ragas.txt
    ragas_env\Scripts\python -m eval.answer.ragas_check --set human --limit 5

The judge is whatever `LLM_API_URL` points at — the same self-hosted model the
pipeline answers with, so nothing leaves the machines it already runs on.
Judgements and embeddings are cached on disk, so re-scoring after a change
costs nothing.

By default similarity reuses the production embedding model and every output
labels that result as correlated with retrieval. Set both
`EVAL_EMBED_API_URL` and `EVAL_EMBED_MODEL_NAME` to use a separately configured
audit model. A separate endpoint is not by itself proof of independence; model
identity is saved with the result.

Current output names carry the set, embedding mode and a deterministic
evaluation-configuration ID, for example
`ragas_diagnostic_v2_human_production_embedding_correlated_<id>_limit5.json`.
The same ID is saved inside the file with the metric set, model identities,
calibration status and per-metric timing. Reports must compare or aggregate
scores only when this ID and `embedding_mode` are carried with the score.
An existing result is never overwritten. Use `--run-tag tekrar-1` when the
same configuration must be run again; a limited smoke run and a full run
already receive different filenames.
Unversioned historical `ragas_<set>.json` files predate this contract and are
retired; they must not be quoted as current measurements. Start every new
configuration with `--limit 5` and price the full run from the recorded timing.

### Two engines, one measurement

The answering engine is pluggable, like the table engine. `native` is the
pipeline described above; `llamaindex` is LlamaIndex retrieving over the same
chunks, kept as a second opinion rather than a replacement.

    pip install -r requirements-llamaindex.txt
    python -m pipeline.retrieval.rag_llamaindex build       # copy chunks into its own table
    python -m eval.retrieval.rag_eval --set human --backend llamaindex

Pick one with `RAG_BACKEND`, or per conversation in OpenWebUI by choosing the
`ragtest-rag-llamaindex` model. The source chunks, the embedding model, the LLM
and the answer prompt are identical for both — only the retrieval strategy
differs, so a difference in the numbers is attributable to the thing being
compared.

## Controlled task runs (agent-loop)

`tools/agent_loop` runs a task through an implementer and an evaluator under a
written contract: the task names what may be edited, which acceptance commands
decide the outcome, and what the run may spend. Nothing about that is inferred
at runtime — a run refuses rather than widening its own permissions.

### 1. Write the task manifest

A manifest is a JSON file inside the repository. It *names* acceptance commands
by id (`pytest_full`, `pytest_selected`, `p0_gate`, `leak_scan`); it never
spells an argv, because the argv lives in the code registry:

    {
      "protocol_version": "1.0",
      "objective": "Fix the Turkish number normalisation in table cells",
      "baseline_sha": "<40-hex commit the run starts from>",
      "allowed_paths": ["pipeline/extraction/"],
      "forbidden_paths": ["tools/agent_loop/", "data/", "eval/"],
      "acceptance_commands": [
        {"command_id": "pytest_selected", "paths": ["tests/test_tables.py"]},
        {"command_id": "leak_scan"}
      ],
      "acceptance_criteria": ["Turkish decimal commas survive normalisation"],
      "max_implementation_rounds": 1,
      "max_repair_rounds": 1,
      "max_wall_clock_minutes": 30,
      "max_budget_usd": 3,
      "max_output_bytes": 1048576,
      "leak_policy": {"command_id": "leak_scan", "max_hard_findings": 0},
      "dirty_tree_allowlist": []
    }

`allowed_paths` may not cover `tools/agent_loop/` — a task that could edit the
loop would be a task that rewrites the rules it is judged by. The human gates
below are likewise not a manifest field: there is no setting that removes them.

### 2. Point at the binaries explicitly

Model binaries are never discovered from PATH: the implementer and evaluator
paths are mandatory explicit arguments. A forgotten binary is an error, never a
surprise call to whatever happened to be installed. (Runner-owned acceptance
commands are the one exception, and only for registry-named tools such as
`python`, `bash` and `git`, resolved through a narrowed PATH.)

    binaries = {
        "implementer": "/usr/local/bin/claude",   # Windows: r"C:\...\claude.exe"
        "evaluator":   "/usr/local/bin/codex",
    }

### 3. Preflight, then run

Preflight is every gate that must pass before a model could be called. It makes
no model call and builds no workspace or run state; it may launch contained,
free CLI version and auth-status probes — so run it first and read its answer
before paying for anything.

    from tools.agent_loop import runner

    result = runner.preflight("tasks/normalise.json", repo=".", binaries=binaries)
    if result.stop_reason == "completed":
        result = runner.run("tasks/normalise.json", repo=".", binaries=binaries)

The implementer works in a flat copy of the tree and is allowed to read and
edit files only — it holds no shell, so it cannot install, fetch or commit
anything. The acceptance commands are run by the runner, not by the model, and
that split is what makes the gates mean anything.

### 4. Read the result

`RunResult` carries identities, counts and closed codes — `state`,
`stop_reason`, `applied_files`, `acceptance_passed`, `pending_approval` — and
never raw model output or a patch. The durable record is in `.agent-loop/`:

    .agent-loop/state.json      # where the run got to, and why it stopped
    .agent-loop/events.jsonl    # the transition journal
    .agent-loop/findings.json   # what the evaluator reported

An accepted candidate is moved into your working tree behind a write-ahead
journal, and only when your copy of each target still matches the baseline —
byte for byte, or by cleaning to the same Git object, so a checkout whose line
endings came from `core.autocrlf` is not mistaken for an edit. Mode is compared
exactly, and a path configured with a custom clean filter, `ident` or a
working-tree encoding is refused rather than guessed at. If a target has really
drifted, the run refuses instead of discarding your edits, and your own bytes
are what a rollback puts back. `runner.resume(repo=".", binaries=binaries)`
reads back an interrupted run and rolls back a crashed application; it does not
continue the loop.

### 5. Commit and push are yours

**The loop never stages, commits, pushes, rewrites history or deletes
branches.** It does use Git read-only, to materialise objects and gather
evidence, and it creates disposable Git metadata inside acceptance mirrors.
`git add`, `git commit`, `git push`, branch deletion, history rewriting,
dependency installation and contract changes are on a frozen human-approval
list, and no task file can shorten it. A run that needs one of them stops with
`user_approval_required` and names it in `pending_approval`. The changes are in
your working tree; review the diff and commit them yourself:

    git diff
    git add -p && git commit
    git push

### 6. Close the run, then start the next one

A finished run leaves its evidence behind on purpose — the state document, the
events journal, the acceptance receipt, and for a run that stopped, the only
copy of the candidate. `runner.finalize` archives all of it byte-exact and then
resets the repository, in one call:

    runner.finalize(
        repo=".",
        archive_root="/somewhere/outside/the/repo",
        expected_run_id=result.run_id,
        task_path="tasks/normalise.json",   # optional
        shipped_commit=head_sha,            # optional
        ci_run_id="32047794474",            # optional, recorded only
    )

It takes no `binaries` because it starts no process: nothing is launched and no
network request is made. `shipped_commit` and `ci_run_id` are recorded, and the
shipment claim is checked against your repository — an approved run whose work
is not committed is archived as `unshipped` rather than filed as shipped.

Nothing is removed until every file has been copied, read back off the disk and
matched by digest, and a completion record is durable; each source is measured
again in the instant before it is unlinked. `run.lock` and the workspace ledger
directory stay. The task manifest is removed only if it is yours to remove —
inside the repository, an ordinary file, carrying this run's exact digest, and
unknown to git in both the index and the tree. If anything cannot be proven the
call refuses and removes nothing.

## Stack

PaddleOCR-VL and HunyuanOCR (table extraction), PaddleOCR PP-OCRv5 (OCR),
Docling (PDF parsing), bge-m3 (embeddings), BM25 (sparse), PostgreSQL + pgvector
(vector store), a reranker, and an LLM — served with vLLM. Fully open-source and
on-premise.
