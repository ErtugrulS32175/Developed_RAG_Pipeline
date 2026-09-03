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

Set separate migration/runtime passwords and a random context-signing key in
your shell first. They are never stored in the repo.

```bash
export DB_MIGRATION_PASSWORD=...
export DB_RUNTIME_PASSWORD=...
export RAG_DB_CONTEXT_SECRET=...    # at least 32 random bytes
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

For scripts and direct API clients, set `API_KEY` in `.env`. Every endpoint that
reads or adds documents then wants `Authorization: Bearer <key>`. Missing API
and OpenWebUI credentials fail startup closed. The sole exception is an
explicit development opt-in with `ALLOW_INSECURE_LOCAL=1` and an
`API_BIND_HOST` that is literally a loopback IP; a hostname or wildcard bind is
refused.

OpenWebUI uses a different, two-part boundary. Generate two different random
values of at least 32 bytes for `OPENWEBUI_GATEWAY_KEY` and
`OPENWEBUI_USER_JWT_SECRET`. Compose configures the first as the provider key
and the second as OpenWebUI's 60-second signed user assertion. The API requires
both; plain `X-OpenWebUI-User-*` headers, an unsigned user id, and OpenWebUI's
own `admin` role grant no RAG permission. The deployment pins OpenWebUI v0.11.0
by OCI digest so this identity contract cannot move under a mutable image tag.
Set a third independent value, `EVIDENCE_HMAC_SECRET`, for stable opaque
citation references. Keep it when applications restart or roll forward;
after rotation, registering a new citation for a chunk replaces that chunk's
old reference. References not registered again remain valid until their
mapping is replaced, so rotation is not a global revocation operation.

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

The connection context is HMAC-signed with `RAG_DB_CONTEXT_SECRET`; PostgreSQL
custom settings by themselves are not authority. Use a runtime `PG_DSN` role
that is neither superuser nor `BYPASSRLS`, cannot create in the product schema,
and cannot read `rag_context_secrets`. Give `scripts.migrate_db` a separate
schema-owner `PG_MIGRATION_DSN`. The API checks both the exact schema receipt
and the restricted runtime role before its first product query, and never runs
DDL on a request path.

Three endpoints stay open on purpose so monitoring can reach them: `/health`
says the process is alive;
`/ready` checks the database, exact schema version/digest, and embedding
service; `/metrics` publishes only bounded route templates, methods, status
classes, counts and summed duration. It never records a query, document,
tenant or raw URL. Every handled HTTP response carries an `X-Request-ID` for
correlation without copying request content into logs.

### Organization hierarchy in OpenWebUI

Tenant is the company boundary. Inside one tenant, visibility is a directed
tree: the root/CEO can monitor every descendant; a manager can monitor only its
own descendants; a leaf can monitor nobody. Peers and ancestors are never
returned, so nobody below the CEO can see the CEO. A position may additionally
be protected from monitoring. These decisions come from PostgreSQL closure
rows, never from a client-supplied level number or OpenWebUI profile role.

Architecture administration is a separate capability and gives no document
read access by itself. Bootstrap the first system architect with their own
OpenWebUI user id (shown at `/ragtest-org` after the Event Function is installed):

```bash
python -m scripts.bootstrap_org \
  --tenant-id 11111111-1111-1111-1111-111111111111 \
  --tenant-name "Example Company" \
  --openwebui-subject "the-user-id-shown-by-openwebui"
```

In OpenWebUI Admin → Functions, import
`openwebui/functions/ragtest_org_portal.py` and enable it as a global Event
Function. Signed-in users then open `/ragtest-org`: everyone sees their own
level and title, managers see only the descendant list they are authorized to
monitor, and their content-free review queue. A system architect receives the
production administration panel: a visual hierarchy and membership editor,
filtered governance-event history, retention-policy controls, content-blind
document lifecycle inventory, legal-hold management and purge scheduling.
Architecture and policy writes use version/epoch compare-and-swap gates. The
topology save replaces the whole draft atomically, while individual membership,
retention, hold and purge actions use their own narrow forms; a stale browser
receives 409 instead of overwriting newer authority. Queue decisions likewise
carry both case revision and current policy epoch. Protected positions remain
absent even from a root user's monitoring results.

The administration panel deliberately never displays filenames, document
content, hashes, status notes or candidate identifiers. Its document inventory
is made only of opaque ids and lifecycle/governance fields, so architecture
authority does not become document-reading authority. Mutations require the
same-origin portal header and closed JSON bodies; inline script and style bytes
are pinned by the response CSP. The route keeps both identity-bridge secrets on
the server and sends no-store, nosniff and no-referrer headers.

### Governed evaluation datasets in OpenWebUI

Import `openwebui/functions/ragtest_eval_portal.py` as a global Event Function
and open `/ragtest-eval`. The portal creates tenant-bound evaluation sets,
opens one draft at a time, imports a closed list of 1–500 cases and publishes
an immutable version. The list exposes content-free version history, can fill
the lifecycle forms without hand-copying hidden identifiers, and retires a set
only when no draft remains. It reuses the signed OpenWebUI identity bridge; an
OpenWebUI admin flag or organization-architect capability is not an eval-data
bypass. Owners can work on their own sets, and an active editor above an owner
may curate a visible descendant's set. Peers, descendants, unrelated branches,
suspended identities and protected targets remain hidden.

Lists and ordinary mutation responses contain metadata only. Case text is
available solely from the explicit version-cases endpoint, which rechecks the
current hierarchy and returns `Cache-Control: no-store`. Import is bounded
before JSON parsing, requires exact case fields and canonical ordering, and
rejects duplicate keys. Publication locks the tenant policy before the set,
then checks the version revision, current policy epoch and the SHA-256 of the
exact draft the browser observed. Published versions, their cases and their
content-free event trail cannot be updated or deleted; later work starts a new
server-numbered draft.

The digest proves the exact evaluation-case document, not the provenance of
the source corpus. A `pages` list by itself does not bind document versions or
a retrieval snapshot, so this lifecycle must not be described as source-level
reproducibility until that separate binding is added.

### Citation evidence in OpenWebUI

Import `openwebui/functions/ragtest_citations.py` as a global Filter and
`openwebui/functions/ragtest_evidence_action.py` and
`openwebui/functions/ragtest_feedback_action.py` as Actions. Checked RAG
answers then publish OpenWebUI `source` events containing only a document
label, page and opaque evidence reference. The initial answer never contains a
passage, source path, ticket or raw chunk id.
Keep streaming enabled on the OpenWebUI provider connection: the Filter emits
the persisted `source` event while processing streamed chunks. Non-streaming
direct API clients still receive the closed `rag_citations` response metadata,
but do not run the OpenWebUI Filter.

The **Kanıtı Göster** action exchanges the reference through two server-side
JSON POSTs. The API first rechecks the signed OpenWebUI actor, active content
membership, tenant, document lifecycle and exact active chunk, then issues a
single-use 50-second ticket. Preview consumption repeats those checks and
returns only the bounded passage. Tickets are actor/tenant/purpose bound,
stored only as SHA-256 digests and never placed in a URL or browser storage.
Organization-architecture authority alone grants no evidence access. The
preview is shown transiently rather than saved into chat history, so revoking
access is not defeated by a durable copy created by the plugin itself.

The feedback Action offers only closed helpful/not-helpful choices and closed
reason codes. It sends an opaque reference, never the question, answer,
passage, chat id or free text. A negative answer opens one idempotent review
case. A `review_required` result opens its case server-side without inventing a
browser feedback target. Direct API-key calls, citationless answers and
non-persisted publications deliberately expose no feedback target.

### Actor-bound table exports

Table extraction never publishes a storage filename or a guessable download
URL. A successful OpenWebUI table reply carries only an opaque
`ragtest-export:` reference. A signed-in content actor exchanges that reference
with `POST /v1/exports/tickets`, then consumes the returned 50-second ticket
exactly once through `POST /v1/exports/download`. Both steps recheck the active
tenant membership and the exact actor that created the export; an API key,
organization-architect capability, peer, other tenant or replay is not a
download authority.

Generated workbooks use random code names. PostgreSQL stores only that code
name, byte length and SHA-256, never workbook cells or source content. The
download opens the file through no-follow directory handles under a 32 MiB
ceiling and returns it only if its current length and digest still match the
registration. The current OpenWebUI table reply displays the opaque reference;
a browser Action for the binary hand-off remains a separate UI integration.

### Local speech dictation in OpenWebUI

An opt-in, local speech-to-text path for OpenWebUI's microphone button. It is
a separate process, `services/speech_service.py`, that loads one
faster-whisper model once and speaks the OpenAI transcription shape:

```bash
pip install -r requirements-speech.txt   # its own environment, not requirements.txt
export SPEECH_API_KEY=...                # at least 32 random characters
export SPEECH_HOTWORD_PROFILE=capital_markets_tr  # optional, deployment-owned
export SPEECH_TERMINOLOGY_FILE=/run/secrets/capital-markets-tr.json  # optional
export SPEECH_TERMINOLOGY_CONTEXT=equities                           # optional
python -m services.speech_service        # 0.0.0.0:8012, one process, one worker
docker compose -f docker-compose.yml -f docker-compose.speech.yml up -d open-webui
```

The override only adds `AUDIO_STT_*` variables to the `open-webui` service;
without `-f docker-compose.speech.yml` on the command line nothing changes.
OpenWebUI then posts each recording as multipart to
`POST /v1/audio/transcriptions` with the bearer key, and the service answers
`{"text": "..."}` and nothing else.

The profile is closed and pinned by tests: model `large-v3`, device `cuda`,
compute type `float16`, `language="tr"`, `vad_filter=True`. A request naming
another `model` or `language` is refused. There is no `turbo`, `int8` or CPU
fallback: a GPU that cannot hold the model leaves `/readyz` answering 503
rather than a smaller model. `GET /healthz` says only that the process is
alive; `GET /readyz` reports the profile above once the model has loaded. Both
are unauthenticated monitoring endpoints, like `/health` and `/ready` on the
main API. The transcription endpoint requires
`Authorization: Bearer $SPEECH_API_KEY`, compared timing-safely. That key is a
service identity between OpenWebUI and this process, not a user or tenant
identity, so this path records no transcript, audit entry or ownership: the
audio and the text exist only for the duration of the request. The service
never opens PostgreSQL, never writes to the RAG index, calls no external model
API and never logs a filename, the audio or the transcript; log lines carry
closed event codes and numbers only.

First local dictation profile, pinned in `tests/test_speech_service.py` until
the E5 quota policy takes these values over: uploads of at most 25 MiB, at
most 120 seconds of audio (measured by decoding before the model runs, so a
small but very long compressed file is refused early), a 90-second
transcription timeout, one transcription at a time with at most two callers
waiting up to 20 seconds each; a further caller gets a closed `speech_busy`
503 at once. Accepted formats are WAV, WebM, MP3, MP4/M4A, OGG and FLAC, by
extension and audio content type; the user's filename is never used as a path.
Uploads are written in 64 KiB chunks to a task-specific temporary file that is
removed by exact path on success, failure, timeout and cancellation. To serve
a model you downloaded beforehand, point `HF_HUB_CACHE` at that cache and set
`HF_HUB_OFFLINE=1`; the model identity stays `large-v3`.

On a single 8 GB card the model shares the GPU with nothing. The compose GPU
profile reserves about 20% of the card for `bge-m3` embeddings and 15% for the
reranker; loading `large-v3` in float16 alone held about 3.9 GB on an RTX 4070
Laptop (8 GB) in the first local run, and decoding real speech needs more on
top, so the local order of operations is manual:

1. Do not run the embed and reranker containers while transcribing.
2. Stop the speech service when dictation is done.
3. Start the embed and reranker containers for the next ingest/index run.

Automatic GPU orchestration is a later E5/E6 package. Persistent recordings,
transcription jobs, transcript revisions and RAG indexing of transcripts are
the E4/E6 packages and are deliberately not scaffolded here.

`SPEECH_HOTWORD_PROFILE=capital_markets_tr` enables a deployment-owned,
versioned terminology registry. The checked-in registry is only the measured
eleven-phrase seed; a licensed or organization-owned full registry can be kept
outside Git and selected with `SPEECH_TERMINOLOGY_FILE`. Validate such a file
without printing its terms or path with:

```bash
python -m services.speech_terminology /run/secrets/capital-markets-tr.json
```

The registry stores the whole catalog in memory, but the loaded large-v3
tokenizer deterministically projects only the highest-priority phrases for the
closed `SPEECH_TERMINOLOGY_CONTEXT` into a maximum 96-token/16-phrase hotword
pack. Requests cannot provide or override the profile, context, terms or token
budget. With a pack enabled, decoding also disables previous-window
conditioning because that was the smallest measured setting which kept the
real two-speaker Turkish benchmark complete without a fabricated closing line.
Leaving the profile empty preserves the general dictation behaviour exactly.

Registry files are strict UTF-8 JSON: root fields are `schema_version` (1),
`profile_id`, `revision`, `language` (`tr`) and `terms`. Each term contains only
`canonical`, `aliases`, `contexts` and `priority`; definitions and source prose
do not belong there. The loader refuses unknown fields, links, invalid or
control-bearing text, normalization collisions, overlong values and unbounded
files. Keep the private registry on a read-only deployment mount. The current
OpenWebUI microphone relay authenticates the service, not the human or tenant,
so this version deliberately supports only deployment contexts. Tenant-specific
overlays require a later signed identity bridge and server-side tenant lookup;
browser-supplied tenant/profile fields are not an authority.

### Production operations

Apply migrations before sending traffic to a new application version:

```bash
# PG_MIGRATION_DSN is the schema owner; PG_DSN remains the restricted runtime.
python -m scripts.migrate_db
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/metrics
```

The migration takes a transaction-scoped PostgreSQL advisory lock, applies the
DDL and its schema receipt in one transaction, and prints closed JSON.
Readiness remains false if either the version or digest differs, so a process
cannot silently serve against a partially or incorrectly migrated database.
Schema receipts are monotonic: an older binary is refused before product DDL,
and a database trigger prevents a schema-state downgrade even if a stale
process reaches the receipt table.

Backups use PostgreSQL custom format and a closed SHA-256 sidecar. Version 2 of
that sidecar also records the exact schema receipt plus tenant, purge-tombstone
and retention-event counts. Those counts and `pg_dump` share one exported MVCC
snapshot, so concurrent writes cannot make the manifest describe a different
database instant. Restore is limited to an empty target and succeeds only when
the restored database reproduces every closed evidence field. Credentials reach
`pg_dump` and `pg_restore` through the process environment, never argv or JSON
output:

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
content-free Trace V2: a random trace id, backend, planner policy version,
query class, retrieval mode, fallback policy, closed scope kind, tenant-policy
epoch, bounded result limits, result/context counts, context byte count and
per-stage milliseconds. It contains no question, query digest, passage text,
score, filename, tenant, actor, document, chunk, version or citation identity.
The field is omitted by default, so existing OpenAI-compatible clients keep
their prior response shape.

Checked retrieval first validates a deterministic V1 plan, then binds the
verified tenant and actor, tenant-policy epoch, resolved document scope and
retrieval itself to one repeatable-read database transaction. Unknown or
cross-tenant explicit identifiers resolve to an empty scope and stop before
embedding or search; they never widen to the corpus. V1 deliberately ships one
measured mode (`hybrid_balanced`) with no automatic backend fallback, 15 final
passages, 60 candidates, and closed candidate/context byte ceilings. Runtime
`TOP_K` or rerank settings that disagree with that plan fail before retrieval
instead of producing a misleading trace. Actor-level per-document visibility
beyond the current tenant and explicit-scope authorities is not claimed here;
that requires its own schema/RLS policy package.

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

Irreversible deletion is deliberately a separate architect-only workflow.
`GET`/`PUT /v1/org/admin/retention-policy` publishes and CAS-updates the tenant
archive-retention period. Legal holds are created, listed and released under
`/documents/{document_id}/legal-holds`; an active hold cancels a pending purge
and prevents another one from being scheduled. Once an archived document has
outlived the tenant period, `POST /documents/{document_id}/purge-jobs` creates a
durable job. Run the content worker separately:

```bash
python -m pipeline.index.purge_worker
```

The worker rechecks holds and ingest activity while retaining the document row
lock, removes only the database-authorized immutable source objects and chunks,
then writes a terminal document tombstone. Version identities, lifecycle events,
the job and closed governance audit events remain as deletion evidence; source
bytes, chunk text, filename, content digests and collection/tag membership do
not. A purged document cannot be restored or otherwise mutated.

`GET /documents` lists active documents by default; use `archived=true` to list
only archived ones. An archived document cannot start processing. Lifecycle
authority is checked again while the ingest lease is taken and is held through
snapshot-backed LlamaIndex retrieval, so a concurrent request cannot publish a
document halfway through its archive transition.

Inventory responses include `next_cursor` when another page exists. Pass its
`before_uploaded_at` and `before_id` values together to walk deep inventories
with keyset pagination; the pair cannot be mixed with a non-zero `offset`.
Offset pagination remains available for existing clients. Active, archived,
status and file-type inventory paths have tenant-leading PostgreSQL indexes;
the total order remains `uploaded_at DESC, id DESC`, so cursor pages neither
overlap nor leave gaps at equal timestamps.

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
pgvector as the store, a reranker, an LLM served with vLLM, and faster-whisper
`large-v3` for local dictation. All open source, all on-premise.
