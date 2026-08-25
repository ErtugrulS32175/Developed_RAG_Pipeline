# Enterprise Roadmap and Progress

This document is the tracked authority for moving RAGTest from a secure
self-hosted RAG product to an enterprise platform. It is intentionally kept in
Git: phases are closed only by changing their checkboxes in the same commit as
their measured evidence. The older local `ROADMAP.md` remains an ignored
working note and is not a release authority.

## Status vocabulary

- `[ ]` not started
- `[~]` in progress
- `[x]` completed and verified
- `[!]` blocked; the progress log must name the closed reason

No phase is complete because code exists. Completion requires its tests,
security checks, leak scan, migration/recovery evidence where relevant, and a
green CI run on the shipped commit.

## Verified starting point

Snapshot recorded on 2026-08-25:

- Baseline commit: `0cd0e10572e854bd3b03f2a49f6bf9b18ff6a6cb`
- Baseline CI: `32828993184`, `completed | success`
- Working tree: clean; `HEAD == origin/main`
- Full suite: 2965 passed, 141 skipped
- Final security/product audit: 366 focused tests
- Real PostgreSQL audit: 84 tests
- Leak scan hard counters: 0 / 0 / 0 / 0 / 0 / 0

The product already owns hierarchical tenant authorization, forced PostgreSQL
RLS, immutable document versions, authorization-bound retrieval plans,
evidence previews, review workflows, governed evaluation releases, retention,
legal hold, purge, and an OpenWebUI organization portal. The remaining program
is primarily enterprise platform work, not another round of isolated RAG
features.

## Program invariants

- One codebase serves `local`, `team`, and `enterprise` deployment profiles;
  security is never maintained through product forks.
- Local unauthenticated use is allowed only on a literal loopback address.
- Tenant identity comes from verified authority, never a request-selected
  tenant header or body field.
- Authorization is applied before retrieval; an unauthorized chunk never
  reaches ranking, generation, validation, logs, traces, or evaluation.
- OpenWebUI remains a working fallback until the custom frontend has passed
  parity, migration, rollback, and pilot gates.
- PostgreSQL remains the authority for relational metadata, policy, chunks,
  vectors, conversations, and audit evidence.
- Large immutable bytes belong behind a storage backend; remote object storage
  is not emulated by mounting it as a local filesystem.
- Every enterprise boundary has permanent cross-tenant, stale-policy,
  concurrency, recovery, and privacy tests.
- Each phase ends with focused tests, relevant real-PostgreSQL tests, full
  suite, leak scan, hygiene checks, human review, commit, push, and green CI.

## E0 - Architecture baseline and decision register

Status: `[x]` completed on 2026-08-25.

- [x] Map the current API, identity, database, storage, model-service, and UI
  trust boundaries from source.
- [x] Record the deployment-profile strategy: `local`, `team`, `enterprise`.
- [x] Record that OpenWebUI retirement is the final migration phase, never the
  starting move.
- [x] Record the shared-RLS and dedicated-data-plane isolation tiers.
- [x] Record the durable data split: PostgreSQL metadata and vectors; object
  storage for immutable sources and large artifacts.
- [x] Record the conversation-governance gap: OpenWebUI currently owns chat
  history, while the RAG database does not yet own conversations/messages.
- [x] Record the first implementation dependency: typed API contracts before a
  generated custom-frontend client.

Provider selections remain deploy-time decisions, not code defaults:

- [x] Select and record Keycloak as the first OIDC provider for the pilot.
- [ ] Select and record the first S3-compatible or managed object store.
- [x] Select and record a server-held BFF/session model for browser identity.
- [ ] Approve pilot RPO, RTO, SLO, data-residency, and compliance requirements.

## E1 - Closed API contracts and domain boundaries

Status: `[x]` complete.

- [x] Introduce closed response models without changing response bytes.
  - [x] Liveness, readiness, and OpenAI model-list response contracts.
  - [x] Organization and membership responses.
    - [x] Self context, visible members, topology read/replace, and membership
      mutation responses.
    - [x] Governance audit and retention-administration responses.
  - [x] Document inventory, detail, version, and lifecycle responses.
    - [x] Inventory, detail, archive/restore, version inventory, and version
      activation responses.
    - [x] Upload/process/ingest-job and retention/legal-hold/purge responses.
      - [x] Retention inventory/policy, legal-hold, and purge responses.
      - [x] Upload, process, and ingest-job responses.
  - [x] Evidence, export, review, and evaluation responses.
    - [x] Evidence ticket/preview, export ticket/download media, and review
      feedback/queue/decision responses.
    - [x] Evaluation dataset and release responses.
  - [x] Chat non-streaming response and SSE event contracts.
  - [x] Collection, tag, and document-organization responses.
- [x] Introduce one versioned, content-free error envelope and error-code
  vocabulary with an explicit compatibility transition from FastAPI `detail`.
- [x] Add an OpenAPI snapshot and backwards-compatibility CI gate.
- [x] Generate and compile a TypeScript client from the checked OpenAPI schema.
- [x] Standardize cursor, pagination, idempotency, conflict, and deprecation
  contracts.
- [x] Split `pipeline/api/app.py` into domain routers without changing route
  behavior or authorization dependencies.
- [x] Split the database facade into bounded domain repositories without
  creating a second SQL or authorization authority.

Exit gate: every JSON route has a closed response contract; OpenAPI drift is
reviewable; the generated TypeScript client compiles; OpenWebUI compatibility,
full tests, leak scan, and CI remain green.

## E2 - Enterprise identity and content-free control plane

Status: `[~]` in progress.

- [ ] Verify OIDC authorization-code/PKCE sessions through issuer, audience,
  client, expiry, and rotating JWKS keys.
- [ ] Add service-account issuance, rotation, revocation, and audit evidence.
- [ ] Add controlled provisioning/SCIM and IdP group-to-role policy.
- [ ] Separate platform administration from customer-tenant administration.
- [ ] Add tenant routing, deployment profile, feature, region, and quota facts
  to a content-free control plane.
- [ ] Add time-bounded, reasoned, audited break-glass authorization.
- [ ] Keep the OpenWebUI signed bridge during migration, mapping both identity
  roads to the same closed principal contract.

Exit gate: forged issuer/audience/tenant/role, stale keys, disabled membership,
revoked service accounts, replay, and cross-tenant identity collisions all fail
closed in permanent tests.

## E3 - Conversation ownership and hierarchy governance

Status: `[ ]` not started.

- [ ] Add tenant-bound conversations, members, messages, citations, feedback,
  access grants, and immutable access events.
- [ ] Enforce owner-only employee access, descendant-only manager monitoring,
  protected-manager privacy, and content-blind architect capability.
- [ ] Require a closed reason for management access and break-glass use.
- [ ] Add conversation retention, legal hold, export, deletion, and tenant
  offboarding behavior.
- [ ] Define and test OpenWebUI history import/read-only migration.

Exit gate: a real-PostgreSQL actor matrix proves self, descendant, peer,
protected ancestor, architect, break-glass, stale policy, and cross-tenant
decisions; every management content read creates immutable audit evidence.

## E4 - Object storage and unified disaster recovery

Status: `[ ]` not started.

- [ ] Define one `ObjectStore` contract and retain the current local backend.
- [ ] Add a native S3-compatible/managed backend with conditional immutable
  writes, bounded streaming, digest verification, and tenant isolation.
- [ ] Add encryption/KMS, lifecycle, legal-hold, purge, and audit integration.
- [ ] Move immutable sources and large export/evaluation artifacts behind the
  backend without exposing permanent object URLs.
- [ ] Bind PostgreSQL snapshot evidence, object inventory, schema receipt,
  model/index fingerprints, and key references into one restore set.
- [ ] Automate restore drills into an empty environment.

Exit gate: API and workers can run on separate hosts; database and object
restore is digest-complete; missing/tampered/cross-tenant objects fail closed.

## E5 - Tenant isolation tiers, quotas, and fair scheduling

Status: `[ ]` not started.

- [ ] Keep shared PostgreSQL plus forced RLS for standard tenants.
- [ ] Add dedicated database, pool, bucket, KMS key, backup, and region profiles
  for enterprise tenants.
- [ ] Add tenant request, concurrency, ingest, storage, document, evaluation,
  export, and model/token quotas.
- [ ] Add content-free usage metering and fair worker scheduling.
- [ ] Add deterministic tenant provisioning and offboarding workflows.

Exit gate: noisy-neighbor load tests stay within approved SLO impact; dedicated
tenant bytes and rows are absent from shared data planes.

## E6 - Model-service client platform and resilience

Status: `[ ]` not started.

- [ ] Put LLM, embedding, reranker, OCR, and table services behind closed client
  interfaces.
- [ ] Standardize pooling, timeout, retry, cancellation, circuit breaker,
  bulkhead, bounded payload, health, and closed failure codes.
- [ ] Authenticate remote services through private networking and mTLS or an
  equivalent approved service identity.
- [ ] Add a model registry binding artifact revision/digest, tokenizer, prompt,
  image, capability, evaluation release, and approval state.

Exit gate: one failed model service cannot exhaust unrelated work; vendor JSON
never crosses into core domain contracts; unapproved model releases cannot be
selected in production.

## E7 - SRE, observability, and production deployment

Status: `[ ]` not started.

- [ ] Containerize API, ingest worker, purge worker, migration, and evaluation
  jobs as reproducible, digest-pinned releases.
- [ ] Add TLS ingress, secret management, network policy, graceful shutdown,
  rollout, rollback, and resource controls.
- [ ] Add OpenTelemetry and durable Prometheus-compatible request, database,
  queue, object, and model latency/error metrics.
- [ ] Define SLOs, alert rules, dashboards, runbooks, RPO, and RTO.
- [ ] Run load, soak, DB restart, worker crash, object degradation, model
  outage, disk pressure, network partition, and rollback exercises.

Exit gate: measured p50/p95/p99, tested alerts, a successful restore drill, and
documented incident/rollback evidence exist for the pilot release.

## E8 - AI governance, privacy, and supply-chain security

Status: `[ ]` not started.

- [ ] Make blind holdout, human construct-validity review, retrieval quality,
  citation accuracy, abstention, false-review cost, and selective risk part of
  model promotion.
- [ ] Add permanent indirect prompt-injection and hostile-document corpora,
  including OCR and table-cell instructions.
- [ ] Add data classification, PII detection/redaction, DLP, residency,
  subject-access/deletion, and SIEM/WORM audit export.
- [ ] Pin dependencies and images; generate SBOM and provenance; run dependency,
  container, secret, and static security scans.

Exit gate: no model/config/image is promoted without its bound gates; hostile
documents cannot widen authority; sensitive data cannot enter logs, traces, or
unapproved evaluation data.

## E9 - Custom frontend foundation

Status: `[ ]` not started.

- [ ] Build the branded design system, responsive shell, accessibility and
  internationalization foundations.
- [ ] Use only the generated TypeScript API client.
- [ ] Add secure OIDC/BFF login, logout, session expiry, tenant selection, and
  user-visible position/level/policy facts.
- [ ] Establish browser, component, CSP, XSS, CSRF, and accessibility tests.

Exit gate: desktop/mobile shell and identity flows pass browser security and
accessibility baselines without removing OpenWebUI.

## E10 - Custom chat and document workspace

Status: `[ ]` not started.

- [ ] Add streaming/cancel/reconnect chat, governed conversation history,
  model and retrieval scope selection, citations, evidence, review state,
  feedback, attachments, and errors.
- [ ] Add upload/progress, inventory, search/filter, collections/tags, immutable
  versions, activation/rollback, archive/restore, evidence, and table exports.

Exit gate: the custom UI and OpenWebUI produce equivalent authorization,
retrieval scope, answer, citation, evidence, and persistence behavior against
the same backend scenarios.

## E11 - Custom administration and governance experience

Status: `[ ]` not started.

- [ ] Add platform tenant provisioning, isolation, region, quota, IdP, service
  account, feature, model-policy, and health controls.
- [ ] Add organization topology, membership, protected positions,
  conversation-monitoring policy, break-glass, audit, review, evaluation,
  retention, legal hold, and purge controls.
- [ ] Show each user their tenant, level, role, visibility, monitoring, privacy,
  retention, and quota facts without granting extra authority.

Exit gate: UI hiding is never treated as authorization; every direct API attack
still fails, and every management content access is audited.

## E12 - Dual-client enterprise pilot

Status: `[ ]` not started.

- [ ] Run OpenWebUI and the custom frontend against the same backend.
- [ ] Pilot root, manager, leaf, architect, reviewer, evaluation writer, shared
  tenant, and dedicated tenant personas.
- [ ] Measure task completion, latency, ingest reliability, citations,
  authorization failures, UI errors, migration integrity, and fallback use.
- [ ] Prove rollout, rollback, backup, restore, and tenant offboarding.

Exit gate: critical security/data-loss/cross-tenant findings are zero; SLO,
migration, restore, rollback, and pilot acceptance gates are green.

## E13 - OpenWebUI retirement

Status: `[ ]` not started and deliberately last.

- [ ] Make the custom frontend default while OpenWebUI remains a fallback.
- [ ] Move OpenWebUI to read-only after measured parity.
- [ ] Complete conversation migration and inventory/digest comparison.
- [ ] Revoke bridge credentials and remove OpenWebUI functions/service only
  after the rollback window.
- [ ] Retain an encrypted migration archive for the approved retention period.

Exit gate: all acceptance tests pass without OpenWebUI; old bridge credentials
are invalid; migration evidence is complete; the custom UI meets production
SLOs; the rollback window has closed.

## Progress log

### 2026-08-25 - Enterprise program opened

- Established this tracked roadmap from the verified `0cd0e10` baseline.
- Closed E0's source-measured architecture baseline and recorded unresolved
  provider/SLO decisions explicitly rather than guessing them.
- Started E1 with the smallest backwards-compatible response-contract slice:
  liveness, readiness, and OpenAI model discovery.
- Verified that slice with 88 focused API/operations tests. The generated
  OpenAPI components and runtime payloads are closed, undeclared fields fail,
  and the previous wire shapes remain unchanged.
- Passed `py_compile`, `pyflakes`, `git diff --check`, LF/BOM/tab/control
  hygiene, and all six hard leak counters. The scanner still reports its
  standing human-triage corpus plus prose overlap from this roadmap; neither is
  represented as a clean leak verdict.

### 2026-08-25 - E1 organization and document contract slice

- Attached closed response models to eleven additional route operations:
  organization self/visibility/topology/membership and document
  inventory/detail/archive/restore/version inventory/version activation.
- Added nested-field refusal, OpenAPI closure, route-binding, projection, and
  timestamp wire-compatibility tests. Database datetimes retain legacy ISO
  spelling instead of being silently rewritten to `Z`.
- Verified the slice with 307 focused org, document, auth, RBAC, and RAG tests.
  Governance audit/retention and the remaining document lifecycle operations
  stay open and are not counted as complete.

### 2026-08-25 - E1 closed API contracts and domain boundaries complete

- Closed every JSON response and chat SSE event contract, introduced the
  versioned content-free error vocabulary, and preserved the compatibility
  transition for existing FastAPI `detail` clients.
- Added checked OpenAPI current/floor snapshots, backwards-compatibility gates,
  and a generated TypeScript client that compiles under strict settings.
- Standardized pagination, cursor, idempotency, conflict, and deprecation
  metadata in one machine-readable protocol contract.
- Assigned all 53 HTTP handlers to seven domain routers. Split API database
  access into six frozen, fail-closed domain repositories while retaining
  `pipeline.index.db` as the only SQL and authorization authority.
- Verified the completed phase with 3062 passed and 141 skipped Python tests,
  a current OpenAPI snapshot, a passing OpenAPI compatibility check, and a
  clean generated TypeScript diff and typecheck. Final hygiene and leak-scan
  evidence is taken from this completed roadmap state before publication.
