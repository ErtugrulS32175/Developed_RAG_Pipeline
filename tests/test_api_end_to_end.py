"""Local end-to-end contracts for the production API wiring.

External services are replaced at their narrow network/storage seams.  The
application router, backend selection, structured generation, deterministic
guard and public response projection remain real.

Every filename, passage and figure below is invented.
"""
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@contextmanager
def _fake_db_conn():
    """Stands in for the pooled per-request connection; the db helpers are
    monkeypatched, so the connection object itself is never touched."""
    yield object()

from pipeline.validation.rag.answer_guard import (
    ABSTAINED,
    ANSWERED,
    REVIEW_REQUIRED,
)


CHUNKS = [
    {
        "filename": "kurgu-belge.pdf",
        "page": 42,
        "type": "text",
        "text": "Zeta uretimi 47 000 birimdir.",
        "headings": [],
        "table_data": None,
    },
]


def _headers(api):
    return (
        {"Authorization": f"Bearer {api.API_KEY}"}
        if api.API_KEY else {}
    )


_ABSENT = object()


def _chat(api, model, stream=False, document_ids=_ABSENT):
    """One chat request. `document_ids` is OMITTED unless a test supplies
    one, so an unscoped request is byte-for-byte the request it always
    was -- an absent field and a null field are not the same evidence."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "zeta uretimi nedir?"}],
        "stream": stream,
    }
    if document_ids is not _ABSENT:
        body["document_ids"] = document_ids
    return TestClient(api.app).post(
        "/v1/chat/completions",
        headers=_headers(api),
        json=body,
    )


def _public_reply(response, stream):
    if not stream:
        body = response.json()
        return body["rag_status"], body["choices"][0]["message"]["content"]

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    statuses = {payload["rag_status"] for payload in payloads}
    assert len(statuses) == 1
    text = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )
    return statuses.pop(), text


def _wire_retrieval(monkeypatch, backend):
    from pipeline.generation import answer as generation

    if backend == "native":
        from pipeline.retrieval import query

        monkeypatch.setattr(query, "retrieve", lambda _question: CHUNKS)
        monkeypatch.setattr(
            query,
            "rerank",
            lambda _question, chunks: chunks,
        )
    else:
        from pipeline.retrieval import rag_llamaindex

        monkeypatch.setattr(
            rag_llamaindex,
            "retrieve",
            lambda _question: CHUNKS,
        )
    return generation


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        ("ragtest-rag", "native"),
        ("ragtest-rag-llamaindex", "llamaindex"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_public_api_runs_the_real_checked_path(
        monkeypatch, model, backend, stream):
    from pipeline.api import app as api

    generation = _wire_retrieval(monkeypatch, backend)
    seen = []

    def complete(policy, user_content):
        seen.append((policy, user_content))
        return json.dumps({
            "dayanak": [{
                "pasaj": 1,
                "alinti": "Zeta uretimi 47 000 birimdir.",
            }],
            "cevap": "Sayfa 42'ye gore 47 000 birim.",
        })

    monkeypatch.setattr(generation, "complete", complete)

    response = _chat(api, model, stream)

    assert response.status_code == 200
    status, text = _public_reply(response, stream)
    assert status == ANSWERED
    assert text == "Sayfa 42'ye gore 47 000 birim."
    assert len(seen) == 1
    policy, user_content = seen[0]
    assert policy.index("dayanak") < policy.index('"cevap"')
    assert "[P1]" in user_content
    assert "zeta uretimi nedir?" in user_content
    assert "Zeta uretimi" not in policy


@pytest.mark.parametrize(
    ("model", "backend"),
    [
        ("ragtest-rag", "native"),
        ("ragtest-rag-llamaindex", "llamaindex"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_public_api_withholds_a_reply_rejected_by_the_real_guard(
        monkeypatch, model, backend, stream):
    from pipeline.api import app as api

    generation = _wire_retrieval(monkeypatch, backend)
    raw = "Sayfa 42'ye gore 88 000 birim."
    monkeypatch.setattr(
        generation,
        "complete",
        lambda _policy, _user_content: json.dumps({
            "dayanak": [{
                "pasaj": 1,
                "alinti": "Zeta uretimi 47 000 birimdir.",
            }],
            "cevap": raw,
        }),
    )

    response = _chat(api, model, stream)

    assert response.status_code == 200
    status, text = _public_reply(response, stream)
    assert status == REVIEW_REQUIRED
    assert text == api.REVIEW_MESSAGE
    assert raw not in response.text


PUBLISHED = "published"


DONE_RUN = ("done", None)


DOCUMENT_UUID = "11111111-1111-4111-8111-111111111111"
VERSION_UUID = "22222222-2222-4222-8222-222222222222"


def _version_source(api, root, document_id, version_id):
    from pipeline.index import publication

    return publication.version_source_path(
        root, str(api.db.DEFAULT_TENANT_ID), document_id, version_id)


def _document_api(monkeypatch, tmp_path, *, ingest_outcome=DONE_RUN,
                  ingest_error=None):
    """The document routes with only their DATABASE seams replaced.

    Package 3C moved the endpoint's hand-written publish sequence into
    the shared service, so the REAL ``publish_candidate`` runs here --
    its lock order, its temp-file-then-rename and its canonical-name
    check are all exercised. What is modelled is the row underneath it:
    stage, finalize, the lease, the guarded status stamp."""
    from pipeline.index import publication
    from pipeline.index.attempt_contract import (
        CandidateConflict,
        CandidateNotPublished,
        IngestAttempt,
    )
    from pipeline.api import app as api

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(api, "UPLOAD_DIR", upload_dir)
    # Rule 13: the destination belongs to the publication service, so the
    # directory to redirect is the SERVICE's -- the endpoint's own copy
    # only decides where `process` LOOKS for the file.
    monkeypatch.setattr(publication, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(api, "db_conn", _fake_db_conn)

    document_id = DOCUMENT_UUID
    state = {}
    calls = []
    minted = {"n": 0}
    attempts = []
    closed = []

    def stage_candidate(_conn, filename, file_type, content_sha256=None,
                        allow_replace=False):
        # the real gate in miniature: the stored spelling is canonical,
        # the candidate id changes only with the bytes, and different
        # bytes need explicit authority
        row = state.get(document_id)
        canonical = row["filename"] if row else filename
        if not (allow_replace or row is None
                or row.get("content_sha256") in (None, content_sha256)):
            raise CandidateConflict(
                "ayni dosya adi farkli icerikle zaten kayitli")
        if (row and row.get("content_sha256") == content_sha256
                and row.get("candidate_id")):
            cid = row["candidate_id"]
        else:
            minted["n"] += 1
            cid = str(uuid.UUID(int=minted["n"] + 32))
        state[document_id] = {
            "id": document_id,
            "filename": canonical,
            "file_type": file_type,
            # an upload does NOT move the served status -- that column
            # describes the version being answered from
            "status": (row or {}).get("status", "pending"),
            "content_sha256": content_sha256,
            "candidate_id": cid,
            "candidate_state": "staged",
            "active_generation": (row or {}).get("active_generation", 0),
            "archived_at": (row or {}).get("archived_at"),
            "attempt_id": (row or {}).get("attempt_id"),
        }
        return document_id, cid, canonical

    def finalize_candidate_publication(_conn, wanted, candidate_id):
        row = state.get(wanted)
        if row is None or row.get("candidate_id") != candidate_id:
            return False
        row["candidate_state"] = PUBLISHED
        return True

    def begin_attempt(_conn, wanted, owner=None):
        row = state[wanted]
        if row.get("archived_at") is not None:
            raise api.db.DocumentLifecycleConflict("kurgu arsiv")
        if row.get("candidate_state") != PUBLISHED:
            raise CandidateNotPublished("aday yayimlanmis degil")
        attempt = IngestAttempt(
            attempt_id=f"kurgu-deneme-{len(attempts) + 1}",
            document_id=wanted,
            candidate_id=row["candidate_id"],
            candidate_sha=row["content_sha256"],
            observed_active=int(row.get("active_generation") or 0),
        )
        attempts.append(attempt)
        return attempt

    def lookup_document(_conn, filename):
        for row in state.values():
            if row["filename"].casefold() == filename.casefold():
                return dict(row)
        return None

    def get_document(_conn, wanted):
        value = state.get(wanted)
        return dict(value) if value is not None else None

    def set_document_status(_conn, wanted, status, note=None,
                            expected_active=None, candidate_id=None):
        # the guarded-stamp contract, same as the real statement's WHERE:
        # a stamp whose run identity is stale writes nothing
        row = state[wanted]
        if (expected_active is not None
                and row.get("active_generation", 0) != expected_active):
            return False
        if (candidate_id is not None
                and row.get("candidate_id") != candidate_id):
            return False
        row["status"] = status
        return True

    def set_document_archived(_conn, wanted, archived):
        row = state.get(wanted)
        if row is None:
            return None
        already = row.get("archived_at") is not None
        if already == archived:
            return {
                "document_id": wanted,
                "archived": already,
                "archived_at": row.get("archived_at"),
            }
        if row.get("attempt_id") is not None:
            raise api.db.DocumentLifecycleConflict("kurgu cakisma")
        row["archived_at"] = (
            f"{998:04d}-01-01T00:00:00+00:00" if archived else None)
        return {
            "document_id": wanted,
            "archived": archived,
            "archived_at": row["archived_at"],
        }

    def run_ingest(object_root, tenant_id, wanted_document, version_id,
                   canonical_filename, attempt, *, expected_sha256,
                   max_bytes):
        """The core as it REALLY behaves, which is the point of this fake.

        The previous version wrote the document's status directly, and
        that single line hid a live defect: a real PARTIAL run leaves
        the row untouched -- its verdict belongs to the attempt (rule 5)
        -- so an endpoint reading the row back saw `processing` and
        called a truthful partial run a failure. Here only a PROMOTION
        moves the served status, in the same breath as the generation,
        and the run REPORTS its verdict to its caller."""
        calls.append((object_root, tenant_id, wanted_document, version_id,
                      canonical_filename, attempt, expected_sha256,
                      max_bytes))
        if ingest_error is not None:
            raise ingest_error
        if ingest_outcome is None:
            return None                 # a run that reported nothing
        status, note = ingest_outcome
        if status == "done":
            row = state[wanted_document]
            row["status"] = "done"
            row["active_generation"] = int(row.get("active_generation") or 0) + 1
        return status, note

    @contextmanager
    def publish_lock(_conn, _filename):
        yield

    def abandon_attempt(att, note):
        # the real helper opens its own short connection and records
        # ERROR on the attempt, clearing the lease in the same statement
        closed.append((att.attempt_id, note))

    monkeypatch.setattr(api.ingest, "abandon_attempt", abandon_attempt)
    monkeypatch.setattr(api.db, "stage_candidate", stage_candidate)
    monkeypatch.setattr(api.db, "finalize_candidate_publication",
                        finalize_candidate_publication)
    monkeypatch.setattr(api.db, "begin_attempt", begin_attempt)
    monkeypatch.setattr(api.db, "lookup_document", lookup_document)
    monkeypatch.setattr(api.db, "get_document", get_document)
    monkeypatch.setattr(api.db, "active_ingest_job",
                        lambda _conn, _document_id: None)
    monkeypatch.setattr(api.db, "set_document_status", set_document_status)
    monkeypatch.setattr(api.db, "set_document_archived",
                        set_document_archived)
    monkeypatch.setattr(api.db, "document_publish_lock", publish_lock)
    # Immutable-source and legacy-migration behaviour has its own storage
    # battery.  This fixture models the HTTP/attempt contract, so keep that
    # storage prerequisite neutral unless a test replaces it deliberately.
    monkeypatch.setattr(
        api.publication, "ensure_bound_version_source",
        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api.ingest, "ingest_version_source", run_ingest)
    return api, TestClient(api.app), state, calls, upload_dir, closed


def test_document_upload_process_and_read_use_the_production_routes(
        monkeypatch, tmp_path):
    api, client, state, calls, upload_dir, _closed = _document_api(
        monkeypatch,
        tmp_path,
    )

    uploaded = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"KURGU_PDF", "application/pdf")},
    )
    assert uploaded.status_code == 200
    document_id = uploaded.json()["document_id"]
    assert uploaded.json()["status"] == "pending"
    first_candidate = uploaded.json()["candidate_id"]
    assert _version_source(
        api, upload_dir, document_id, first_candidate).read_bytes() == b"KURGU_PDF"

    # Round 14: same name + different bytes is a CONFLICT by default -- the
    # old contract announced the replacement only after doing it. The old
    # file must survive a refused upload untouched.
    conflict = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"KURGU_PDF_V2", "application/pdf")},
    )
    assert conflict.status_code == 409
    assert _version_source(
        api, upload_dir, document_id, first_candidate).read_bytes() == b"KURGU_PDF"
    # replacement happens only when the caller claims it out loud
    replaced = client.post(
        "/documents/upload?replace=true",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"KURGU_PDF_V2", "application/pdf")},
    )
    assert replaced.status_code == 200
    # Package 3C: the response no longer carries a `replaced` flag -- it
    # was computed by hashing the disk, which the shared service does not
    # report back and which could only be re-derived OUTSIDE the publish
    # lock. The candidate id answers the same question truthfully.
    assert replaced.json()["candidate_id"] != first_candidate
    assert _version_source(
        api, upload_dir, document_id,
        replaced.json()["candidate_id"]).read_bytes() == b"KURGU_PDF_V2"
    # same bytes again: the SAME candidate, so nothing was replaced
    again = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"KURGU_PDF_V2", "application/pdf")},
    )
    assert again.status_code == 200
    assert again.json()["candidate_id"] == replaced.json()["candidate_id"]

    processed = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )
    assert processed.status_code == 200
    assert processed.json() == {"document_id": document_id, "status": "done"}
    # Package 3C: the binding travels inside the ATTEMPT the endpoint
    # took before anything was parsed -- processing "whatever the disk
    # holds now" was the P0, and a tuple the endpoint read a moment
    # earlier was still not a run identity.
    (root_called, tenant_called, document_called, version_called,
     name_called, attempt, expected_sha, _max_bytes) = calls[0]
    assert root_called == upload_dir
    assert tenant_called == api.db.DEFAULT_TENANT_ID
    assert document_called == document_id
    assert version_called == state[document_id]["candidate_id"]
    assert name_called == "kurgu-belge.pdf"
    assert expected_sha == state[document_id]["content_sha256"]
    assert attempt.candidate_id == state[document_id]["candidate_id"]
    assert attempt.candidate_sha == state[document_id]["content_sha256"]

    read = client.get(
        f"/documents/{document_id}",
        headers=_headers(api),
    )
    assert read.status_code == 200
    assert read.json()["status"] == "done"
    assert state[document_id]["status"] == "done"


def test_upload_publishes_db_and_disk_inside_one_held_lock(
        monkeypatch, tmp_path):
    """Round 17: the transaction-scoped lock released at the upsert's own
    commit and os.replace ran OUTSIDE it -- a two-worker probe left the
    database carrying the second hash and the disk the first bytes. The
    structural fix is checkable on a single request: the database decision
    and the disk publish both land inside the session lock's span.

    Package 3C: the span is now the SERVICE's, and the sequence gained a
    finalisation -- the candidate becomes published only once the row and
    the bytes agree, which is what closed the gap a process request used
    to fall into."""
    api, client, state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    events = []

    @contextmanager
    def recording_lock(_conn, filename):
        events.append("kilit-al")
        try:
            yield
        finally:
            events.append("kilit-birak")

    def recording_stage(_conn, filename, file_type, content_sha256=None,
                        allow_replace=False):
        events.append("db-evrele")
        return DOCUMENT_UUID, VERSION_UUID, filename

    def recording_finalize(_conn, _document_id, _candidate_id):
        events.append("db-yayimla")
        return True

    from pipeline.index import publication
    real_publish = publication.publish_version_source

    def recording_publish(*args, **kwargs):
        events.append("surum-kaynagi")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(api.db, "document_publish_lock", recording_lock)
    monkeypatch.setattr(api.db, "stage_candidate", recording_stage)
    monkeypatch.setattr(api.db, "finalize_candidate_publication",
                        recording_finalize)
    monkeypatch.setattr(publication, "publish_version_source",
                        recording_publish)

    response = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu.pdf", b"KURGU_PDF", "application/pdf")},
    )
    assert response.status_code == 200
    assert events == ["kilit-al", "db-evrele", "surum-kaynagi", "db-yayimla",
                      "kilit-birak"]


def test_two_workers_cannot_split_the_database_from_the_disk(
        monkeypatch, tmp_path):
    """The probe's shape: two workers, same name, no shared process lock.

    Package 3C removed the endpoint's in-process lock entirely -- the
    publication service holds a database SESSION lock, which serialises
    across processes and therefore already covers what the weaker one
    covered. So this is now simply what two workers look like, and the
    invariant that must hold either way the race lands is that the
    recorded candidate hash and the published bytes agree."""
    api, client, state, _calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    gate = threading.Lock()

    @contextmanager
    def session_lock(_conn, _filename):
        with gate:
            yield

    def slow_stage(_conn, filename, file_type, content_sha256=None,
                   allow_replace=False):
        candidate_id = str(uuid.UUID(content_sha256[:32]))
        state[DOCUMENT_UUID] = {
            "id": DOCUMENT_UUID,
            "filename": filename,
            "file_type": file_type,
            "status": "pending",
            "content_sha256": content_sha256,
            "candidate_id": candidate_id,
            "candidate_state": "staged",
        }
        time.sleep(0.15)  # the historic gap between DB commit and os.replace
        return (DOCUMENT_UUID, candidate_id, filename)

    monkeypatch.setattr(api.db, "document_publish_lock", session_lock)
    monkeypatch.setattr(api.db, "stage_candidate", slow_stage)

    codes = []

    def worker(body, replace):
        response = TestClient(api.app).post(
            f"/documents/upload?replace={'true' if replace else 'false'}",
            headers=_headers(api),
            files={"file": ("kurgu.pdf", body, "application/pdf")},
        )
        codes.append(response.status_code)

    first = threading.Thread(target=worker, args=(b"ILK_KURGU", False))
    second = threading.Thread(target=worker, args=(b"IKINCI_KURGU", True))
    first.start()
    time.sleep(0.05)
    second.start()
    first.join()
    second.join()

    assert sorted(codes) == [200, 200]
    winning = state[DOCUMENT_UUID]
    source = _version_source(
        api, upload_dir, DOCUMENT_UUID, winning["candidate_id"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == winning[
        "content_sha256"]


def test_a_recased_upload_targets_the_canonical_disk_file(
        monkeypatch, tmp_path):
    """Round 18: the database canonicalises spellings but the disk target
    was built from the REQUEST spelling -- on a case-sensitive filesystem
    the row pointed at one file while the bytes landed in another. The
    destination now comes from the canonical name."""
    api, client, state, _calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    first = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"ILK_KURGU", "application/pdf")},
    )
    assert first.status_code == 200
    assert first.json()["filename"] == "kurgu-belge.pdf"
    recased = client.post(
        "/documents/upload?replace=true",
        headers=_headers(api),
        files={"file": ("KURGU-Belge.PDF", b"IKINCI_KURGU",
                        "application/pdf")},
    )
    assert recased.status_code == 200
    # the response and the disk both speak the CANONICAL spelling
    assert recased.json()["filename"] == "kurgu-belge.pdf"
    assert _version_source(
        api, upload_dir, DOCUMENT_UUID,
        recased.json()["candidate_id"]).read_bytes() == b"IKINCI_KURGU"
    assert state[DOCUMENT_UUID]["filename"] == "kurgu-belge.pdf"


def test_a_process_bound_to_a_stale_candidate_cannot_touch_the_index(
        monkeypatch, tmp_path):
    """Round 18, the P0 replayed at the seam: the process step hands its
    ingest the candidate identity it READ, and an ingest that finds the
    disk carrying other bytes refuses. Here the disk moved after the row
    was read -- the refusal is the contract."""
    api, client, state, calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = DOCUMENT_UUID
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
        "content_sha256": hashlib.sha256(b"ILK_KURGU").hexdigest(),
        "candidate_id": VERSION_UUID,
        "candidate_state": PUBLISHED,
    }
    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )
    assert response.status_code == 200
    attempt = calls[0][5]
    assert attempt.candidate_id == VERSION_UUID
    assert attempt.candidate_sha == hashlib.sha256(b"ILK_KURGU").hexdigest()


def test_a_second_process_request_is_refused_while_a_run_holds_the_lease(
        monkeypatch, tmp_path):
    """Package 3C: the lease is taken by the ENDPOINT, before anything is
    parsed, so a document already being indexed is answered rather than
    indexed twice. 409 and nothing else: no ingest, no status change --
    the run that holds the lease owns the document's state, and a second
    request must not write over it on its way to being refused."""
    from pipeline.index.attempt_contract import AttemptAlreadyRunning

    api, client, state, calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "done",
        "active_generation": 3,
        "content_sha256": "c" * 64,
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    def busy(*_a, **_k):
        raise AttemptAlreadyRunning("canli bir deneme var")

    monkeypatch.setattr(api.db, "begin_attempt", busy)

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 409
    assert calls == []
    assert state[document_id]["status"] == "done"
    assert state[document_id]["active_generation"] == 3


def test_a_document_without_a_recorded_candidate_cannot_be_processed(
        monkeypatch, tmp_path):
    """A legacy row with no candidate identity has nothing for the ingest
    to bind to: processing it would be exactly the unbound run the P0
    exploited -- 409, not a guess."""
    api, client, state, calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )
    assert response.status_code == 409
    assert calls == []


def test_a_lost_promotion_race_does_not_relabel_the_winners_done(
        monkeypatch, tmp_path):
    """Round 17: the losing run's error handler stamped 'error' over the
    winner's 'done' -- active generation 1, final status error, a healthy
    index wearing a failure label. Status stamps are scoped to the run's
    own observation now, so the loser's verdict lands nowhere."""
    api, client, state, _calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
        "active_generation": 0,
        "content_sha256": "c" * 64,
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    def losing_ingest(_root, _tenant, _document, _version, _filename,
                      _attempt, **_kwargs):
        # a concurrent run promotes first...
        state[document_id]["active_generation"] = 1
        state[document_id]["status"] = "done"
        # ...and THIS run's promotion fails its CAS loudly
        raise RuntimeError("es zamanli terfi kazandi")

    monkeypatch.setattr(api.ingest, "ingest_version_source", losing_ingest)

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 500          # the REQUEST failed...
    assert state[document_id]["status"] == "done"        # ...the DOCUMENT did not
    assert state[document_id]["active_generation"] == 1


def test_process_does_not_claim_done_when_the_run_reported_no_verdict(
        monkeypatch, tmp_path):
    """A run that comes back without a terminal verdict is not a success
    and not a partial -- it is an answer nobody can act on. The REQUEST
    fails; the served version, which that run never touched, keeps
    saying exactly what it said."""
    api, client, state, _calls, upload_dir, closed = _document_api(
        monkeypatch,
        tmp_path,
        ingest_outcome=None,
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "done",
        "active_generation": 4,
        "content_sha256": "c" * 64,
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert state[document_id]["status"] == "done"
    assert state[document_id]["active_generation"] == 4
    # ...and the lease does not outlive the request. The HTTP side was
    # already fail-closed here while the LIFECYCLE was not: a retry had
    # to wait out the whole lease window for a run that was long over.
    assert closed == [("kurgu-deneme-1", "IncompleteIngest")]


def test_a_partial_ingest_is_reported_without_touching_the_served_version(
        monkeypatch, tmp_path):
    """Round 14: "partial" is a TRUE statement -- pages were lost, stored
    chunks are real, and the last COMPLETE generation is still what
    answers questions. The API used to rewrite it as error + HTTP 500.

    Package 3C: the partial verdict comes from the RUN, not from the
    document row. A real partial closes its attempt PARTIAL and leaves
    the row alone by design -- and the endpoint, which used to stamp
    `processing` first and then read that row back, called the truthful
    partial "never finished". The served status must come out of this
    request exactly as it went in."""
    api, client, state, _calls, upload_dir, _closed = _document_api(
        monkeypatch,
        tmp_path,
        ingest_outcome=("partial", '[{"kaynak": "page2:scanned"}]'),
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "done",
        "active_generation": 4,
        "content_sha256": "c" * 64,
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert "page2:scanned" in body["status_note"]
    # the SERVED version is untouched: a partial run promoted nothing
    assert state[document_id]["status"] == "done"
    assert state[document_id]["active_generation"] == 4


def test_a_missing_source_file_is_generic_and_leaves_the_index_alone(
        monkeypatch, tmp_path):
    api, client, state, calls, _upload_dir, closed = _document_api(
        monkeypatch,
        tmp_path,
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "done",
        "active_generation": 4,
        "content_sha256": "c" * 64,
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }

    from pipeline.index.publication import VersionSourceMissing

    def missing_source(*_args, **_kwargs):
        raise VersionSourceMissing("invented private storage detail")

    monkeypatch.setattr(api.ingest, "ingest_version_source", missing_source)

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert response.json()["detail"] == api.DOCUMENT_PROCESSING_FAILURE_MESSAGE
    # The SERVED version is untouched. Its chunks are in the index and
    # still answering questions -- the source file is only needed to
    # build the NEXT generation, so a missing one is a storage problem,
    # not a verdict on what is being served. Stamping `error` here left
    # a healthy index wearing a failure label.
    assert state[document_id]["status"] == "done"
    assert state[document_id]["active_generation"] == 4
    assert calls == []
    assert closed == [("kurgu-deneme-1", "VersionSourceMissing")]
    assert "kurgu-belge.pdf" not in response.text


def test_process_failure_is_generic_and_leaves_the_served_version_alone(
        monkeypatch, tmp_path, caplog):
    private = "OZEL_KURGU_INGEST_AYRINTISI"
    api, client, state, _calls, upload_dir, closed = _document_api(
        monkeypatch,
        tmp_path,
        ingest_error=RuntimeError(private),
    )
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "done",
        "active_generation": 4,
        "content_sha256": "c" * 64,
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"KURGU_PDF")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert response.json()["detail"] == api.DOCUMENT_PROCESSING_FAILURE_MESSAGE
    # Package 3C: the failure is the RUN's, and it is recorded on the
    # run's own attempt by the core. The endpoint no longer stamps the
    # document -- a crashed run did not make the served version any
    # worse, and labelling a healthy active generation `error` was how a
    # loser used to brand the winner's index.
    assert state[document_id]["status"] == "done"
    assert state[document_id]["active_generation"] == 4
    # the lease is released, and the note carries the exception TYPE only
    assert closed == [("kurgu-deneme-1", "RuntimeError")]
    assert private not in response.text
    assert private not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize(
    "filename",
    [
        "../disari.pdf",
        "..\\disari.pdf",
        "/disari.pdf",
        "alt/disari.pdf",
        "alt\\disari.pdf",
        "kurgu.pdf.",
        "...",
        "NUL",
        "nul.pdf",
        "PRN.pdf",
    ],
)
def test_upload_rejects_a_path_or_noncanonical_filename_before_writing(
        monkeypatch, tmp_path, filename):
    api, client, _state, calls, upload_dir, _closed = _document_api(
        monkeypatch,
        tmp_path,
    )

    response = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": (filename, b"KURGU_PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert calls == []
    assert not (tmp_path / "disari.pdf").exists()
    assert list(upload_dir.iterdir()) == []


def test_rejected_alias_cannot_overwrite_an_existing_upload(
        monkeypatch, tmp_path):
    api, client, state, calls, upload_dir, _closed = _document_api(
        monkeypatch,
        tmp_path,
    )
    original = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu.pdf", b"ILK_KURGU_PDF", "application/pdf")},
    )
    alias = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu.pdf.", b"IKINCI_KURGU_PDF", "application/pdf")},
    )

    assert original.status_code == 200
    assert alias.status_code == 400
    source = _version_source(
        api, upload_dir, original.json()["document_id"],
        original.json()["candidate_id"])
    assert source.read_bytes() == b"ILK_KURGU_PDF"
    assert len(state) == 1
    assert calls == []


@pytest.mark.parametrize(
    "filename",
    [
        "C:\\disari.pdf",
        "kurgu.pdf\n",
        "kurgu\x00.pdf",
        "kurgu.pdf:ek",
        "kurgu.pdf.",
        "...",
        "NUL",
        "nul.pdf",
        "CON",
        "PRN.pdf",
        "AUX",
        "com1",
        "COM¹.txt",
        "LPT9.txt",
        "NUL .pdf",
        "..",
        None,
    ],
)
def test_unsafe_filename_is_rejected_before_path_construction(filename):
    from fastapi import HTTPException
    from pipeline.api import app as api

    with pytest.raises(HTTPException) as error:
        api._safe_upload_filename(filename)

    assert error.value.status_code == 400


# --- reversible document lifecycle -------------------------------------


def _lifecycle_api_row(*, archived_at=None, attempt_id=None):
    return {
        "id": "kurgu-yasam-belgesi",
        "filename": "kurgu-yasam.pdf",
        "file_type": "pdf",
        "status": "done",
        "content_sha256": "kurgu-ozet",
        "candidate_id": "kurgu-aday",
        "candidate_state": PUBLISHED,
        "active_generation": 4,
        "archived_at": archived_at,
        "attempt_id": attempt_id,
    }


def test_archive_and_restore_are_idempotent_metadata_transitions(
        monkeypatch, tmp_path):
    api, client, state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-yasam-belgesi"
    state[document_id] = _lifecycle_api_row()

    archived = client.post(
        f"/documents/{document_id}/archive", headers=_headers(api))
    repeated = client.post(
        f"/documents/{document_id}/archive", headers=_headers(api))
    restored = client.post(
        f"/documents/{document_id}/restore", headers=_headers(api))
    restored_again = client.post(
        f"/documents/{document_id}/restore", headers=_headers(api))

    assert archived.status_code == repeated.status_code == 200
    assert archived.json() == repeated.json()
    assert archived.json()["archived"] is True
    assert archived.json()["archived_at"] is not None
    assert restored.status_code == restored_again.status_code == 200
    assert restored.json() == restored_again.json() == {
        "document_id": document_id,
        "archived": False,
        "archived_at": None,
    }
    # The lifecycle is metadata-only: the publication identity remains.
    assert state[document_id]["candidate_id"] == "kurgu-aday"
    assert state[document_id]["active_generation"] == 4


@pytest.mark.parametrize("operation", ["archive", "restore"])
def test_a_missing_document_has_no_lifecycle_to_change(
        monkeypatch, tmp_path, operation):
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)

    response = client.post(
        f"/documents/yok/{operation}", headers=_headers(api))

    assert response.status_code == 404


def test_an_active_ingest_attempt_blocks_archive(monkeypatch, tmp_path):
    api, client, state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-yasam-belgesi"
    state[document_id] = _lifecycle_api_row(attempt_id="aktif-deneme")

    response = client.post(
        f"/documents/{document_id}/archive", headers=_headers(api))

    assert response.status_code == 409
    assert state[document_id]["archived_at"] is None


def test_an_archived_document_cannot_start_processing(monkeypatch, tmp_path):
    api, client, state, calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-yasam-belgesi"
    state[document_id] = _lifecycle_api_row(archived_at="once")

    response = client.post(
        f"/documents/{document_id}/process", headers=_headers(api))

    assert response.status_code == 409
    assert calls == []


def test_lifecycle_routes_require_the_admin_dependency(
        monkeypatch, tmp_path):
    api, _client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)

    def dependencies(path, method):
        for route in api.app.routes:
            if (getattr(route, "path", None) == path
                    and method in getattr(route, "methods", ())):
                return [depends.dependency for depends in route.dependencies]
        raise AssertionError(path)

    assert dependencies(
        "/documents/{document_id}/archive", "POST") == [api.require_admin]
    assert dependencies(
        "/documents/{document_id}/restore", "POST") == [api.require_admin]


# --- the document inventory --------------------------------------------


def _inventory_row(document_id, uploaded_at, **overrides):
    """A row as the DATABASE holds it -- candidate columns included.

    These tests deliberately feed the endpoint MORE than the real query
    selects. The narrow SELECT is the first guard and is checked in
    test_db_lifecycle; what is measured here is the second one, the
    endpoint's own projection, which is the guard that still holds if a
    column is later added to that SELECT list.
    """
    row = {
        "document_id": document_id,
        "id": document_id,
        "filename": f"{document_id}.pdf",
        "file_type": "pdf",
        "uploaded_at": uploaded_at,
        "status": "done",
        "status_note": None,
        "active_generation": 2,
        "archived_at": None,
        "content_sha256": "KURGU_SHA_" + document_id,
        "candidate_id": "KURGU_ADAY_" + document_id,
        "candidate_state": PUBLISHED,
    }
    row.update(overrides)
    return row


INVENTORY = [
    _inventory_row("kurgu-uc", f"{999:04d}-03-03T00:00:00+00:00"),
    _inventory_row("kurgu-iki", f"{999:04d}-02-02T00:00:00+00:00",
                   status="partial", status_note="sayfa 2 kayip"),
    _inventory_row("kurgu-bir", f"{999:04d}-01-01T00:00:00+00:00"),
]


class _InventoryCalls(list):
    """The four page/equality arguments per call, positionally -- exactly
    the record this fixture always kept -- with the two date bounds of the
    same calls alongside in `.dates` and the search value in `.searches`.

    Each addition went NEXT TO that record rather than into it: a test
    that pinned `[(20, 0, None, None)]` still pins it, whether the call
    also carried a window, a search, both or neither."""

    def __init__(self):
        super().__init__()
        self.dates = []
        self.searches = []
        self.archives = []
        self.cursors = []
        self.tenants = []


def _like_matches(value, pattern, escape="!"):
    """`value ILIKE pattern ESCAPE escape`, over the ASCII subset.

    MEASURED: no live PostgreSQL is reachable in this loop -- the local
    Docker daemon is down and CI declares no database service -- so the
    real operator runs NOWHERE here, and a fixture that merely compared
    substrings would pass whatever the escaping did or failed to do. This
    models the operator's PATTERN LANGUAGE, written from the definition
    and knowing nothing about the transform it is used to judge: the
    escape character consumes the next character literally, `%` is any
    run of characters, `_` is exactly one, the match is anchored, and
    ASCII letters compare case-insensitively.

    WHAT IT DOES NOT PROVE: it is not PostgreSQL. Case folding here is
    Python's; the server's is decided by the column's collation, which
    this model neither knows nor consults. The metacharacter, escape and
    anchoring behaviour it pins holds for the ASCII fixtures in this
    battery; real collation and full Unicode case folding are NOT
    established here and wait on a live server."""
    import re

    parts = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == escape:
            index += 1
            if index >= len(pattern):
                # the server's own error: a pattern may not end on a lone
                # escape character
                raise ValueError("LIKE deseni kacis karakteriyle bitemez")
            parts.append(re.escape(pattern[index]))
        elif char == "%":
            parts.append("(?s:.)*")
        elif char == "_":
            parts.append("(?s:.)")
        else:
            parts.append(re.escape(char))
        index += 1
    return re.fullmatch("".join(parts), value, re.IGNORECASE) is not None


def _wire_inventory(monkeypatch, api, rows=INVENTORY):
    """Replace the query with one that records how it was CALLED.

    It returns the ``limit + 1`` window the real helper returns, so the
    endpoint's `has_more` is computed from the same evidence in the test
    as in production -- and, like the real helper, it filters BEFORE it
    pages, so offset and the sentinel walk the filtered sequence.

    The date bounds are applied here the way the SQL applies them: both
    EXCLUSIVE, so a row sitting exactly on a bound is filtered out.

    `q` is applied the way the SQL applies THAT: the production transform
    builds the pattern -- it is the code under test and is not reimplemented
    here -- and `_like_matches` interprets it exactly as the server would.
    So a search's behaviour is decided by the real escaping run through a
    real LIKE model, which is the strongest evidence available without a
    live server."""
    asked = _InventoryCalls()

    def list_documents(_conn, limit, offset, status=None, file_type=None,
                       uploaded_after=None, uploaded_before=None, q=None,
                       archived=False, collection_id=None, tag=None,
                       before=None, tenant_id=None):
        asked.append((limit, offset, status, file_type))
        asked.dates.append((uploaded_after, uploaded_before))
        asked.searches.append(q)
        asked.archives.append(archived)
        asked.cursors.append(before)
        asked.tenants.append(tenant_id)
        pattern = (None if q is None
                   else "%" + api.db.escape_like_pattern(q) + "%")
        matched = [
            dict(row) for row in rows
            if (status is None or row["status"] == status)
            and (file_type is None or row["file_type"] == file_type)
            and (uploaded_after is None
                 or row["uploaded_at"] > uploaded_after)
            and (uploaded_before is None
                 or row["uploaded_at"] < uploaded_before)
            and ((row.get("archived_at") is not None) == archived)
            and (pattern is None
                 or _like_matches(row["filename"], pattern))
            and (before is None
                 or (row["uploaded_at"], str(row["document_id"])) < before)
        ]
        return matched[offset:offset + limit + 1]

    monkeypatch.setattr(api.db, "list_documents", list_documents)
    return asked


def test_the_document_inventory_pages_with_a_limit_plus_one_probe(
        monkeypatch, tmp_path):
    """`has_more` comes out of the page's OWN scan: the endpoint asks for
    one row past the page and publishes the flag, never the row."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api)

    first = client.get("/documents?limit=2", headers=_headers(api))

    assert first.status_code == 200
    body = first.json()
    assert [doc["document_id"] for doc in body["documents"]] == [
        "kurgu-uc", "kurgu-iki"]
    assert (body["limit"], body["offset"], body["has_more"]) == (2, 0, True)
    # the sentinel row is EVIDENCE, not content -- it is not published
    assert len(body["documents"]) == 2
    # an unfiltered listing sends NO filter to the query seam
    assert asked == [(2, 0, None, None)]

    last = client.get("/documents?limit=2&offset=2", headers=_headers(api))

    assert last.status_code == 200
    tail = last.json()
    assert [doc["document_id"] for doc in tail["documents"]] == ["kurgu-bir"]
    assert (tail["limit"], tail["offset"], tail["has_more"]) == (2, 2, False)


def test_the_document_inventory_walks_deep_pages_with_an_exact_cursor(
        monkeypatch, tmp_path):
    """The published cursor is the last visible total-order key, not an
    opaque offset. Feeding it back yields the next rows without overlap."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    rows = [
        _inventory_row(
            f"00000000-0000-0000-0000-00000000000{number}",
            datetime(2026, 1, number, tzinfo=timezone.utc),
        )
        for number in (4, 3, 2, 1)
    ]
    asked = _wire_inventory(monkeypatch, api, rows)

    first = client.get("/documents?limit=2", headers=_headers(api))

    assert first.status_code == 200
    first_body = first.json()
    assert [item["document_id"] for item in first_body["documents"]] == [
        rows[0]["document_id"], rows[1]["document_id"]]
    assert first_body["next_cursor"] == {
        "before_uploaded_at": "2026-01-03T00:00:00+00:00",
        "before_id": rows[1]["document_id"],
    }

    second = client.get(
        "/documents",
        params={"limit": 2, **first_body["next_cursor"]},
        headers=_headers(api),
    )

    assert second.status_code == 200
    second_body = second.json()
    assert [item["document_id"] for item in second_body["documents"]] == [
        rows[2]["document_id"], rows[3]["document_id"]]
    assert second_body["has_more"] is False
    assert second_body["next_cursor"] is None
    assert asked.cursors == [
        None,
        (datetime(2026, 1, 3, tzinfo=timezone.utc), rows[1]["document_id"]),
    ]


@pytest.mark.parametrize(
    "query",
    [
        "before_uploaded_at=2026-01-03T00:00:00Z",
        "before_id=00000000-0000-0000-0000-000000000003",
        ("before_uploaded_at=2026-01-03T00:00:00Z&"
         "before_id=00000000-0000-0000-0000-000000000003&offset=1"),
    ],
)
def test_an_incomplete_or_mixed_inventory_cursor_costs_no_database_work(
        monkeypatch, tmp_path, query):
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed, queried = _recording_gates(monkeypatch, api)

    response = client.get("/documents?" + query, headers=_headers(api))

    assert response.status_code == 422
    assert borrowed == []
    assert queried == []


def test_the_inventory_publishes_only_the_safe_document_fields(
        monkeypatch, tmp_path):
    """The recorded candidate's bytes and its immutable identity are
    single-document detail. A listing is read by anyone holding the key,
    and it must not hand them out one page at a time."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    _wire_inventory(monkeypatch, api)

    response = client.get("/documents", headers=_headers(api))

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 3
    for document in documents:
        assert set(document) == {
            "document_id", "filename", "file_type", "uploaded_at",
            "status", "status_note", "active_generation", "archived_at"}
    # not merely absent as keys: the VALUES never appear in the response
    assert "content_sha256" not in response.text
    assert "candidate_id" not in response.text
    assert "KURGU_SHA_" not in response.text
    assert "KURGU_ADAY_" not in response.text
    assert documents[1]["status_note"] == "sayfa 2 kayip"


def test_the_inventory_separates_active_and_archived_documents(
        monkeypatch, tmp_path):
    archived_at = f"{996:04d}-01-01T00:00:00+00:00"
    rows = [
        _inventory_row("aktif", f"{997:04d}-01-01T00:00:00+00:00"),
        _inventory_row("arsiv", f"{996:04d}-01-01T00:00:00+00:00",
                       archived_at=archived_at),
    ]
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows)

    active = client.get("/documents", headers=_headers(api))
    archived = client.get("/documents?archived=true", headers=_headers(api))

    assert active.status_code == archived.status_code == 200
    assert [row["document_id"] for row in active.json()["documents"]] == [
        "aktif"]
    assert [row["document_id"] for row in archived.json()["documents"]] == [
        "arsiv"]
    assert archived.json()["documents"][0]["archived_at"] == archived_at
    assert asked.archives == [False, True]


def test_an_invalid_archive_filter_is_refused_before_a_connection_is_borrowed(
        monkeypatch, tmp_path):
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed = []

    @contextmanager
    def recording_connection():
        borrowed.append(True)
        yield object()

    monkeypatch.setattr(api, "db_conn", recording_connection)

    response = client.get("/documents?archived=perhaps", headers=_headers(api))

    assert response.status_code == 422
    assert borrowed == []


def test_the_inventory_defaults_to_the_first_twenty(monkeypatch, tmp_path):
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api)

    body = client.get("/documents", headers=_headers(api)).json()

    assert asked == [(20, 0, None, None)]
    assert (body["limit"], body["offset"], body["has_more"]) == (20, 0, False)


# Variety on BOTH filter columns, so each filter provably discards rows
# the other would keep. Values are invented, like everything else here --
# neither column has a closed vocabulary in the schema.
FILTERED_INVENTORY = [
    _inventory_row("kurgu-dort", f"{999:04d}-04-04T00:00:00+00:00",
                   status="done", file_type="pdf"),
    _inventory_row("kurgu-uc", f"{999:04d}-03-03T00:00:00+00:00",
                   status="partial", file_type="pdf"),
    _inventory_row("kurgu-iki", f"{999:04d}-02-02T00:00:00+00:00",
                   status="done", file_type="docx"),
    _inventory_row("kurgu-bir", f"{999:04d}-01-01T00:00:00+00:00",
                   status="done", file_type="pdf"),
]


@pytest.mark.parametrize(
    ("query", "expected_ids", "expected_ask"),
    [
        ("status=done",
         ["kurgu-dort", "kurgu-iki", "kurgu-bir"],
         (20, 0, "done", None)),
        ("file_type=pdf",
         ["kurgu-dort", "kurgu-uc", "kurgu-bir"],
         (20, 0, None, "pdf")),
        # both together are AND: pdf alone keeps kurgu-uc, done alone
        # keeps kurgu-iki, and each is discarded by the OTHER filter
        ("status=done&file_type=pdf",
         ["kurgu-dort", "kurgu-bir"],
         (20, 0, "done", "pdf")),
        # no vocabulary anywhere: an unknown value is a valid filter that
        # simply matches nothing, never an error
        ("status=kurgu-bilinmeyen",
         [],
         (20, 0, "kurgu-bilinmeyen", None)),
    ],
)
def test_the_inventory_filters_by_exact_equality(
        monkeypatch, tmp_path, query, expected_ids, expected_ask):
    """Each filter alone narrows the listing; together they AND. The
    endpoint forwards the values to the query seam and invents nothing."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=FILTERED_INVENTORY)

    response = client.get(f"/documents?{query}", headers=_headers(api))

    assert response.status_code == 200
    body = response.json()
    assert [doc["document_id"] for doc in body["documents"]] == expected_ids
    assert body["has_more"] is False
    assert asked == [expected_ask]
    for document in body["documents"]:
        assert set(document) == {
            "document_id", "filename", "file_type", "uploaded_at",
            "status", "status_note", "active_generation", "archived_at"}


def test_filters_are_applied_before_pagination(monkeypatch, tmp_path):
    """`offset`, `limit` and `has_more` all walk the FILTERED sequence:
    page one of status=done carries the sentinel's evidence that a done
    row follows, and page two starts where the filtered page ended --
    not two rows into the raw table."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=FILTERED_INVENTORY)

    first = client.get("/documents?status=done&limit=2",
                       headers=_headers(api))

    assert first.status_code == 200
    body = first.json()
    assert [doc["document_id"] for doc in body["documents"]] == [
        "kurgu-dort", "kurgu-iki"]
    assert (body["limit"], body["offset"], body["has_more"]) == (2, 0, True)
    # the limit+1 probe was asked WITH the filter, of the same scan
    assert asked == [(2, 0, "done", None)]

    last = client.get("/documents?status=done&limit=2&offset=2",
                      headers=_headers(api))

    assert last.status_code == 200
    tail = last.json()
    # offset 2 skips two FILTERED rows -- the partial kurgu-uc between
    # them does not consume any of the offset
    assert [doc["document_id"] for doc in tail["documents"]] == ["kurgu-bir"]
    assert (tail["limit"], tail["offset"], tail["has_more"]) == (2, 2, False)


# --- the upload window ---------------------------------------------------
#
# Four instants, one per row, a month apart. They are written as OBJECTS
# because an object is what the endpoint must hand the query seam; the
# query strings in the tests below are merely TEXTS that denote them, and
# several different texts denote the same one.

INSTANT_BIR = datetime(999, 1, 1, tzinfo=timezone.utc)
INSTANT_IKI = datetime(999, 2, 2, tzinfo=timezone.utc)
INSTANT_UC = datetime(999, 3, 3, tzinfo=timezone.utc)
INSTANT_DORT = datetime(999, 4, 4, tzinfo=timezone.utc)

DATED_INVENTORY = [
    _inventory_row("kurgu-dort", INSTANT_DORT),
    _inventory_row("kurgu-uc", INSTANT_UC, status="partial"),
    _inventory_row("kurgu-iki", INSTANT_IKI, file_type="docx"),
    _inventory_row("kurgu-bir", INSTANT_BIR),
]


def _recording_gates(monkeypatch, api):
    """Wire the two probes that measure what a refusal COST.

    A `db_conn` that records every borrow and a query seam that records
    every call, so "refused before the database is touched" is a
    measurement -- zero checkouts, zero statements -- and not an
    inference from the status code.
    """
    borrowed = []
    queried = []

    @contextmanager
    def recording_conn():
        borrowed.append(1)
        yield object()

    monkeypatch.setattr(api, "db_conn", recording_conn)
    monkeypatch.setattr(
        api.db, "list_documents",
        lambda *a, **k: queried.append((a, k)) or [])
    return borrowed, queried


@pytest.mark.parametrize(
    ("params", "expected_ids", "expected_bounds"),
    [
        # after alone: STRICTLY greater, so kurgu-iki -- sitting exactly
        # on the bound -- is outside the window it names
        ({"uploaded_after": "0999-02-02T00:00:00Z"},
         ["kurgu-dort", "kurgu-uc"], (INSTANT_IKI, None)),
        # before alone: strictly less, so kurgu-uc is excluded the same way
        ({"uploaded_before": "0999-03-03T00:00:00Z"},
         ["kurgu-iki", "kurgu-bir"], (None, INSTANT_UC)),
        # both together are AND, and both ends are open: the two rows
        # ON the bounds go, the two strictly inside stay
        ({"uploaded_after": "0999-01-01T00:00:00Z",
          "uploaded_before": "0999-04-04T00:00:00Z"},
         ["kurgu-uc", "kurgu-iki"], (INSTANT_BIR, INSTANT_DORT)),
    ],
)
def test_the_inventory_window_on_uploaded_at_excludes_both_bounds(
        monkeypatch, tmp_path, params, expected_ids, expected_bounds):
    """Each bound may stand alone, together they AND, and neither is
    inclusive -- a row whose `uploaded_at` IS the bound is not in the
    window. That is what makes two adjoining windows a partition instead
    of a pair that both claim the row on the seam."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=DATED_INVENTORY)

    response = client.get("/documents", params=params, headers=_headers(api))

    assert response.status_code == 200
    body = response.json()
    assert [doc["document_id"] for doc in body["documents"]] == expected_ids
    assert body["has_more"] is False
    # the other four arguments are untouched by a date filter
    assert asked == [(20, 0, None, None)]
    # the bounds reached the seam as the instants they denote, and an
    # unsupplied one reached it as None
    assert asked.dates == [expected_bounds]
    for bound in asked.dates[0]:
        assert bound is None or (isinstance(bound, datetime)
                                 and bound.utcoffset() is not None)
    # the window changes WHICH rows are listed, never what a row shows
    assert set(response.json()) == {"documents", "limit", "offset",
                                    "has_more", "next_cursor"}
    for document in body["documents"]:
        assert set(document) == {
            "document_id", "filename", "file_type", "uploaded_at",
            "status", "status_note", "active_generation", "archived_at"}


@pytest.mark.parametrize(
    "text",
    [
        "0999-02-02T00:00:00Z",         # UTC written with a trailing Z
        "0999-02-02T03:00:00+03:00",    # a positive offset
        "0999-02-01T19:00:00-05:00",    # a negative offset, previous day
    ],
)
def test_an_upload_bound_is_the_instant_it_denotes_not_the_text_typed(
        monkeypatch, tmp_path, text):
    """Three different texts, three different wall clocks, ONE instant.

    The window they select is therefore identical, and the value that
    reaches the query seam compares equal in all three cases -- the
    comparison is by absolute instant, and the text is only how the
    caller spelled it."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=DATED_INVENTORY)

    response = client.get("/documents", params={"uploaded_after": text},
                          headers=_headers(api))

    assert response.status_code == 200
    assert [doc["document_id"] for doc in response.json()["documents"]] == [
        "kurgu-dort", "kurgu-uc"]
    bound = asked.dates[0][0]
    # not the text, and not a wall clock: the instant, whatever offset it
    # was written with
    assert isinstance(bound, datetime)
    assert bound == INSTANT_IKI
    assert bound.utcoffset() is not None


@pytest.mark.parametrize(
    ("params", "offender"),
    [
        # naive: no offset and no Z names a wall clock, not an instant
        ({"uploaded_after": "0999-02-02T00:00:00"}, "uploaded_after"),
        ({"uploaded_before": "0999-02-02T00:00:00"}, "uploaded_before"),
        # date-only: the same gap, spelled shorter
        ({"uploaded_after": "0999-02-02"}, "uploaded_after"),
        # a well-formed filter next to it buys the naive one nothing
        ({"status": "done", "uploaded_before": "0999-02-02T00:00:00"},
         "uploaded_before"),
    ],
)
def test_a_naive_upload_bound_is_refused_before_the_database_is_touched(
        monkeypatch, tmp_path, params, offender):
    """`uploaded_at` is `timestamptz`: comparing it against a value with
    no offset means silently choosing a timezone for the caller. The
    aware constraint is declared on the PARAMETER, so FastAPI refuses
    before the body -- and therefore before any checkout or statement --
    runs.

    The refusal is pinned three ways: the parameter-declared SHAPE (a
    list of loc/type objects, not the text `detail` an HTTPException
    carries), the offending parameter, and the `timezone_aware` type
    that names this specific gate rather than some earlier one."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed, queried = _recording_gates(monkeypatch, api)

    refused = client.get("/documents", params=params, headers=_headers(api))

    assert refused.status_code == 422
    # zero cost: no pooled connection was borrowed, no SQL was executed
    assert borrowed == []
    assert queried == []
    detail = refused.json()["detail"]
    assert isinstance(detail, list)
    assert [tuple(err["loc"]) for err in detail] == [("query", offender)]
    assert [err["type"] for err in detail] == ["timezone_aware"]

    # control: the same value WITH an offset passes the identical wiring
    # and the body runs, so the 422 came from the aware constraint on
    # this parameter and not from a gate in front of it
    allowed = client.get("/documents",
                         params={offender: "0999-02-02T00:00:00Z"},
                         headers=_headers(api))

    assert allowed.status_code == 200
    assert borrowed == [1]
    assert len(queried) == 1
    assert queried[0][1][offender] == INSTANT_IKI


@pytest.mark.parametrize("offender", ["uploaded_after", "uploaded_before"])
def test_a_malformed_upload_bound_is_refused_before_the_database_is_touched(
        monkeypatch, tmp_path, offender):
    """Text that denotes no datetime at all fails at the same declared
    gate, with the same parameter-declared shape and the same zero cost.
    The error type is not `timezone_aware` here -- there was nothing to
    check an offset on -- but it still comes from the datetime
    declaration, which is what pins the source of this 422."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed, queried = _recording_gates(monkeypatch, api)

    refused = client.get("/documents",
                         params={offender: "kurgu-tarih-degil"},
                         headers=_headers(api))

    assert refused.status_code == 422
    assert borrowed == []
    assert queried == []
    detail = refused.json()["detail"]
    assert isinstance(detail, list)
    assert [tuple(err["loc"]) for err in detail] == [("query", offender)]
    assert detail[0]["type"].startswith("datetime")

    # control: well-formed text through the identical wiring runs the body
    allowed = client.get("/documents",
                         params={offender: "0999-02-02T00:00:00Z"},
                         headers=_headers(api))

    assert allowed.status_code == 200
    assert borrowed == [1]
    assert len(queried) == 1


@pytest.mark.parametrize(
    "params",
    [
        # equal, written identically
        {"uploaded_after": "0999-02-02T00:00:00Z",
         "uploaded_before": "0999-02-02T00:00:00Z"},
        # equal as INSTANTS while the texts differ: 03:00+03:00 is the
        # same moment as midnight Z, so the refusal cannot be a string
        # comparison dressed up as a range check
        {"uploaded_after": "0999-02-02T00:00:00Z",
         "uploaded_before": "0999-02-02T03:00:00+03:00"},
        # reversed
        {"uploaded_after": "0999-03-03T00:00:00Z",
         "uploaded_before": "0999-02-02T00:00:00Z"},
    ],
)
def test_an_impossible_upload_window_is_refused_before_the_database_is_touched(
        monkeypatch, tmp_path, params):
    """`after < before` is a statement about BOTH values, so no parameter
    declaration can carry it -- a declaration sees only its own value.
    It is checked as the first thing in the body, above `db_conn()`, so
    an empty or reversed window still costs zero checkouts and zero
    statements.

    THE SHAPE HERE IS THE OTHER ONE. This gate raises `HTTPException`,
    whose 422 carries a TEXT detail, where the declared gate above
    carries a list of loc/type objects. Each test pins the shape its own
    gate produces; neither shape covers both."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed, queried = _recording_gates(monkeypatch, api)

    refused = client.get("/documents", params=params, headers=_headers(api))

    assert refused.status_code == 422
    assert borrowed == []
    assert queried == []
    detail = refused.json()["detail"]
    assert isinstance(detail, str)
    assert "uploaded_after" in detail and "uploaded_before" in detail

    # the gate reached is provably THIS one: each value on its own clears
    # the declared constraint and runs the body, so what was refused is
    # the pair, not either bound's own shape
    for name, value in params.items():
        alone = client.get("/documents", params={name: value},
                           headers=_headers(api))
        assert alone.status_code == 200

    assert borrowed == [1, 1]
    assert len(queried) == 2


def test_the_upload_window_ands_with_the_status_and_file_type_filters(
        monkeypatch, tmp_path):
    """All four filters at once. Each provably discards something the
    others would keep: the window drops kurgu-bir, `status=done` drops
    the partial kurgu-uc, `file_type=pdf` drops the docx kurgu-iki."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=DATED_INVENTORY)

    windowed = client.get(
        "/documents",
        params={"uploaded_after": "0999-01-01T00:00:00Z"},
        headers=_headers(api))

    # the window alone keeps three of the four rows
    assert [doc["document_id"] for doc in windowed.json()["documents"]] == [
        "kurgu-dort", "kurgu-uc", "kurgu-iki"]

    response = client.get(
        "/documents",
        params={"uploaded_after": "0999-01-01T00:00:00Z",
                "uploaded_before": "0999-05-05T00:00:00Z",
                "status": "done", "file_type": "pdf"},
        headers=_headers(api))

    assert response.status_code == 200
    body = response.json()
    assert [doc["document_id"] for doc in body["documents"]] == ["kurgu-dort"]
    assert asked[1] == (20, 0, "done", "pdf")
    assert asked.dates[1] == (INSTANT_BIR, datetime(999, 5, 5,
                                                    tzinfo=timezone.utc))


def test_the_window_is_applied_before_pagination(monkeypatch, tmp_path):
    """`limit + 1`, `offset` and `has_more` all walk the WINDOWED
    sequence: page one carries the sentinel's evidence that a third
    windowed row follows, and page two starts where the windowed page
    ended -- not two rows into the raw table."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=DATED_INVENTORY)
    window = {"uploaded_after": "0999-01-01T00:00:00Z"}

    first = client.get("/documents", params={**window, "limit": 2},
                       headers=_headers(api))

    assert first.status_code == 200
    body = first.json()
    assert [doc["document_id"] for doc in body["documents"]] == [
        "kurgu-dort", "kurgu-uc"]
    assert (body["limit"], body["offset"], body["has_more"]) == (2, 0, True)
    # the limit+1 probe was asked WITH the window, of the same scan
    assert asked == [(2, 0, None, None)]
    assert asked.dates == [(INSTANT_BIR, None)]

    last = client.get("/documents",
                      params={**window, "limit": 2, "offset": 2},
                      headers=_headers(api))

    assert last.status_code == 200
    tail = last.json()
    # offset 2 skips two WINDOWED rows; kurgu-bir, excluded by the bound,
    # consumes none of the offset and never appears
    assert [doc["document_id"] for doc in tail["documents"]] == ["kurgu-iki"]
    assert (tail["limit"], tail["offset"], tail["has_more"]) == (2, 2, False)


def test_an_inventory_without_a_window_sends_no_date_bound(
        monkeypatch, tmp_path):
    """The unfiltered listing is unchanged: neither bound is invented,
    both reach the seam as None, and the page is the whole inventory."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=DATED_INVENTORY)

    body = client.get("/documents", headers=_headers(api)).json()

    assert [doc["document_id"] for doc in body["documents"]] == [
        "kurgu-dort", "kurgu-uc", "kurgu-iki", "kurgu-bir"]
    assert asked == [(20, 0, None, None)]
    assert asked.dates == [(None, None)]


# --- the filename search --------------------------------------------------
#
# Six rows whose NAMES are the subject. Three carry the word being
# searched for in a different position and a different case; the other
# three each carry exactly one of LIKE's three interesting characters, so
# a search for `%`, `_` or `!` has both a row it must find and rows it
# must not.

INSTANT_BES = datetime(999, 5, 5, tzinfo=timezone.utc)
INSTANT_ALTI = datetime(999, 6, 6, tzinfo=timezone.utc)

SEARCHED_INVENTORY = [
    _inventory_row("kurgu-alti", INSTANT_ALTI,
                   filename="Zeta-Rapor-2029.pdf"),
    _inventory_row("kurgu-bes", INSTANT_BES,
                   filename="yillik-zeta-ozeti.docx", file_type="docx"),
    _inventory_row("kurgu-dort", INSTANT_DORT,
                   filename="butce-notlari-zeta.pdf", status="partial"),
    _inventory_row("kurgu-uc", INSTANT_UC,
                   filename="kurgu-%100-tablo.pdf"),
    _inventory_row("kurgu-iki", INSTANT_IKI,
                   filename="kurgu_100_tablo.pdf"),
    _inventory_row("kurgu-bir", INSTANT_BIR,
                   filename="kurgu-!100-tablo.pdf"),
]


@pytest.mark.parametrize(
    ("q", "expected_ids"),
    [
        # a plain literal substring, and the SAME query in three cases:
        # the search is case-insensitive in both directions -- lower-case
        # text finds the upper-case name, upper-case text finds the
        # lower-case ones
        ("zeta", ["kurgu-alti", "kurgu-bes", "kurgu-dort"]),
        ("ZETA", ["kurgu-alti", "kurgu-bes", "kurgu-dort"]),
        ("Zeta", ["kurgu-alti", "kurgu-bes", "kurgu-dort"]),
        # the START of a filename
        ("zeta-rapor", ["kurgu-alti"]),
        # the MIDDLE
        ("-notlari-", ["kurgu-dort"]),
        # the END, including the extension
        ("tablo.pdf", ["kurgu-uc", "kurgu-iki", "kurgu-bir"]),
        # a substring of no filename at all: an empty list, never an error
        ("kurgu-hicbir-yerde-yok", []),
    ],
)
def test_the_inventory_searches_filenames_by_literal_substring(
        monkeypatch, tmp_path, q, expected_ids):
    """One column, case-insensitively, anywhere in the name.

    The match is decided by the production escaping run through a LIKE
    interpreter (see `_like_matches`), so these are not substring
    comparisons dressed up as a search."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    response = client.get("/documents", params={"q": q},
                          headers=_headers(api))

    assert response.status_code == 200
    body = response.json()
    assert [doc["document_id"] for doc in body["documents"]] == expected_ids
    assert body["has_more"] is False
    # the search reached the seam as the RAW value; the endpoint invents
    # nothing and transforms nothing
    assert asked.searches == [q]
    # the other four arguments are untouched by a search
    assert asked == [(20, 0, None, None)]
    assert asked.dates == [(None, None)]
    # the search changes WHICH rows are listed, never what a row shows
    assert set(body) == {
        "documents", "limit", "offset", "has_more", "next_cursor"}
    for document in body["documents"]:
        assert set(document) == {
            "document_id", "filename", "file_type", "uploaded_at",
            "status", "status_note", "active_generation", "archived_at"}


@pytest.mark.parametrize(
    ("q", "expected_ids"),
    [
        # `%` is LIKE's "any run of characters". Searched for literally it
        # must find the ONE name that carries one -- not all six
        ("%", ["kurgu-uc"]),
        # `_` is LIKE's "any single character". Literally, it must not
        # match an arbitrary one
        ("_", ["kurgu-iki"]),
        # the escape character itself, which must still find itself
        ("!", ["kurgu-bir"]),
        # and in longer runs, where a wildcard reading would widen the
        # result instead of narrowing it
        ("%100", ["kurgu-uc"]),
        ("u_1", ["kurgu-iki"]),
        ("-!100-", ["kurgu-bir"]),
    ],
)
def test_a_metacharacter_search_matches_only_that_literal_character(
        monkeypatch, tmp_path, q, expected_ids):
    """The wildcard cases, proven THROUGH the in-memory LIKE model.

    MEASURED: no live PostgreSQL is reachable and CI declares no database
    service, so this fixture is the only place the operator's semantics
    exist in this loop. Each case names the rows it must find AND is
    checked against six rows it could have matched, so "only the literal
    character" is a measurement rather than a claim."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    response = client.get("/documents", params={"q": q},
                          headers=_headers(api))

    assert response.status_code == 200
    listed = [doc["document_id"] for doc in response.json()["documents"]]
    assert listed == expected_ids
    assert len(listed) < len(SEARCHED_INVENTORY)     # not every document


@pytest.mark.parametrize(
    ("q", "expected_ids"),
    [
        # all three metacharacters in one value, matched character for
        # character: the name that carries the run is found, and the one
        # carrying the same characters in another order is not
        ("!100-tablo", ["kurgu-bir"]),
        ("%100-tablo", ["kurgu-uc"]),
        ("u_100_t", ["kurgu-iki"]),
    ],
)
def test_a_combined_metacharacter_search_is_matched_literally(
        monkeypatch, tmp_path, q, expected_ids):
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    response = client.get("/documents", params={"q": q},
                          headers=_headers(api))

    assert response.status_code == 200
    assert [doc["document_id"]
            for doc in response.json()["documents"]] == expected_ids


def test_an_empty_search_is_refused_before_the_database_is_touched(
        monkeypatch, tmp_path):
    """A search that is present but empty cannot mean anything: as a
    pattern it would be `%%`, which is every row wearing the costume of a
    filter. Its shape lives in the signature, so FastAPI refuses it with
    422 before the body -- and therefore before any connection checkout
    or statement -- runs.

    The gate reached is pinned three ways: the parameter-declared SHAPE
    (a list of loc/type objects, not the text `detail` an HTTPException
    carries), the offending parameter, and a control that clears exactly
    this constraint through the identical wiring and runs the body."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed, queried = _recording_gates(monkeypatch, api)

    refused = client.get("/documents", params={"q": ""},
                         headers=_headers(api))

    assert refused.status_code == 422
    # zero cost: no pooled connection was borrowed, no SQL was executed
    assert borrowed == []
    assert queried == []
    detail = refused.json()["detail"]
    assert isinstance(detail, list)
    assert [tuple(err["loc"]) for err in detail] == [("query", "q")]
    assert detail[0]["type"] == "string_too_short"

    # control: one character clears the bound and the body runs, so the
    # 422 came from THIS parameter's declared shape and not from a gate
    # in front of it
    allowed = client.get("/documents", params={"q": "z"},
                         headers=_headers(api))

    assert allowed.status_code == 200
    assert borrowed == [1]
    assert len(queried) == 1
    assert queried[0][1]["q"] == "z"


@pytest.mark.parametrize(
    ("params", "expected_ids"),
    [
        # each existing filter, ANDed with the search: the search alone
        # keeps three rows, and each filter provably discards some of them
        ({"q": "zeta", "status": "done"}, ["kurgu-alti", "kurgu-bes"]),
        ({"q": "zeta", "file_type": "pdf"}, ["kurgu-alti", "kurgu-dort"]),
        ({"q": "zeta", "uploaded_after": "0999-04-04T00:00:00Z"},
         ["kurgu-alti", "kurgu-bes"]),
        ({"q": "zeta", "uploaded_before": "0999-06-06T00:00:00Z"},
         ["kurgu-bes", "kurgu-dort"]),
        # and all five at once
        ({"q": "zeta", "status": "done", "file_type": "pdf",
          "uploaded_after": "0999-01-01T00:00:00Z",
          "uploaded_before": "0999-12-12T00:00:00Z"},
         ["kurgu-alti"]),
    ],
)
def test_the_search_ands_with_every_existing_filter(
        monkeypatch, tmp_path, params, expected_ids):
    """`q` narrows what the others left, and they narrow what it left --
    the combination is AND, in every pairing and all together.

    The date cases also pin that the EXCLUSIVE bounds are unchanged by a
    search next to them: the row sitting exactly on each bound is
    outside the window, which is why `kurgu-dort` and `kurgu-alti` drop
    out of their respective cases."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    response = client.get("/documents", params=params, headers=_headers(api))

    assert response.status_code == 200
    body = response.json()
    assert [doc["document_id"] for doc in body["documents"]] == expected_ids
    # every supplied filter reached the seam, each in its own argument
    assert asked.searches == ["zeta"]
    assert asked[0][2] == params.get("status")
    assert asked[0][3] == params.get("file_type")
    for bound, sent in zip(("uploaded_after", "uploaded_before"),
                           asked.dates[0]):
        if bound in params:
            assert isinstance(sent, datetime) and sent.utcoffset() is not None
        else:
            assert sent is None


def test_the_search_is_applied_before_pagination(monkeypatch, tmp_path):
    """`limit + 1`, `offset` and `has_more` all walk the SEARCHED
    sequence: page one carries the sentinel's evidence that a third
    matching row follows, and page two starts where the searched page
    ended -- not two rows into the raw table."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    first = client.get("/documents", params={"q": "zeta", "limit": 2},
                       headers=_headers(api))

    assert first.status_code == 200
    body = first.json()
    assert [doc["document_id"] for doc in body["documents"]] == [
        "kurgu-alti", "kurgu-bes"]
    assert (body["limit"], body["offset"], body["has_more"]) == (2, 0, True)
    # the limit+1 probe was asked WITH the search, of the same scan
    assert asked == [(2, 0, None, None)]
    assert asked.searches == ["zeta"]

    last = client.get("/documents",
                      params={"q": "zeta", "limit": 2, "offset": 2},
                      headers=_headers(api))

    assert last.status_code == 200
    tail = last.json()
    # offset 2 skips two MATCHING rows; the three names that do not carry
    # "zeta" consume none of the offset and never appear
    assert [doc["document_id"] for doc in tail["documents"]] == ["kurgu-dort"]
    assert (tail["limit"], tail["offset"], tail["has_more"]) == (2, 2, False)


def test_an_inventory_without_a_search_sends_no_q(monkeypatch, tmp_path):
    """The unfiltered listing is unchanged: no search value is invented,
    it reaches the seam as None, and the page is the whole inventory in
    the order it always had."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    body = client.get("/documents", headers=_headers(api)).json()

    assert [doc["document_id"] for doc in body["documents"]] == [
        "kurgu-alti", "kurgu-bes", "kurgu-dort", "kurgu-uc", "kurgu-iki",
        "kurgu-bir"]
    assert asked == [(20, 0, None, None)]
    assert asked.dates == [(None, None)]
    assert asked.searches == [None]


@pytest.mark.parametrize(
    "q",
    [
        "alt/klasor",                   # a slash
        "kurgu:ek",                     # a colon
        "kurgu ",                       # a trailing space
        "..",
        "kurgu' OR 1=1 --",             # quote syntax
        "kurgu\\ters",                  # a backslash
        "belge-özet-Şubat",             # non-ASCII
        "kurgu-" + "u" * 250,           # long
    ],
)
def test_a_search_is_not_validated_as_an_upload_filename(
        monkeypatch, tmp_path, q):
    """MEASURED: `_safe_upload_filename` is an UPLOAD validator and is NOT
    reused here. It rejects slashes, colons, control characters and
    trailing spaces -- every one of them a legitimate thing to SEARCH
    for -- so reusing it would silently narrow the search instead of
    protecting anything. There is no length authority for `filename`
    anywhere either (the column is unbounded `text`,
    `_safe_upload_filename` has no length check, and UPLOAD_MAX_BYTES is
    a body cap), so no maximum is invented for `q`.

    Each of these is therefore an ordinary search: accepted, forwarded
    character for character, and answered with a page."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    asked = _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    response = client.get("/documents", params={"q": q},
                          headers=_headers(api))

    assert response.status_code == 200
    assert asked.searches == [q]
    assert len(asked.searches[0]) == len(q)     # intact, never truncated
    assert response.json()["documents"] == []
    assert set(response.json()) == {"documents", "limit", "offset",
                                    "has_more", "next_cursor"}


def test_a_searched_listing_publishes_only_the_safe_document_fields(
        monkeypatch, tmp_path):
    """The projection does not widen because a search narrowed: the
    recorded candidate's bytes and its immutable identity stay out of the
    response, exactly as in an unsearched listing."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    _wire_inventory(monkeypatch, api, rows=SEARCHED_INVENTORY)

    response = client.get("/documents", params={"q": "zeta"},
                          headers=_headers(api))

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 3
    for document in documents:
        assert set(document) == {
            "document_id", "filename", "file_type", "uploaded_at",
            "status", "status_note", "active_generation", "archived_at"}
    # not merely absent as keys: the VALUES never appear in the response
    assert "content_sha256" not in response.text
    assert "candidate_id" not in response.text
    assert "KURGU_SHA_" not in response.text
    assert "KURGU_ADAY_" not in response.text


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=101", "limit=-1", "offset=-1", "limit=abc",
     "limit=20&offset=-5"],
)
def test_an_unsized_page_is_refused_before_the_database_is_touched(
        monkeypatch, tmp_path, query):
    """A page the endpoint cannot size must cost NOTHING: no pooled
    connection, no scan. The bounds live in the signature, so FastAPI
    refuses the request before the body -- and therefore before the
    connection checkout -- ever runs."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed = []
    queried = []

    @contextmanager
    def recording_conn():
        borrowed.append(1)
        yield object()

    monkeypatch.setattr(api, "db_conn", recording_conn)
    monkeypatch.setattr(
        api.db, "list_documents",
        lambda *a, **k: queried.append(1) or [])

    response = client.get(f"/documents?{query}", headers=_headers(api))

    assert response.status_code == 422
    assert borrowed == []
    assert queried == []


@pytest.mark.parametrize(
    ("query", "offender"),
    [
        ("status=", "status"),                        # empty: below min_length
        ("file_type=", "file_type"),
        # two parameters supplied, one of them empty: the refusal still
        # names only the offender, and the well-formed one buys nothing
        ("status=done&file_type=", "file_type"),
    ],
)
def test_a_malformed_filter_is_refused_before_the_database_is_touched(
        monkeypatch, tmp_path, query, offender):
    """EMPTY is the only malformed shape there is. A filter that is
    present but empty cannot mean anything, so its shape lives in the
    signature and FastAPI refuses it with 422 before the body -- and
    therefore before any connection checkout or statement -- runs. The
    refusal is pinned to the offending parameter, and a well-formed
    filter through the identical wiring reaches the query seam, so the
    422 provably comes from this parameter's declared bound and not
    from some earlier gate.

    LENGTH IS NOT A SHAPE HERE: both columns are unbounded `text`, so
    there is no long-value case in this list -- see
    test_a_long_filter_value_is_a_valid_filter."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    borrowed = []
    queried = []

    @contextmanager
    def recording_conn():
        borrowed.append(1)
        yield object()

    monkeypatch.setattr(api, "db_conn", recording_conn)
    monkeypatch.setattr(
        api.db, "list_documents",
        lambda *a, **k: queried.append((a, k)) or [])

    refused = client.get(f"/documents?{query}", headers=_headers(api))

    assert refused.status_code == 422
    # zero cost: no pooled connection was borrowed, no SQL was executed
    assert borrowed == []
    assert queried == []
    # the refusal names the malformed QUERY parameter, nothing else
    assert [tuple(err["loc"]) for err in refused.json()["detail"]] == [
        ("query", offender)]

    # control: a well-formed value passes the same gate and the body runs
    allowed = client.get(
        f"/documents?{offender}=kurgu-deger", headers=_headers(api))

    assert allowed.status_code == 200
    assert borrowed == [1]
    assert len(queried) == 1
    assert queried[0][1][offender] == "kurgu-deger"


@pytest.mark.parametrize("parameter", ["status", "file_type"])
def test_a_long_filter_value_is_a_valid_filter(monkeypatch, tmp_path,
                                               parameter):
    """A REGRESSION, not a bound: `documents.status` and `file_type` are
    unbounded `text`, so a 128-character value is an ordinary value the
    database can hold. It must reach the query seam intact -- character
    for character, neither truncated nor rejected -- and travel from
    there as a parameter, exactly like a short one.

    The API layer once declared a 64-character cap of its own, which made
    every longer-but-valid row unfilterable through this endpoint while
    the db seam happily accepted it. This test is what would fail if that
    cap came back."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    queried = []

    monkeypatch.setattr(
        api.db, "list_documents",
        lambda *a, **k: queried.append((a, k)) or [])

    long_value = "kurgu-" + "u" * 122
    assert len(long_value) == 128

    response = client.get(f"/documents?{parameter}={long_value}",
                          headers=_headers(api))

    assert response.status_code == 200
    # it arrived intact: the same characters, the same length, no
    # truncation at 64 or anywhere else
    assert queried[0][1][parameter] == long_value
    assert len(queried[0][1][parameter]) == 128
    # and the other filter is still absent, as an unsupplied filter is
    other = "file_type" if parameter == "status" else "status"
    assert queried[0][1][other] is None
    # the response shape is untouched by the length of a filter
    assert set(response.json()) == {"documents", "limit", "offset",
                                    "has_more", "next_cursor"}


def test_the_inventory_sits_behind_the_same_key_as_the_other_document_routes(
        monkeypatch, tmp_path):
    """Not "it has some dependency" -- the SAME one, by identity. A
    listing walks the whole corpus, so an inventory left one dependency
    short is a bigger opening than any single-document route."""
    api, client, _state, _calls, _upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    _wire_inventory(monkeypatch, api)

    def dependencies(path):
        for route in api.app.routes:
            if (getattr(route, "path", None) == path
                    and "GET" in getattr(route, "methods", ())):
                return [depends.dependency for depends in route.dependencies]
        raise AssertionError(f"{path} yok")

    assert dependencies("/documents") == [api.require_api_key]
    assert dependencies("/documents") == dependencies("/documents/{document_id}")


def test_the_inventory_is_refused_without_the_configured_key(monkeypatch):
    """The dependency identity above is structure; this is the behaviour."""
    import importlib

    import pipeline.api.app as api

    # the key is built, not written out: a literal `Bearer <long token>`
    # in a source file is a key-shaped object to the leak scanner even
    # when the token is invented
    key = "zeta-gamma-envanter-anahtari"
    monkeypatch.setenv("API_KEY", key)
    importlib.reload(api)
    try:
        monkeypatch.setattr(api, "db_conn", _fake_db_conn)
        monkeypatch.setattr(api.db, "list_documents", lambda *a, **k: [])
        client = TestClient(api.app)

        assert client.get("/documents").status_code == 401
        assert client.get(
            "/documents",
            headers={"Authorization": "Bearer yanlis"},
        ).status_code == 401
        allowed = client.get(
            "/documents",
            headers={"Authorization": f"Bearer {key}"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["documents"] == []
    finally:
        monkeypatch.delenv("API_KEY", raising=False)
        importlib.reload(api)


def test_unhandled_dependency_error_never_copies_its_message_to_logs(
        monkeypatch, caplog):
    from pipeline.api import app as api

    private = "OZEL_KURGU_BAGLANTI_AYRINTISI"
    monkeypatch.setattr(api, "db_conn", _fake_db_conn)

    def fail(_conn, _document_id):
        raise RuntimeError(private)

    monkeypatch.setattr(api.db, "get_document", fail)
    response = TestClient(
        api.app,
        raise_server_exceptions=False,
    ).get(
        "/documents/kurgu-belge-kimligi",
        headers=_headers(api),
    )

    assert response.status_code == 500
    assert private not in response.text
    assert private not in caplog.text
    assert "RuntimeError" in caplog.text
    assert '"iz":' in caplog.text
    assert '"fonksiyon": "fail"' in caplog.text


# --- answering from a named subset of documents --------------------------
#
# `document_ids` narrows a RAG question. Absent, everything below behaves
# exactly as the tests above it do. Supplied, it must scope the whole path:
# both retrieval statements, the reranker's candidates, the assembled
# context and every published citation. The tests come in pairs on purpose
# -- each scoped assertion is paired with the UNSCOPED run that shows the
# excluded document really would have matched, so a passing scope is
# evidence about the filter rather than about a corpus with one document
# in it.

IC_BELGE = "11111111-1111-1111-1111-111111111111"
DIS_BELGE = "22222222-2222-2222-2222-222222222222"
YOK_BELGE = "33333333-3333-3333-3333-333333333333"

KAPSAM_CORPUS = [
    {
        "id": "ic-1",
        "document_id": IC_BELGE,
        "filename": "kapsam-icinde.pdf",
        "page": 7,
        "type": "text",
        "text": "Zeta uretimi 47 000 birimdir.",
        "source_tag": "kurgu",
        "headings": [],
        "table_data": None,
    },
    {
        "id": "dis-1",
        "document_id": DIS_BELGE,
        "filename": "kapsam-disinda.pdf",
        "page": 9,
        "type": "text",
        "text": "Omega uretimi 88 000 birimdir.",
        "source_tag": "kurgu",
        "headings": [],
        "table_data": None,
    },
]

ICERIDEN = json.dumps({
    "dayanak": [{"pasaj": 1, "alinti": "Zeta uretimi 47 000 birimdir."}],
    "cevap": "Sayfa 7'ye gore 47 000 birim.",
})
CEKIMSER = json.dumps({
    "dayanak": [],
    "cevap": "Bu bilgi mevcut belgelerde bulunamadi.",
})


def _wire_scoped_corpus(monkeypatch):
    """The NATIVE chain with only its database and network seams replaced.

    The request model, the endpoint, `rag_backends`, `query.ask_checked`,
    `query.retrieve`, the real `db.hybrid_search` and the real guard all
    run. The fake cursor applies the scope clause the way the server would,
    so what comes back is decided by the statement that was actually sent.
    """
    from pipeline.generation import answer as generation
    from pipeline.index import db
    from pipeline.retrieval import query

    seen = {"statements": [], "checkouts": 0, "dense": 0, "sparse": 0,
            "ranked": [], "prompts": [], "tenant": [], "rollbacks": 0,
            "scope_resolutions": []}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, sql, params=None):
            seen["statements"].append((sql, params))
            rows = list(KAPSAM_CORPUS)
            if db.DOCUMENT_SCOPE_CLAUSE in sql:
                scope = set(params[0])
                rows = [row for row in rows if row["document_id"] in scope]
            self._rows = rows

        def fetchall(self):
            return [dict(row) for row in self._rows]

    class Conn:
        def cursor(self, row_factory=None):
            return Cursor()

        def rollback(self):
            seen["rollbacks"] += 1

    class Pool:
        @contextmanager
        def connection(self):
            seen["checkouts"] += 1
            yield Conn()

    def embed_dense(_question):
        seen["dense"] += 1
        return [0.0]

    def embed_sparse(_question):
        seen["sparse"] += 1
        return ([1], [1.0])

    def rerank(_question, chunks, *_args, **_kwargs):
        seen["ranked"].append(list(chunks))
        return chunks

    def complete(policy, user_content):
        seen["prompts"].append((policy, user_content))
        return (ICERIDEN if "Zeta uretimi 47 000 birimdir." in user_content
                else CEKIMSER)

    monkeypatch.setattr(db, "get_pool", lambda: Pool())
    monkeypatch.setattr(
        db, "current_execution_tenant",
        lambda: (db.DEFAULT_TENANT_ID, False))
    monkeypatch.setattr(db, "current_execution_actor", lambda: None)
    monkeypatch.setattr(db, "begin_retrieval_snapshot", lambda _conn: None)
    monkeypatch.setattr(db, "retrieval_policy_epoch", lambda _conn: 1)
    def resolve_document_scope(
            _conn, *, document_ids=None, collection_ids=None, tags=None):
        resolved = tuple(sorted(
            set(document_ids or ())
            & {row["document_id"] for row in KAPSAM_CORPUS}))
        seen["scope_resolutions"].append({
            "document_ids": document_ids,
            "collection_ids": collection_ids,
            "tags": tags,
            "resolved": resolved,
        })
        return resolved

    monkeypatch.setattr(db, "resolve_document_scope", resolve_document_scope)
    monkeypatch.setattr(
        db, "set_tenant_context",
        lambda _conn, tenant_id, *, service=False:
        seen["tenant"].append(("set", tenant_id, service)))
    monkeypatch.setattr(
        db, "clear_tenant_context",
        lambda _conn: seen["tenant"].append(("clear",)))
    monkeypatch.setattr(query, "embed_dense", embed_dense)
    monkeypatch.setattr(query, "embed_sparse", embed_sparse)
    monkeypatch.setattr(query, "rerank", rerank)
    monkeypatch.setattr(generation, "complete", complete)
    return seen


def _citations(response, stream):
    if not stream:
        return response.json()["rag_citations"]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    published = {json.dumps(payload["rag_citations"], sort_keys=True)
                 for payload in payloads}
    assert len(published) == 1
    return json.loads(published.pop())


def _filenames(chunks):
    return [chunk["filename"] for chunk in chunks]


@pytest.mark.parametrize("stream", [False, True])
def test_a_scoped_request_answers_from_that_document_alone(
        monkeypatch, stream):
    """One identifier, and the scope holds all the way down."""
    from pipeline.api import app as api
    from pipeline.index import db

    seen = _wire_scoped_corpus(monkeypatch)

    response = _chat(api, "ragtest-rag", stream, document_ids=[IC_BELGE])

    assert response.status_code == 200
    status, text = _public_reply(response, stream)
    assert status == ANSWERED
    assert text == "Sayfa 7'ye gore 47 000 birim."
    # BOTH retrieval statements were scoped, by the same clause, with the
    # identifier travelling as a parameter and never as statement text
    assert len(seen["statements"]) == 3
    lock_sql, lock_params = seen["statements"][0]
    assert "FOR SHARE" in lock_sql
    assert lock_params[0] == [IC_BELGE]
    for sql, params in seen["statements"][1:]:
        assert db.DOCUMENT_SCOPE_CLAUSE in sql
        assert params[0] == [IC_BELGE]
        assert IC_BELGE not in sql
    # the reranker never saw a candidate from outside the scope
    assert _filenames(seen["ranked"][0]) == ["kapsam-icinde.pdf"]
    # neither did the assembled context, so neither did any citation line
    (_policy, user_content), = seen["prompts"]
    assert "[kapsam-icinde.pdf | Sayfa 7]" in user_content
    assert "kapsam-disinda.pdf" not in user_content
    assert "Omega uretimi" not in user_content
    # and the published citations name a page of the scoped document only
    assert _citations(response, stream) == [{"page": 7, "source": "model"}]
    assert "kapsam-disinda" not in response.text
    assert "88 000" not in response.text


def test_the_excluded_document_really_would_have_matched(monkeypatch):
    """THE NEGATIVE HALF. Same corpus, same question, no scope: the other
    document is retrieved, reranked and put in front of the model. Without
    this, the test above would pass over a corpus that simply had nothing
    else in it."""
    from pipeline.api import app as api
    from pipeline.index import db

    seen = _wire_scoped_corpus(monkeypatch)

    response = _chat(api, "ragtest-rag")

    assert response.status_code == 200
    assert _filenames(seen["ranked"][0]) == [
        "kapsam-icinde.pdf", "kapsam-disinda.pdf"]
    (_policy, user_content), = seen["prompts"]
    assert "kapsam-disinda.pdf" in user_content
    # ... and an unscoped request added NO clause and NO parameter
    lock_sql, lock_params = seen["statements"][0]
    assert "FOR SHARE" in lock_sql
    assert lock_params == ()
    for sql, params in seen["statements"][1:]:
        assert db.DOCUMENT_SCOPE_CLAUSE not in sql
        assert len(params) == 2


def test_a_scope_of_several_documents_returns_only_that_set(monkeypatch):
    from pipeline.api import app as api

    seen = _wire_scoped_corpus(monkeypatch)

    response = _chat(api, "ragtest-rag",
                     document_ids=[IC_BELGE, DIS_BELGE])

    assert response.status_code == 200
    assert _filenames(seen["ranked"][0]) == [
        "kapsam-icinde.pdf", "kapsam-disinda.pdf"]
    for _sql, params in seen["statements"]:
        assert sorted(params[0]) == sorted([IC_BELGE, DIS_BELGE])


def test_an_unknown_identifier_scopes_to_nothing_not_to_everything(
        monkeypatch):
    """An identifier that names no document must NARROW the search to
    nothing. Falling back to the corpus would answer a question the caller
    never asked, and would look like a good answer while doing it."""
    from pipeline.api import app as api

    seen = _wire_scoped_corpus(monkeypatch)

    response = _chat(api, "ragtest-rag", document_ids=[YOK_BELGE])

    assert response.status_code == 200
    status, _text = _public_reply(response, False)
    assert status == ABSTAINED
    assert seen["ranked"] == [[]]
    assert seen["scope_resolutions"] == [{
        "document_ids": (YOK_BELGE,),
        "collection_ids": None,
        "tags": None,
        "resolved": (),
    }]
    # The authority resolved the requested identifier to the empty set.  That
    # result short-circuits before embedding or retrieval SQL; it must never
    # be reinterpreted as an unscoped corpus search.
    assert seen["dense"] == 0
    assert seen["sparse"] == 0
    assert seen["statements"] == []
    assert _citations(response, False) == []


def test_a_mixed_scope_is_scoped_to_the_known_identifier_alone(monkeypatch):
    from pipeline.api import app as api

    seen = _wire_scoped_corpus(monkeypatch)

    response = _chat(api, "ragtest-rag",
                     document_ids=[IC_BELGE, YOK_BELGE])

    assert response.status_code == 200
    assert _filenames(seen["ranked"][0]) == ["kapsam-icinde.pdf"]
    assert seen["scope_resolutions"] == [{
        "document_ids": (IC_BELGE, YOK_BELGE),
        "collection_ids": None,
        "tags": None,
        "resolved": (IC_BELGE,),
    }]
    for _sql, params in seen["statements"]:
        assert params[0] == [IC_BELGE]


def test_a_repeated_identifier_neither_widens_nor_repeats_the_filter(
        monkeypatch):
    """Set semantics, applied ONCE before the first backend call. The
    repetition must not reach the database as a repeated filter value, and
    the answer must be the one the single identifier produces."""
    from pipeline.api import app as api

    seen = _wire_scoped_corpus(monkeypatch)
    repeated = _chat(api, "ragtest-rag",
                     document_ids=[IC_BELGE, IC_BELGE, IC_BELGE])

    for _sql, params in seen["statements"]:
        assert params[0] == [IC_BELGE]
    assert _filenames(seen["ranked"][0]) == ["kapsam-icinde.pdf"]

    once = _wire_scoped_corpus(monkeypatch)
    single = _chat(api, "ragtest-rag", document_ids=[IC_BELGE])

    assert repeated.json()["rag_status"] == single.json()["rag_status"]
    assert (repeated.json()["choices"][0]["message"]["content"]
            == single.json()["choices"][0]["message"]["content"])
    assert repeated.json()["rag_citations"] == single.json()["rag_citations"]
    assert ([params for _sql, params in seen["statements"]]
            == [params for _sql, params in once["statements"]])


def test_the_collapse_happens_before_the_first_backend_call(monkeypatch):
    """Pinned at the seam itself: the backend is handed ONE canonical
    scope, not a list with a repetition for something further down to
    deduplicate (or fail to)."""
    from pipeline.api import app as api
    from pipeline.validation.rag.answer_guard import ANSWERED, GuardResult

    handed = []

    def checked(question, backend=None, *, document_ids=None):
        handed.append(document_ids)
        return GuardResult(ANSWERED, "kurgu cevap", ())

    monkeypatch.setattr(api.rag_backends, "answer_checked", checked)

    _chat(api, "ragtest-rag",
          document_ids=[DIS_BELGE, IC_BELGE, IC_BELGE])

    assert handed == [(IC_BELGE, DIS_BELGE)]
    assert len(handed[0]) == len(set(handed[0]))


def test_an_unscoped_request_hands_the_backend_no_scope_at_all(monkeypatch):
    """Absent stays absent: the backend is called exactly as it always
    was, so every caller that passes no scope keeps working untouched."""
    from pipeline.api import app as api
    from pipeline.validation.rag.answer_guard import ANSWERED, GuardResult

    handed = []

    def checked(question, backend=None, **kwargs):
        handed.append(kwargs)
        return GuardResult(ANSWERED, "kurgu cevap", ())

    monkeypatch.setattr(api.rag_backends, "answer_checked", checked)

    _chat(api, "ragtest-rag")

    assert handed == [{}]


def test_streaming_and_non_streaming_agree_under_the_same_scope(monkeypatch):
    """One closure serves both response shapes, so the two cannot be
    scoped differently: same retrieval, same status, same citations."""
    from pipeline.api import app as api

    streamed_seen = _wire_scoped_corpus(monkeypatch)
    streamed = _chat(api, "ragtest-rag", True, document_ids=[IC_BELGE])
    plain_seen = _wire_scoped_corpus(monkeypatch)
    plain = _chat(api, "ragtest-rag", False, document_ids=[IC_BELGE])

    assert _public_reply(streamed, True) == _public_reply(plain, False)
    assert _citations(streamed, True) == _citations(plain, False)
    assert ([params for _sql, params in streamed_seen["statements"]]
            == [params for _sql, params in plain_seen["statements"]])
    assert (_filenames(streamed_seen["ranked"][0])
            == _filenames(plain_seen["ranked"][0]))


@pytest.mark.parametrize("stream", [False, True])
def test_the_llamaindex_model_is_given_the_same_scope(monkeypatch, stream):
    """The other engine is scoped through the same field and the same
    closure; its own route to the store is pinned in
    tests/test_llamaindex_build.py."""
    from pipeline.api import app as api
    from pipeline.generation import answer as generation
    from pipeline.retrieval import rag_llamaindex

    handed = []

    def retrieve(_question, **kwargs):
        handed.append(kwargs)
        return [KAPSAM_CORPUS[0]]

    monkeypatch.setattr(rag_llamaindex, "retrieve", retrieve)
    monkeypatch.setattr(generation, "complete",
                        lambda _policy, _user_content: ICERIDEN)

    response = _chat(api, "ragtest-rag-llamaindex", stream,
                     document_ids=[IC_BELGE])

    assert response.status_code == 200
    assert _public_reply(response, stream)[0] == ANSWERED
    assert handed == [{"document_ids": (IC_BELGE,)}]


# --- refusing a scope that is not one ------------------------------------
#
# Every refusal below is proven to cost NOTHING: the request model refuses
# the shape before the endpoint body runs, so no backend is called, no
# pooled connection is borrowed and neither embedding is computed. Each
# test also pins WHICH gate refused it -- a validation error located at
# `body -> document_ids` -- so a refusal cannot quietly move to (or away
# from) the declaration that is supposed to carry it.


def _refusal_gates(monkeypatch):
    """Counters on every seam a refused request must never reach."""
    from pipeline.api import app as api
    from pipeline.index import db
    from pipeline.retrieval import query

    counted = {"backend": 0, "checkouts": 0, "dense": 0, "sparse": 0}

    class Pool:
        @contextmanager
        def connection(self):
            counted["checkouts"] += 1
            yield object()

    def backend(*_args, **_kwargs):
        counted["backend"] += 1
        raise AssertionError("reddedilen istek backende ulasti")

    def dense(_question):
        counted["dense"] += 1
        return [0.0]

    def sparse(_question):
        counted["sparse"] += 1
        return ([1], [1.0])

    monkeypatch.setattr(db, "get_pool", lambda: Pool())
    monkeypatch.setattr(api.rag_backends, "answer_checked", backend)
    monkeypatch.setattr(query, "embed_dense", dense)
    monkeypatch.setattr(query, "embed_sparse", sparse)
    return counted


@pytest.mark.parametrize(
    ("document_ids", "gerekce"),
    [
        ([], "bos liste"),
        ([f"{n:08d}-0000-0000-0000-000000000000" for n in range(51)],
         "elli birden fazla"),
        ("kurgu", "liste degil"),
        ({"belge": IC_BELGE}, "liste degil"),
        (7, "liste degil"),
        ([IC_BELGE, "belge-bir"], "gecersiz uuid"),
        (["  "], "gecersiz uuid"),
        ([IC_BELGE, 7], "uuid degil"),
        ([[IC_BELGE]], "uuid degil"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_a_malformed_scope_is_refused_and_costs_nothing(
        monkeypatch, document_ids, gerekce, stream):
    from pipeline.api import app as api

    counted = _refusal_gates(monkeypatch)

    response = _chat(api, "ragtest-rag", stream, document_ids=document_ids)

    assert response.status_code == 422, gerekce
    # the gate that refused it: the request model's own declaration, which
    # is why the error carries a body location instead of a text detail
    locations = [tuple(error["loc"]) for error in response.json()["detail"]]
    assert any(location[:2] == ("body", "document_ids")
               for location in locations), locations
    # ... and it was refused BEFORE any of this could happen
    assert counted == {"backend": 0, "checkouts": 0, "dense": 0, "sparse": 0}


def test_the_scope_bound_is_the_list_as_sent(monkeypatch):
    """Fifty is the cap and fifty is accepted: the refusal above is about
    the bound, not about long lists being unsupported."""
    from pipeline.api import app as api
    from pipeline.validation.rag.answer_guard import ANSWERED, GuardResult

    handed = []

    def checked(question, backend=None, *, document_ids=None):
        handed.append(document_ids)
        return GuardResult(ANSWERED, "kurgu cevap", ())

    monkeypatch.setattr(api.rag_backends, "answer_checked", checked)

    fifty = [f"{n:08d}-0000-0000-0000-000000000000" for n in range(50)]
    response = _chat(api, "ragtest-rag", document_ids=fifty)

    assert response.status_code == 200
    assert api.DOCUMENT_SCOPE_MAX == 50
    assert len(handed[0]) == 50


def test_an_unknown_model_is_still_refused_before_any_scope_work(monkeypatch):
    from pipeline.api import app as api

    counted = _refusal_gates(monkeypatch)

    response = _chat(api, "ragtest-bilinmeyen", document_ids=[IC_BELGE])

    assert response.status_code == 404
    assert counted["backend"] == 0


def test_the_table_route_is_unaffected_by_the_new_field(monkeypatch):
    """The table model keeps its separate service path. The field is not
    its business, and it must not become a reason to touch RAG."""
    from pipeline.api import app as api

    counted = _refusal_gates(monkeypatch)
    handed = {}

    def table_reply(_messages, **kwargs):
        handed.update(kwargs)
        return "KURGU_TABLO_CEVABI"

    monkeypatch.setattr(api.owui_chat, "tables_reply", table_reply)

    response = _chat(api, "ragtest-table", document_ids=[IC_BELGE])

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "KURGU_TABLO_CEVABI")
    assert "rag_status" not in response.json()
    assert counted["backend"] == 0
    assert callable(handed["export_ref_for"])
    # A direct API-key principal may extract a table but never acquires an
    # OpenWebUI actor-bound export reference.
    assert handed["export_ref_for"]("a" * 32 + ".xlsx") is None


def test_the_chat_route_keeps_the_same_auth_dependency_object():
    """Widening the request model must not have rewired the route: the
    dependency list is the SAME object every other protected route uses."""
    from pipeline.api import app as api

    route = next(route for route in api.app.routes
                 if getattr(route, "path", None) == "/v1/chat/completions")
    assert len(route.dependencies) == 1
    assert route.dependencies[0] is api.AUTH[0]
    assert api.AUTH[0].dependency is api.require_api_key
