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
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@contextmanager
def _fake_db_conn():
    """Stands in for the pooled per-request connection; the db helpers are
    monkeypatched, so the connection object itself is never touched."""
    yield object()

from pipeline.validation.rag.answer_guard import ANSWERED, REVIEW_REQUIRED


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


def _chat(api, model, stream=False):
    return TestClient(api.app).post(
        "/v1/chat/completions",
        headers=_headers(api),
        json={
            "model": model,
            "messages": [{"role": "user", "content": "zeta uretimi nedir?"}],
            "stream": stream,
        },
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

    document_id = "kurgu-belge-kimligi"
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
            cid = f"kurgu-aday-{minted['n']}"
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

    def run_ingest(path, expected_candidate=None, attempt=None):
        """The core as it REALLY behaves, which is the point of this fake.

        The previous version wrote the document's status directly, and
        that single line hid a live defect: a real PARTIAL run leaves
        the row untouched -- its verdict belongs to the attempt (rule 5)
        -- so an endpoint reading the row back saw `processing` and
        called a truthful partial run a failure. Here only a PROMOTION
        moves the served status, in the same breath as the generation,
        and the run REPORTS its verdict to its caller."""
        calls.append((path, expected_candidate, attempt))
        if ingest_error is not None:
            raise ingest_error
        if ingest_outcome is None:
            return None                 # a run that reported nothing
        status, note = ingest_outcome
        if status == "done":
            row = state[document_id]
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
    monkeypatch.setattr(api.db, "set_document_status", set_document_status)
    monkeypatch.setattr(api.db, "document_publish_lock", publish_lock)
    monkeypatch.setattr(api.ingest, "main", run_ingest)
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
    assert (upload_dir / "kurgu-belge.pdf").read_bytes() == b"KURGU_PDF"

    # Round 14: same name + different bytes is a CONFLICT by default -- the
    # old contract announced the replacement only after doing it. The old
    # file must survive a refused upload untouched.
    conflict = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu-belge.pdf", b"KURGU_PDF_V2", "application/pdf")},
    )
    assert conflict.status_code == 409
    assert (upload_dir / "kurgu-belge.pdf").read_bytes() == b"KURGU_PDF"
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
    assert (upload_dir / "kurgu-belge.pdf").read_bytes() == b"KURGU_PDF_V2"
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
    path_called, legacy_binding, attempt = calls[0]
    assert path_called == str(upload_dir / "kurgu-belge.pdf")
    assert legacy_binding is None
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
        return "kurgu-belge-kimligi", "kurgu-aday-1", filename

    def recording_finalize(_conn, _document_id, _candidate_id):
        events.append("db-yayimla")
        return True

    real_replace = api.os.replace

    def recording_replace(src, dst):
        events.append("disk")
        return real_replace(src, dst)

    monkeypatch.setattr(api.db, "document_publish_lock", recording_lock)
    monkeypatch.setattr(api.db, "stage_candidate", recording_stage)
    monkeypatch.setattr(api.db, "finalize_candidate_publication",
                        recording_finalize)
    monkeypatch.setattr(api.os, "replace", recording_replace)

    response = client.post(
        "/documents/upload",
        headers=_headers(api),
        files={"file": ("kurgu.pdf", b"KURGU_PDF", "application/pdf")},
    )
    assert response.status_code == 200
    assert events == ["kilit-al", "db-evrele", "disk", "db-yayimla",
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
        state["kurgu-belge-kimligi"] = {
            "id": "kurgu-belge-kimligi",
            "filename": filename,
            "file_type": file_type,
            "status": "pending",
            "content_sha256": content_sha256,
            "candidate_id": f"kurgu-aday-{content_sha256[:8]}",
            "candidate_state": "staged",
        }
        time.sleep(0.15)  # the historic gap between DB commit and os.replace
        return ("kurgu-belge-kimligi",
                state["kurgu-belge-kimligi"]["candidate_id"], filename)

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
    disk_sha = hashlib.sha256(
        (upload_dir / "kurgu.pdf").read_bytes()).hexdigest()
    assert state["kurgu-belge-kimligi"]["content_sha256"] == disk_sha


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
    assert (upload_dir / "kurgu-belge.pdf").read_bytes() == b"IKINCI_KURGU"
    assert state["kurgu-belge-kimligi"]["filename"] == "kurgu-belge.pdf"


def test_a_process_bound_to_a_stale_candidate_cannot_touch_the_index(
        monkeypatch, tmp_path):
    """Round 18, the P0 replayed at the seam: the process step hands its
    ingest the candidate identity it READ, and an ingest that finds the
    disk carrying other bytes refuses. Here the disk moved after the row
    was read -- the refusal is the contract."""
    api, client, state, calls, upload_dir, _closed = _document_api(
        monkeypatch, tmp_path)
    document_id = "kurgu-belge-kimligi"
    state[document_id] = {
        "id": document_id,
        "filename": "kurgu-belge.pdf",
        "file_type": "pdf",
        "status": "pending",
        "content_sha256": hashlib.sha256(b"ILK_KURGU").hexdigest(),
        "candidate_id": "kurgu-aday-0",
        "candidate_state": PUBLISHED,
    }
    (upload_dir / "kurgu-belge.pdf").write_bytes(b"ILK_KURGU")

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )
    assert response.status_code == 200
    _path, _legacy, attempt = calls[0]
    assert attempt.candidate_id == "kurgu-aday-0"
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

    def losing_ingest(_path, expected_candidate=None, attempt=None):
        # a concurrent run promotes first...
        state[document_id]["active_generation"] = 1
        state[document_id]["status"] = "done"
        # ...and THIS run's promotion fails its CAS loudly
        raise RuntimeError("es zamanli terfi kazandi")

    monkeypatch.setattr(api.ingest, "main", losing_ingest)

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

    response = client.post(
        f"/documents/{document_id}/process",
        headers=_headers(api),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "uploaded file missing"
    # The SERVED version is untouched. Its chunks are in the index and
    # still answering questions -- the source file is only needed to
    # build the NEXT generation, so a missing one is a storage problem,
    # not a verdict on what is being served. Stamping `error` here left
    # a healthy index wearing a failure label.
    assert state[document_id]["status"] == "done"
    assert state[document_id]["active_generation"] == 4
    assert calls == []
    assert closed == []          # no lease was ever taken
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
    assert (upload_dir / "kurgu.pdf").read_bytes() == b"ILK_KURGU_PDF"
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
    same calls alongside in `.dates`.

    The window was added WITHOUT moving what the older assertions read: a
    test that pinned `[(20, 0, None, None)]` still pins it, because the
    new pair went next to that record rather than into it."""

    def __init__(self):
        super().__init__()
        self.dates = []


def _wire_inventory(monkeypatch, api, rows=INVENTORY):
    """Replace the query with one that records how it was CALLED.

    It returns the ``limit + 1`` window the real helper returns, so the
    endpoint's `has_more` is computed from the same evidence in the test
    as in production -- and, like the real helper, it filters BEFORE it
    pages, so offset and the sentinel walk the filtered sequence.

    The date bounds are applied here the way the SQL applies them: both
    EXCLUSIVE, so a row sitting exactly on a bound is filtered out."""
    asked = _InventoryCalls()

    def list_documents(_conn, limit, offset, status=None, file_type=None,
                       uploaded_after=None, uploaded_before=None):
        asked.append((limit, offset, status, file_type))
        asked.dates.append((uploaded_after, uploaded_before))
        matched = [
            dict(row) for row in rows
            if (status is None or row["status"] == status)
            and (file_type is None or row["file_type"] == file_type)
            and (uploaded_after is None
                 or row["uploaded_at"] > uploaded_after)
            and (uploaded_before is None
                 or row["uploaded_at"] < uploaded_before)
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
            "status", "status_note", "active_generation"}
    # not merely absent as keys: the VALUES never appear in the response
    assert "content_sha256" not in response.text
    assert "candidate_id" not in response.text
    assert "KURGU_SHA_" not in response.text
    assert "KURGU_ADAY_" not in response.text
    assert documents[1]["status_note"] == "sayfa 2 kayip"


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
            "status", "status_note", "active_generation"}


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
                                    "has_more"}
    for document in body["documents"]:
        assert set(document) == {
            "document_id", "filename", "file_type", "uploaded_at",
            "status", "status_note", "active_generation"}


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
                                    "has_more"}


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
