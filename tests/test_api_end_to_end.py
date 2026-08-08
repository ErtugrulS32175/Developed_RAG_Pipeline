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
