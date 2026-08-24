"""Public contracts for immutable document-version inventory and activation."""
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api
from pipeline.api import auth
from pipeline.index import db, publication


DOCUMENT = "11111111-1111-4111-8111-111111111111"
VERSION = "22222222-2222-4222-8222-222222222222"
TENANT = "33333333-3333-4333-8333-333333333333"
DIGEST = "a" * 64
SAFE_VERSION_FIELDS = {
    "version_id", "version_number", "created_at", "is_active",
    "index_ready", "document_revision",
}


@contextmanager
def _conn():
    yield object()


@pytest.fixture
def version_api(monkeypatch):
    rows = [
        {"key": "reader-key", "tenant_id": TENANT, "role": "reader"},
        {"key": "editor-key", "tenant_id": TENANT, "role": "editor"},
        {"key": "admin-key", "tenant_id": TENANT, "role": "admin"},
    ]
    monkeypatch.setattr(
        api, "AUTH_REGISTRY", auth.load_registry("", json.dumps(rows)))
    monkeypatch.setattr(api, "db_conn", _conn)
    return TestClient(api.app)


def _headers(role):
    return {"Authorization": f"Bearer {role}-key"}


def _version(number, *, active=False, ready=True):
    return {
        "version_id": str(number).zfill(8) + "-0000-4000-8000-000000000000",
        "version_number": number,
        "created_at": f"0999-01-{number:02d}T00:00:00+00:00",
        "is_active": active,
        "index_ready": ready,
        "document_revision": 7,
        # A database helper is narrow today, but the HTTP boundary keeps its
        # own allowlist so a future SELECT expansion cannot become an API leak.
        "content_sha256": "private-digest",
        "candidate_id": "private-candidate",
        "source_path": "private-path",
    }


def test_version_inventory_is_safe_bounded_and_cursor_paginated(
        version_api, monkeypatch):
    calls = []
    monkeypatch.setattr(
        api.db, "list_document_versions",
        lambda _conn, document, **kwargs: calls.append((document, kwargs)) or [
            _version(11, active=True), _version(10), _version(9)])

    response = version_api.get(
        f"/documents/{DOCUMENT}/versions?limit=2&before_version_number=12",
        headers=_headers("reader"))

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 2
    assert body["before_version_number"] == 12
    assert body["has_more"] is True
    assert body["next_before_version_number"] == 10
    assert [row["version_number"] for row in body["versions"]] == [11, 10]
    assert all(set(row) == SAFE_VERSION_FIELDS for row in body["versions"])
    assert "private" not in response.text
    assert calls == [(DOCUMENT, {"limit": 2, "before_version_number": 12})]


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=101", "before_version_number=0"])
def test_invalid_version_page_is_refused_before_database(
        version_api, monkeypatch, query):
    borrowed = []

    @contextmanager
    def borrowing():
        borrowed.append(True)
        yield object()

    monkeypatch.setattr(api, "db_conn", borrowing)
    response = version_api.get(
        f"/documents/{DOCUMENT}/versions?{query}",
        headers=_headers("reader"))
    assert response.status_code == 422
    assert borrowed == []


def test_admin_activation_freshly_proves_source_and_passes_revision_cas(
        version_api, monkeypatch):
    calls = []
    monkeypatch.setattr(
        api.db, "document_version_source_digest",
        lambda _conn, document, version: calls.append(
            ("digest", document, version)) or DIGEST)

    def verify(root, tenant, document, version, **kwargs):
        calls.append(("proof", root, tenant, document, version, kwargs))
        return publication.VersionSourceProof(
            str(tenant), document, version, DIGEST, 17)

    def activate(_conn, document, version, revision, **kwargs):
        calls.append(("activate", document, version, revision, kwargs))
        return {"document_id": document, "active_version_id": version,
                "active_generation": 4, "revision": 8, "changed": True}

    monkeypatch.setattr(api.publication, "verify_version_source", verify)
    monkeypatch.setattr(api.db, "activate_document_version", activate)

    response = version_api.post(
        f"/documents/{DOCUMENT}/versions/{VERSION}/activate",
        headers=_headers("admin"), json={"expected_revision": 7})

    assert response.status_code == 200
    assert response.json()["revision"] == 8
    assert [call[0] for call in calls] == ["digest", "proof", "activate"]
    assert tuple(map(str, calls[1][2:5])) == (TENANT, DOCUMENT, VERSION)
    assert calls[1][5]["expected_sha256"] == DIGEST
    assert calls[2][1:] == (
        DOCUMENT, VERSION, 7, {"verified_source_sha256": DIGEST})


def test_unknown_version_is_404_without_touching_source_or_activation(
        version_api, monkeypatch):
    touched = []
    monkeypatch.setattr(api.db, "document_version_source_digest",
                        lambda *_args: None)
    monkeypatch.setattr(api.publication, "verify_version_source",
                        lambda *_a, **_k: touched.append("proof"))
    monkeypatch.setattr(api.db, "activate_document_version",
                        lambda *_a, **_k: touched.append("activate"))

    response = version_api.post(
        f"/documents/{DOCUMENT}/versions/{VERSION}/activate",
        headers=_headers("admin"), json={"expected_revision": 7})
    assert response.status_code == 404
    assert touched == []


@pytest.mark.parametrize(
    "failure, expected_detail",
    [
        (publication.VersionSourceMissing("private missing path"),
         "document version source unavailable"),
        (publication.VersionSourceCorrupt("private corrupt bytes"),
         "document version source invalid"),
    ],
)
def test_source_proof_refusal_is_closed_409_without_activation(
        version_api, monkeypatch, failure, expected_detail):
    activated = []
    monkeypatch.setattr(api.db, "document_version_source_digest",
                        lambda *_args: DIGEST)
    monkeypatch.setattr(
        api.publication, "verify_version_source",
        lambda *_a, **_k: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(api.db, "activate_document_version",
                        lambda *_a, **_k: activated.append(True))

    response = version_api.post(
        f"/documents/{DOCUMENT}/versions/{VERSION}/activate",
        headers=_headers("admin"), json={"expected_revision": 7})
    assert response.status_code == 409
    assert response.json()["detail"] == expected_detail
    assert "private" not in response.text
    assert activated == []


def test_revision_conflict_is_closed_409(version_api, monkeypatch):
    private = "private revision and row detail"
    monkeypatch.setattr(api.db, "document_version_source_digest",
                        lambda *_args: DIGEST)
    monkeypatch.setattr(
        api.publication, "verify_version_source",
        lambda _root, tenant, document, version, **_kwargs:
        publication.VersionSourceProof(
            str(tenant), document, version, DIGEST, 17))
    monkeypatch.setattr(
        api.db, "activate_document_version",
        lambda *_a, **_k: (_ for _ in ()).throw(
            db.DocumentVersionConflict(private)))

    response = version_api.post(
        f"/documents/{DOCUMENT}/versions/{VERSION}/activate",
        headers=_headers("admin"), json={"expected_revision": 6})
    assert response.status_code == 409
    assert "private" not in response.text


def test_version_reads_allow_reader_but_activation_requires_admin(
        version_api, monkeypatch):
    monkeypatch.setattr(api.db, "list_document_versions",
                        lambda *_args, **_kwargs: [])
    path = f"/documents/{DOCUMENT}/versions"
    assert version_api.get(path, headers=_headers("reader")).status_code == 200
    activation = path + f"/{VERSION}/activate"
    body = {"expected_revision": 0}
    assert version_api.post(activation, json=body).status_code == 401
    assert version_api.post(
        activation, headers=_headers("reader"), json=body).status_code == 403
    assert version_api.post(
        activation, headers=_headers("editor"), json=body).status_code == 403
