"""Collections and tags are metadata scopes, never document ownership."""
from contextlib import contextmanager
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from pipeline.api import app as api
from pipeline.index import db
from pipeline.validation.rag.answer_guard import ANSWERED, GuardResult


COLLECTION = "11111111-1111-1111-1111-111111111111"
DOCUMENT = "22222222-2222-2222-2222-222222222222"
TAG = "33333333-3333-3333-3333-333333333333"


def _headers():
    return ({"Authorization": f"Bearer {api.API_KEY}"}
            if api.API_KEY else {})


@contextmanager
def _conn():
    yield object()


def test_label_identity_is_trimmed_casefolded_and_closed_to_controls():
    assert db._canonical_label("  Strasse  ") == ("Strasse", "strasse")
    assert db._canonical_label("STRASSE")[1] == "strasse"
    with pytest.raises(ValueError):
        db._canonical_label("   ")
    with pytest.raises(ValueError):
        db._canonical_label("alpha\n beta")


def test_collection_management_routes_use_one_authenticated_db_seam(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(
        api.db, "create_collection",
        lambda _conn, name: calls.append(("create", name)) or {
            "collection_id": COLLECTION, "name": name, "created_at": None})
    monkeypatch.setattr(
        api.db, "list_collections",
        lambda _conn: [{"collection_id": COLLECTION, "name": "Alpha",
                        "created_at": None, "document_count": 1}])
    monkeypatch.setattr(
        api.db, "delete_collection",
        lambda _conn, identity: calls.append(("delete", identity)) or True)
    client = TestClient(api.app)

    created = client.post("/collections", headers=_headers(),
                          json={"name": "Alpha"})
    listed = client.get("/collections", headers=_headers())
    deleted = client.delete(f"/collections/{COLLECTION}", headers=_headers())

    assert created.status_code == 200
    assert created.json()["collection_id"] == COLLECTION
    assert listed.json()["collections"][0]["document_count"] == 1
    assert deleted.status_code == 204
    assert calls == [("create", "Alpha"), ("delete", COLLECTION)]


def test_tag_vocabulary_can_be_listed_and_deleted_without_a_document_delete(
        monkeypatch):
    calls = []
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(
        api.db, "list_tags",
        lambda _conn: [{"tag_id": TAG, "name": "Finance",
                        "created_at": None, "document_count": 2}])
    monkeypatch.setattr(
        api.db, "delete_tag",
        lambda _conn, identity: calls.append(identity) or True)
    client = TestClient(api.app)

    listed = client.get("/tags", headers=_headers())
    deleted = client.delete(f"/tags/{TAG}", headers=_headers())

    assert listed.json()["tags"][0]["document_count"] == 2
    assert deleted.status_code == 204
    assert calls == [TAG]


def test_membership_and_tag_routes_are_idempotent_metadata_operations(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(
        api.db, "set_collection_document",
        lambda _conn, collection, document, present: calls.append(
            (collection, document, present)) or present)
    monkeypatch.setattr(
        api.db, "replace_document_tags",
        lambda _conn, document, tags: calls.append((document, tuple(tags))) or {
            "document_id": document, "tags": tags})
    client = TestClient(api.app)

    added = client.put(
        f"/collections/{COLLECTION}/documents/{DOCUMENT}", headers=_headers())
    removed = client.delete(
        f"/collections/{COLLECTION}/documents/{DOCUMENT}", headers=_headers())
    tagged = client.put(f"/documents/{DOCUMENT}/tags", headers=_headers(),
                        json={"tags": ["Finance", "Urgent"]})
    cleared = client.put(f"/documents/{DOCUMENT}/tags", headers=_headers(),
                         json={"tags": []})

    assert (added.json()["present"], removed.json()["present"]) == (True, False)
    assert tagged.json()["tags"] == ["Finance", "Urgent"]
    assert cleared.json()["tags"] == []
    assert calls == [
        (COLLECTION, DOCUMENT, True), (COLLECTION, DOCUMENT, False),
        (DOCUMENT, ("Finance", "Urgent")), (DOCUMENT, ())]


def test_unknown_collection_or_document_is_a_closed_not_found(monkeypatch):
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(api.db, "set_collection_document",
                        lambda *_args, **_kwargs: None)
    response = TestClient(api.app).put(
        f"/collections/{COLLECTION}/documents/{DOCUMENT}", headers=_headers())
    assert response.status_code == 404


def test_chat_hands_canonical_scope_dimensions_to_the_checked_backend(
        monkeypatch):
    seen = []
    monkeypatch.setattr(
        api.rag_backends, "answer_checked",
        lambda _question, backend=None, **scope: seen.append(
            {"backend": backend, **scope}) or GuardResult(
            status=ANSWERED, answer="kurgu cevap", diagnostics=(),
            citations=()))
    response = TestClient(api.app).post(
        "/v1/chat/completions", headers=_headers(), json={
            "model": api.RAG_MODEL_ID,
            "messages": [{"role": "user", "content": "soru"}],
            "document_ids": [DOCUMENT],
            "collection_ids": [COLLECTION],
            "tags": ["Finance", "URGENT"],
        })

    assert response.status_code == 200
    assert seen == [{
        "backend": "native", "document_ids": (DOCUMENT,),
        "collection_ids": (COLLECTION,),
        "tags": ("Finance", "URGENT")},
    ]


def test_metadata_scope_is_never_dropped_before_the_checked_backend(
        monkeypatch):
    scopes = []
    monkeypatch.setattr(
        api.rag_backends, "answer_checked",
        lambda _question, backend=None, **scope: scopes.append(scope) or
        GuardResult(status=ANSWERED, answer="kurgu cevap", diagnostics=(),
                    citations=()))
    response = TestClient(api.app).post(
        "/v1/chat/completions", headers=_headers(), json={
            "model": api.RAG_MODEL_ID,
            "messages": [{"role": "user", "content": "soru"}],
            "collection_ids": [COLLECTION],
        })
    assert response.status_code == 200
    assert scopes == [{"collection_ids": (COLLECTION,)}]


def test_inventory_forwards_organization_filters_without_widening_projection(
        monkeypatch):
    asked = []
    monkeypatch.setattr(api, "db_conn", _conn)

    def listing(_conn, limit, offset, **filters):
        asked.append((limit, offset, filters))
        return [{"document_id": DOCUMENT, "filename": "alpha.pdf",
                 "file_type": "pdf", "uploaded_at": None, "status": "done",
                 "status_note": None, "active_generation": 1,
                 "archived_at": None, "private": "never"}]

    monkeypatch.setattr(api.db, "list_documents", listing)
    response = TestClient(api.app).get(
        "/documents", params={"collection_id": COLLECTION, "tag": "Finance"},
        headers=_headers())
    assert response.status_code == 200
    assert asked[0][2]["collection_id"] == COLLECTION
    assert asked[0][2]["tag"] == "Finance"
    assert set(response.json()["documents"][0]) == set(api.DOCUMENT_LIST_FIELDS)
    assert "private" not in response.json()["documents"][0]


def test_scope_types_are_refused_before_a_connection_is_borrowed(monkeypatch):
    borrowed = []

    @contextmanager
    def borrowing():
        borrowed.append(True)
        yield object()

    monkeypatch.setattr(api, "db_conn", borrowing)
    client = TestClient(api.app)
    for body in (
        {"collection_ids": []}, {"tags": []},
        {"collection_ids": ["not-a-uuid"]}, {"tags": [""]},
    ):
        payload = {"model": api.RAG_MODEL_ID,
                   "messages": [{"role": "user", "content": "soru"}], **body}
        assert client.post("/v1/chat/completions", headers=_headers(),
                           json=payload).status_code == 422
    assert borrowed == []


def test_whitespace_inventory_tag_is_a_422_not_a_server_error(monkeypatch):
    monkeypatch.setattr(api, "db_conn", _conn)
    monkeypatch.setattr(
        api.db, "list_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("tag bos olmayan bir metin olmali")))
    response = TestClient(api.app).get(
        "/documents", params={"tag": "   "}, headers=_headers())
    assert response.status_code == 422


def test_collection_paths_use_uuid_types_in_openapi():
    schema = api.app.openapi()
    path = schema["paths"]["/collections/{collection_id}"]["delete"]
    parameter = next(item for item in path["parameters"]
                     if item["name"] == "collection_id")
    assert parameter["schema"]["format"] == "uuid"
    assert UUID(COLLECTION)
