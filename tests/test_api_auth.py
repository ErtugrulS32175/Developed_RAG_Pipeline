"""Authentication on the OpenAI-compatible API, and the liveness/readiness split.

The API used to have no authentication at all, so anything that could reach the
port could ask questions of the indexed documents and upload new ones.
"""
import importlib

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, key):
    """Reimport the module with API_KEY set, since it is read once at import."""
    monkeypatch.setenv("API_KEY", key)
    import pipeline.api as api
    importlib.reload(api)
    return TestClient(api.app), api


@pytest.fixture
def secured(monkeypatch):
    client, api = _client(monkeypatch, "zeta-gamma-test-key")
    yield client
    monkeypatch.delenv("API_KEY", raising=False)
    importlib.reload(api)


# --- with a key configured ---

def test_a_request_without_a_key_is_refused(secured):
    assert secured.get("/v1/models").status_code == 401


def test_a_wrong_key_is_refused(secured):
    r = secured.get("/v1/models", headers={"Authorization": "Bearer yanlis"})
    assert r.status_code == 401


def test_the_right_key_is_accepted(secured):
    r = secured.get("/v1/models", headers={"Authorization": "Bearer zeta-gamma-test-key"})
    assert r.status_code == 200
    # the third id is the alternative engine, selectable per conversation
    assert [m["id"] for m in r.json()["data"]] == [
        "ragtest-rag", "ragtest-table", "ragtest-rag-llamaindex"]


def test_the_scheme_has_to_be_bearer(secured):
    r = secured.get("/v1/models", headers={"Authorization": "Basic zeta-gamma-test-key"})
    assert r.status_code == 401


def test_every_data_endpoint_is_covered(secured):
    """A single unprotected route is enough to read or add documents, so the
    whole surface is checked rather than a sample of it."""
    for method, path in [
        ("get", "/v1/models"),
        ("post", "/v1/chat/completions"),
        ("post", "/documents/upload"),
        ("post", "/documents/abc/process"),
        ("get", "/documents/abc"),
    ]:
        r = getattr(secured, method)(path)
        assert r.status_code == 401, f"{method.upper()} {path} korumasiz"


def test_liveness_stays_open(secured):
    """Monitoring must be able to reach it without holding a credential."""
    assert secured.get("/health").status_code == 200


def test_the_api_surface_is_not_published_once_a_key_is_set(secured):
    """No reason to hand the endpoint list to a caller who cannot use any of it."""
    assert secured.get("/openapi.json").status_code == 404
    assert secured.get("/docs").status_code == 404


def test_readiness_reports_status_without_leaking_connection_detail(monkeypatch):
    """/ready has to stay reachable for a load balancer, so its body must not
    carry the host, port or user a failed connection was trying.

    The dependencies are stubbed rather than contacted: a test that waits on a
    real connection timeout takes minutes and fails for reasons of its own."""
    import pipeline.api as api

    def unreachable(*a, **k):
        raise ConnectionError("postgresql://rag:gizli@db-host:5433/ragdb erisilemedi")

    monkeypatch.setattr(api, "get_conn", unreachable)
    monkeypatch.setattr("requests.get", unreachable)

    r = TestClient(api.app).get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["kontroller"] == {"veritabani": False, "embedding": False}
    assert "gizli" not in r.text and "db-host" not in r.text


def test_the_download_link_stays_open(secured):
    """The xlsx link is opened by the user's browser, which cannot send a bearer
    header. The filename is a sha256 prefix, so knowing the URL is the
    capability -- but the route must still reject a traversal attempt."""
    assert secured.get("/files/..%2Fsecret.xlsx").status_code in (400, 404)


# --- with no key configured ---

def test_without_a_key_the_api_is_open(monkeypatch):
    """Documented behaviour, not an oversight: a local run stays friction-free
    and startup prints a warning. Anything exposed beyond localhost must set it."""
    client, api = _client(monkeypatch, "")
    try:
        assert client.get("/v1/models").status_code == 200
    finally:
        monkeypatch.delenv("API_KEY", raising=False)
        importlib.reload(api)
