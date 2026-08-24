"""Authentication on the OpenAI-compatible API, and the liveness/readiness split.

The API used to have no authentication at all, so anything that could reach the
port could ask questions of the indexed documents and upload new ones.
"""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, key):
    """Reimport the module with API_KEY set, since it is read once at import."""
    monkeypatch.setenv("API_KEY", key)
    import pipeline.api.app as api
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
    for method, path, expected in [
        ("get", "/v1/models", 401),
        ("post", "/v1/chat/completions", 401),
        ("post", "/documents/upload", 401),
        ("post", "/documents/abc/process", 401),
        ("post", "/documents/11111111-1111-1111-1111-111111111111/ingest-jobs", 401),
        ("get", "/ingest-jobs/22222222-2222-2222-2222-222222222222", 401),
        ("delete", "/ingest-jobs/22222222-2222-2222-2222-222222222222", 401),
        ("get", "/documents/abc", 401),
        # Actor-bound content routes reject the API-key principal one layer
        # later: authenticated, but not an OpenWebUI content actor.
        ("post", "/v1/exports/tickets", 403),
        ("post", "/v1/exports/download", 403),
    ]:
        r = getattr(secured, method)(path)
        assert r.status_code == expected, f"{method.upper()} {path} korumasiz"


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
    import pipeline.api.app as api

    def unreachable(*a, **k):
        raise ConnectionError(
            "postgresql://db-user:Enter_Your_Password_Here@db-host:5433/db-name erisilemedi")

    monkeypatch.setattr(api, "db_conn", unreachable)
    monkeypatch.setattr("requests.get", unreachable)

    r = TestClient(api.app).get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["kontroller"] == {
        "veritabani": False,
        "sema": False,
        "embedding": False,
    }
    assert "gizli" not in r.text and "db-host" not in r.text


def test_the_download_link_is_not_a_public_filename_capability(secured):
    assert secured.get("/files/guessed.xlsx").status_code == 404
    assert secured.post("/v1/exports/tickets").status_code == 403
    assert secured.post("/v1/exports/download").status_code == 403


# --- deliberately insecure local startup ---

def _startup(tmp_path, *, allow=None, bind=None):
    """Import the application in an isolated process with no auth source.

    Import is the startup configuration boundary.  A subprocess keeps a
    deliberately failed import from leaving this test process with a partly
    initialised module, and disabling dotenv prevents a developer's private
    file from silently supplying a credential to the test.
    """
    environment = os.environ.copy()
    for name in (
            "API_KEY", "API_KEYS_JSON", "OPENWEBUI_GATEWAY_KEY",
            "OPENWEBUI_USER_JWT_SECRET", "EVIDENCE_HMAC_SECRET",
            "ALLOW_INSECURE_LOCAL", "API_BIND_HOST"):
        environment.pop(name, None)
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    environment["UPLOAD_DIR"] = str(tmp_path / "uploads")
    if allow is not None:
        environment["ALLOW_INSECURE_LOCAL"] = allow
    if bind is not None:
        environment["API_BIND_HOST"] = bind
    return subprocess.run(
        [sys.executable, "-c", "import pipeline.api.app"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(("allow", "bind"), [
    (None, None),
    (None, "127.0.0.1"),
    ("1", None),
    ("1", "0.0.0.0"),
    ("1", "localhost"),
    ("true", "127.0.0.1"),
])
def test_no_auth_configuration_fails_startup_closed(
        tmp_path, allow, bind):
    result = _startup(tmp_path, allow=allow, bind=bind)
    assert result.returncode != 0
    assert "AuthConfigurationError" in result.stderr


def test_explicit_insecure_mode_requires_the_documented_loopback_binding(
        tmp_path):
    result = _startup(
        tmp_path, allow="1", bind="127.0.0.1")
    assert result.returncode == 0, result.stderr
