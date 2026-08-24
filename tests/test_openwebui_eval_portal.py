"""Static, cryptographic and closed-shape tests for the eval portal."""
import asyncio
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.api import identity


ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "openwebui" / "functions" / "ragtest_eval_portal.py"


def _module(monkeypatch):
    monkeypatch.setenv(
        "OPENWEBUI_GATEWAY_KEY", "eval-gateway-key-with-more-than-32-bytes")
    monkeypatch.setenv(
        "FORWARD_USER_INFO_HEADER_JWT_SECRET",
        "eval-identity-key-with-more-than-32-bytes")
    spec = importlib.util.spec_from_file_location("ragtest_eval_portal", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uuid(number):
    return f"00000000-0000-4000-8000-{number:012d}"


def test_eval_portal_assertion_matches_the_backend_verifier(monkeypatch):
    module = _module(monkeypatch)
    user = SimpleNamespace(id="eval-subject", email="display@example.invalid",
                           name="Display", role="admin")
    token = module.signed_identity(user, now=1_000)
    verifier = identity.Verifier.configured(
        "eval-identity-key-with-more-than-32-bytes")
    resolved = verifier.verify(token, now=1_030)
    assert resolved.subject == "eval-subject"
    assert not hasattr(resolved, "role")


def test_every_route_is_session_authenticated_and_role_is_not_a_bypass():
    source = PLUGIN.read_text(encoding="utf-8")
    assert source.count("Depends(get_verified_user)") == 8
    assert source.count("app.add_api_route(") == 8
    assert 'Authorization": f"Bearer {GATEWAY_KEY}' in source
    assert "X-OpenWebUI-User-Jwt" in source
    register = source[source.index("    def _register"):]
    assert ".role" not in register
    assert "architecture_admin" not in register


def test_browser_keeps_secrets_and_run_content_out_of_the_page(monkeypatch):
    module = _module(monkeypatch)
    html = module.PORTAL_HTML
    for forbidden in (
        "OPENWEBUI_GATEWAY_KEY", "FORWARD_USER_INFO_HEADER_JWT_SECRET",
        "innerHTML", "localStorage", "sessionStorage", "indexedDB",
        ".context", ".passage", "run_answer"):
        assert forbidden not in html
    assert "textContent" in html
    assert "replaceChildren" in html
    assert "for(const c of" not in html


def test_proxy_caps_success_responses_before_decoding(monkeypatch):
    module = _module(monkeypatch)
    user = SimpleNamespace(id="u", email="", name="", role="user")

    class Response:
        status = 200
        length = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, amount):
            assert amount == module.MAX_RESPONSE_BYTES + 1
            return b"x" * amount

    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: Response())
    with pytest.raises(RuntimeError, match="response is too large"):
        module._request(user, "GET", "/v1/eval/datasets?limit=100")


def test_backend_error_content_never_reaches_the_browser(monkeypatch):
    module = _module(monkeypatch)
    user = SimpleNamespace(id="u", email="", name="", role="user")
    secret = b'{"detail":"PRIVATE_RUN_ANSWER_AND_SECRET"}'

    def refuse(request, timeout):
        raise module.urllib.error.HTTPError(
            request.full_url, 409, "conflict", {}, io.BytesIO(secret))

    monkeypatch.setattr(module.urllib.request, "urlopen", refuse)
    status, body = module._request(user, "POST", "/v1/eval/datasets", {})
    assert status == 409
    assert body == {"detail": "evaluation request refused"}
    assert "PRIVATE" not in json.dumps(body)


def test_create_draft_and_publish_payloads_are_exact(monkeypatch):
    module = _module(monkeypatch)
    assert module._dataset_create({"slug": "kalite-v1", "label": "Kalite"}) == {
        "slug": "kalite-v1", "label": "Kalite"}
    assert module._dataset_create({
        "slug": "kalite-v1", "label": "Kalite", "admin": True}) is None
    assert module._draft_create({"expected_revision": 2}) == {
        "expected_revision": 2}
    assert module._draft_create({"expected_revision": True}) is None
    assert module._publish({
        "expected_revision": 2, "expected_policy_epoch": 3,
        "expected_draft_sha256": "a" * 64}) == {
            "expected_revision": 2, "expected_policy_epoch": 3,
            "expected_draft_sha256": "a" * 64}
    assert module._publish({
        "expected_revision": 2, "expected_policy_epoch": 3,
        "expected_draft_sha256": "a" * 64, "force": True}) is None
    assert module._publish({
        "expected_revision": 2, "expected_policy_epoch": 3,
        "expected_draft_sha256": "A" * 64}) is None
    assert module._retire({
        "expected_revision": 4, "expected_policy_epoch": 3}) == {
            "expected_revision": 4, "expected_policy_epoch": 3}
    assert module._retire({
        "expected_revision": 4, "expected_policy_epoch": 3,
        "force": True}) is None


def test_case_import_is_bounded_closed_and_carries_no_run_artifacts(
        monkeypatch):
    module = _module(monkeypatch)
    case = {
        "case_key": _uuid(1), "q": "Soru", "key": "belge-a",
        "answer": "Beklenen", "pages": [1, 2], "type": "metin",
    }
    accepted = module._import_cases({"expected_revision": 4, "cases": [case]})
    assert accepted == {"expected_revision": 4, "cases": [case]}
    for forbidden in ("context", "passage", "run_answer", "secret"):
        hostile = dict(case, **{forbidden: "sentinel"})
        assert module._import_cases({
            "expected_revision": 4, "cases": [hostile]}) is None
    assert module._import_cases({
        "expected_revision": 4, "cases": [case] * 501}) is None
    assert module._import_cases({
        "expected_revision": 4,
        "cases": [dict(case, pages=[1, 1])]}) is None
    assert module._import_cases({
        "expected_revision": 4,
        "cases": [dict(case, pages=[2, 1])]}) is None


def test_portal_bounds_import_before_decode_and_rejects_duplicate_keys(
        monkeypatch):
    module = _module(monkeypatch)
    case = {
        "case_key": _uuid(1), "q": "Soru", "key": "belge-a",
        "answer": "Beklenen", "pages": [1], "type": "metin",
    }

    class Request:
        def __init__(self, raw, length=None):
            self.raw = raw
            self.headers = {}
            if length is not None:
                self.headers["content-length"] = str(length)

        async def stream(self):
            midpoint = len(self.raw) // 2
            yield self.raw[:midpoint]
            yield self.raw[midpoint:]

    payload = json.dumps({
        "expected_revision": 4, "cases": [case]},
        separators=(",", ":")).encode()
    assert asyncio.run(module._read_import_request(Request(payload))) == {
        "expected_revision": 4, "cases": [case]}
    with pytest.raises(module._ImportTooLarge):
        asyncio.run(module._read_import_request(
            Request(b"{}", module.MAX_IMPORT_BYTES + 1)))
    duplicate = payload.replace(
        b'{"expected_revision":4',
        b'{"expected_revision":4,"expected_revision":4')
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(module._read_import_request(Request(duplicate)))


def test_list_contract_is_metadata_only_and_statuses_are_closed(monkeypatch):
    module = _module(monkeypatch)
    body = {"datasets": [{
        "dataset_id": _uuid(1), "slug": "kalite", "label": "Kalite",
        "state": "active", "revision": 2, "owner_label": "Ekip",
        "policy_epoch": 4, "current_version_id": _uuid(2),
        "current_version_number": 1,
        "versions": [{
            "version_id": _uuid(2), "version_number": 1,
            "state": "published", "revision": 3, "case_count": 20,
            "content_sha256": "a" * 64, "sealed_at": "2026-08-24T00:00:00Z",
        }],
    }]}
    assert module._closed_metadata(body)
    assert module._closed_dataset_result({"dataset": body["datasets"][0]})
    assert module._closed_version_result({
        "version": body["datasets"][0]["versions"][0]})
    assert module._closed_versions({
        "versions": body["datasets"][0]["versions"]})
    bad_state = json.loads(json.dumps(body))
    bad_state["datasets"][0]["versions"][0]["state"] = "retired"
    assert not module._closed_metadata(bad_state)
    for forbidden in ("answer", "q", "context", "passage", "secret"):
        leaked = json.loads(json.dumps(body))
        leaked["datasets"][0][forbidden] = "sentinel"
        assert not module._closed_metadata(leaked)


def test_command_results_cannot_return_case_or_run_content(monkeypatch):
    module = _module(monkeypatch)
    version = {
        "version_id": _uuid(3), "version_number": 1, "state": "draft",
        "revision": 1, "case_count": 0, "content_sha256": None,
        "sealed_at": None,
    }
    assert module._closed_version_result({"version": version})
    assert not module._closed_import_result({"version": version})
    imported = dict(version, content_sha256="a" * 64)
    assert module._closed_import_result({"version": imported})
    for forbidden in ("q", "answer", "context", "passage", "secret"):
        assert not module._closed_version_result({
            "version": dict(version, **{forbidden: "sentinel"})})


def test_browser_repeats_the_closed_case_shape_and_size_gate(monkeypatch):
    module = _module(monkeypatch)
    html = module.PORTAL_HTML
    assert "const caseOk=" in html
    assert "cases.every(caseOk)" in html
    assert "bytes(body)>16777216" in html
    assert "['case_key','q','key','answer','pages','type']" in html
    assert "result.version.content_sha256" in html
    assert "document.getElementById('draft-sha256').value=" in html
    assert "publish-revision').value=result.version.revision" in html
    assert "Sürüm geçmişi" in html
    assert "Set formlarına aktar" in html
    assert "retire-policy-epoch" in html


def test_request_uses_only_the_documented_backend_routes():
    source = PLUGIN.read_text(encoding="utf-8")
    assert '"/v1/eval/datasets?limit=100"' in source
    assert '"/v1/eval/datasets"' in source
    assert '"/versions"' in source
    assert '"/drafts"' in source
    assert '"/cases/import"' in source
    assert '"/publish"' in source
    assert '"/retire"' in source
    assert "RAGTEST_EVAL_API_URL" in source
    assert "owner_actor_id" not in source
