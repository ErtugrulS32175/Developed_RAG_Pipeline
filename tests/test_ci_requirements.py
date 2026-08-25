"""The CI files list their packages explicitly; this keeps that list honest.

They used to derive it with `grep -v '^vllm$' requirements.txt`. The filter
stopped matching once the file was stored with CRLF -- silently, because grep
still exits 0 -- and CI installed the multi-GB CUDA serving stack the filter
existed to keep out, failing before a single test ran. An explicit list cannot
fail silently, but it can fall behind requirements.txt, so that is what is
checked here: a dependency added to the pipeline and not to CI means CI is
testing a different tree than the one that ships.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CI_FILES = [ROOT / ".github" / "workflows" / "tests.yml", ROOT / ".gitlab-ci.yml"]

# Deliberately not installed in CI: vLLM serves models, nothing under test
# imports it, and it pins torch to an exact version this file does not.
CI_OMITS = {"vllm"}


def required_packages():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


@pytest.mark.parametrize("ci_file", CI_FILES, ids=lambda p: p.name)
def test_ci_installs_every_requirement(ci_file):
    text = ci_file.read_text(encoding="utf-8")
    missing = sorted(p for p in required_packages() - CI_OMITS if p not in text)
    assert not missing, f"{ci_file.name} bu paketleri kurmuyor: {missing}"


@pytest.mark.parametrize("ci_file", CI_FILES, ids=lambda p: p.name)
def test_ci_does_not_install_the_requirements_file_wholesale(ci_file):
    """`-r requirements.txt` would quietly pull vLLM back in."""
    assert "-r requirements.txt" not in ci_file.read_text(encoding="utf-8")


def test_the_omitted_package_is_actually_in_requirements():
    """Guards the guard: if vLLM leaves requirements.txt, CI_OMITS is stale and
    the exclusion above stops meaning anything."""
    assert CI_OMITS <= required_packages()


@pytest.mark.parametrize("ci_file", CI_FILES, ids=lambda p: p.name)
def test_ci_checks_openapi_compatibility_and_the_generated_client(ci_file):
    text = ci_file.read_text(encoding="utf-8")
    for command in (
            "npm ci",
            "scripts/export_openapi.py --check contracts/openapi.json",
            "scripts/check_openapi_compat.py contracts/openapi.v1.json",
            "npm run api:generate",
            "git diff --exit-code -- clients/typescript/src/schema.ts",
            "npm run typecheck"):
        assert command in text


@pytest.mark.parametrize("ci_file", CI_FILES, ids=lambda p: p.name)
def test_eval_governance_uses_real_postgresql_in_every_ci(ci_file):
    text = ci_file.read_text(encoding="utf-8")
    assert "pgvector/pgvector:pg17" in text
    assert "RAGTEST_EVAL_PG_DSN" in text
    assert "RAGTEST_PG_TEST_DSN" in text
    assert "RAGTEST_P0_GATE" in text
