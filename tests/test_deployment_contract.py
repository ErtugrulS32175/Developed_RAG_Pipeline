"""Deployment files must point at the schema that actually ships."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "pipeline" / "index" / "schema.sql"


def test_every_postgres_bootstrap_names_the_real_schema_file():
    assert SCHEMA.is_file()
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup_postgres.sh").read_text(
        encoding="utf-8")

    expected = "pipeline/index/schema.sql"
    assert expected in compose
    assert expected in setup
    assert "pipeline/schema.sql" not in compose
    assert "pipeline/schema.sql" not in setup


def test_the_scope_index_is_an_idempotent_part_of_the_schema():
    text = SCHEMA.read_text(encoding="utf-8")
    statement = (
        "CREATE INDEX IF NOT EXISTS chunks_document_id_idx "
        "ON chunks(document_id);"
    )
    assert text.count(statement) == 1


def test_openwebui_is_immutable_and_forwards_only_a_signed_user_assertion():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "open-webui:v0.11.0@sha256:" in compose
    assert "open-webui:main" not in compose
    assert 'ENABLE_FORWARD_USER_INFO_HEADERS: "true"' in compose
    assert "FORWARD_USER_INFO_HEADER_JWT_SECRET:" in compose
    assert 'FORWARD_USER_INFO_HEADER_JWT_EXPIRES_SECONDS: "60"' in compose
    assert "OPENAI_API_KEY: ${OPENWEBUI_GATEWAY_KEY:?" in compose


def test_the_operator_rule_names_the_cache_that_invalidated_a_real_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "treat the main checkout as\nread-only evidence" in readme
    assert "including `--collect-only`" in readme
    assert "even `.pytest_cache`" in readme
    assert "disposable clone pinned to the run's baseline" in readme
