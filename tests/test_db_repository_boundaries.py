"""Architectural gates for the API's bounded database repositories."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pipeline.index import db
from pipeline.index import repositories
from pipeline.index.repositories._boundary import RepositoryBoundaryError


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "pipeline" / "api" / "app.py"
REPOSITORY_ROOT = ROOT / "pipeline" / "index" / "repositories"

EXPECTED_OPERATIONS = {
    "documents": {
        "activate_document_version",
        "active_ingest_job",
        "begin_attempt",
        "cancel_ingest_job",
        "create_collection",
        "delete_collection",
        "delete_tag",
        "document_version_source_digest",
        "enqueue_ingest_job",
        "get_document",
        "get_ingest_job",
        "list_collections",
        "list_document_versions",
        "list_documents",
        "list_tags",
        "replace_document_tags",
        "set_collection_document",
        "set_document_archived",
    },
    "evaluation": {
        "create_eval_dataset",
        "create_eval_draft",
        "list_eval_datasets",
        "list_eval_versions",
        "publish_eval_version",
        "read_eval_cases",
        "replace_eval_cases",
        "retire_eval_dataset",
    },
    "evidence": {
        "consume_evidence_preview_ticket",
        "consume_table_export_ticket",
        "mint_evidence_preview_ticket",
        "mint_table_export_ticket",
        "register_evidence_references",
        "register_table_export",
    },
    "governance": {
        "create_document_legal_hold",
        "get_tenant_retention_policy",
        "list_document_legal_holds",
        "list_document_purge_jobs",
        "list_org_audit_events",
        "list_retention_documents",
        "lock_service_account_redeemer",
        "org_context",
        "org_topology",
        "record_org_decision",
        "release_document_legal_hold",
        "replace_org_topology",
        "schedule_document_purge",
        "update_org_member",
        "update_tenant_retention_policy",
        "visible_org_members",
    },
    "reviews": {
        "create_review_interaction",
        "decide_review_case",
        "list_review_cases",
        "submit_review_feedback",
    },
    "runtime": {
        "bind_execution_tenant",
        "clear_tenant_context",
        "close_pool",
        "get_conn",
        "get_pool",
        "require_runtime_ready",
        "reset_execution_tenant",
        "resolve_org_identity",
        "schema_is_current",
        "set_tenant_context",
    },
}


def _app_tree():
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def test_repository_operation_sets_are_exact_and_do_not_overlap():
    measured = {
        repository.name: set(repository.operations)
        for repository in repositories.DOMAIN_REPOSITORIES
    }
    assert measured == EXPECTED_OPERATIONS

    owners = {}
    for domain, operations in measured.items():
        for operation in operations:
            assert operation not in owners, (
                f"{operation} is owned by both {owners.get(operation)} and {domain}"
            )
            owners[operation] = domain


def test_every_repository_operation_resolves_to_the_single_db_authority():
    for repository in repositories.DOMAIN_REPOSITORIES:
        for operation_name in repository.operations:
            assert getattr(repository, operation_name) is getattr(db, operation_name)


def test_repository_lookup_tracks_the_existing_monkeypatch_seam(monkeypatch):
    sentinel = object()

    def replacement(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(db, "list_documents", replacement)
    assert repositories.documents.list_documents(None, limit=1, offset=0) is sentinel


def test_repository_objects_are_frozen_slotted_and_fail_closed():
    with pytest.raises(FrozenInstanceError):
        repositories.documents.name = "widened"
    assert not hasattr(repositories.documents, "__dict__")
    with pytest.raises(RepositoryBoundaryError):
        repositories.documents.record_org_decision
    with pytest.raises(RepositoryBoundaryError):
        repositories.documents._private


def test_repository_modules_own_no_sql_or_authorization_policy():
    forbidden_import_roots = {
        "pipeline.api.auth",
        "pipeline.api.identity",
        "pipeline.api.org_policy",
    }
    sql_tokens = {
        "select", "insert", "update", "delete", "alter", "create", "drop",
    }

    for path in sorted(REPOSITORY_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        string_words = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_words.update(node.value.lower().split())
        assert imported.isdisjoint(forbidden_import_roots), path.name
        assert string_words.isdisjoint(sql_tokens), path.name


def test_app_uses_repositories_for_operations_and_db_only_for_error_types():
    tree = _app_tree()
    db_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "db"
    }
    assert db_attributes
    assert all(name[0].isupper() for name in db_attributes)

    repository_bindings = {
        "document_repository": repositories.documents,
        "evaluation_repository": repositories.evaluation,
        "evidence_repository": repositories.evidence,
        "governance_repository": repositories.governance,
        "review_repository": repositories.reviews,
        "runtime_repository": repositories.runtime,
    }
    calls = {name: set() for name in repository_bindings}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if isinstance(owner, ast.Name) and owner.id in calls:
            calls[owner.id].add(node.func.attr)

    for binding, repository in repository_bindings.items():
        assert calls[binding]
        assert calls[binding] <= repository.operations
