"""Domain routers own the HTTP surface while the legacy handlers stay stable."""
import ast
from pathlib import Path

from pipeline.api import app as api


APP_SOURCE = Path(api.__file__)
EXPECTED_ENDPOINTS = {
    "governance": {
        "organization_me", "organization_visible_members",
        "organization_topology", "organization_audit_events",
        "replace_organization_topology", "update_organization_membership",
        "organization_retention_policy", "organization_retention_documents",
        "update_organization_retention_policy", "document_legal_holds",
        "create_document_legal_hold", "release_document_legal_hold",
        "document_purge_jobs", "schedule_document_purge",
    },
    "evaluation": {
        "eval_dataset_list", "eval_dataset_create", "eval_version_list",
        "eval_draft_create", "eval_cases_import", "eval_cases_read",
        "eval_version_publish", "eval_dataset_retire",
    },
    "system": {
        "list_models", "oidc_login", "oidc_callback",
        "oidc_browser_session", "oidc_logout", "health",
        "prometheus_metrics", "ready",
    },
    "chat": {"chat_completions"},
    "evidence": {
        "create_evidence_ticket", "preview_evidence",
        "create_export_ticket", "download_export",
    },
    "reviews": {
        "submit_review_feedback", "review_queue", "decide_review_case",
    },
    "documents": {
        "upload_document", "process_document", "enqueue_ingest_job",
        "read_ingest_job", "cancel_ingest_job", "list_documents",
        "create_collection", "list_collections", "list_tags", "delete_tag",
        "delete_collection", "add_collection_document",
        "remove_collection_document", "replace_document_tags",
        "archive_document", "restore_document", "list_document_versions",
        "activate_document_version", "read_document",
    },
}


def test_every_http_handler_has_one_domain_owner():
    observed = {}
    all_routes = []
    for domain, router in api.ROUTERS_BY_DOMAIN.items():
        endpoints = {route.endpoint.__name__ for route in router.routes}
        observed[domain] = endpoints
        all_routes.extend(
            (method, route.path, route.endpoint.__name__)
            for route in router.routes for method in route.methods)
    assert observed == EXPECTED_ENDPOINTS
    assert len(all_routes) == 57
    assert len(all_routes) == len(set(all_routes))


def test_the_application_keeps_only_process_wide_middleware_decorators():
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    app_decorators = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "app"):
                app_decorators.append(decorator.func.attr)
    assert app_decorators == ["middleware", "middleware"]


def test_the_application_assembles_each_domain_router_once():
    assert tuple(api.ROUTERS_BY_DOMAIN.values()) == api.DOMAIN_ROUTERS
    source = APP_SOURCE.read_text(encoding="utf-8")
    assert source.count("app.router.routes.extend(domain_router.routes)") == 1
    runtime_endpoints = {
        route.endpoint.__name__
        for route in api.app.routes
        if getattr(route, "endpoint", None) is not None
    }
    expected = set().union(*EXPECTED_ENDPOINTS.values())
    assert expected <= runtime_endpoints
