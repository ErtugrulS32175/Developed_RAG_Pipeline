"""Organization, retention, legal-hold, and purge boundary."""

from pipeline.index.repositories._boundary import BoundedRepository


repository = BoundedRepository(
    name="governance",
    operations=frozenset({
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
    }),
)
