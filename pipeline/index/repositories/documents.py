"""Document inventory, lifecycle, version, and ingest-job boundary."""

from pipeline.index.repositories._boundary import BoundedRepository


repository = BoundedRepository(
    name="documents",
    operations=frozenset({
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
    }),
)
