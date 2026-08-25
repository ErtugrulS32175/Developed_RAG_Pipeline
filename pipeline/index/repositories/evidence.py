"""Citation evidence and export-ticket boundary."""

from pipeline.index.repositories._boundary import BoundedRepository


repository = BoundedRepository(
    name="evidence",
    operations=frozenset({
        "consume_evidence_preview_ticket",
        "consume_table_export_ticket",
        "mint_evidence_preview_ticket",
        "mint_table_export_ticket",
        "register_evidence_references",
        "register_table_export",
    }),
)
