"""Evaluation dataset and version boundary."""

from pipeline.index.repositories._boundary import BoundedRepository


repository = BoundedRepository(
    name="evaluation",
    operations=frozenset({
        "create_eval_dataset",
        "create_eval_draft",
        "list_eval_datasets",
        "list_eval_versions",
        "publish_eval_version",
        "read_eval_cases",
        "replace_eval_cases",
        "retire_eval_dataset",
    }),
)
