"""Human-review queue and feedback boundary."""

from pipeline.index.repositories._boundary import BoundedRepository


repository = BoundedRepository(
    name="reviews",
    operations=frozenset({
        "create_review_interaction",
        "decide_review_case",
        "list_review_cases",
        "submit_review_feedback",
    }),
)
