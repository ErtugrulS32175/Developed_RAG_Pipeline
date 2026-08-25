"""Bounded domain repositories backed by the single ``db`` authority."""

from pipeline.index.repositories.documents import repository as documents
from pipeline.index.repositories.evaluation import repository as evaluation
from pipeline.index.repositories.evidence import repository as evidence
from pipeline.index.repositories.governance import repository as governance
from pipeline.index.repositories.reviews import repository as reviews
from pipeline.index.repositories.runtime import repository as runtime


DOMAIN_REPOSITORIES = (
    documents,
    evaluation,
    evidence,
    governance,
    reviews,
    runtime,
)

__all__ = (
    "DOMAIN_REPOSITORIES",
    "documents",
    "evaluation",
    "evidence",
    "governance",
    "reviews",
    "runtime",
)
