"""Domain-owned HTTP routers assembled by :mod:`pipeline.api.app`."""
from pipeline.api.routers import chat
from pipeline.api.routers import documents
from pipeline.api.routers import evaluation
from pipeline.api.routers import evidence
from pipeline.api.routers import governance
from pipeline.api.routers import reviews
from pipeline.api.routers import system


ROUTER_MODULES = {
    "governance": governance,
    "evaluation": evaluation,
    "system": system,
    "chat": chat,
    "evidence": evidence,
    "reviews": reviews,
    "documents": documents,
}
