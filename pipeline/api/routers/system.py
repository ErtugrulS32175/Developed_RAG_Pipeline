"""Health, readiness, metrics, and model-discovery route ownership."""
from pipeline.api.routers._factory import domain_router


def new_router():
    return domain_router()


router = domain_router()
