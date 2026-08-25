"""Document, ingest-job, collection, and tag route ownership."""
from pipeline.api.routers._factory import domain_router


def new_router():
    return domain_router()
