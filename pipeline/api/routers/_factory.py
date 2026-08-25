"""One constructor so every domain router inherits the public error surface."""
from fastapi import APIRouter

from pipeline.api import errors


def domain_router():
    return APIRouter(responses=dict(errors.ERROR_RESPONSES))
