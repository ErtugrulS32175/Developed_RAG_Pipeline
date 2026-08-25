"""Closed repository views over the existing database authority.

The repository layer owns no SQL and no authorization decision.  It only
narrows which operations a domain may ask the existing ``db`` module to run.
Looking up the operation at call time deliberately preserves the established
test and instrumentation seam on ``pipeline.index.db``.
"""

from dataclasses import dataclass
from typing import Any

from pipeline.index import db


class RepositoryBoundaryError(AttributeError):
    """Raised when a caller asks a repository for an undeclared operation."""


@dataclass(frozen=True, slots=True)
class BoundedRepository:
    """Expose only a closed set of callable operations for one domain."""

    name: str
    operations: frozenset[str]

    def __getattr__(self, operation_name: str) -> Any:
        if operation_name.startswith("_") or operation_name not in self.operations:
            raise RepositoryBoundaryError(
                f"{self.name} repository does not expose {operation_name!r}"
            )
        operation = getattr(db, operation_name, None)
        if not callable(operation):
            raise RepositoryBoundaryError(
                f"{self.name} repository operation is unavailable: "
                f"{operation_name!r}"
            )
        return operation
