"""Database runtime and tenant-context boundary."""

from pipeline.index.repositories._boundary import BoundedRepository


repository = BoundedRepository(
    name="runtime",
    operations=frozenset({
        "bind_execution_tenant",
        "clear_tenant_context",
        "close_pool",
        "get_conn",
        "get_pool",
        "require_runtime_ready",
        "reset_execution_tenant",
        "resolve_org_identity",
        "schema_is_current",
        "set_tenant_context",
    }),
)
