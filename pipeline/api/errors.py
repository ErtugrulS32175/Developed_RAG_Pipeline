"""One content-free error vocabulary with a bounded legacy-detail bridge."""
from types import MappingProxyType

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from pipeline.api import contracts


ERROR_CONTRACT_VERSION = 1
ERROR_CODE_BY_STATUS = MappingProxyType({
    400: "invalid_request",
    401: "authentication_required",
    403: "permission_denied",
    404: "resource_not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_failed",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_failed",
    503: "service_unavailable",
})
ERROR_RESPONSES = MappingProxyType({
    status: {
        "model": contracts.ErrorResponse,
        "description": f"Closed error response ({code})",
    }
    for status, code in ERROR_CODE_BY_STATUS.items()
})


def _request_id(request):
    value = getattr(request.state, "request_id", None)
    if (type(value) is str and len(value) == 8
            and all(char in "0123456789abcdef" for char in value)):
        return value
    return "unavailable"


def _body(request, status_code, detail):
    code = ERROR_CODE_BY_STATUS.get(status_code, "request_failed")
    return contracts.ErrorResponse(
        error={
            "version": ERROR_CONTRACT_VERSION,
            "code": code,
            "request_id": _request_id(request),
        },
        detail=detail,
    ).model_dump(mode="json")


def _response(request, status_code, detail, *, headers=None):
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(_body(request, status_code, detail)),
        headers=response_headers,
    )


async def http_exception(request, error):
    return _response(
        request,
        error.status_code,
        error.detail,
        headers=error.headers,
    )


async def request_validation(request, error):
    issues = [
        {
            "type": issue["type"],
            "loc": list(issue["loc"]),
            "msg": issue["msg"],
        }
        for issue in error.errors()
    ]
    return _response(request, 422, issues)


async def unexpected_exception(request, _error):
    return _response(request, 500, "internal server error")


def install(app):
    """Bind every HTTP failure road to the same response vocabulary."""
    app.add_exception_handler(StarletteHTTPException, http_exception)
    app.add_exception_handler(RequestValidationError, request_validation)
    app.add_exception_handler(Exception, unexpected_exception)
