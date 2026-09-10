"""
api/errors.py
-------------
Global exception handlers that enforce a single, consistent JSON error
shape across the entire production API:

    { "error": "<error_code>", "detail": "<human-readable message>" }

No raw framework error pages, no stack traces, no FastAPI default 422 dumps
are ever returned to the caller.

Register all handlers by calling register_handlers(app) from api/main.py.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("api.errors")


# ---------------------------------------------------------------------------
# Error body helper
# ---------------------------------------------------------------------------

def _body(error_code: str, detail: str) -> dict:
    return {"error": error_code, "detail": detail}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_validation_error(
    request: Request,  # noqa: ARG001
    exc: RequestValidationError,
) -> JSONResponse:
    """Custom 422 handler — replaces FastAPI's default verbose dump."""
    errors = exc.errors()
    parts = []
    for err in errors:
        # Skip the "body" prefix Pydantic includes in loc tuples
        loc_parts = [str(x) for x in err["loc"] if x != "body"]
        loc = " -> ".join(loc_parts) if loc_parts else "request"
        parts.append(f"{loc}: {err['msg']}")
    detail = "; ".join(parts) if parts else "Invalid request payload"
    return JSONResponse(
        status_code=422,
        content=_body("validation_error", detail),
    )


async def _handle_http_exception(
    request: Request,  # noqa: ARG001
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Convert HTTPException (including our 503 sentinel) to the standard shape."""
    if exc.status_code == 503:
        return JSONResponse(
            status_code=503,
            content=_body("service_unavailable", str(exc.detail)),
        )
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content=_body("not_found", str(exc.detail)),
        )
    # Fallback for any other HTTP error raised explicitly in route handlers
    return JSONResponse(
        status_code=exc.status_code,
        content=_body("http_error", str(exc.detail)),
    )


async def _handle_unhandled_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all 500 handler — logs full detail server-side, returns generic message."""
    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content=_body(
            "internal_error",
            "An unexpected error occurred. Please try again or contact support.",
        ),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to a FastAPI application instance."""
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException,  _handle_http_exception)
    app.add_exception_handler(Exception,               _handle_unhandled_exception)
