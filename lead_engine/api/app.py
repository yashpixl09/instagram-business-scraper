"""The application: one error shape, one lifespan, one composition root.

    uvicorn --factory lead_engine.api.app:create_app

A factory rather than a module-level `app` on purpose. `create_app()` reads the environment
and opens a connection pool, and doing that at import time means every `import
lead_engine.api.app` -- a test, a linter, a doc build -- pays for it and can fail on it.

EVERY FAILURE IS THE SAME SHAPE
-------------------------------
`{"error": {"code", "message", "details"}}`, from all four handlers below and from the 404
FastAPI would otherwise answer with `{"detail": "Not Found"}`. A client that has to parse
two failure shapes parses neither well: it grows one code path per shape, and the second one
is the one nobody tests.

That is why `RequestValidationError` is handled here at all. FastAPI's default body is a
list of `loc`/`msg`/`type` objects under `detail`, which is a third shape, and it appears
the moment a caller sends `?run_id=not-a-uuid`.

THE UNHANDLED HANDLER RETURNS CANNED TEXT
-----------------------------------------
An exception message is written by whoever raised it, and this codebase is not the only
thing that raises: psycopg puts the failing statement -- parameters and all -- into
`DiagnosticsError`, and a DSN, an API key or a customer's phone number reaches a caller
through nothing more exotic than `str(exc)`. So the 500 body is a constant, the detail goes
to the log, and `tests/test_api.py` plants a secret in an exception message and asserts it
appears nowhere in the response.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import Settings
from ..providers.errors import ProviderError
from .deps import Engine, build_engine
from .routes import router
from .schemas import ApiProblem, envelope

logger = logging.getLogger("lead_engine.api")

TITLE = "lead-engine"
VERSION = "0.1.0"

#: The control panel: static HTML/JS/CSS calling this same API, no build step. Mounted
#: rather than templated, because it is not server-rendered -- every value on the page
#: comes from a `fetch()` to a route this file already defines, so a missing feature is a
#: route to write, never markup to keep in sync with one.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

#: What a 500 says. A constant, so no exception's text can become part of it.
INTERNAL_ERROR_MESSAGE = (
    "The server hit an unexpected error. The failure has been logged; nothing about it is "
    "reported here."
)

#: HTTP statuses FastAPI raises on its own, mapped into this API's vocabulary so that a
#: wrong path and a wrong payload are described the same way.
_HTTP_CODES: dict[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    415: "unsupported_media_type",
    422: "invalid_request",
}


def _problem_response(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=envelope(code, message, details or {}))


async def handle_api_problem(request: Request, exc: Exception) -> JSONResponse:
    """A request this system classified. The status travels with the classification."""
    problem = exc if isinstance(exc, ApiProblem) else ApiProblem(500, "internal_error", "")
    return _problem_response(problem.status, problem.code, problem.message, problem.details)


async def handle_provider_error(request: Request, exc: Exception) -> JSONResponse:
    """An upstream failure, already classified by the shared taxonomy.

    `as_dict()` is the payload the error itself says is safe to publish -- it can carry no
    URL and no response body -- and it is poured into the standard envelope rather than
    returned bare, so a caller still parses one shape. `retryable` rides along in `details`
    because it is the field that decides whether a client should try again.
    """
    if not isinstance(exc, ProviderError):  # pragma: no cover - registered by class
        raise exc
    body = exc.as_dict()
    return _problem_response(
        exc.status,
        str(body["code"]),
        str(body["message"]),
        {"retryable": body["retryable"], "provider": exc.provider},
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """FastAPI's own validation failures, in this API's envelope.

    These come from path and query parameters -- `run_id` that is not a UUID, `limit` out of
    range -- because request *bodies* are validated by `schemas.parse_*`, which produces a
    specific code per field. The `loc` path is kept in `details` since for a query parameter
    it is genuinely the useful part.
    """
    errors = getattr(exc, "errors", lambda: [])()
    fields = [
        {"field": ".".join(str(part) for part in error.get("loc", ())), "message": error.get("msg")}
        for error in errors
    ]
    return _problem_response(
        422,
        "invalid_request",
        "The request could not be validated.",
        {"fields": fields},
    )


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """404s, 405s and anything else Starlette raises before a route is reached."""
    status = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", "") or ""
    return _problem_response(status, _HTTP_CODES.get(status, "http_error"), str(detail))


async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Anything nobody classified. Logged in full, reported as a constant.

    The log line names the method and path so the entry can be found; the exception, its
    message and its traceback go to the logger and nowhere else.
    """
    logger.exception("unhandled error serving %s %s", request.method, request.url.path)
    return _problem_response(500, "internal_error", INTERNAL_ERROR_MESSAGE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the pool on the way up, close it on the way down.

    An Engine injected by a test or by the CLI is used as-is and is NOT closed here: the
    thing that opened the pool is the thing that owns it.
    """
    engine: Engine | None = getattr(app.state, "engine", None)
    owns = engine is None
    if engine is None:
        engine = build_engine(app.state.settings)
        app.state.engine = engine
    # Seeds this key's lifetime allowance if the ledger has never seen it. Idempotent, and
    # it never resets a spend.
    engine.ensure_budget()
    try:
        yield
    finally:
        if owns:
            engine.close()


def create_app(settings: Settings | None = None, *, engine: Engine | None = None) -> FastAPI:
    """Build the application.

    `engine` is the injection seam: the tests hand in one wired to a throwaway schema and a
    provider that raises if it is ever called, which is how "this endpoint does not search"
    is proved rather than asserted.
    """
    app = FastAPI(title=TITLE, version=VERSION, lifespan=lifespan)
    app.state.settings = settings if settings is not None else Settings()
    app.state.engine = engine

    app.add_exception_handler(ApiProblem, handle_api_problem)
    app.add_exception_handler(ProviderError, handle_provider_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    # Last, and the only one that must never say what went wrong.
    app.add_exception_handler(Exception, handle_unexpected)

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    if FRONTEND_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return app
