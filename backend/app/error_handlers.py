"""
VoiceGuard Backend — Global Exception Handlers

Catches all VoiceGuard exceptions and unhandled errors, returning
consistent JSON error responses with structured logging.
"""

import traceback
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import VoiceGuardError

logger = structlog.get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI app."""

    @app.exception_handler(VoiceGuardError)
    async def voiceguard_error_handler(request: Request, exc: VoiceGuardError) -> JSONResponse:
        """Handle all VoiceGuard custom exceptions."""
        logger.warning(
            "error.handled",
            error_code=exc.error_code,
            message=exc.message,
            path=str(request.url),
            method=request.method,
            **exc.context,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "error_code": exc.error_code,
                "message": exc.message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors."""
        logger.warning(
            "error.validation",
            message=str(exc),
            path=str(request.url),
            method=request.method,
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "error_code": "VALIDATION_ERROR",
                "message": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all for unhandled exceptions.
        Logs the full traceback but returns a safe message to the client.
        """
        logger.error(
            "error.unhandled",
            error_type=type(exc).__name__,
            message=str(exc),
            path=str(request.url),
            method=request.method,
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
