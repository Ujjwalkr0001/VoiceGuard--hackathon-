"""
VoiceGuard Backend — Main FastAPI Application

AI-powered real-time voice clone detection for phone calls.
Ingests Twilio Media Streams, runs ML inference, and dispatches alerts.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Configure structured logging first
    from app.logging_config import setup_logging

    setup_logging(
        json_logs=(settings.app_env == "production"),
        log_level="DEBUG" if settings.app_debug else "INFO",
    )

    logger.info(
        "voiceguard.startup",
        version=__version__,
        time=datetime.now(timezone.utc).isoformat(),
    )

    # TODO (Phase 2+): Load ML models into memory here
    # TODO (Phase 6):  Initialize Redis connection pool
    # TODO (Phase 8):  Initialize DynamoDB client

    yield  # ---- app is running ----

    # Cleanup
    logger.info("voiceguard.shutdown")
    # TODO: Close Redis pool, flush pending DynamoDB writes


# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VoiceGuard API",
    description=(
        "Real-time AI-powered voice clone detection. "
        "Ingests Twilio Media Streams, computes rolling risk scores, "
        "and dispatches alerts via FCM push and SMS."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the React Native app and dev tools
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------------------------------------------
# Health check — detailed (with Redis ping) in health.py
# ---------------------------------------------------------------------------
@app.get("/", tags=["system"])
async def root():
    """Root endpoint — service info."""
    return {
        "service": "VoiceGuard API",
        "version": __version__,
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------
from app.health import router as health_router  # noqa: E402
from app.routes.twilio_webhook import router as twilio_webhook_router  # noqa: E402
from app.routes.media_stream import router as media_stream_router  # noqa: E402

app.include_router(health_router)
app.include_router(twilio_webhook_router)
app.include_router(media_stream_router)
# TODO (Phase 10): app.include_router(api_router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
from app.error_handlers import register_exception_handlers  # noqa: E402

register_exception_handlers(app)
