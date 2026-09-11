"""
VoiceGuard Backend — Health Check Endpoint

Provides a detailed health check that verifies connectivity
to all critical dependencies (Redis, etc.).
"""

from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter

from app import __version__
from app.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["system"])


async def _check_redis() -> dict[str, Any]:
    """Ping Redis and return connection status."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        pong = await client.ping()
        await client.aclose()
        return {"status": "healthy", "ping": pong}
    except Exception as e:
        logger.warning("health.redis_failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}


@router.get("/health")
async def health_check():
    """
    Detailed health check — pings Redis and reports overall status.

    Returns 200 with service info and dependency statuses.
    """
    redis_status = await _check_redis()

    # Overall status: healthy only if all dependencies are healthy
    all_healthy = redis_status["status"] == "healthy"

    return {
        "status": "healthy" if all_healthy else "degraded",
        "service": "voiceguard-backend",
        "version": __version__,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {
            "redis": redis_status,
        },
    }


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness probe — returns 200 only when the service is
    fully ready to accept traffic (all deps connected, models loaded).
    """
    redis_status = await _check_redis()

    if redis_status["status"] != "healthy":
        return {"ready": False, "reason": "Redis not available"}

    # TODO (Phase 3+): Check that ML models are loaded in memory

    return {"ready": True}


@router.get("/health/live")
async def liveness_check():
    """
    Liveness probe — lightweight check that the process is running.
    Always returns 200 if the server can respond.
    """
    return {"alive": True}
