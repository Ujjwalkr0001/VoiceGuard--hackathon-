"""
VoiceGuard Backend — Async Redis Client (Step 115)

Provides a shared async Redis connection pool for the application.
Used by the RiskEngine to store per-call hot-path session state
(rolling scores, peak scores, threshold history).

Uses `redis.asyncio` (redis-py 4.x+ has built-in async support,
superseding the standalone `aioredis` package).

Usage:
    from app.services.redis_client import get_redis, close_redis

    # At startup
    redis = await get_redis()

    # Read/write
    await redis.set("key", "value")
    val = await redis.get("key")

    # At shutdown
    await close_redis()
"""

import logging
from typing import Optional

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level connection pool singleton
# ---------------------------------------------------------------------------

_redis_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """
    Get or create the shared async Redis connection.

    Uses a connection pool under the hood for efficient multiplexing
    of concurrent requests across call sessions.

    Returns:
        An async Redis client backed by a connection pool.
    """
    global _redis_pool

    if _redis_pool is None:
        logger.info(
            "redis_client.connecting",
            extra={"url": _sanitize_url(settings.redis_url)},
        )
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,       # Return str instead of bytes
            max_connections=20,          # Enough for concurrent call sessions
            socket_timeout=5.0,          # Per-command timeout
            socket_connect_timeout=5.0,  # Initial connection timeout
            retry_on_timeout=True,
        )

        # Verify connection
        try:
            pong = await _redis_pool.ping()
            logger.info(
                "redis_client.connected",
                extra={"ping": pong},
            )
        except Exception as e:
            logger.error(
                "redis_client.connection_failed",
                extra={"error": str(e), "url": _sanitize_url(settings.redis_url)},
            )
            _redis_pool = None
            raise

    return _redis_pool


async def close_redis() -> None:
    """
    Close the Redis connection pool gracefully.

    Call this during application shutdown.
    """
    global _redis_pool

    if _redis_pool is not None:
        logger.info("redis_client.closing")
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("redis_client.closed")


async def ping_redis() -> bool:
    """
    Health check: ping Redis and return True if responsive.

    Returns:
        True if Redis responds to PING, False otherwise.
    """
    try:
        r = await get_redis()
        return await r.ping()
    except Exception as e:
        logger.warning(
            "redis_client.ping_failed",
            extra={"error": str(e)},
        )
        return False


# ---------------------------------------------------------------------------
# Call-session key helpers
# ---------------------------------------------------------------------------

# Key schema:
#   call:{call_sid}:risk     → Hash  (current_score, peak_score, chunk_count, last_updated)
#   call:{call_sid}:history  → Sorted Set  (score keyed by timestamp)
#   TTL: 24 hours

CALL_KEY_TTL = 86400  # 24 hours in seconds


def risk_key(call_sid: str) -> str:
    """Redis key for per-call risk state hash."""
    return f"call:{call_sid}:risk"


def history_key(call_sid: str) -> str:
    """Redis key for per-call score history sorted set."""
    return f"call:{call_sid}:history"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _sanitize_url(url: str) -> str:
    """Mask password in Redis URL for safe logging."""
    if "@" in url:
        # redis://:password@host:port/db → redis://***@host:port/db
        scheme_rest = url.split("://", 1)
        if len(scheme_rest) == 2:
            auth_host = scheme_rest[1].split("@", 1)
            if len(auth_host) == 2:
                return f"{scheme_rest[0]}://***@{auth_host[1]}"
    return url
