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
# Per-call state storage (Step 116)
# ---------------------------------------------------------------------------

async def store_risk_state(
    call_sid: str,
    current_score: float,
    peak_score: float,
    chunk_count: int,
) -> None:
    """
    Store / update the per-call risk state hash in Redis.

    Key: call:{call_sid}:risk
    Fields: current_score, peak_score, chunk_count, last_updated
    TTL: 24 hours (reset on each update)
    """
    import time

    r = await get_redis()
    key = risk_key(call_sid)

    await r.hset(key, mapping={
        "current_score": str(round(current_score, 2)),
        "peak_score": str(round(peak_score, 2)),
        "chunk_count": str(chunk_count),
        "last_updated": str(round(time.time(), 3)),
    })
    await r.expire(key, CALL_KEY_TTL)

    logger.debug(
        "redis_client.risk_state_stored",
        extra={
            "call_sid": call_sid,
            "current_score": round(current_score, 2),
            "peak_score": round(peak_score, 2),
            "chunk_count": chunk_count,
        },
    )


async def get_risk_state(call_sid: str) -> Optional[dict]:
    """
    Retrieve the per-call risk state hash from Redis.

    Returns:
        Dict with keys: current_score, peak_score, chunk_count, last_updated
        Or None if the key doesn't exist.
    """
    r = await get_redis()
    data = await r.hgetall(risk_key(call_sid))

    if not data:
        return None

    return {
        "current_score": float(data.get("current_score", 0)),
        "peak_score": float(data.get("peak_score", 0)),
        "chunk_count": int(data.get("chunk_count", 0)),
        "last_updated": float(data.get("last_updated", 0)),
    }


async def append_score_history(
    call_sid: str,
    timestamp: float,
    score: float,
) -> None:
    """
    Append a (timestamp, score) entry to the per-call score history.

    Key: call:{call_sid}:history (sorted set, scored by timestamp)
    TTL: 24 hours
    """
    r = await get_redis()
    key = history_key(call_sid)

    # Sorted set: member = "timestamp:score", score = timestamp (for ordering)
    member = f"{round(timestamp, 3)}:{round(score, 2)}"
    await r.zadd(key, {member: timestamp})
    await r.expire(key, CALL_KEY_TTL)


async def get_score_history(call_sid: str) -> list:
    """
    Retrieve the full score history for a call.

    Returns:
        List of (timestamp, score) tuples, sorted by timestamp ascending.
    """
    r = await get_redis()
    members = await r.zrangebyscore(
        history_key(call_sid),
        min="-inf",
        max="+inf",
    )

    history = []
    for member in members:
        parts = member.split(":", 1)
        if len(parts) == 2:
            history.append({
                "timestamp": float(parts[0]),
                "score": float(parts[1]),
            })

    return history


async def cleanup_call_state(call_sid: str) -> None:
    """
    Delete all Redis keys for a completed call.

    Called when a call ends to free memory immediately
    (rather than waiting for the 24h TTL).
    """
    r = await get_redis()
    await r.delete(risk_key(call_sid), history_key(call_sid))
    logger.info(
        "redis_client.call_state_cleaned",
        extra={"call_sid": call_sid},
    )


# ---------------------------------------------------------------------------
# Convenience readers (Steps 117-118)
# ---------------------------------------------------------------------------

async def get_current_risk(call_sid: str) -> float:
    """
    Read the current risk score for an active call from Redis.

    Args:
        call_sid: Twilio call SID.

    Returns:
        Current smoothed risk score (0-100), or 0.0 if no state exists.
    """
    state = await get_risk_state(call_sid)
    if state is None:
        return 0.0
    return state["current_score"]


async def check_threshold_crossed(call_sid: str) -> Optional[str]:
    """
    Check whether the current risk score exceeds a threshold.

    Reads the current score from Redis and compares it against the
    configured medium and high thresholds.

    Args:
        call_sid: Twilio call SID.

    Returns:
        "high"   — if current_score >= high threshold (default 85)
        "medium" — if current_score >= medium threshold (default 70)
        None     — if below both thresholds or no state exists
    """
    state = await get_risk_state(call_sid)
    if state is None:
        return None

    score = state["current_score"]

    if score >= settings.risk_threshold_high:
        return "high"
    elif score >= settings.risk_threshold_medium:
        return "medium"
    return None


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
