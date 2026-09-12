"""
VoiceGuard Backend — REST API Endpoints (Steps 153–165)

Provides the REST API for the mobile app and admin tools:
    - Call History:  List / detail / risk-timeline for past calls
    - Contacts:     Manage trusted phone numbers
    - Config:       View / update runtime thresholds and weights
    - Metrics:      Pipeline performance stats

All endpoints require API key authentication via the
`X-API-Key` header (Step 162).
"""

import logging
import re
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.services.dynamo_client import dynamo_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 162 — API Key Authentication
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Dependency that validates the API key from the X-API-Key header.

    Raises 401 if key is missing or invalid.
    """
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Step 164 — Simple Rate Limiting (in-memory)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Simple token-bucket rate limiter: max `rate` requests per second per key.
    """

    def __init__(self, rate: int = 10):
        self.rate = rate
        self._buckets: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        now = time.time()
        window = self._buckets.setdefault(key, [])
        # Remove entries older than 1 second
        window[:] = [t for t in window if now - t < 1.0]
        if len(window) >= self.rate:
            return False
        window.append(now)
        return True


_rate_limiter = _RateLimiter(rate=10)


async def check_rate_limit(
    request: Request,
    api_key: str = Depends(verify_api_key),
) -> str:
    """Dependency that enforces rate limiting (10 req/s per API key)."""
    if not _rate_limiter.check(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded (10 requests/second)",
        )
    return api_key


# ---------------------------------------------------------------------------
# Step 163 — Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------

# --- Call History ---

class CallSummary(BaseModel):
    """Summary of a single call session (for list view)."""
    call_sid: str
    caller_number: str
    user_id: str
    start_time: float
    end_time: float = 0
    duration_seconds: float = 0
    peak_risk_score: float = 0
    final_verdict: str = "in_progress"


class CallListResponse(BaseModel):
    """Paginated list of call sessions."""
    sessions: list[CallSummary]
    has_more: bool = False
    count: int


class RiskTimelineEntry(BaseModel):
    """Single entry in the risk score timeline."""
    timestamp: float
    score: float
    signals: dict = Field(default_factory=dict)


class CallDetailResponse(BaseModel):
    """Full detail of a single call session."""
    call_sid: str
    caller_number: str
    user_id: str
    start_time: float
    end_time: float = 0
    duration_seconds: float = 0
    peak_risk_score: float = 0
    final_verdict: str = "in_progress"
    risk_score_timeline: list[dict] = Field(default_factory=list)
    alerts_sent: list[dict] = Field(default_factory=list)


class RiskTimelineResponse(BaseModel):
    """Risk score timeline for a call (for charting)."""
    call_sid: str
    timeline: list[dict]
    peak_score: float = 0
    count: int


# --- Contacts ---

class ContactCreate(BaseModel):
    """Request body for adding a trusted contact."""
    phone_number: str = Field(
        ...,
        description="Phone number in E.164 format (e.g. +919876543210)",
        examples=["+919876543210"],
    )
    name: Optional[str] = Field(
        None,
        description="Optional display name for the contact",
        max_length=100,
    )

    @field_validator("phone_number")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not re.match(r"^\+[1-9]\d{6,14}$", v):
            raise ValueError("Phone number must be in E.164 format (e.g. +919876543210)")
        return v


class ContactResponse(BaseModel):
    """A single trusted contact."""
    phone_number: str
    name: Optional[str] = None
    added_at: float = 0


class ContactListResponse(BaseModel):
    """List of trusted contacts."""
    contacts: list[ContactResponse]
    count: int


# --- Config ---

class ConfigResponse(BaseModel):
    """Current runtime configuration."""
    risk_threshold_medium: int
    risk_threshold_high: int
    ensemble_weight_model_a: float
    ensemble_weight_model_b: float
    caller_multiplier_unknown: float
    caller_multiplier_flagged: float
    rolling_window_size: int


class ConfigUpdate(BaseModel):
    """Partial config update (all fields optional)."""
    risk_threshold_medium: Optional[int] = Field(None, ge=0, le=100)
    risk_threshold_high: Optional[int] = Field(None, ge=0, le=100)
    ensemble_weight_model_a: Optional[float] = Field(None, ge=0.0, le=1.0)
    ensemble_weight_model_b: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("risk_threshold_high")
    @classmethod
    def high_ge_medium(cls, v, info):
        if v is not None:
            medium = info.data.get("risk_threshold_medium")
            if medium is not None and v < medium:
                raise ValueError("risk_threshold_high must be >= risk_threshold_medium")
        return v


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1",
    tags=["api"],
    dependencies=[Depends(check_rate_limit)],
)


# ===========================================================================
# 10.1 — Call History API (Steps 153–156)
# ===========================================================================

@router.get("/calls", response_model=CallListResponse)
async def list_calls(
    user_id: str,
    limit: int = 20,
):
    """
    List recent calls for a user, sorted by start_time descending.

    Query params:
        user_id: The enrolled VoiceGuard user ID.
        limit: Max results per page (default 20, max 100).
    """
    limit = min(limit, 100)

    result = await dynamo_client.list_sessions(user_id=user_id, limit=limit)
    sessions = result["sessions"]

    return CallListResponse(
        sessions=[CallSummary(**s) for s in sessions],
        has_more=result["last_evaluated_key"] is not None,
        count=len(sessions),
    )


@router.get("/calls/{call_sid}", response_model=CallDetailResponse)
async def get_call_detail(call_sid: str):
    """
    Get full call session details including risk timeline and alerts.
    """
    session = await dynamo_client.get_session(call_sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Call session '{call_sid}' not found")

    return CallDetailResponse(**session)


@router.get("/calls/{call_sid}/risk-timeline", response_model=RiskTimelineResponse)
async def get_risk_timeline(call_sid: str):
    """
    Return risk scores over time for a call (for live chart / v2).
    """
    session = await dynamo_client.get_session(call_sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Call session '{call_sid}' not found")

    timeline = session.get("risk_score_timeline", [])

    return RiskTimelineResponse(
        call_sid=call_sid,
        timeline=timeline,
        peak_score=session.get("peak_risk_score", 0),
        count=len(timeline),
    )


# ===========================================================================
# 10.2 — Contacts / Trusted Numbers API (Steps 157–159)
# ===========================================================================

# In-memory contacts store (replace with DynamoDB in production)
# Key: user_id → list of ContactResponse
_contacts_store: dict[str, list[dict]] = {}


@router.post("/contacts", response_model=ContactResponse, status_code=201)
async def add_contact(body: ContactCreate, user_id: str = "default_user"):
    """
    Add a trusted phone number. Calls from trusted numbers get
    a lower risk multiplier (1.0x instead of 1.3x).
    """
    contacts = _contacts_store.setdefault(user_id, [])

    # Check for duplicate
    for c in contacts:
        if c["phone_number"] == body.phone_number:
            raise HTTPException(
                status_code=409,
                detail=f"Contact '{body.phone_number}' already exists",
            )

    contact = {
        "phone_number": body.phone_number,
        "name": body.name,
        "added_at": time.time(),
    }
    contacts.append(contact)

    logger.info(
        "api.contact_added",
        extra={
            "user_id": user_id,
            "phone_number": body.phone_number,
        },
    )

    return ContactResponse(**contact)


@router.get("/contacts", response_model=ContactListResponse)
async def list_contacts(user_id: str = "default_user"):
    """List all trusted contacts for a user."""
    contacts = _contacts_store.get(user_id, [])
    return ContactListResponse(
        contacts=[ContactResponse(**c) for c in contacts],
        count=len(contacts),
    )


@router.delete("/contacts/{phone_number}", status_code=204)
async def delete_contact(phone_number: str, user_id: str = "default_user"):
    """Remove a trusted contact by phone number."""
    contacts = _contacts_store.get(user_id, [])

    # URL-decode + in phone numbers (from path encoding)
    phone_number = phone_number.replace("%2B", "+").replace(" ", "+")

    original_len = len(contacts)
    contacts[:] = [c for c in contacts if c["phone_number"] != phone_number]

    if len(contacts) == original_len:
        raise HTTPException(
            status_code=404,
            detail=f"Contact '{phone_number}' not found",
        )

    _contacts_store[user_id] = contacts

    logger.info(
        "api.contact_removed",
        extra={
            "user_id": user_id,
            "phone_number": phone_number,
        },
    )


# ===========================================================================
# 10.3 — Configuration API (Steps 160–161)
# ===========================================================================

@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Return current thresholds, ensemble weights, and multipliers."""
    return ConfigResponse(
        risk_threshold_medium=settings.risk_threshold_medium,
        risk_threshold_high=settings.risk_threshold_high,
        ensemble_weight_model_a=settings.ensemble_weight_model_a,
        ensemble_weight_model_b=settings.ensemble_weight_model_b,
        caller_multiplier_unknown=settings.caller_multiplier_unknown,
        caller_multiplier_flagged=settings.caller_multiplier_flagged,
        rolling_window_size=settings.rolling_window_size,
    )


@router.patch("/config", response_model=ConfigResponse)
async def update_config(body: ConfigUpdate):
    """
    Update thresholds and weights at runtime (for demo tuning).

    Only provided fields are updated; others remain unchanged.
    """
    if body.risk_threshold_medium is not None:
        settings.risk_threshold_medium = body.risk_threshold_medium
    if body.risk_threshold_high is not None:
        settings.risk_threshold_high = body.risk_threshold_high
    if body.ensemble_weight_model_a is not None:
        settings.ensemble_weight_model_a = body.ensemble_weight_model_a
    if body.ensemble_weight_model_b is not None:
        settings.ensemble_weight_model_b = body.ensemble_weight_model_b

    logger.info(
        "api.config_updated",
        extra={
            "medium": settings.risk_threshold_medium,
            "high": settings.risk_threshold_high,
            "weight_a": settings.ensemble_weight_model_a,
            "weight_b": settings.ensemble_weight_model_b,
        },
    )

    return ConfigResponse(
        risk_threshold_medium=settings.risk_threshold_medium,
        risk_threshold_high=settings.risk_threshold_high,
        ensemble_weight_model_a=settings.ensemble_weight_model_a,
        ensemble_weight_model_b=settings.ensemble_weight_model_b,
        caller_multiplier_unknown=settings.caller_multiplier_unknown,
        caller_multiplier_flagged=settings.caller_multiplier_flagged,
        rolling_window_size=settings.rolling_window_size,
    )


# ===========================================================================
# Bonus — Pipeline Metrics endpoint
# ===========================================================================

@router.get("/metrics")
async def get_metrics():
    """
    Return pipeline performance metrics (P50/P95/P99 latencies, counts).
    """
    from app.pipeline.metrics import pipeline_metrics
    return pipeline_metrics.summary()


# ===========================================================================
# Bonus — Model status endpoint
# ===========================================================================

@router.get("/models/status")
async def get_model_status():
    """Return current ML model loading status."""
    from app.pipeline.model_manager import model_manager
    return model_manager.status
