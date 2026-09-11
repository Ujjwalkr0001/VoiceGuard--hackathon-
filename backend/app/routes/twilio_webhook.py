"""
VoiceGuard Backend — Twilio Voice Webhook

Handles incoming/outgoing call setup via Twilio. Returns TwiML that:
1. Bridges the call to the target number via <Dial>
2. Starts a Media Stream (inbound_track only) pointing at our WebSocket endpoint

Twilio hits this endpoint when a call comes in to our Twilio number.
"""

import structlog
from fastapi import APIRouter, Form, Request
from fastapi.responses import Response

from app.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])


@router.post("/voice-webhook")
async def voice_webhook(
    request: Request,
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallSid: str = Form(default=""),
    CallStatus: str = Form(default=""),
):
    """
    Twilio Voice webhook — called when someone dials our Twilio number.

    Returns TwiML that:
    - Bridges the call to the target number
    - Opens a Media Stream (caller's inbound audio only) to our WebSocket
    """
    logger.info(
        "twilio.incoming_call",
        call_sid=CallSid,
        from_number=From,
        to_number=To,
        status=CallStatus,
    )

    # Determine the WebSocket URL for media streaming
    # In production this comes from the ALB domain; in dev, use ngrok or similar
    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/media-stream"

    # Build TwiML response
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{stream_url}" track="inbound_track">
            <Parameter name="caller_number" value="{From}" />
            <Parameter name="called_number" value="{To}" />
        </Stream>
    </Start>
    <Dial callerId="{settings.twilio_phone_number}">
        <Number>{To}</Number>
    </Dial>
</Response>"""

    logger.info(
        "twilio.twiml_generated",
        call_sid=CallSid,
        stream_url=stream_url,
    )

    return Response(content=twiml, media_type="application/xml")


@router.post("/call-status")
async def call_status_callback(
    CallSid: str = Form(default=""),
    CallStatus: str = Form(default=""),
    CallDuration: str = Form(default="0"),
    From: str = Form(default=""),
    To: str = Form(default=""),
):
    """
    Twilio status callback — called when call status changes
    (ringing, in-progress, completed, failed, etc.).
    """
    logger.info(
        "twilio.call_status",
        call_sid=CallSid,
        status=CallStatus,
        duration=CallDuration,
        from_number=From,
        to_number=To,
    )

    # TODO (Phase 9): Trigger session finalization in DynamoDB on 'completed'

    return Response(content="<Response/>", media_type="application/xml")
