"""
VoiceGuard Backend — Alert Dispatcher (Steps 125-134)

Responsible for dispatching risk alerts through multiple channels:
  1. AWS SNS topic publish  (→ consumed by FCM Lambda + audit)
  2. Twilio SMS backup      (high-risk only)
  3. FCM push notification  (inline, for simplicity over Lambda)

Includes:
  - Per-call rate limiting (max 1 alert per 60s per call SID)
  - Redis-backed cooldown tracking
  - Structured audit logging of all dispatched alerts
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert payload
# ---------------------------------------------------------------------------

@dataclass
class AlertPayload:
    """Structured alert payload sent to all channels."""
    call_sid: str
    caller_number: str
    risk_score: float
    risk_level: str                   # "medium" or "high"
    signals: List[str]                # top contributing signal descriptions
    timestamp: float = field(default_factory=time.time)
    alert_id: str = ""                # set by dispatcher

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Alert Dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """
    Multi-channel alert dispatcher with rate limiting.

    Usage:
        dispatcher = AlertDispatcher()
        await dispatcher.dispatch_alert(
            call_sid="CA123",
            caller_number="+919876543210",
            risk_score=92.5,
            risk_level="high",
            signals=["OTP requested", "Unknown caller", "High acoustic score"],
        )
    """

    def __init__(self, redis_client=None, fcm_device_tokens: list = None):
        """
        Args:
            redis_client:       Optional async Redis client for cooldown tracking.
            fcm_device_tokens:  List of FCM device tokens to send push notifications to.
        """
        self._redis = redis_client
        self._sns_client = None
        self._twilio_client = None
        self._fcm_initialized = False
        self._fcm_device_tokens = fcm_device_tokens or []
        self._alert_counter = 0

        self._init_fcm()
        logger.info("alert_dispatcher.initialized")

    # ----- lazy AWS/Twilio clients -----

    def _get_sns_client(self):
        """Lazy-init SNS client."""
        if self._sns_client is None:
            self._sns_client = boto3.client(
                "sns",
                region_name=settings.aws_region,
            )
        return self._sns_client

    def _get_twilio_client(self):
        """Lazy-init Twilio REST client."""
        if self._twilio_client is None:
            from twilio.rest import Client
            self._twilio_client = Client(
                settings.twilio_account_sid,
                settings.twilio_auth_token,
            )
        return self._twilio_client

    def _init_fcm(self):
        """Initialize Firebase Admin SDK for FCM push notifications."""
        try:
            import firebase_admin
            from firebase_admin import credentials

            # Only init once globally
            if not firebase_admin._apps:
                cred_path = settings.fcm_credentials_path
                if cred_path:
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                    self._fcm_initialized = True
                    logger.info("alert_dispatcher.fcm_initialized")
                else:
                    logger.warning(
                        "alert_dispatcher.fcm_skipped",
                        extra={"reason": "FCM_CREDENTIALS_PATH not configured"},
                    )
            else:
                self._fcm_initialized = True
        except ImportError:
            logger.warning(
                "alert_dispatcher.fcm_skipped",
                extra={"reason": "firebase-admin package not installed"},
            )
        except Exception as e:
            logger.error(
                "alert_dispatcher.fcm_init_failed",
                extra={"error": str(e)},
            )

    # ----- public API -----

    async def dispatch_alert(
        self,
        call_sid: str,
        caller_number: str,
        risk_score: float,
        risk_level: str,
        signals: List[str],
    ) -> Optional[str]:
        """
        Dispatch a risk alert through all configured channels.

        Steps:
          1. Check per-call cooldown (max 1 alert per 60s)
          2. Build AlertPayload
          3. Publish to SNS topic
          4. If risk_level == "high", send Twilio SMS backup
          5. Log for audit trail

        Args:
            call_sid:       Twilio call SID
            caller_number:  Caller's phone number
            risk_score:     Current risk score (0-100)
            risk_level:     "medium" or "high"
            signals:        List of human-readable signal descriptions

        Returns:
            alert_id if dispatched, None if rate-limited
        """
        # --- Step 132: cooldown check ---
        if await self._is_rate_limited(call_sid):
            logger.info(
                "alert_dispatcher.rate_limited",
                extra={"call_sid": call_sid, "risk_level": risk_level},
            )
            return None

        # Build payload
        self._alert_counter += 1
        alert_id = f"alert-{call_sid}-{self._alert_counter}-{int(time.time())}"

        payload = AlertPayload(
            call_sid=call_sid,
            caller_number=caller_number,
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            signals=signals[:5],  # cap at 5 signals
            alert_id=alert_id,
        )

        # --- Step 126: Publish to SNS ---
        sns_success = await self._publish_to_sns(payload)

        # --- Step 127-128: FCM push notification ---
        fcm_success = await self._send_fcm_push(payload)

        # --- Step 130: SMS backup for high risk only ---
        sms_success = False
        if risk_level == "high":
            sms_success = await self._send_sms_alert(payload)

        # --- Step 133: Record cooldown in Redis ---
        await self._set_cooldown(call_sid)

        # --- Step 134: Audit log ---
        logger.warning(
            "alert_dispatcher.alert_dispatched",
            extra={
                "alert_id": alert_id,
                "call_sid": call_sid,
                "caller_number": caller_number,
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "signals": signals[:5],
                "sns_published": sns_success,
                "fcm_sent": fcm_success,
                "sms_sent": sms_success,
            },
        )

        return alert_id

    # ----- SNS publishing (Step 126) -----

    async def _publish_to_sns(self, payload: AlertPayload) -> bool:
        """
        Publish alert payload to the voiceguard-alerts SNS topic.

        Returns True on success, False on failure (non-fatal).
        """
        topic_arn = settings.sns_topic_arn
        if not topic_arn:
            logger.warning(
                "alert_dispatcher.sns_skipped",
                extra={"reason": "SNS_TOPIC_ARN not configured"},
            )
            return False

        try:
            client = self._get_sns_client()
            response = client.publish(
                TopicArn=topic_arn,
                Subject=f"VoiceGuard Alert: {payload.risk_level.upper()} risk on {payload.call_sid}",
                Message=payload.to_json(),
                MessageAttributes={
                    "risk_level": {
                        "DataType": "String",
                        "StringValue": payload.risk_level,
                    },
                    "call_sid": {
                        "DataType": "String",
                        "StringValue": payload.call_sid,
                    },
                },
            )
            logger.info(
                "alert_dispatcher.sns_published",
                extra={
                    "alert_id": payload.alert_id,
                    "message_id": response.get("MessageId"),
                },
            )
            return True

        except ClientError as e:
            logger.error(
                "alert_dispatcher.sns_failed",
                extra={"alert_id": payload.alert_id, "error": str(e)},
            )
            return False

    # ----- FCM push notification (Steps 127-128) -----

    async def _send_fcm_push(self, payload: AlertPayload) -> bool:
        """
        Send FCM push notification to registered device tokens.

        Notification config:
          - Priority: high
          - Title: "⚠️ VoiceGuard Alert"
          - Body: "Risk Level: {HIGH/MEDIUM} — {top_signal}"
          - Data payload: {type, call_sid, risk_score, signals}
          - Android: high-importance channel for heads-up + vibration

        Returns True if at least one notification sent, False otherwise.
        """
        if not self._fcm_initialized or not self._fcm_device_tokens:
            logger.debug(
                "alert_dispatcher.fcm_skipped",
                extra={
                    "reason": "not initialized" if not self._fcm_initialized
                    else "no device tokens",
                },
            )
            return False

        try:
            from firebase_admin import messaging

            top_signal = payload.signals[0] if payload.signals else "Suspicious activity"

            notification = messaging.Notification(
                title="⚠️ VoiceGuard Alert",
                body=f"Risk Level: {payload.risk_level.upper()} — {top_signal}",
            )

            android_config = messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="voiceguard_high_risk",
                    priority="max",
                    default_vibrate_timings=True,
                    visibility="public",
                ),
            )

            data_payload = {
                "type": "high_risk_call",
                "call_sid": payload.call_sid,
                "risk_score": str(payload.risk_score),
                "risk_level": payload.risk_level,
                "signals": json.dumps(payload.signals),
                "alert_id": payload.alert_id,
            }

            sent_count = 0
            for token in self._fcm_device_tokens:
                try:
                    message = messaging.Message(
                        notification=notification,
                        android=android_config,
                        data=data_payload,
                        token=token,
                    )
                    response = messaging.send(message)
                    sent_count += 1
                    logger.info(
                        "alert_dispatcher.fcm_sent",
                        extra={
                            "alert_id": payload.alert_id,
                            "fcm_message_id": response,
                            "token_suffix": token[-8:],
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "alert_dispatcher.fcm_token_failed",
                        extra={
                            "alert_id": payload.alert_id,
                            "token_suffix": token[-8:],
                            "error": str(e),
                        },
                    )

            return sent_count > 0

        except Exception as e:
            logger.error(
                "alert_dispatcher.fcm_failed",
                extra={"alert_id": payload.alert_id, "error": str(e)},
            )
            return False

    # ----- SMS backup (Step 130) -----

    async def _send_sms_alert(self, payload: AlertPayload) -> bool:
        """
        Send Twilio SMS backup alert for HIGH risk calls only.

        Returns True on success, False on failure (non-fatal).
        """
        user_phone = settings.alert_sms_to
        if not user_phone:
            logger.warning(
                "alert_dispatcher.sms_skipped",
                extra={"reason": "ALERT_SMS_TO not configured"},
            )
            return False

        message_body = (
            f"[VoiceGuard] HIGH RISK detected on your current call "
            f"(score: {payload.risk_score:.0f}). "
            f"The caller's voice shows signs of AI generation. "
            f"Verify the caller's identity before sharing sensitive information."
        )

        try:
            client = self._get_twilio_client()
            msg = client.messages.create(
                body=message_body,
                from_=settings.twilio_phone_number,
                to=user_phone,
            )
            logger.info(
                "alert_dispatcher.sms_sent",
                extra={
                    "alert_id": payload.alert_id,
                    "sms_sid": msg.sid,
                    "to": user_phone,
                },
            )
            return True

        except Exception as e:
            logger.error(
                "alert_dispatcher.sms_failed",
                extra={"alert_id": payload.alert_id, "error": str(e)},
            )
            return False

    # ----- Rate limiting (Steps 132-133) -----

    _COOLDOWN_SECONDS = 60  # max 1 alert per 60s per call SID

    async def _is_rate_limited(self, call_sid: str) -> bool:
        """Check if an alert was sent for this call within the cooldown window."""
        if self._redis is None:
            return False

        key = f"alert:cooldown:{call_sid}"
        try:
            exists = await self._redis.exists(key)
            return bool(exists)
        except Exception as e:
            logger.warning(
                "alert_dispatcher.cooldown_check_failed",
                extra={"call_sid": call_sid, "error": str(e)},
            )
            return False  # fail open — allow alert

    async def _set_cooldown(self, call_sid: str) -> None:
        """Set cooldown flag in Redis with TTL."""
        if self._redis is None:
            return

        key = f"alert:cooldown:{call_sid}"
        try:
            await self._redis.set(key, "1", ex=self._COOLDOWN_SECONDS)
        except Exception as e:
            logger.warning(
                "alert_dispatcher.cooldown_set_failed",
                extra={"call_sid": call_sid, "error": str(e)},
            )
