"""
VoiceGuard — Twilio SMS Alert Test Script (Step 131)

Sends a test SMS alert to verify end-to-end delivery
via the AlertDispatcher's SMS channel.

Prerequisites:
  1. Set in .env:
     - TWILIO_ACCOUNT_SID
     - TWILIO_AUTH_TOKEN
     - TWILIO_PHONE_NUMBER  (your Twilio sender number)
     - ALERT_SMS_TO          (recipient phone number)
  2. Twilio account must have SMS capability

Usage:
  cd backend
  python -m scripts.test_sms_alert

The script sends a single test SMS and reports success/failure.
"""

import asyncio
import sys

sys.path.insert(0, ".")

from app.config import settings
from app.services.alert_dispatcher import AlertDispatcher


async def main():
    print("📱 VoiceGuard SMS Alert Test")
    print("=" * 40)

    # Preflight checks
    if not settings.twilio_account_sid or settings.twilio_account_sid == "your_account_sid":
        print("❌ TWILIO_ACCOUNT_SID not configured in .env")
        sys.exit(1)

    if not settings.alert_sms_to:
        print("❌ ALERT_SMS_TO not configured in .env")
        sys.exit(1)

    print(f"  From: {settings.twilio_phone_number}")
    print(f"  To:   {settings.alert_sms_to}")
    print()

    dispatcher = AlertDispatcher()

    alert_id = await dispatcher.dispatch_alert(
        call_sid="TEST-SMS-VERIFY",
        caller_number="+910000000000",
        risk_score=91.0,
        risk_level="high",  # SMS only fires for "high"
        signals=[
            "Test SMS — ignore this alert",
            "Simulated high-risk detection",
        ],
    )

    if alert_id:
        print(f"✅ Alert dispatched: {alert_id}")
        print(f"   Check your phone ({settings.alert_sms_to}) for the SMS.")
    else:
        print("❌ Alert was rate-limited or failed. Check logs.")


if __name__ == "__main__":
    asyncio.run(main())
