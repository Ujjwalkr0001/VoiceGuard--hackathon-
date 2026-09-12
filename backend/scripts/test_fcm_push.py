"""
VoiceGuard — FCM Push Notification Test Script (Step 129)

Sends a test FCM push notification to a real Android device
to verify end-to-end delivery.

Prerequisites:
  1. Set FCM_CREDENTIALS_PATH in .env (path to Firebase service account JSON)
  2. Get a device token from your Android app and pass it as an argument

Usage:
  python -m scripts.test_fcm_push <DEVICE_TOKEN>

Example:
  python -m scripts.test_fcm_push "dK1x...your_device_token"
"""

import asyncio
import sys

# Add backend to path
sys.path.insert(0, ".")

from app.services.alert_dispatcher import AlertDispatcher


async def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_fcm_push <DEVICE_TOKEN>")
        print("")
        print("Get your device token from the React Native app's")
        print("Firebase messaging registration.")
        sys.exit(1)

    device_token = sys.argv[1]
    print(f"📱 Sending test FCM push to token: ...{device_token[-12:]}")
    print()

    dispatcher = AlertDispatcher(fcm_device_tokens=[device_token])

    if not dispatcher._fcm_initialized:
        print("❌ FCM not initialized. Check FCM_CREDENTIALS_PATH in .env")
        sys.exit(1)

    alert_id = await dispatcher.dispatch_alert(
        call_sid="TEST-FCM-VERIFY",
        caller_number="+910000000000",
        risk_score=88.5,
        risk_level="high",
        signals=[
            "Test notification — ignore this alert",
            "OTP keyword detected",
            "Unknown caller",
        ],
    )

    if alert_id:
        print(f"✅ Alert dispatched successfully!")
        print(f"   Alert ID: {alert_id}")
        print(f"   Check your Android device for the notification.")
        print()
        print("Expected notification:")
        print('   Title: "⚠️ VoiceGuard Alert"')
        print('   Body:  "Risk Level: HIGH — Test notification — ignore this alert"')
    else:
        print("❌ Alert was rate-limited or failed. Check logs.")


if __name__ == "__main__":
    asyncio.run(main())
