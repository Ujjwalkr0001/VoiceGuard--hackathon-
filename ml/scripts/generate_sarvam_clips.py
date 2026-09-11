"""
VoiceGuard — Sarvam AI TTS Clip Generator

Generates synthetic speech clips using Sarvam AI's TTS API for use as
spoof samples in the training/evaluation dataset.

Critical for the VoiceGuard use case: the eval set MUST include clips from
the actual vocoder we'll demo against (Sarvam AI), since anti-spoofing
artifacts are vocoder-specific.

Usage:
    # Set SARVAM_API_KEY in your .env or environment
    python scripts/generate_sarvam_clips.py --count 200
    python scripts/generate_sarvam_clips.py --count 50 --language hi-IN
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load .env from backend
load_dotenv(Path(__file__).parent.parent.parent / "backend" / ".env")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sarvam_tts" / "spoof"

# ---- Sarvam AI TTS Configuration ----
SARVAM_API_URL = "https://api.sarvam.ai/text-to-speech"

# Texts to synthesize — mix of scam scenarios, casual speech, and neutral content
# Each entry: (text, language, category)
TEXTS = [
    # --- English: Scam/Social Engineering ---
    ("Your bank account has been compromised. Please share your OTP immediately.", "en-IN", "scam"),
    ("This is the Reserve Bank of India calling. Your KYC needs immediate verification.", "en-IN", "scam"),
    ("Please transfer ten thousand rupees to this account to avoid penalty.", "en-IN", "scam"),
    ("We have detected suspicious activity on your account. Please verify your identity.", "en-IN", "scam"),
    ("Your credit card will be blocked in 24 hours unless you verify your details now.", "en-IN", "scam"),
    ("I am calling from the income tax department. You have outstanding dues.", "en-IN", "scam"),
    ("Please share your Aadhaar number for verification purposes.", "en-IN", "scam"),
    ("Your UPI PIN has been compromised. Please reset it immediately.", "en-IN", "scam"),
    ("This is an urgent matter regarding your insurance policy. Please call back.", "en-IN", "scam"),
    ("We need your account number to process the refund.", "en-IN", "scam"),

    # --- English: Casual/Neutral ---
    ("Hello, how are you doing today? I wanted to discuss our meeting schedule.", "en-IN", "casual"),
    ("The weather has been quite pleasant this week, don't you think?", "en-IN", "casual"),
    ("I'll be arriving at the airport around six in the evening.", "en-IN", "casual"),
    ("Could you please pick up some groceries on your way home?", "en-IN", "casual"),
    ("Let me know when you're free so we can catch up over coffee.", "en-IN", "casual"),
    ("The project deadline has been extended by two weeks.", "en-IN", "casual"),
    ("I just finished reading a wonderful book about Indian history.", "en-IN", "casual"),
    ("We should plan a trip to the mountains this weekend.", "en-IN", "casual"),
    ("The new restaurant near the office serves excellent biryani.", "en-IN", "casual"),
    ("Happy birthday! Wishing you a wonderful year ahead.", "en-IN", "casual"),

    # --- Hindi: Scam/Social Engineering ---
    ("Aapka bank account compromise ho gaya hai. Kripya apna OTP turant share karein.", "hi-IN", "scam"),
    ("Yeh Reserve Bank of India se call hai. Aapka KYC turant verify karna zaroori hai.", "hi-IN", "scam"),
    ("Kripya das hazaar rupaye is account mein transfer karein, nahi toh penalty lagegi.", "hi-IN", "scam"),
    ("Aapke account mein suspicious activity detect hui hai. Kripya apni identity verify karein.", "hi-IN", "scam"),
    ("Aapka credit card 24 ghante mein block ho jayega agar aap verify nahi karte.", "hi-IN", "scam"),
    ("Main income tax department se bol raha hoon. Aap par outstanding dues hain.", "hi-IN", "scam"),
    ("Kripya verification ke liye apna Aadhaar number share karein.", "hi-IN", "scam"),
    ("Aapka UPI PIN compromise ho gaya hai. Turant reset karein.", "hi-IN", "scam"),

    # --- Hindi: Casual/Neutral ---
    ("Namaste, aap kaise hain? Main aaj ki meeting ke baare mein baat karna chahta tha.", "hi-IN", "casual"),
    ("Mausam bahut achha hai aaj. Kya aap bahar ghumne chalein?", "hi-IN", "casual"),
    ("Main shaam ko chhe baje airport pahunch jaunga.", "hi-IN", "casual"),
    ("Ghar aate waqt thoda sabzi le aana please.", "hi-IN", "casual"),
    ("Jab free ho toh bata dena, coffee pe milte hain.", "hi-IN", "casual"),
    ("Project ki deadline do hafte badh gayi hai.", "hi-IN", "casual"),
    ("Janamdin mubarak ho! Bahut saari shubhkamnayein.", "hi-IN", "casual"),
    ("Naye restaurant ka khana bahut achha hai, zaroor try karo.", "hi-IN", "casual"),
]

# Available Sarvam AI voices (update based on actual API documentation)
VOICES = [
    "meera",      # Female Indian English
    "arvind",     # Male Indian English
    "amol",       # Male Hindi
    "pavithra",   # Female Hindi
]


def generate_clip(
    text: str,
    language: str,
    voice: str,
    api_key: str,
    output_path: Path,
) -> bool:
    """
    Generate a single TTS clip using Sarvam AI API.

    Returns True on success, False on failure.
    """
    headers = {
        "Content-Type": "application/json",
        "API-Subscription-Key": api_key,
    }

    payload = {
        "inputs": [text],
        "target_language_code": language,
        "speaker": voice,
        "model": "bulbul:v1",
        "enable_preprocessing": True,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(SARVAM_API_URL, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()

            # Sarvam API returns base64-encoded audio
            if "audios" in result and len(result["audios"]) > 0:
                import base64
                audio_b64 = result["audios"][0]
                audio_bytes = base64.b64decode(audio_b64)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

                return True
            else:
                print(f"  [WARN] No audio in response for: {text[:50]}...")
                return False

    except httpx.HTTPStatusError as e:
        print(f"  [ERROR] HTTP {e.response.status_code}: {text[:50]}...")
        return False
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate Sarvam AI TTS clips")
    parser.add_argument("--count", type=int, default=200,
                        help="Target number of clips to generate")
    parser.add_argument("--language", type=str, default=None,
                        help="Filter by language (e.g., hi-IN, en-IN)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without calling API")
    args = parser.parse_args()

    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: SARVAM_API_KEY not set. Add it to backend/.env or environment.")
        sys.exit(1)

    # Filter texts by language if specified
    texts = TEXTS
    if args.language:
        texts = [(t, l, c) for t, l, c in TEXTS if l == args.language]

    # Generate enough clips to hit the target count by cycling through texts × voices
    clips_to_generate = []
    clip_idx = 0
    while len(clips_to_generate) < args.count:
        for text, lang, category in texts:
            for voice in VOICES:
                if len(clips_to_generate) >= args.count:
                    break
                filename = f"sarvam_{clip_idx:04d}_{voice}_{lang}_{category}.wav"
                clips_to_generate.append({
                    "text": text,
                    "language": lang,
                    "voice": voice,
                    "category": category,
                    "filename": filename,
                })
                clip_idx += 1
            if len(clips_to_generate) >= args.count:
                break

    print(f"\nSarvam AI TTS Generator")
    print(f"  Target clips: {args.count}")
    print(f"  Unique texts: {len(texts)}")
    print(f"  Voices: {len(VOICES)}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would generate:")
        for clip in clips_to_generate[:10]:
            print(f"  {clip['filename']}: {clip['text'][:60]}...")
        if len(clips_to_generate) > 10:
            print(f"  ... and {len(clips_to_generate) - 10} more")
        return

    # Generate clips
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0

    for i, clip in enumerate(clips_to_generate):
        output_path = OUTPUT_DIR / clip["filename"]

        if output_path.exists():
            print(f"  [{i+1}/{len(clips_to_generate)}] SKIP (exists): {clip['filename']}")
            success_count += 1
            continue

        print(f"  [{i+1}/{len(clips_to_generate)}] Generating: {clip['filename']}")
        ok = generate_clip(
            text=clip["text"],
            language=clip["language"],
            voice=clip["voice"],
            api_key=api_key,
            output_path=output_path,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting — be respectful to the API
        time.sleep(0.5)

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(clips_to_generate, f, indent=2)

    print(f"\nDone! Generated {success_count}/{len(clips_to_generate)} clips")
    print(f"  Failures: {fail_count}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
