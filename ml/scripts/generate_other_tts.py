"""
VoiceGuard — Multi-TTS Clip Generator

Generates synthetic speech clips using freely available TTS systems
to diversify the spoof training data across different vocoders.

Supported backends:
  - gTTS (Google Text-to-Speech) — no API key, always available
  - Coqui TTS — local neural TTS, install with: pip install TTS
  - Edge TTS — Microsoft Edge online TTS, install with: pip install edge-tts

Usage:
    python scripts/generate_other_tts.py --backend gtts --count 50
    python scripts/generate_other_tts.py --backend coqui --count 50
    python scripts/generate_other_tts.py --backend edge --count 50
    python scripts/generate_other_tts.py --all --count 50
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "other_tts" / "spoof"

# Shared text corpus — same scam + casual mix for cross-TTS comparison
TEXTS_EN = [
    "Your bank account has been compromised. Please share your OTP immediately.",
    "This is the Reserve Bank of India calling about your KYC verification.",
    "Please transfer the amount to avoid account suspension.",
    "We detected suspicious activity on your credit card.",
    "Your insurance policy requires immediate renewal.",
    "Hello, how are you doing today? Let's discuss the meeting.",
    "The weather has been quite pleasant this week.",
    "I will be arriving at the airport around six in the evening.",
    "Could you please pick up some groceries on your way home?",
    "The project deadline has been extended by two weeks.",
    "Let me know when you are free so we can catch up.",
    "Happy birthday! Wishing you a wonderful year ahead.",
    "The new restaurant serves excellent food, you should try it.",
    "I just finished reading a wonderful book about history.",
    "We should plan a trip to the mountains this weekend.",
]

TEXTS_HI = [
    "Aapka bank account compromise ho gaya hai. OTP share karein.",
    "Yeh Reserve Bank se call hai. KYC verify karna zaroori hai.",
    "Kripya das hazaar rupaye transfer karein.",
    "Aapke account mein suspicious activity detect hui hai.",
    "Namaste, aap kaise hain? Meeting ke baare mein baat karte hain.",
    "Mausam bahut achha hai aaj.",
    "Ghar aate waqt sabzi le aana please.",
    "Janamdin mubarak ho! Shubhkamnayein.",
]


def generate_gtts_clips(count: int) -> int:
    """Generate clips using Google Text-to-Speech (gTTS)."""
    try:
        from gtts import gTTS
    except ImportError:
        print("  [SKIP] gTTS not installed. Run: pip install gtts")
        return 0

    out_dir = OUTPUT_DIR / "gtts"
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = TEXTS_EN + TEXTS_HI
    langs = ["en"] * len(TEXTS_EN) + ["hi"] * len(TEXTS_HI)
    generated = 0

    for i in range(min(count, len(texts))):
        filename = f"gtts_{i:04d}_{langs[i]}.mp3"
        filepath = out_dir / filename

        if filepath.exists():
            print(f"  [{i+1}/{count}] SKIP (exists): {filename}")
            generated += 1
            continue

        try:
            tts = gTTS(text=texts[i], lang=langs[i])
            tts.save(str(filepath))
            print(f"  [{i+1}/{count}] Generated: {filename}")
            generated += 1
        except Exception as e:
            print(f"  [{i+1}/{count}] ERROR: {e}")

    return generated


def generate_edge_tts_clips(count: int) -> int:
    """Generate clips using Microsoft Edge TTS."""
    try:
        import edge_tts
    except ImportError:
        print("  [SKIP] edge-tts not installed. Run: pip install edge-tts")
        return 0

    out_dir = OUTPUT_DIR / "edge_tts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Edge TTS voices
    voices = [
        ("en-IN-NeerjaNeural", "en"),
        ("en-IN-PrabhatNeural", "en"),
        ("hi-IN-SwaraNeural", "hi"),
        ("hi-IN-MadhurNeural", "hi"),
    ]

    texts = TEXTS_EN + TEXTS_HI
    generated = 0

    async def _generate():
        nonlocal generated
        idx = 0
        while generated < count and idx < len(texts) * len(voices):
            text_idx = idx % len(texts)
            voice_idx = (idx // len(texts)) % len(voices)
            voice_name, lang = voices[voice_idx]

            # Match language
            if lang == "en" and text_idx >= len(TEXTS_EN):
                idx += 1
                continue
            if lang == "hi" and text_idx < len(TEXTS_EN):
                idx += 1
                continue

            filename = f"edge_{idx:04d}_{voice_name}.mp3"
            filepath = out_dir / filename

            if filepath.exists():
                generated += 1
                idx += 1
                continue

            try:
                communicate = edge_tts.Communicate(texts[text_idx], voice_name)
                await communicate.save(str(filepath))
                print(f"  [{generated+1}/{count}] Generated: {filename}")
                generated += 1
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

            idx += 1

    asyncio.run(_generate())
    return generated


def generate_coqui_clips(count: int) -> int:
    """Generate clips using Coqui TTS (local neural TTS)."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("  [SKIP] Coqui TTS not installed. Run: pip install TTS")
        return 0

    out_dir = OUTPUT_DIR / "coqui"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a lightweight English model
    print("  Loading Coqui TTS model (first run downloads ~100MB)...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)

    generated = 0
    for i in range(min(count, len(TEXTS_EN))):
        filename = f"coqui_{i:04d}_en.wav"
        filepath = out_dir / filename

        if filepath.exists():
            print(f"  [{i+1}/{count}] SKIP (exists): {filename}")
            generated += 1
            continue

        try:
            tts.tts_to_file(text=TEXTS_EN[i], file_path=str(filepath))
            print(f"  [{i+1}/{count}] Generated: {filename}")
            generated += 1
        except Exception as e:
            print(f"  [{i+1}/{count}] ERROR: {e}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate TTS clips from multiple backends")
    parser.add_argument("--backend", choices=["gtts", "edge", "coqui"],
                        help="TTS backend to use")
    parser.add_argument("--all", action="store_true",
                        help="Run all available backends")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of clips per backend (default: 50)")
    args = parser.parse_args()

    if not args.backend and not args.all:
        parser.print_help()
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    backends = {
        "gtts": ("gTTS (Google)", generate_gtts_clips),
        "edge": ("Edge TTS (Microsoft)", generate_edge_tts_clips),
        "coqui": ("Coqui TTS (Neural)", generate_coqui_clips),
    }

    targets = backends.keys() if args.all else [args.backend]

    for backend_key in targets:
        name, func = backends[backend_key]
        print(f"\n{'='*50}")
        print(f"  {name} — generating up to {args.count} clips")
        print(f"{'='*50}\n")
        count = func(args.count)
        total += count
        print(f"\n  {name}: {count} clips generated")

    print(f"\nTotal clips generated: {total}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
