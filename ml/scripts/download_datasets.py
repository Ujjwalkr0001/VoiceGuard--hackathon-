"""
VoiceGuard — Dataset Download & Setup Script

Downloads and organizes all required datasets for training the
voice clone detection models.

IMPORTANT: Some datasets require manual registration before downloading.
Follow the instructions for each dataset below.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --asvspoof2019
    python scripts/download_datasets.py --setup-dirs-only
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def setup_directory_structure():
    """Create the expected directory layout for all datasets."""
    dirs = [
        # ASVspoof 2019 LA
        DATA_DIR / "asvspoof2019" / "LA" / "train" / "bonafide",
        DATA_DIR / "asvspoof2019" / "LA" / "train" / "spoof",
        DATA_DIR / "asvspoof2019" / "LA" / "dev" / "bonafide",
        DATA_DIR / "asvspoof2019" / "LA" / "dev" / "spoof",
        DATA_DIR / "asvspoof2019" / "LA" / "eval" / "bonafide",
        DATA_DIR / "asvspoof2019" / "LA" / "eval" / "spoof",
        # ASVspoof 2021 DF
        DATA_DIR / "asvspoof2021" / "DF",
        # In-the-Wild
        DATA_DIR / "in_the_wild" / "bonafide",
        DATA_DIR / "in_the_wild" / "spoof",
        # Sarvam AI TTS generated clips
        DATA_DIR / "sarvam_tts" / "spoof",
        # Other TTS systems (Coqui, Bark, etc.)
        DATA_DIR / "other_tts" / "spoof",
        # Combined dataset (created by prepare_dataset.py)
        DATA_DIR / "combined" / "train" / "bonafide",
        DATA_DIR / "combined" / "train" / "spoof",
        DATA_DIR / "combined" / "eval" / "bonafide",
        DATA_DIR / "combined" / "eval" / "spoof",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.relative_to(DATA_DIR.parent)}")

    print(f"\nDirectory structure created at: {DATA_DIR}")


def print_download_instructions():
    """Print manual download instructions for each dataset."""
    instructions = """
╔══════════════════════════════════════════════════════════════════╗
║              VoiceGuard — Dataset Download Guide                ║
╚══════════════════════════════════════════════════════════════════╝

1. ASVspoof 2019 LA  (Required — primary training data)
   ─────────────────────────────────────────────────────
   • Register at: https://www.asvspoof.org/
   • Download: LA.zip from ASVspoof 2019 challenge page
   • Contains: ~25,000 bonafide + ~108,000 spoof utterances
   • Extract to: ml/data/asvspoof2019/LA/
   • After extraction, run: python scripts/prepare_dataset.py --organize-asvspoof2019

2. ASVspoof 2021 DF  (Recommended — cross-dataset evaluation)
   ─────────────────────────────────────────────────────
   • Download from: https://www.asvspoof.org/
   • Contains: Deepfake track audio with latest TTS/VC attacks
   • Extract to: ml/data/asvspoof2021/DF/

3. In-the-Wild Deepfake Audio  (Recommended — real-world evaluation)
   ─────────────────────────────────────────────────────
   • Paper: "In-the-Wild Audio Deepfake Detection"
   • Download from: https://deepfake-demo.aisec.fraunhofer.de/in_the_wild
   • Extract to: ml/data/in_the_wild/

4. Sarvam AI TTS Clips  (Critical — must test against demo vocoder)
   ─────────────────────────────────────────────────────
   • Generate using: python scripts/generate_sarvam_clips.py
   • Requires: SARVAM_API_KEY in .env
   • Output: ml/data/sarvam_tts/spoof/

5. Other TTS Systems  (Optional — diversity)
   ─────────────────────────────────────────────────────
   • Generate using Coqui TTS, Bark, etc.
   • Place in: ml/data/other_tts/spoof/
"""
    print(instructions)


def check_dataset_status():
    """Check which datasets are present and report status."""
    datasets = {
        "ASVspoof 2019 LA": DATA_DIR / "asvspoof2019" / "LA",
        "ASVspoof 2021 DF": DATA_DIR / "asvspoof2021" / "DF",
        "In-the-Wild": DATA_DIR / "in_the_wild",
        "Sarvam AI TTS": DATA_DIR / "sarvam_tts",
        "Other TTS": DATA_DIR / "other_tts",
    }

    print("\n📊 Dataset Status:")
    print("─" * 50)
    for name, path in datasets.items():
        if path.exists():
            # Count audio files
            audio_count = sum(
                1 for f in path.rglob("*")
                if f.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
            )
            if audio_count > 0:
                print(f"  ✅ {name}: {audio_count} audio files")
            else:
                print(f"  ⚠️  {name}: directory exists but no audio files found")
        else:
            print(f"  ❌ {name}: not downloaded")
    print()


def main():
    parser = argparse.ArgumentParser(description="VoiceGuard dataset setup")
    parser.add_argument("--setup-dirs-only", action="store_true",
                        help="Only create directory structure")
    parser.add_argument("--status", action="store_true",
                        help="Check which datasets are present")
    parser.add_argument("--instructions", action="store_true",
                        help="Print download instructions")
    parser.add_argument("--all", action="store_true",
                        help="Setup dirs + print instructions + check status")

    args = parser.parse_args()

    if args.all or args.setup_dirs_only or not any(vars(args).values()):
        print("\n📁 Setting up directory structure...\n")
        setup_directory_structure()

    if args.all or args.instructions:
        print_download_instructions()

    if args.all or args.status:
        check_dataset_status()


if __name__ == "__main__":
    main()
