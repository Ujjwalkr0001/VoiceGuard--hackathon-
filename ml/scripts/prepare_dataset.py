"""
VoiceGuard — Dataset Preprocessing & Preparation Script

Collects audio from all source datasets (ASVspoof 2019, ASVspoof 2021,
In-the-Wild, Sarvam TTS, Other TTS), preprocesses them to a uniform format,
and splits into stratified train/eval partitions.

Preprocessing steps:
  1. Resample to 16 kHz mono
  2. Trim leading/trailing silence
  3. Normalize amplitude (peak normalization)
  4. Stratified train/eval split (80/20 by default)

Outputs:
  - Preprocessed WAV files in ml/data/train/{bonafide,spoof}/ and ml/data/eval/{bonafide,spoof}/
  - Manifest CSV: ml/data/manifest.csv  (path, label, source, duration)

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --eval-ratio 0.2 --seed 42
    python scripts/prepare_dataset.py --dry-run
    python scripts/prepare_dataset.py --organize-asvspoof2019
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# ---- Directory Layout ----
ML_DIR = Path(__file__).parent.parent
DATA_DIR = ML_DIR / "data"

# Source dataset directories
SOURCE_DIRS = {
    "asvspoof2019_train_bonafide": DATA_DIR / "asvspoof2019" / "LA" / "train" / "bonafide",
    "asvspoof2019_train_spoof":    DATA_DIR / "asvspoof2019" / "LA" / "train" / "spoof",
    "asvspoof2019_dev_bonafide":   DATA_DIR / "asvspoof2019" / "LA" / "dev" / "bonafide",
    "asvspoof2019_dev_spoof":      DATA_DIR / "asvspoof2019" / "LA" / "dev" / "spoof",
    "asvspoof2019_eval_bonafide":  DATA_DIR / "asvspoof2019" / "LA" / "eval" / "bonafide",
    "asvspoof2019_eval_spoof":     DATA_DIR / "asvspoof2019" / "LA" / "eval" / "spoof",
    "asvspoof2021_df":             DATA_DIR / "asvspoof2021" / "DF",
    "in_the_wild_bonafide":        DATA_DIR / "in_the_wild" / "bonafide",
    "in_the_wild_spoof":           DATA_DIR / "in_the_wild" / "spoof",
    "sarvam_tts":                  DATA_DIR / "sarvam_tts" / "spoof",
    "other_tts":                   DATA_DIR / "other_tts" / "spoof",
}

# Output directories
OUTPUT_TRAIN_BONAFIDE = DATA_DIR / "train" / "bonafide"
OUTPUT_TRAIN_SPOOF    = DATA_DIR / "train" / "spoof"
OUTPUT_EVAL_BONAFIDE  = DATA_DIR / "eval" / "bonafide"
OUTPUT_EVAL_SPOOF     = DATA_DIR / "eval" / "spoof"
MANIFEST_PATH         = DATA_DIR / "manifest.csv"

# Audio file extensions to scan
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}

# Target format
TARGET_SR = 16000
TARGET_CHANNELS = 1  # mono


def _lazy_import_audio_libs():
    """Lazy-import heavy audio libraries so --dry-run and --help are fast."""
    global librosa, sf
    try:
        import librosa as _librosa
        import soundfile as _sf
        librosa = _librosa
        sf = _sf
    except ImportError as e:
        print(f"ERROR: Missing audio library: {e}")
        print("Install with:  pip install librosa soundfile")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Audio preprocessing functions
# ---------------------------------------------------------------------------

def resample_to_mono(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Resample audio to target sample rate and convert to mono.

    Args:
        audio: Input audio array. Shape (samples,) or (channels, samples).
        orig_sr: Original sample rate.
        target_sr: Target sample rate (default 16000).

    Returns:
        Mono audio at target sample rate as float32 numpy array.
    """
    # Convert to mono if stereo/multi-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Resample if needed
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    return audio.astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int = TARGET_SR,
                 top_db: int = 30, frame_length: int = 2048,
                 hop_length: int = 512) -> np.ndarray:
    """
    Trim leading and trailing silence from audio using librosa.

    Args:
        audio: Input mono audio array.
        sr: Sample rate.
        top_db: Threshold in dB below peak to consider as silence.
        frame_length: Frame length for energy computation.
        hop_length: Hop length for energy computation.

    Returns:
        Trimmed audio array. Returns original if trimming would produce
        fewer than 1600 samples (0.1s at 16kHz).
    """
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # Safety: don't trim to near-nothing
    min_samples = int(0.1 * sr)  # 100ms minimum
    if len(trimmed) < min_samples:
        return audio

    return trimmed


def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak-normalize audio so max absolute amplitude equals target_peak.

    Args:
        audio: Input audio array.
        target_peak: Target peak amplitude (0.0 to 1.0).

    Returns:
        Normalized audio array. Returns original if audio is silence (max < 1e-8).
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        # Audio is essentially silence — skip normalization
        return audio
    return audio * (target_peak / peak)


def preprocess_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SR,
) -> Optional[float]:
    """
    Load, preprocess, and save a single audio file.

    Preprocessing: resample → mono → trim silence → normalize amplitude.

    Args:
        input_path: Path to source audio file.
        output_path: Path to save processed WAV.
        target_sr: Target sample rate.

    Returns:
        Duration in seconds of the processed audio, or None on failure.
    """
    try:
        # Load audio (librosa always loads as float32)
        audio, sr = librosa.load(str(input_path), sr=None, mono=False)

        # Step 1: Resample to target SR + mono
        audio = resample_to_mono(audio, sr, target_sr)

        # Step 2: Trim silence
        audio = trim_silence(audio, target_sr)

        # Step 3: Normalize amplitude
        audio = normalize_amplitude(audio)

        # Validate: skip very short clips (< 0.3s)
        duration = len(audio) / target_sr
        if duration < 0.3:
            return None

        # Save as 16-bit PCM WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, target_sr, subtype="PCM_16")

        return duration

    except Exception as e:
        print(f"  [WARN] Failed to process {input_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Dataset scanning and collection
# ---------------------------------------------------------------------------

def _determine_label(source_key: str) -> str:
    """Determine 'bonafide' or 'spoof' label from source directory key."""
    if "bonafide" in source_key:
        return "bonafide"
    elif "spoof" in source_key or "tts" in source_key:
        return "spoof"
    else:
        # For ambiguous sources like asvspoof2021_df, default to unknown
        # (these need a metadata file or separate handling)
        return "unknown"


def _determine_source_name(source_key: str) -> str:
    """Extract a human-readable source name from the source key."""
    if source_key.startswith("asvspoof2019"):
        return "asvspoof2019"
    elif source_key.startswith("asvspoof2021"):
        return "asvspoof2021"
    elif source_key.startswith("in_the_wild"):
        return "in_the_wild"
    elif source_key.startswith("sarvam"):
        return "sarvam_tts"
    elif source_key.startswith("other"):
        return "other_tts"
    return source_key


def scan_source_datasets() -> List[dict]:
    """
    Scan all source dataset directories and collect audio file metadata.

    Returns:
        List of dicts with keys: source_path, source_key, label, source_name
    """
    files = []

    for source_key, source_dir in SOURCE_DIRS.items():
        if not source_dir.exists():
            continue

        label = _determine_label(source_key)
        source_name = _determine_source_name(source_key)

        for f in sorted(source_dir.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file():
                files.append({
                    "source_path": f,
                    "source_key": source_key,
                    "label": label,
                    "source_name": source_name,
                })

    return files


def _generate_output_filename(source_path: Path, source_name: str, idx: int) -> str:
    """
    Generate a unique output filename based on source + hash to avoid collisions.

    Format: {source}_{idx:06d}_{hash8}.wav
    """
    # Use a hash of the original path for uniqueness
    path_hash = hashlib.md5(str(source_path).encode()).hexdigest()[:8]
    stem = source_path.stem[:30]  # Truncate long filenames
    return f"{source_name}_{idx:06d}_{stem}_{path_hash}.wav"


# ---------------------------------------------------------------------------
# Stratified train/eval split
# ---------------------------------------------------------------------------

def stratified_split(
    files: List[dict],
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Split files into train/eval with stratification by (label, source_name).

    Ensures proportional representation of each label × source combination
    in both train and eval sets.

    Args:
        files: List of file metadata dicts.
        eval_ratio: Fraction to hold out for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        (train_files, eval_files) tuple
    """
    rng = np.random.RandomState(seed)

    # Group by (label, source_name)
    groups = {}
    for f in files:
        key = (f["label"], f["source_name"])
        groups.setdefault(key, []).append(f)

    train_files = []
    eval_files = []

    for key, group_files in sorted(groups.items()):
        # Shuffle deterministically within each group
        indices = np.arange(len(group_files))
        rng.shuffle(indices)

        n_eval = max(1, int(len(group_files) * eval_ratio))
        eval_indices = set(indices[:n_eval])

        for i, f in enumerate(group_files):
            if i in eval_indices:
                eval_files.append(f)
            else:
                train_files.append(f)

    return train_files, eval_files


# ---------------------------------------------------------------------------
# Manifest CSV
# ---------------------------------------------------------------------------

def write_manifest(
    records: List[dict],
    manifest_path: Path,
) -> None:
    """
    Write a manifest CSV with columns: path, label, source, duration.

    Args:
        records: List of dicts with keys: output_path, label, source_name, duration
        manifest_path: Path to write the CSV file.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "label", "source", "duration"])

        for rec in sorted(records, key=lambda r: str(r["output_path"])):
            # Use relative path from ml/data/ for portability
            try:
                rel_path = rec["output_path"].relative_to(DATA_DIR)
            except ValueError:
                rel_path = rec["output_path"]

            writer.writerow([
                str(rel_path),
                rec["label"],
                rec["source_name"],
                f"{rec['duration']:.3f}",
            ])

    print(f"\n📄 Manifest written: {manifest_path}")
    print(f"   Total records: {len(records)}")


# ---------------------------------------------------------------------------
# ASVspoof 2019 organization helper
# ---------------------------------------------------------------------------

def organize_asvspoof2019():
    """
    Organize raw ASVspoof 2019 LA dataset into bonafide/spoof directories.

    Expects the standard ASVspoof 2019 layout with protocol files:
      - ASVspoof2019.LA.cm.train.trn.txt
      - ASVspoof2019.LA.cm.dev.trl.txt
      - ASVspoof2019.LA.cm.eval.trl.txt

    Each protocol line: SPEAKER_ID AUDIO_FILE_ID - ATTACK_TYPE LABEL
    where LABEL is 'bonafide' or 'spoof'.
    """
    la_dir = DATA_DIR / "asvspoof2019" / "LA"

    # Map partition names to their protocol + audio directories
    partitions = {
        "train": {
            "protocol": la_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt",
            "audio": la_dir / "ASVspoof2019_LA_train" / "flac",
        },
        "dev": {
            "protocol": la_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
            "audio": la_dir / "ASVspoof2019_LA_dev" / "flac",
        },
        "eval": {
            "protocol": la_dir / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt",
            "audio": la_dir / "ASVspoof2019_LA_eval" / "flac",
        },
    }

    for part_name, part_info in partitions.items():
        protocol_path = part_info["protocol"]
        audio_dir = part_info["audio"]

        if not protocol_path.exists():
            print(f"  ⚠️  Protocol file not found: {protocol_path}")
            print(f"     Trying alternative paths...")

            # Try common alternative layouts
            alt_paths = [
                la_dir / f"ASVspoof2019.LA.cm.{part_name}.trn.txt",
                la_dir / "protocols" / f"ASVspoof2019.LA.cm.{part_name}.trn.txt",
            ]
            found = False
            for alt in alt_paths:
                if alt.exists():
                    protocol_path = alt
                    found = True
                    break

            if not found:
                print(f"  ❌ Skipping {part_name} — no protocol file found")
                continue

        if not audio_dir.exists():
            print(f"  ⚠️  Audio directory not found: {audio_dir}")
            continue

        print(f"\n  Organizing {part_name} partition...")
        bonafide_dir = la_dir / part_name / "bonafide"
        spoof_dir = la_dir / part_name / "spoof"
        bonafide_dir.mkdir(parents=True, exist_ok=True)
        spoof_dir.mkdir(parents=True, exist_ok=True)

        bonafide_count = 0
        spoof_count = 0
        missing_count = 0

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # Format: SPEAKER_ID AUDIO_ID - ATTACK_TYPE LABEL
                audio_id = parts[1]
                label = parts[4]  # 'bonafide' or 'spoof'

                # Find the audio file
                audio_file = audio_dir / f"{audio_id}.flac"
                if not audio_file.exists():
                    # Try .wav
                    audio_file = audio_dir / f"{audio_id}.wav"
                    if not audio_file.exists():
                        missing_count += 1
                        continue

                # Create symlink or copy to organized directory
                target_dir = bonafide_dir if label == "bonafide" else spoof_dir
                target_path = target_dir / audio_file.name

                if not target_path.exists():
                    try:
                        # Try symlink first (saves disk space)
                        target_path.symlink_to(audio_file.resolve())
                    except (OSError, NotImplementedError):
                        # Fall back to copy on Windows or if symlinks not supported
                        shutil.copy2(str(audio_file), str(target_path))

                    if label == "bonafide":
                        bonafide_count += 1
                    else:
                        spoof_count += 1

        print(f"    ✅ {part_name}: {bonafide_count} bonafide, {spoof_count} spoof")
        if missing_count > 0:
            print(f"    ⚠️  {missing_count} audio files not found")


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    eval_ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
    max_files: Optional[int] = None,
) -> None:
    """
    Run the full preprocessing pipeline:
      1. Scan all source datasets
      2. Stratified train/eval split
      3. Preprocess (resample, trim, normalize)
      4. Save to output directories
      5. Write manifest CSV

    Args:
        eval_ratio: Fraction of data for evaluation.
        seed: Random seed for reproducibility.
        dry_run: If True, only scan and report — don't process.
        max_files: If set, limit total files processed (for testing).
    """
    print("\n" + "=" * 60)
    print("  VoiceGuard — Dataset Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Scan source datasets
    print("\n📂 Scanning source datasets...")
    all_files = scan_source_datasets()

    if not all_files:
        print("\n❌ No audio files found in any source directory!")
        print("   Make sure you've downloaded datasets first.")
        print("   Run: python scripts/download_datasets.py --instructions")
        return

    # Filter out unknowns
    labeled_files = [f for f in all_files if f["label"] != "unknown"]
    unknown_files = [f for f in all_files if f["label"] == "unknown"]

    # Report what was found
    print(f"\n📊 Source dataset summary:")
    source_counts = {}
    for f in all_files:
        key = (f["source_name"], f["label"])
        source_counts[key] = source_counts.get(key, 0) + 1

    for (source, label), count in sorted(source_counts.items()):
        print(f"   {source:20s} | {label:10s} | {count:>7,} files")

    print(f"\n   Total labeled files:   {len(labeled_files):>7,}")
    if unknown_files:
        print(f"   Unknown label (skipped): {len(unknown_files):>5,}")

    if max_files and len(labeled_files) > max_files:
        print(f"\n   ⚠️  Limiting to {max_files} files (--max-files)")
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(labeled_files), max_files, replace=False)
        labeled_files = [labeled_files[i] for i in sorted(indices)]

    # Step 2: Stratified split
    print(f"\n🔀 Stratified split (eval_ratio={eval_ratio}, seed={seed})...")
    train_files, eval_files = stratified_split(labeled_files, eval_ratio, seed)

    n_train_b = sum(1 for f in train_files if f["label"] == "bonafide")
    n_train_s = sum(1 for f in train_files if f["label"] == "spoof")
    n_eval_b  = sum(1 for f in eval_files if f["label"] == "bonafide")
    n_eval_s  = sum(1 for f in eval_files if f["label"] == "spoof")

    print(f"   Train: {len(train_files):>7,} ({n_train_b:,} bonafide, {n_train_s:,} spoof)")
    print(f"   Eval:  {len(eval_files):>7,} ({n_eval_b:,} bonafide, {n_eval_s:,} spoof)")

    if dry_run:
        print("\n[DRY RUN] No files will be processed or written.")
        print("   Remove --dry-run to execute preprocessing.")
        return

    # Step 3: Preprocess and save
    if not dry_run:
        _lazy_import_audio_libs()

    manifest_records = []

    # Process train set
    print(f"\n🔧 Processing train set ({len(train_files)} files)...")
    for i, fmeta in enumerate(train_files):
        output_dir = OUTPUT_TRAIN_BONAFIDE if fmeta["label"] == "bonafide" else OUTPUT_TRAIN_SPOOF
        output_name = _generate_output_filename(fmeta["source_path"], fmeta["source_name"], i)
        output_path = output_dir / output_name

        if output_path.exists():
            # Already processed — just read duration for manifest
            try:
                info = sf.info(str(output_path))
                duration = info.duration
            except Exception:
                duration = 0.0
        else:
            duration = preprocess_audio_file(fmeta["source_path"], output_path)

        if duration is not None and duration > 0:
            manifest_records.append({
                "output_path": output_path,
                "label": fmeta["label"],
                "source_name": fmeta["source_name"],
                "duration": duration,
                "split": "train",
            })

        if (i + 1) % 500 == 0 or i == len(train_files) - 1:
            print(f"   [{i+1}/{len(train_files)}] processed")

    # Process eval set
    print(f"\n🔧 Processing eval set ({len(eval_files)} files)...")
    for i, fmeta in enumerate(eval_files):
        output_dir = OUTPUT_EVAL_BONAFIDE if fmeta["label"] == "bonafide" else OUTPUT_EVAL_SPOOF
        output_name = _generate_output_filename(fmeta["source_path"], fmeta["source_name"], i)
        output_path = output_dir / output_name

        if output_path.exists():
            try:
                info = sf.info(str(output_path))
                duration = info.duration
            except Exception:
                duration = 0.0
        else:
            duration = preprocess_audio_file(fmeta["source_path"], output_path)

        if duration is not None and duration > 0:
            manifest_records.append({
                "output_path": output_path,
                "label": fmeta["label"],
                "source_name": fmeta["source_name"],
                "duration": duration,
                "split": "eval",
            })

        if (i + 1) % 500 == 0 or i == len(eval_files) - 1:
            print(f"   [{i+1}/{len(eval_files)}] processed")

    # Step 4: Write manifest
    write_manifest(manifest_records, MANIFEST_PATH)

    # Final summary
    total_duration = sum(r["duration"] for r in manifest_records)
    print(f"\n{'=' * 60}")
    print(f"  ✅ Preprocessing complete!")
    print(f"     Total files processed: {len(manifest_records):,}")
    print(f"     Total audio duration:  {total_duration / 3600:.1f} hours")
    print(f"     Train output: {OUTPUT_TRAIN_BONAFIDE.parent}")
    print(f"     Eval output:  {OUTPUT_EVAL_BONAFIDE.parent}")
    print(f"     Manifest:     {MANIFEST_PATH}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VoiceGuard — Dataset preprocessing and preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_dataset.py                     # Run full preprocessing
  python scripts/prepare_dataset.py --dry-run            # Scan only, don't process
  python scripts/prepare_dataset.py --eval-ratio 0.15    # 15% eval split
  python scripts/prepare_dataset.py --max-files 1000     # Process only 1000 files (testing)
  python scripts/prepare_dataset.py --organize-asvspoof2019  # Organize raw ASVspoof data
        """,
    )
    parser.add_argument(
        "--eval-ratio", type=float, default=0.2,
        help="Fraction of data to hold out for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only scan and report — don't process or write files",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit total files to process (useful for testing)",
    )
    parser.add_argument(
        "--organize-asvspoof2019", action="store_true",
        help="Organize raw ASVspoof 2019 LA data into bonafide/spoof dirs using protocol files",
    )

    args = parser.parse_args()

    if args.organize_asvspoof2019:
        print("\n📁 Organizing ASVspoof 2019 LA dataset...")
        organize_asvspoof2019()
        return

    run_preprocessing(
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
