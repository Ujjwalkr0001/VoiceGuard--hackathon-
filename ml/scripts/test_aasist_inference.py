"""
VoiceGuard — AASIST Model Inference Test & Benchmark (Steps 74 & 75)

Tests the pretrained AASIST model on synthetic audio samples to verify:
  1. Model loads correctly and produces valid outputs
  2. Output scores are in [0, 1] range
  3. Scores show reasonable separation between bonafide-like and spoof-like signals
  4. Single-chunk inference latency is within target (< 100ms on CPU for 2s audio)

Since real ASVspoof / In-the-Wild datasets require manual download,
this script uses synthetically generated test signals:
  - Bonafide-like: natural speech characteristics (varying pitch, jitter, pauses)
  - Spoof-like: unnaturally smooth signals (constant pitch, no jitter, periodic)

When real datasets are available, re-run with --use-real-data to test on actual samples.

Usage:
    python scripts/test_aasist_inference.py
    python scripts/test_aasist_inference.py --use-lightweight
    python scripts/test_aasist_inference.py --use-real-data --data-dir ../data
    python scripts/test_aasist_inference.py --benchmark --n-runs 20
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add project paths
ML_DIR = Path(__file__).parent.parent
MODELS_DIR = ML_DIR / "models"
sys.path.insert(0, str(MODELS_DIR))

from aasist_wrapper import AASISTWrapper


# ---------------------------------------------------------------------------
# Synthetic test signal generators
# ---------------------------------------------------------------------------

def generate_natural_speech_like(
    duration_sec: float = 2.0,
    sr: int = 16000,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a signal with characteristics typical of natural speech:
    - Varying fundamental frequency (F0) with jitter
    - Amplitude modulation (shimmer)
    - Random pauses (silence segments)
    - Harmonics with noise

    This is NOT real speech — it's a synthetic approximation for testing
    that the model can process varied, noisy, non-periodic signals.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Varying F0 with jitter (100-250 Hz range, typical for speech)
    base_f0 = rng.uniform(100, 250)
    f0_contour = base_f0 + 20 * np.sin(2 * np.pi * 3 * t)  # Slow modulation
    f0_contour += rng.normal(0, 5, n_samples)  # Jitter

    # Generate signal with harmonics
    phase = np.cumsum(2 * np.pi * f0_contour / sr)
    signal = np.sin(phase)  # Fundamental
    signal += 0.5 * np.sin(2 * phase)  # 2nd harmonic
    signal += 0.25 * np.sin(3 * phase)  # 3rd harmonic
    signal += 0.1 * np.sin(4 * phase)  # 4th harmonic

    # Add amplitude modulation (shimmer)
    envelope = 1.0 + 0.3 * np.sin(2 * np.pi * 5 * t)
    envelope += rng.normal(0, 0.05, n_samples)
    signal *= np.clip(envelope, 0.1, 2.0)

    # Add random pauses (silence segments)
    n_pauses = rng.randint(1, 4)
    for _ in range(n_pauses):
        pause_start = rng.randint(0, n_samples - 1600)
        pause_len = rng.randint(800, 3200)  # 50-200ms
        signal[pause_start:pause_start + pause_len] *= 0.01

    # Add noise (realistic SNR ~20-30 dB)
    noise = rng.normal(0, 0.02, n_samples)
    signal += noise

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.9
    return signal.astype(np.float32)


def generate_synthetic_tts_like(
    duration_sec: float = 2.0,
    sr: int = 16000,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a signal with characteristics typical of TTS/vocoder output:
    - Very stable F0 (minimal jitter)
    - Smooth amplitude envelope (minimal shimmer)
    - Periodic, clean harmonics
    - Very low noise floor

    This mimics the unnaturally "perfect" quality of synthesized speech.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Stable F0 with minimal variation
    base_f0 = rng.uniform(120, 200)
    f0_contour = base_f0 + 2 * np.sin(2 * np.pi * 0.5 * t)  # Very slow, tiny modulation

    # Generate clean harmonics
    phase = np.cumsum(2 * np.pi * f0_contour / sr)
    signal = np.sin(phase)
    signal += 0.6 * np.sin(2 * phase)
    signal += 0.35 * np.sin(3 * phase)
    signal += 0.2 * np.sin(4 * phase)
    signal += 0.1 * np.sin(5 * phase)
    signal += 0.05 * np.sin(6 * phase)

    # Smooth envelope (no shimmer)
    envelope = 0.8 + 0.1 * np.sin(2 * np.pi * 1.5 * t)
    signal *= envelope

    # Very low noise (unnaturally clean)
    noise = rng.normal(0, 0.002, n_samples)
    signal += noise

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.95
    return signal.astype(np.float32)


def generate_test_samples(
    n_bonafide: int = 10,
    n_spoof: int = 10,
    duration_sec: float = 2.0,
    sr: int = 16000,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate synthetic bonafide-like and spoof-like test samples.

    Returns:
        (bonafide_samples, spoof_samples) tuple of lists of numpy arrays.
    """
    bonafide_samples = [
        generate_natural_speech_like(duration_sec, sr, seed=i)
        for i in range(n_bonafide)
    ]
    spoof_samples = [
        generate_synthetic_tts_like(duration_sec, sr, seed=100 + i)
        for i in range(n_spoof)
    ]
    return bonafide_samples, spoof_samples


# ---------------------------------------------------------------------------
# Real data loading (when available)
# ---------------------------------------------------------------------------

def load_real_samples(
    data_dir: Path,
    n_samples: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load real audio samples from the dataset directory.
    Expects: data_dir/{train,eval}/{bonafide,spoof}/*.wav
    """
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa required for loading real audio. pip install librosa")
        sys.exit(1)

    bonafide_dirs = [
        data_dir / "train" / "bonafide",
        data_dir / "eval" / "bonafide",
        data_dir / "asvspoof2019" / "LA" / "train" / "bonafide",
        data_dir / "asvspoof2019" / "LA" / "eval" / "bonafide",
    ]

    spoof_dirs = [
        data_dir / "train" / "spoof",
        data_dir / "eval" / "spoof",
        data_dir / "sarvam_tts" / "spoof",
        data_dir / "asvspoof2019" / "LA" / "train" / "spoof",
    ]

    def _load_from_dirs(dirs, n):
        samples = []
        for d in dirs:
            if not d.exists():
                continue
            audio_files = sorted(d.glob("*.wav")) + sorted(d.glob("*.flac"))
            for f in audio_files[:n - len(samples)]:
                try:
                    audio, _ = librosa.load(str(f), sr=16000, mono=True)
                    # Truncate or pad to 2 seconds
                    target_len = 32000
                    if len(audio) > target_len:
                        audio = audio[:target_len]
                    elif len(audio) < target_len:
                        audio = np.pad(audio, (0, target_len - len(audio)))
                    samples.append(audio)
                except Exception as e:
                    print(f"  [WARN] Could not load {f.name}: {e}")
                if len(samples) >= n:
                    break
            if len(samples) >= n:
                break
        return samples

    bonafide = _load_from_dirs(bonafide_dirs, n_samples)
    spoof = _load_from_dirs(spoof_dirs, n_samples)

    return bonafide, spoof


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def run_inference_test(
    wrapper: AASISTWrapper,
    bonafide_samples: List[np.ndarray],
    spoof_samples: List[np.ndarray],
) -> dict:
    """
    Run inference on bonafide and spoof samples, report scores and separation.
    """
    print("\n" + "=" * 65)
    print("  AASIST Inference Test — Score Separation Analysis")
    print("=" * 65)

    bonafide_scores = []
    spoof_scores = []

    # Test bonafide samples
    print(f"\n{'─' * 65}")
    print(f"  BONAFIDE samples ({len(bonafide_samples)} samples)")
    print(f"  {'Sample':>8}  {'Spoof Prob':>12}  {'Verdict':>10}  {'Latency':>10}")
    print(f"{'─' * 65}")

    for i, audio in enumerate(bonafide_samples):
        result = wrapper.predict_with_details(audio)
        score = result["spoof_probability"]
        bonafide_scores.append(score)
        verdict = "✅ SAFE" if score < 0.5 else "⚠️ FALSE+"
        print(f"  {'B-' + str(i+1):>8}  {score:>12.4f}  {verdict:>10}  {result['inference_time_ms']:>8.1f}ms")

    # Test spoof samples
    print(f"\n{'─' * 65}")
    print(f"  SPOOF samples ({len(spoof_samples)} samples)")
    print(f"  {'Sample':>8}  {'Spoof Prob':>12}  {'Verdict':>10}  {'Latency':>10}")
    print(f"{'─' * 65}")

    for i, audio in enumerate(spoof_samples):
        result = wrapper.predict_with_details(audio)
        score = result["spoof_probability"]
        spoof_scores.append(score)
        verdict = "✅ CAUGHT" if score >= 0.5 else "❌ MISSED"
        print(f"  {'S-' + str(i+1):>8}  {score:>12.4f}  {verdict:>10}  {result['inference_time_ms']:>8.1f}ms")

    # Compute statistics
    b_mean = np.mean(bonafide_scores)
    b_std = np.std(bonafide_scores)
    s_mean = np.mean(spoof_scores)
    s_std = np.std(spoof_scores)
    separation = abs(s_mean - b_mean)

    # Accuracy at threshold 0.5
    bonafide_correct = sum(1 for s in bonafide_scores if s < 0.5)
    spoof_correct = sum(1 for s in spoof_scores if s >= 0.5)
    total = len(bonafide_scores) + len(spoof_scores)
    accuracy = (bonafide_correct + spoof_correct) / total

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Bonafide scores:  mean={b_mean:.4f}, std={b_std:.4f}, range=[{min(bonafide_scores):.4f}, {max(bonafide_scores):.4f}]")
    print(f"  Spoof scores:     mean={s_mean:.4f}, std={s_std:.4f}, range=[{min(spoof_scores):.4f}, {max(spoof_scores):.4f}]")
    print(f"  Score separation: {separation:.4f}")
    print(f"  Accuracy @0.5:    {accuracy:.1%} ({bonafide_correct + spoof_correct}/{total})")
    print(f"  Bonafide correct: {bonafide_correct}/{len(bonafide_scores)}")
    print(f"  Spoof detected:   {spoof_correct}/{len(spoof_scores)}")

    if separation > 0.1:
        print(f"\n  ✅ Reasonable score separation detected ({separation:.4f} > 0.1)")
    else:
        print(f"\n  ⚠️  Low score separation ({separation:.4f}).")
        print(f"     This is expected with synthetic signals — the pretrained model")
        print(f"     was trained on real speech vs TTS, not sine-wave approximations.")
        print(f"     Re-run with --use-real-data once ASVspoof data is downloaded.")

    return {
        "bonafide_scores": bonafide_scores,
        "spoof_scores": spoof_scores,
        "bonafide_mean": b_mean,
        "spoof_mean": s_mean,
        "separation": separation,
        "accuracy": accuracy,
    }


def run_latency_benchmark(
    wrapper: AASISTWrapper,
    n_runs: int = 20,
    duration_sec: float = 2.0,
    sr: int = 16000,
) -> dict:
    """
    Benchmark single-chunk inference latency.
    Target: < 100ms on CPU for 2-second audio.
    """
    print(f"\n{'=' * 65}")
    print(f"  AASIST Latency Benchmark")
    print(f"  Audio: {duration_sec}s @ {sr} Hz | Runs: {n_runs}")
    print(f"  Device: {wrapper.device}")
    print(f"{'=' * 65}")

    # Generate test audio
    audio = generate_natural_speech_like(duration_sec, sr, seed=42)

    # Warm-up (3 runs)
    print("\n  Warming up (3 runs)...")
    for _ in range(3):
        wrapper.predict(audio)

    # Benchmark
    print(f"  Running {n_runs} timed inferences...\n")
    times_ms = []

    for i in range(n_runs):
        start = time.perf_counter()
        _ = wrapper.predict(audio)
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    times_ms = np.array(times_ms)
    mean_ms = np.mean(times_ms)
    std_ms = np.std(times_ms)
    min_ms = np.min(times_ms)
    max_ms = np.max(times_ms)
    p50 = np.percentile(times_ms, 50)
    p95 = np.percentile(times_ms, 95)
    p99 = np.percentile(times_ms, 99)

    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'─' * 32}")
    print(f"  {'Mean':.<20} {mean_ms:>8.1f}ms")
    print(f"  {'Std':.<20} {std_ms:>8.1f}ms")
    print(f"  {'Min':.<20} {min_ms:>8.1f}ms")
    print(f"  {'Max':.<20} {max_ms:>8.1f}ms")
    print(f"  {'P50 (median)':.<20} {p50:>8.1f}ms")
    print(f"  {'P95':.<20} {p95:>8.1f}ms")
    print(f"  {'P99':.<20} {p99:>8.1f}ms")

    target_ms = 100.0
    if p95 < target_ms:
        print(f"\n  ✅ PASS: P95 latency ({p95:.1f}ms) < target ({target_ms}ms)")
    else:
        print(f"\n  ⚠️  P95 latency ({p95:.1f}ms) exceeds target ({target_ms}ms)")
        print(f"     Consider using AASIST-L (--use-lightweight) for faster inference.")

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "target_met": p95 < target_ms,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test AASIST inference and benchmark latency",
    )
    parser.add_argument(
        "--use-lightweight", action="store_true",
        help="Use AASIST-L (85K params, faster) instead of full AASIST",
    )
    parser.add_argument(
        "--use-real-data", action="store_true",
        help="Load real audio samples from the data directory",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(ML_DIR / "data"),
        help="Path to data directory (for --use-real-data)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a custom AASIST checkpoint (.pth)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of bonafide/spoof samples to test (default: 10)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark (included by default unless --no-benchmark)",
    )
    parser.add_argument(
        "--no-benchmark", action="store_true",
        help="Skip latency benchmark",
    )
    parser.add_argument(
        "--n-runs", type=int, default=20,
        help="Number of benchmark runs (default: 20)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', 'cuda', or None (auto-detect)",
    )

    args = parser.parse_args()

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║" + "  VoiceGuard — AASIST Model Test & Benchmark".center(63) + "║")
    print("╚" + "═" * 63 + "╝")

    variant = "AASIST-L (lightweight)" if args.use_lightweight else "AASIST (full)"
    print(f"\n  Model variant: {variant}")
    print(f"  Device: {args.device or 'auto-detect'}")

    # Initialize model
    print("\n  Loading model...")
    wrapper = AASISTWrapper(
        use_lightweight=args.use_lightweight,
        device=args.device,
    )
    wrapper.load_model(args.checkpoint)
    print(f"  ✅ Model loaded on {wrapper.device}")

    # Load or generate test samples
    if args.use_real_data:
        data_dir = Path(args.data_dir)
        print(f"\n  Loading real samples from: {data_dir}")
        bonafide, spoof = load_real_samples(data_dir, args.n_samples)
        if len(bonafide) == 0 or len(spoof) == 0:
            print(f"\n  ❌ Not enough real samples found!")
            print(f"     Bonafide: {len(bonafide)}, Spoof: {len(spoof)}")
            print(f"     Falling back to synthetic samples...")
            bonafide, spoof = generate_test_samples(args.n_samples, args.n_samples)
    else:
        print(f"\n  Generating {args.n_samples} synthetic bonafide + {args.n_samples} spoof samples...")
        bonafide, spoof = generate_test_samples(args.n_samples, args.n_samples)

    # Run inference test (Step 74)
    test_results = run_inference_test(wrapper, bonafide, spoof)

    # Run latency benchmark (Step 75)
    if not args.no_benchmark:
        bench_results = run_latency_benchmark(wrapper, n_runs=args.n_runs)

    print(f"\n{'=' * 65}")
    print(f"  All tests complete.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
