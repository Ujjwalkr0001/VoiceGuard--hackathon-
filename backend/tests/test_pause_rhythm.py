"""
Unit tests for DSP feature extraction — Pause/Rhythm Statistics.

Uses synthetic audio with known speech/silence patterns to verify
that pause_rhythm_stats() correctly identifies pauses and computes
rhythm features.

NOTE: These tests require torch and Silero VAD (downloaded on first run).
"""

import numpy as np
import pytest

from app.ml.dsp_features import pause_rhythm_stats


def _make_speech_with_pauses(
    sr: int = 8000,
    speech_segments: list[tuple[float, float]] = None,
    total_duration_sec: float = 2.0,
    tone_freq: float = 200.0,
) -> np.ndarray:
    """
    Create audio with alternating speech (loud tone) and silence (zeros).

    Args:
        sr: Sample rate.
        speech_segments: List of (start_sec, end_sec) for speech regions.
        total_duration_sec: Total audio duration.
        tone_freq: Frequency of the tone used to simulate speech.
    """
    if speech_segments is None:
        # Default: speech 0.0-0.5, silence 0.5-1.0, speech 1.0-1.5, silence 1.5-2.0
        speech_segments = [(0.0, 0.5), (1.0, 1.5)]

    n_samples = int(total_duration_sec * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    for start_sec, end_sec in speech_segments:
        start_idx = int(start_sec * sr)
        end_idx = min(int(end_sec * sr), n_samples)
        t = np.arange(end_idx - start_idx) / sr

        # Loud harmonic signal (so VAD detects it as speech)
        segment = np.sin(2 * np.pi * tone_freq * t) * 0.7
        segment += np.sin(2 * np.pi * tone_freq * 2 * t) * 0.3
        # Add slight noise for naturalness
        segment += np.random.randn(len(t)).astype(np.float32) * 0.02
        audio[start_idx:end_idx] = segment.astype(np.float32)

    return audio


def _make_continuous_speech(duration_sec: float = 2.0, sr: int = 8000) -> np.ndarray:
    """Create continuous speech-like audio with no pauses."""
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Continuous loud harmonic signal
    signal = np.sin(2 * np.pi * 200.0 * t) * 0.7
    signal += np.sin(2 * np.pi * 400.0 * t) * 0.3
    signal += np.random.randn(n_samples) * 0.02

    return signal.astype(np.float32)


class TestPauseRhythmStats:
    """Tests for pause_rhythm_stats()."""

    def test_returns_correct_shape(self):
        """Feature vector should be 8-dim."""
        audio = _make_speech_with_pauses()
        features, summary = pause_rhythm_stats(audio, sr=8000)

        assert features.shape == (8,), f"Expected (8,), got {features.shape}"
        assert features.dtype == np.float32
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self):
        """Summary must contain documented keys."""
        audio = _make_speech_with_pauses()
        _, summary = pause_rhythm_stats(audio, sr=8000)

        assert "num_pauses" in summary
        assert "mean_pause_sec" in summary
        assert "speech_rate" in summary
        assert "rhythm_regularity" in summary

    def test_no_nan_or_inf(self):
        """Features must be numerically valid."""
        audio = _make_speech_with_pauses()
        features, _ = pause_rhythm_stats(audio, sr=8000)

        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isinf(features)), "Features contain Inf"

    def test_detects_pauses_in_segmented_audio(self):
        """Audio with clear speech/silence alternation should have pauses detected."""
        # Speech at 0.0-0.5s and 1.0-1.5s → pause at 0.5-1.0s
        audio = _make_speech_with_pauses(
            speech_segments=[(0.0, 0.5), (1.0, 1.5)],
            total_duration_sec=2.0,
        )
        features, summary = pause_rhythm_stats(audio, sr=8000)

        num_pauses = features[0]
        assert num_pauses >= 1, (
            f"Expected at least 1 pause, got {num_pauses}"
        )

    def test_continuous_speech_has_fewer_pauses(self):
        """Continuous speech should have fewer pauses than segmented speech."""
        segmented = _make_speech_with_pauses(
            speech_segments=[(0.0, 0.4), (0.8, 1.2), (1.6, 2.0)],
            total_duration_sec=2.0,
        )
        continuous = _make_continuous_speech(duration_sec=2.0)

        seg_features, _ = pause_rhythm_stats(segmented, sr=8000)
        cont_features, _ = pause_rhythm_stats(continuous, sr=8000)

        # Continuous should have fewer or equal pauses
        assert cont_features[0] <= seg_features[0], (
            f"Continuous ({cont_features[0]}) should have ≤ pauses than segmented ({seg_features[0]})"
        )

    def test_speech_rate_higher_for_continuous(self):
        """Continuous speech should have higher speech rate than segmented."""
        segmented = _make_speech_with_pauses(
            speech_segments=[(0.0, 0.5), (1.0, 1.5)],
            total_duration_sec=2.0,
        )
        continuous = _make_continuous_speech(duration_sec=2.0)

        seg_features, _ = pause_rhythm_stats(segmented, sr=8000)
        cont_features, _ = pause_rhythm_stats(continuous, sr=8000)

        # Speech rate (index 4) should be higher for continuous
        assert cont_features[4] >= seg_features[4], (
            f"Continuous speech rate ({cont_features[4]}) should be ≥ segmented ({seg_features[4]})"
        )

    def test_silence_returns_valid_features(self):
        """Pure silence should return valid features with high pause count."""
        silence = np.zeros(16000, dtype=np.float32)
        features, summary = pause_rhythm_stats(silence, sr=8000)

        assert features.shape == (8,)
        assert not np.any(np.isnan(features))
        # Speech rate should be 0 or very low
        assert features[4] < 0.1

    def test_short_audio_returns_zeros(self):
        """Audio shorter than 200ms should return zero features."""
        short_audio = np.zeros(800, dtype=np.float32)  # 100ms at 8kHz
        features, summary = pause_rhythm_stats(short_audio, sr=8000)

        assert features.shape == (8,)
        assert np.all(features == 0)
        assert summary["status"] == "audio_too_short"


class TestPauseRhythmBenchmark:
    """Benchmark tests for Silero VAD inference latency."""

    def test_vad_inference_under_50ms(self):
        """
        Silero VAD inference on a 2-second chunk must complete in < 50ms.

        First call loads the model (slow); we warm up, then measure 10 runs.
        Target: median < 50ms per chunk on CPU.
        """
        import time

        sr = 8000
        audio = _make_continuous_speech(duration_sec=2.0, sr=sr)

        # Warm-up run (loads model, JIT compiles)
        pause_rhythm_stats(audio, sr=sr)

        # Benchmark 10 runs
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            pause_rhythm_stats(audio, sr=sr)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        median_latency = float(np.median(latencies))
        mean_latency = float(np.mean(latencies))
        max_latency = float(np.max(latencies))

        print(f"\n  VAD Benchmark (2s chunk, 10 runs):")
        print(f"    Median: {median_latency:.1f} ms")
        print(f"    Mean:   {mean_latency:.1f} ms")
        print(f"    Max:    {max_latency:.1f} ms")

        assert median_latency < 50.0, (
            f"VAD median latency {median_latency:.1f} ms exceeds 50 ms target"
        )
