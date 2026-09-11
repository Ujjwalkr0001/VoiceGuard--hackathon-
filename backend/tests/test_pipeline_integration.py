"""
Integration test for the full DSP feature extraction pipeline.

Simulates the real Twilio pipeline path:
    AudioBufferManager emits a 2-second, 8 kHz, float32 PCM chunk
    → extract_all_features() runs all 4 feature families
    → verify output shape (96-dim), no NaN/Inf, valid structure

NOTE: Requires torch, torchcrepe, librosa, and Silero VAD.
      First run downloads the Silero VAD model (~2 MB).
"""

import numpy as np
import pytest

from app.ml.dsp_features import (
    TOTAL_FEATURE_DIM,
    extract_all_features,
    get_feature_normalizer,
)


def _simulate_twilio_chunk(
    duration_sec: float = 2.0,
    sr: int = 8000,
) -> np.ndarray:
    """
    Simulate a 2-second PCM chunk as emitted by AudioBufferManager.

    This mimics what Twilio Media Streams would produce after:
    base64 → mu-law decode → PCM → float32 normalized to [-1, 1].

    Uses a realistic speech-like signal (harmonics + jitter + pauses).
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Speech-like signal: varying pitch with harmonics
    base_f0 = 140.0
    jitter = np.cumsum(np.random.randn(n_samples) * 0.4)
    f0 = base_f0 + jitter
    phase = np.cumsum(2 * np.pi * f0 / sr)

    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.25
    signal += np.sin(3 * phase) * 0.1

    # Add a brief pause in the middle (silence from 0.8s to 1.0s)
    pause_start = int(0.8 * sr)
    pause_end = int(1.0 * sr)
    signal[pause_start:pause_end] = 0.0

    # Amplitude envelope
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 3.5 * t)
    signal *= envelope

    # Background noise (telephone-quality)
    signal += np.random.randn(n_samples) * 0.01

    # Normalize to [-1, 1] as AudioBufferManager does
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal.astype(np.float32)


class TestFullPipelineIntegration:
    """Integration tests for the complete feature extraction pipeline."""

    def test_output_shape_is_96_dim(self):
        """Full pipeline should produce a 96-dim feature vector."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        assert result["features"].shape == (TOTAL_FEATURE_DIM,), (
            f"Expected ({TOTAL_FEATURE_DIM},), got {result['features'].shape}"
        )

    def test_normalized_features_shape_matches(self):
        """Normalized features should have the same shape as raw features."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        assert result["features_normalized"].shape == result["features"].shape

    def test_no_nan_in_features(self):
        """No NaN values in raw or normalized feature vectors."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        assert not np.any(np.isnan(result["features"])), (
            "Raw features contain NaN"
        )
        assert not np.any(np.isnan(result["features_normalized"])), (
            "Normalized features contain NaN"
        )

    def test_no_inf_in_features(self):
        """No Inf values in raw or normalized feature vectors."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        assert not np.any(np.isinf(result["features"])), (
            "Raw features contain Inf"
        )
        assert not np.any(np.isinf(result["features_normalized"])), (
            "Normalized features contain Inf"
        )

    def test_features_dtype_is_float32(self):
        """Both feature vectors should be float32."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        assert result["features"].dtype == np.float32
        assert result["features_normalized"].dtype == np.float32

    def test_result_contains_all_keys(self):
        """Result dict must contain all expected keys."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        expected_keys = {
            "features", "features_normalized", "summaries",
            "timing_ms", "total_time_ms", "normalizer_fitted",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_summaries_contain_all_families(self):
        """Summaries dict must have entries for all 4 feature families."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        expected_families = {"group_delay", "cqcc", "pitch_jitter", "pause_rhythm"}
        assert expected_families.issubset(result["summaries"].keys()), (
            f"Missing summary families: {expected_families - result['summaries'].keys()}"
        )

    def test_timing_is_recorded_for_all_families(self):
        """Timing dict must have entries for all 4 feature families."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()
        result = extract_all_features(audio, sr=8000)

        for family in ["group_delay", "cqcc", "pitch_jitter", "pause_rhythm"]:
            assert family in result["timing_ms"], f"Missing timing for {family}"
            assert result["timing_ms"][family] >= 0, f"Negative timing for {family}"

        assert result["total_time_ms"] > 0

    def test_normalizer_identity_when_unfitted(self):
        """With default normalizer (unfitted), raw and normalized should match."""
        np.random.seed(42)
        audio = _simulate_twilio_chunk()

        normalizer = get_feature_normalizer()
        if not normalizer.is_fitted:
            result = extract_all_features(audio, sr=8000)
            np.testing.assert_array_almost_equal(
                result["features"],
                result["features_normalized"],
                decimal=5,
            )
            assert result["normalizer_fitted"] is False

    def test_pipeline_handles_pure_silence(self):
        """Full pipeline should not crash on silent input."""
        silence = np.zeros(16000, dtype=np.float32)
        result = extract_all_features(silence, sr=8000)

        assert result["features"].shape == (TOTAL_FEATURE_DIM,)
        assert not np.any(np.isnan(result["features"]))

    def test_pipeline_handles_loud_noise(self):
        """Full pipeline should handle random noise without crashing."""
        np.random.seed(123)
        noise = np.random.randn(16000).astype(np.float32) * 0.5
        result = extract_all_features(noise, sr=8000)

        assert result["features"].shape == (TOTAL_FEATURE_DIM,)
        assert not np.any(np.isnan(result["features"]))
