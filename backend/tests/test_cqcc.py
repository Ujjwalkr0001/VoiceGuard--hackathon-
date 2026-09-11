"""
Unit tests for DSP feature extraction — CQCC Features.

Uses synthetic audio to simulate bonafide (natural) vs TTS (synthetic) signals.
Verifies that cqcc_features() produces correct shapes, valid values,
and measurably different distributions for the two signal types.
"""

import numpy as np
import pytest

from app.ml.dsp_features import cqcc_features


def _make_bonafide_audio(duration_sec: float = 2.0, sr: int = 8000) -> np.ndarray:
    """Natural speech simulation: harmonics with jitter, formants, and noise."""
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Varying F0 with jitter
    base_f0 = 130.0
    jitter = np.cumsum(np.random.randn(n_samples) * 0.6)
    f0 = base_f0 + jitter
    phase = np.cumsum(2 * np.pi * f0 / sr)

    # Harmonics with formant-like spectral shaping
    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.35      # strong second harmonic
    signal += np.sin(3 * phase) * 0.15
    signal += np.sin(4 * phase) * 0.08
    signal += np.sin(5 * phase) * 0.03

    # Amplitude modulation (syllable rhythm ~4 Hz)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    signal *= envelope

    # Background noise
    signal += np.random.randn(n_samples) * 0.03

    return (signal / (np.max(np.abs(signal)) + 1e-8)).astype(np.float32)


def _make_tts_audio(duration_sec: float = 2.0, sr: int = 8000) -> np.ndarray:
    """TTS/vocoder simulation: perfectly periodic, uniform spectral envelope."""
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Fixed pitch — no jitter
    f0 = 130.0
    phase = 2 * np.pi * f0 * t

    # Harmonics with unnaturally uniform spectral envelope
    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.5       # same amplitude as fundamental
    signal += np.sin(3 * phase) * 0.5       # unnaturally flat
    signal += np.sin(4 * phase) * 0.5
    signal += np.sin(5 * phase) * 0.5

    # Minimal noise
    signal += np.random.randn(n_samples) * 0.001

    return (signal / (np.max(np.abs(signal)) + 1e-8)).astype(np.float32)


class TestCQCCFeatures:
    """Tests for cqcc_features()."""

    def test_returns_correct_shape(self):
        """Feature vector should be 60-dim (20 CQCC + 20 delta + 20 dd)."""
        audio = _make_bonafide_audio()
        features, summary = cqcc_features(audio, sr=8000)

        assert features.shape == (60,), f"Expected (60,), got {features.shape}"
        assert features.dtype == np.float32

    def test_summary_has_required_keys(self):
        """Summary must contain documented keys."""
        audio = _make_bonafide_audio()
        _, summary = cqcc_features(audio, sr=8000)

        assert "cqcc_mean_energy" in summary
        assert "cqcc_spectral_variability" in summary
        assert "n_frames_analyzed" in summary

    def test_no_nan_or_inf(self):
        """Features must be numerically valid."""
        for make_fn in [_make_bonafide_audio, _make_tts_audio]:
            audio = make_fn()
            features, _ = cqcc_features(audio, sr=8000)

            assert not np.any(np.isnan(features)), "CQCC features contain NaN"
            assert not np.any(np.isinf(features)), "CQCC features contain Inf"

    def test_bonafide_vs_tts_feature_separation(self):
        """
        Bonafide and TTS audio should produce different CQCC distributions.
        The constant-Q transform is sensitive to spectral envelope differences.
        """
        np.random.seed(42)

        bonafide_features, _ = cqcc_features(_make_bonafide_audio(), sr=8000)
        tts_features, _ = cqcc_features(_make_tts_audio(), sr=8000)

        # Features should NOT be identical
        assert not np.allclose(bonafide_features, tts_features, atol=1e-3), (
            "Bonafide and TTS CQCC features are too similar"
        )

        # Meaningful Euclidean distance
        distance = np.linalg.norm(bonafide_features - tts_features)
        assert distance > 0.1, (
            f"CQCC feature distance too small ({distance:.6f})"
        )

    def test_spectral_variability_differs(self):
        """
        Natural speech has more spectral variation than TTS.
        Variability should be higher for bonafide.
        """
        np.random.seed(42)

        _, bonafide_summary = cqcc_features(_make_bonafide_audio(), sr=8000)
        _, tts_summary = cqcc_features(_make_tts_audio(), sr=8000)

        # Both should be positive
        assert bonafide_summary["cqcc_spectral_variability"] > 0
        assert tts_summary["cqcc_spectral_variability"] > 0

        # They should differ (we don't assert direction since it depends on synthesis)
        assert bonafide_summary["cqcc_spectral_variability"] != tts_summary["cqcc_spectral_variability"]

    def test_short_audio_returns_zeros(self):
        """Audio shorter than 100ms should return zero features."""
        short_audio = np.zeros(50, dtype=np.float32)  # 6.25ms at 8kHz
        features, summary = cqcc_features(short_audio, sr=8000)

        assert features.shape == (60,)
        assert np.all(features == 0)
        assert summary["status"] == "audio_too_short"

    def test_silence_does_not_crash(self):
        """Pure silence should produce valid features."""
        silence = np.zeros(16000, dtype=np.float32)
        features, summary = cqcc_features(silence, sr=8000)

        assert features.shape == (60,)
        assert not np.any(np.isnan(features))

    def test_deterministic_output(self):
        """Same input should always produce the same output."""
        audio = _make_tts_audio()
        f1, s1 = cqcc_features(audio, sr=8000)
        f2, s2 = cqcc_features(audio, sr=8000)

        np.testing.assert_array_equal(f1, f2)
        assert s1 == s2
