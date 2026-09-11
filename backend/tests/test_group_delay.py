"""
Unit tests for DSP feature extraction — Group Delay Features.

Uses synthetic audio to simulate bonafide (natural) vs spoof (synthetic) signals:
- Bonafide: harmonic signal with pitch jitter and noise (mimics natural speech)
- Spoof: perfectly periodic, clean signal (mimics TTS/vocoder output)

Verifies that group_delay_features() produces measurably different outputs
for the two signal types, confirming discriminative power.
"""

import numpy as np
import pytest

from app.ml.dsp_features import group_delay_features


def _make_bonafide_audio(duration_sec: float = 2.0, sr: int = 8000) -> np.ndarray:
    """
    Simulate natural speech: harmonic stack with pitch jitter, amplitude
    variation, and additive noise.
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Base pitch with jitter (random walk on F0)
    base_f0 = 120.0  # Hz
    jitter = np.cumsum(np.random.randn(n_samples) * 0.5)
    f0 = base_f0 + jitter

    # Instantaneous phase from varying F0
    phase = np.cumsum(2 * np.pi * f0 / sr)

    # Harmonic stack (fundamental + 3 harmonics with decreasing amplitude)
    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.25
    signal += np.sin(3 * phase) * 0.12
    signal += np.sin(4 * phase) * 0.06

    # Amplitude modulation (natural speech has varying loudness)
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 3.0 * t)
    signal *= envelope

    # Additive noise (natural recordings have background noise)
    signal += np.random.randn(n_samples) * 0.02

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal.astype(np.float32)


def _make_spoof_audio(duration_sec: float = 2.0, sr: int = 8000) -> np.ndarray:
    """
    Simulate TTS/vocoder output: perfectly periodic harmonics, no jitter,
    very clean signal with minimal noise.
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr

    # Fixed pitch — no jitter (perfectly periodic)
    f0 = 120.0
    phase = 2 * np.pi * f0 * t

    # Clean harmonic stack
    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.25
    signal += np.sin(3 * phase) * 0.12
    signal += np.sin(4 * phase) * 0.06

    # No amplitude modulation, very little noise
    signal += np.random.randn(n_samples) * 0.001

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal.astype(np.float32)


class TestGroupDelayFeatures:
    """Tests for group_delay_features()."""

    def test_returns_correct_shape(self):
        """Feature vector should be 20-dim, summary should be a dict."""
        audio = _make_bonafide_audio()
        features, summary = group_delay_features(audio, sr=8000)

        assert features.shape == (20,), f"Expected (20,), got {features.shape}"
        assert features.dtype == np.float32
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self):
        """Summary dict must contain the documented keys."""
        audio = _make_bonafide_audio()
        _, summary = group_delay_features(audio, sr=8000)

        assert "group_delay_deviation" in summary
        assert "phase_coherence" in summary
        assert "n_frames_analyzed" in summary

    def test_no_nan_or_inf_in_features(self):
        """Feature vector must not contain NaN or Inf values."""
        for make_fn in [_make_bonafide_audio, _make_spoof_audio]:
            audio = make_fn()
            features, _ = group_delay_features(audio, sr=8000)

            assert not np.any(np.isnan(features)), "Features contain NaN"
            assert not np.any(np.isinf(features)), "Features contain Inf"

    def test_bonafide_vs_spoof_feature_separation(self):
        """
        Bonafide and spoof audio should produce different feature distributions.
        Natural speech has more group delay variation than clean synthetic speech.
        """
        np.random.seed(42)

        bonafide_features, bonafide_summary = group_delay_features(
            _make_bonafide_audio(), sr=8000
        )
        spoof_features, spoof_summary = group_delay_features(
            _make_spoof_audio(), sr=8000
        )

        # Features should NOT be identical
        assert not np.allclose(bonafide_features, spoof_features, atol=1e-3), (
            "Bonafide and spoof features are too similar — no discriminative power"
        )

        # Euclidean distance between feature vectors should be non-trivial
        distance = np.linalg.norm(bonafide_features - spoof_features)
        assert distance > 0.01, (
            f"Feature distance too small ({distance:.6f}) — insufficient separation"
        )

    def test_phase_coherence_higher_for_spoof(self):
        """
        Synthetic (spoof) audio is more phase-coherent than natural speech.
        Phase coherence should be higher for spoof.
        """
        np.random.seed(42)

        _, bonafide_summary = group_delay_features(_make_bonafide_audio(), sr=8000)
        _, spoof_summary = group_delay_features(_make_spoof_audio(), sr=8000)

        assert spoof_summary["phase_coherence"] > bonafide_summary["phase_coherence"], (
            f"Expected spoof coherence ({spoof_summary['phase_coherence']}) > "
            f"bonafide coherence ({bonafide_summary['phase_coherence']})"
        )

    def test_short_audio_returns_zeros(self):
        """Audio shorter than n_fft should return zero features gracefully."""
        short_audio = np.zeros(100, dtype=np.float32)
        features, summary = group_delay_features(short_audio, sr=8000)

        assert features.shape == (20,)
        assert np.all(features == 0)
        assert summary["status"] == "audio_too_short"

    def test_silence_does_not_crash(self):
        """Pure silence should produce valid (non-NaN) features."""
        silence = np.zeros(16000, dtype=np.float32)
        features, summary = group_delay_features(silence, sr=8000)

        assert features.shape == (20,)
        assert not np.any(np.isnan(features))

    def test_deterministic_output(self):
        """Same input should always produce the same output."""
        audio = _make_spoof_audio()
        f1, s1 = group_delay_features(audio, sr=8000)
        f2, s2 = group_delay_features(audio, sr=8000)

        np.testing.assert_array_equal(f1, f2)
        assert s1 == s2
