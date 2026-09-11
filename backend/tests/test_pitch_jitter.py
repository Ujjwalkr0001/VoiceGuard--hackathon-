"""
Unit tests for DSP feature extraction — Pitch Jitter & Micro-Prosody.

Verifies that pitch_jitter() produces correct shapes, handles edge cases,
and shows discriminative power: near-zero jitter for constant-pitch synthetic
tones vs higher jitter for natural-like speech.

NOTE: These tests require torch and torchcrepe to be installed.
      They use CPU inference with the 'tiny' model.
"""

import numpy as np
import pytest

from app.ml.dsp_features import pitch_jitter


def _make_constant_pitch_tone(
    duration_sec: float = 2.0, sr: int = 8000, f0: float = 150.0
) -> np.ndarray:
    """
    Pure constant-pitch harmonic tone — simulates perfectly periodic TTS.
    Jitter should be near zero.
    """
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr
    phase = 2 * np.pi * f0 * t

    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.25
    signal += np.sin(3 * phase) * 0.12

    return (signal / (np.max(np.abs(signal)) + 1e-8)).astype(np.float32)


def _make_jittery_speech(
    duration_sec: float = 2.0, sr: int = 8000
) -> np.ndarray:
    """
    Harmonic signal with realistic pitch jitter — simulates natural speech.
    Jitter should be measurably higher than the constant-pitch tone.
    """
    n_samples = int(duration_sec * sr)

    # Varying F0 with natural jitter
    base_f0 = 150.0
    jitter = np.cumsum(np.random.randn(n_samples) * 0.8)
    f0 = base_f0 + jitter
    phase = np.cumsum(2 * np.pi * f0 / sr)

    signal = np.sin(phase) * 0.5
    signal += np.sin(2 * phase) * 0.25
    signal += np.sin(3 * phase) * 0.12

    # Add slight noise
    signal += np.random.randn(n_samples) * 0.01

    return (signal / (np.max(np.abs(signal)) + 1e-8)).astype(np.float32)


class TestPitchJitter:
    """Tests for pitch_jitter()."""

    def test_returns_correct_shape(self):
        """Feature vector should be 8-dim."""
        audio = _make_constant_pitch_tone()
        features, summary = pitch_jitter(audio, sr=8000)

        assert features.shape == (8,), f"Expected (8,), got {features.shape}"
        assert features.dtype == np.float32
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self):
        """Summary must contain documented keys."""
        audio = _make_constant_pitch_tone()
        _, summary = pitch_jitter(audio, sr=8000)

        assert "f0_mean_hz" in summary
        assert "jitter_relative" in summary
        assert "shimmer" in summary
        assert "voiced_fraction" in summary

    def test_no_nan_or_inf(self):
        """Features must not contain NaN or Inf."""
        audio = _make_constant_pitch_tone()
        features, _ = pitch_jitter(audio, sr=8000)

        assert not np.any(np.isnan(features)), "Features contain NaN"
        assert not np.any(np.isinf(features)), "Features contain Inf"

    def test_constant_pitch_has_lower_jitter_than_speech(self):
        """
        Core discriminative test: constant-pitch tone should have
        lower jitter than jittery speech-like signal.
        """
        np.random.seed(42)

        tone_features, tone_summary = pitch_jitter(
            _make_constant_pitch_tone(), sr=8000
        )
        speech_features, speech_summary = pitch_jitter(
            _make_jittery_speech(), sr=8000
        )

        # Jitter relative is feature index 5
        tone_jitter = tone_features[5]
        speech_jitter = speech_features[5]

        assert speech_jitter > tone_jitter, (
            f"Expected speech jitter ({speech_jitter}) > tone jitter ({tone_jitter})"
        )

    def test_f0_mean_in_reasonable_range(self):
        """F0 mean for a 150 Hz tone should be close to 150 Hz."""
        audio = _make_constant_pitch_tone(f0=150.0)
        features, summary = pitch_jitter(audio, sr=8000)

        f0_mean = features[0]  # index 0 = F0 mean
        # Allow generous tolerance since 8kHz and CREPE tiny may not be exact
        assert 80.0 < f0_mean < 300.0, (
            f"F0 mean {f0_mean} Hz is outside reasonable range for 150 Hz input"
        )

    def test_short_audio_returns_zeros(self):
        """Audio shorter than 50ms should return zero features."""
        short_audio = np.zeros(200, dtype=np.float32)  # 25ms at 8kHz
        features, summary = pitch_jitter(short_audio, sr=8000)

        assert features.shape == (8,)
        assert np.all(features == 0)
        assert summary["status"] == "audio_too_short"

    def test_silence_returns_mostly_unvoiced(self):
        """Pure silence should be detected as unvoiced."""
        silence = np.zeros(16000, dtype=np.float32)
        features, summary = pitch_jitter(silence, sr=8000)

        assert features.shape == (8,)
        assert features[7] < 0.1  # voiced_fraction should be very low

    def test_voiced_fraction_is_high_for_tonal_audio(self):
        """A loud harmonic tone should have high voiced fraction."""
        audio = _make_constant_pitch_tone()
        features, summary = pitch_jitter(audio, sr=8000)

        voiced_fraction = features[7]
        assert voiced_fraction > 0.5, (
            f"Voiced fraction {voiced_fraction} is too low for a tonal signal"
        )
