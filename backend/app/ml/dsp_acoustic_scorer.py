"""
VoiceGuard — DSP Heuristic Acoustic Scorer

A lightweight acoustic voice-clone scorer that works entirely from the
DSP feature vector already computed by extract_all_features().  No .pth
checkpoint, no torch required.

It acts as the Model-A slot when the real AASIST checkpoint is not
available, giving the pipeline real (non-neutral) scores immediately.

Feature layout (96-dim, see dsp_features.py):
    [  0: 20]  Group-delay features (MODGDF)
    [ 20: 80]  CQCC features (20 static + 20 Δ + 20 ΔΔ)
    [ 80: 88]  Pitch-jitter / shimmer features
    [ 88: 96]  Pause / rhythm / VAD features

Heuristic rationale
-------------------
TTS / voice-clone systems have characteristic DSP fingerprints:

1. **Low group-delay entropy** — synthesised speech has near-perfect
   phase coherence; natural speech is messier.  Low variance in MODGDF
   → elevated spoof score.

2. **Abnormally low jitter / shimmer** — synthesisers produce
   unnaturally clean pitch tracks.  Jitter < 0.5 % and shimmer < 1 dB
   are suspicious.

3. **Unusually stable CQCC delta energy** — natural speech has larger
   short-time spectral flux; TTS is smoother.  Low CQCC-Δ std → elevated
   spoof score.

4. **Unnatural pause statistics** — TTS often has very regular or very
   few pauses.  Both extremes relative to typical phone-call statistics
   contribute to the score.

The four sub-scores are combined with fixed weights, then passed
through a sigmoid to keep the output in (0, 1).
"""

import logging
from typing import Optional

import numpy as np

from app.ml.dsp_features import extract_all_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-slice indices (must match extract_all_features output order)
# ---------------------------------------------------------------------------
_GD_START, _GD_END = 0, 20        # Group-delay features (20-dim)
_CQCC_START, _CQCC_END = 20, 80   # CQCC features (60-dim)
_PJ_START, _PJ_END = 80, 88       # Pitch-jitter / shimmer (8-dim)
_PR_START, _PR_END = 88, 96       # Pause / rhythm (8-dim)

# ---------------------------------------------------------------------------
# Sub-score weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_W_GROUP_DELAY = 0.30
_W_CQCC_DELTA = 0.30
_W_JITTER = 0.25
_W_PAUSE = 0.15

# ---------------------------------------------------------------------------
# Calibration constants (tuned on typical 8 kHz phone-call audio)
# ---------------------------------------------------------------------------

# Group-delay score: based on variance of the GD feature vector
# Low variance → near-uniform GD → TTS-like → high spoof
_GD_VAR_NATURAL = 0.08   # Typical natural-speech GD variance
_GD_VAR_SYNTH   = 0.01   # Typical synthesised-speech GD variance

# CQCC delta score: standard deviation of CQCC-Δ coefficients
# Low std → smooth spectral trajectory → TTS-like → high spoof
# CQCC layout: [0:20] static, [20:40] Δ, [40:60] ΔΔ  (within the 60-dim slice)
_CQCC_DELTA_NATURAL = 2.5   # Typical natural std of Δ coefficients
_CQCC_DELTA_SYNTH   = 0.5   # Typical synthesised std of Δ coefficients

# Jitter score: use index 0 of PJ features (relative jitter in %)
# Very low jitter (< 0.5 %) → suspicious
_JITTER_LOW_THRESH   = 0.005   # Below this → strong spoof signal
_JITTER_NATURAL_MID  = 0.02    # Typical natural jitter (~2 %)
_JITTER_HIGH_THRESH  = 0.15    # Above this → definitely natural

# Pause score: use mean pause duration (index 2 in PR features)
# Unnatural pauses: either 0 (perfect continuity) or very long
_PAUSE_MEAN_NATURAL   = 0.15   # Seconds, typical natural pause
_PAUSE_MEAN_SYNTH_LOW = 0.02   # Very few / short pauses → suspicious
_PAUSE_MEAN_SYNTH_HIGH = 0.60  # Very long pauses → suspicious (text-chunks)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class DSPAcousticScorer:
    """
    Heuristic acoustic voice-clone scorer based on DSP features.

    Implements the same `predict(audio) -> float` interface as the
    AASIST wrapper, so it can be dropped into the `model_a` slot of
    `ModelManager` without any other code changes.

    This class is intentionally stateless across calls — each `predict`
    call is independent.
    """

    def __init__(self, sample_rate: int = 8000):
        """
        Args:
            sample_rate: Expected sample rate of incoming audio.
                         Twilio sends 8 kHz μ-law, decoded to 8 kHz PCM.
        """
        self.sample_rate = sample_rate
        logger.info(
            "dsp_acoustic_scorer.initialized",
            extra={"sample_rate": sample_rate},
        )

    # ------------------------------------------------------------------
    # Public interface (matches AASISTWrapper.predict)
    # ------------------------------------------------------------------

    def predict(self, audio: np.ndarray, sr: Optional[int] = None) -> float:
        """
        Compute spoof probability for a single audio chunk.

        Args:
            audio: 1-D float32 array of PCM samples normalised to [-1, 1].
            sr:    Sample rate. Defaults to self.sample_rate if None.

        Returns:
            Spoof probability in [0.0, 1.0].
            0.0 → very likely genuine voice.
            1.0 → very likely synthesised / cloned voice.
        """
        if sr is None:
            sr = self.sample_rate

        # Guard: silence / very short chunks return neutral
        if len(audio) < sr * 0.1:  # < 100 ms
            return 0.5

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-5:  # Near-silent chunk
            return 0.5

        try:
            features_result = extract_all_features(audio, sr)
            fv = features_result["features"]  # 96-dim float32
        except Exception as e:
            logger.warning(
                "dsp_acoustic_scorer.feature_extraction_failed",
                extra={"error": str(e)},
            )
            return 0.5

        # Compute individual sub-scores
        gd_score    = self._group_delay_score(fv)
        cqcc_score  = self._cqcc_delta_score(fv)
        jitter_score = self._jitter_score(fv)
        pause_score = self._pause_score(fv)

        # Weighted combination → logit-space addition → sigmoid
        raw = (
            _W_GROUP_DELAY * gd_score
            + _W_CQCC_DELTA * cqcc_score
            + _W_JITTER * jitter_score
            + _W_PAUSE * pause_score
        )
        # raw is already in [0, 1] — apply a light S-curve to sharpen decisions
        spoof_prob = self._sharpen(raw)

        logger.debug(
            "dsp_acoustic_scorer.scores",
            extra={
                "gd": round(gd_score, 3),
                "cqcc": round(cqcc_score, 3),
                "jitter": round(jitter_score, 3),
                "pause": round(pause_score, 3),
                "raw": round(raw, 3),
                "spoof_prob": round(spoof_prob, 3),
            },
        )

        return _clamp(spoof_prob)

    # ------------------------------------------------------------------
    # Sub-scorers — each returns a value in [0.0, 1.0]
    # 0.0 = definitely genuine signal; 1.0 = definitely spoof signal
    # ------------------------------------------------------------------

    def _group_delay_score(self, fv: np.ndarray) -> float:
        """
        Score based on group-delay feature variance.

        Synthesised speech → lower variance (too-regular phase).
        """
        gd = fv[_GD_START:_GD_END]
        variance = float(np.var(gd))

        # Map variance from [_GD_VAR_SYNTH, _GD_VAR_NATURAL] → [1.0, 0.0]
        # Values below _GD_VAR_SYNTH get score 1.0 (very suspicious)
        if variance <= _GD_VAR_SYNTH:
            return 1.0
        if variance >= _GD_VAR_NATURAL:
            return 0.0

        # Linear interpolation (inverted)
        t = (variance - _GD_VAR_SYNTH) / (_GD_VAR_NATURAL - _GD_VAR_SYNTH)
        return _clamp(1.0 - t)

    def _cqcc_delta_score(self, fv: np.ndarray) -> float:
        """
        Score based on CQCC-delta coefficient standard deviation.

        Synthesised speech → smoother spectral trajectory → lower std.
        """
        cqcc_slice = fv[_CQCC_START:_CQCC_END]  # 60-dim
        # Layout within the 60-dim CQCC slice: 20 static, 20 Δ, 20 ΔΔ
        cqcc_delta = cqcc_slice[20:40]  # Δ coefficients
        std = float(np.std(cqcc_delta))

        if std <= _CQCC_DELTA_SYNTH:
            return 1.0
        if std >= _CQCC_DELTA_NATURAL:
            return 0.0

        t = (std - _CQCC_DELTA_SYNTH) / (_CQCC_DELTA_NATURAL - _CQCC_DELTA_SYNTH)
        return _clamp(1.0 - t)

    def _jitter_score(self, fv: np.ndarray) -> float:
        """
        Score based on pitch jitter.

        Synthesisers produce unnaturally clean (very low) jitter.
        """
        pj = fv[_PJ_START:_PJ_END]  # 8-dim
        # Index 0 is relative jitter (ddp) — the most discriminative
        jitter = float(abs(pj[0]))

        if jitter <= _JITTER_LOW_THRESH:
            return 1.0
        if jitter >= _JITTER_HIGH_THRESH:
            return 0.0

        # Log-scale mapping (jitter spans many orders of magnitude)
        log_j = np.log(jitter + 1e-9)
        log_lo = np.log(_JITTER_LOW_THRESH + 1e-9)
        log_hi = np.log(_JITTER_HIGH_THRESH + 1e-9)

        t = (log_j - log_lo) / (log_hi - log_lo)
        return _clamp(1.0 - t)

    def _pause_score(self, fv: np.ndarray) -> float:
        """
        Score based on pause / rhythm statistics.

        Both very-short and very-long mean pause durations are suspicious.
        Natural speech sits in the middle (~0.15 s).
        """
        pr = fv[_PR_START:_PR_END]  # 8-dim
        # Index 2 is typically mean pause duration (see pause_rhythm_stats)
        mean_pause = float(abs(pr[2]))

        # Distance from the natural pause centre
        distance = abs(mean_pause - _PAUSE_MEAN_NATURAL)

        # Normalise: distance > 0.4 s → strongly suspicious
        score = _clamp(distance / 0.4)
        return score

    @staticmethod
    def _sharpen(p: float, gain: float = 4.0) -> float:
        """
        Apply a centred sigmoid sharpening to push probabilities away from 0.5.

        Uses logit → scale → sigmoid to preserve the [0, 1] range.
        """
        eps = 1e-6
        p = _clamp(p, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        return float(_sigmoid(logit * gain / 2.0))
