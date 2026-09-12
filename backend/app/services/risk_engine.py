"""
VoiceGuard Backend — Risk Scoring Engine (Steps 109-114)

Computes a composite, per-call risk score that combines:
  - acoustic_score (from the AASIST + XGBoost ensemble)
  - context_score  (from the NLP risk-phrase analyzer)

weighted by a caller reputation multiplier and temporally smoothed
via a rolling average over the last N chunks.

Key behaviours:
  - Composite:  raw = 0.65 × acoustic + 0.35 × context
  - Multiplier: known contact → 1.0, unknown → 1.3, flagged → 1.5
  - Smoothing:  rolling average over a configurable window (default 10)
  - Spike gate: any single chunk ≥ 95 immediately flags, bypassing the
                rolling average

One RiskEngine instance is created per active call SID and holds
all in-memory state for the duration of that call.
"""

import logging
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caller reputation categories
# ---------------------------------------------------------------------------

class CallerReputation(str, Enum):
    """Caller trust level — determines the risk multiplier."""
    TRUSTED = "trusted"      # In the user's enrolled contacts list
    UNKNOWN = "unknown"      # Not recognised
    FLAGGED = "flagged"      # Previously triggered a high-risk alert


# Multiplier mapping (configurable via settings)
_CALLER_MULTIPLIERS: Dict[CallerReputation, float] = {
    CallerReputation.TRUSTED: 1.0,
    CallerReputation.UNKNOWN: settings.caller_multiplier_unknown,   # default 1.3
    CallerReputation.FLAGGED: settings.caller_multiplier_flagged,   # default 1.5
}


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Per-call risk scoring engine.

    Instantiate one per active call SID.  Feed it chunk-level acoustic
    and context scores; it maintains a smoothed rolling risk score and
    detects threshold crossings.

    Usage:
        engine = RiskEngine(call_sid="CA123", caller_number="+919876543210")
        result = engine.process_chunk(acoustic_score=72.0, context_score=45.0)
        # result = {
        #     "current_risk": 63.2,
        #     "peak_risk": 63.2,
        #     "threshold_crossed": None,     # or "medium" / "high"
        #     "chunk_index": 0,
        #     ...
        # }
    """

    # ----- construction -----

    def __init__(
        self,
        call_sid: str,
        caller_number: str = "",
        caller_reputation: CallerReputation = CallerReputation.UNKNOWN,
        rolling_window_size: int = 0,        # 0 → use settings default
        acoustic_weight: float = 0.65,
        context_weight: float = 0.35,
        spike_threshold: float = 95.0,
        medium_threshold: int = 0,           # 0 → use settings default
        high_threshold: int = 0,             # 0 → use settings default
    ):
        self.call_sid = call_sid
        self.caller_number = caller_number
        self.caller_reputation = caller_reputation
        self.start_time = time.time()

        # Weights
        self.acoustic_weight = acoustic_weight
        self.context_weight = context_weight

        # Multiplier
        self.caller_multiplier = _CALLER_MULTIPLIERS.get(
            caller_reputation, settings.caller_multiplier_unknown,
        )

        # Rolling window
        self.rolling_window_size = (
            rolling_window_size if rolling_window_size > 0
            else settings.rolling_window_size
        )
        self._window: deque[float] = deque(maxlen=self.rolling_window_size)

        # Spike detection
        self.spike_threshold = spike_threshold

        # Thresholds
        self.medium_threshold = (
            medium_threshold if medium_threshold > 0
            else settings.risk_threshold_medium
        )
        self.high_threshold = (
            high_threshold if high_threshold > 0
            else settings.risk_threshold_high
        )

        # Hysteresis state — tracks whether each level has already fired
        # and whether the score has dropped far enough to re-arm.
        self._high_fired = False
        self._medium_fired = False
        self._high_rearm_threshold = self.high_threshold - 10   # e.g. 75
        self._medium_rearm_threshold = self.medium_threshold - 10  # e.g. 60

        # Accumulators
        self._chunk_count: int = 0
        self._peak_risk: float = 0.0
        self._all_scores: List[Dict] = []

        logger.info(
            "risk_engine.created",
            extra={
                "call_sid": call_sid,
                "caller_number": caller_number,
                "caller_reputation": caller_reputation.value,
                "caller_multiplier": self.caller_multiplier,
                "rolling_window": self.rolling_window_size,
                "medium_threshold": self.medium_threshold,
                "high_threshold": self.high_threshold,
            },
        )

    # ----- public API -----

    def process_chunk(
        self,
        acoustic_score: float,
        context_score: float,
        timestamp_sec: Optional[float] = None,
    ) -> Dict:
        """
        Ingest scores for a single audio chunk and return the updated
        risk assessment.

        Args:
            acoustic_score: Ensemble spoof score in [0, 100].
            context_score:  NLP risk-phrase score in [0, 100].
            timestamp_sec:  Chunk timestamp within the call (seconds).

        Returns:
            Dict with keys:
              - raw_score          (float)  before multiplier
              - adjusted_score     (float)  after multiplier, before smoothing
              - current_risk       (float)  smoothed rolling risk
              - peak_risk          (float)  highest score so far
              - is_spike           (bool)   True if spike detection fired
              - threshold_crossed  (str|None)  "medium", "high", or None
              - chunk_index        (int)
              - caller_multiplier  (float)
        """
        # Clamp inputs
        acoustic_score = max(0.0, min(100.0, acoustic_score))
        context_score = max(0.0, min(100.0, context_score))

        # --- Step 112: composite score ---
        raw_score = (
            self.acoustic_weight * acoustic_score
            + self.context_weight * context_score
        )
        adjusted_score = min(100.0, raw_score * self.caller_multiplier)

        # --- Step 114: spike detection ---
        is_spike = adjusted_score >= self.spike_threshold

        # --- Step 113: rolling average ---
        self._window.append(adjusted_score)

        if is_spike:
            # Bypass rolling average — use the spike score directly
            current_risk = adjusted_score
        else:
            current_risk = sum(self._window) / len(self._window)

        # Update peak
        self._peak_risk = max(self._peak_risk, current_risk)

        # --- Threshold crossing with hysteresis (Step 119 prep) ---
        threshold_crossed = self._check_threshold(current_risk)

        # Record
        ts = timestamp_sec if timestamp_sec is not None else (
            time.time() - self.start_time
        )
        entry = {
            "chunk_index": self._chunk_count,
            "timestamp_sec": round(ts, 3),
            "acoustic_score": round(acoustic_score, 2),
            "context_score": round(context_score, 2),
            "raw_score": round(raw_score, 2),
            "adjusted_score": round(adjusted_score, 2),
            "current_risk": round(current_risk, 2),
            "is_spike": is_spike,
            "threshold_crossed": threshold_crossed,
        }
        self._all_scores.append(entry)
        self._chunk_count += 1

        logger.debug(
            "risk_engine.chunk_processed",
            extra={
                "call_sid": self.call_sid,
                **entry,
            },
        )

        return {
            **entry,
            "peak_risk": round(self._peak_risk, 2),
            "caller_multiplier": self.caller_multiplier,
        }

    def get_current_risk(self) -> float:
        """Return the current smoothed risk score (or 0 if no chunks yet)."""
        if not self._window:
            return 0.0
        return round(sum(self._window) / len(self._window), 2)

    def get_peak_risk(self) -> float:
        """Return the highest risk score observed during the call."""
        return round(self._peak_risk, 2)

    def get_final_verdict(self) -> str:
        """
        Return a human-readable verdict for the completed call.

        Returns:
            'safe', 'suspicious', or 'high_risk'
        """
        peak = self._peak_risk
        if peak >= self.high_threshold:
            return "high_risk"
        elif peak >= self.medium_threshold:
            return "suspicious"
        return "safe"

    @property
    def chunk_count(self) -> int:
        """Number of chunks processed so far."""
        return self._chunk_count

    @property
    def score_history(self) -> List[Dict]:
        """Full list of per-chunk score records."""
        return list(self._all_scores)

    @property
    def stats(self) -> Dict:
        """Summary statistics for this call."""
        return {
            "call_sid": self.call_sid,
            "caller_number": self.caller_number,
            "caller_reputation": self.caller_reputation.value,
            "caller_multiplier": self.caller_multiplier,
            "chunk_count": self._chunk_count,
            "current_risk": self.get_current_risk(),
            "peak_risk": self.get_peak_risk(),
            "verdict": self.get_final_verdict(),
            "duration_sec": round(time.time() - self.start_time, 1),
        }

    # ----- internals -----

    def _check_threshold(self, current_risk: float) -> Optional[str]:
        """
        Determine if a threshold was crossed, respecting hysteresis.

        Hysteresis logic:
          - Once 'high' fires, it won't fire again until the score
            drops below (high_threshold - 10) and then re-crosses.
          - Same for 'medium'.

        Returns:
            "high", "medium", or None.
        """
        crossed: Optional[str] = None

        # --- HIGH ---
        if current_risk >= self.high_threshold:
            if not self._high_fired:
                crossed = "high"
                self._high_fired = True
                # Also suppress medium (high supersedes)
                self._medium_fired = True
                logger.warning(
                    "risk_engine.threshold_crossed",
                    extra={
                        "call_sid": self.call_sid,
                        "level": "high",
                        "score": round(current_risk, 2),
                    },
                )
        else:
            # Re-arm high if score dropped below re-arm line
            if self._high_fired and current_risk < self._high_rearm_threshold:
                self._high_fired = False

        # --- MEDIUM (only if high didn't just fire) ---
        if crossed is None and current_risk >= self.medium_threshold:
            if not self._medium_fired:
                crossed = "medium"
                self._medium_fired = True
                logger.warning(
                    "risk_engine.threshold_crossed",
                    extra={
                        "call_sid": self.call_sid,
                        "level": "medium",
                        "score": round(current_risk, 2),
                    },
                )
        else:
            if self._medium_fired and current_risk < self._medium_rearm_threshold:
                self._medium_fired = False

        return crossed

    def update_caller_reputation(self, reputation: CallerReputation) -> None:
        """
        Change the caller multiplier mid-call (e.g. after a contact lookup
        completes asynchronously).
        """
        self.caller_reputation = reputation
        self.caller_multiplier = _CALLER_MULTIPLIERS.get(
            reputation, settings.caller_multiplier_unknown,
        )
        logger.info(
            "risk_engine.reputation_updated",
            extra={
                "call_sid": self.call_sid,
                "new_reputation": reputation.value,
                "new_multiplier": self.caller_multiplier,
            },
        )
