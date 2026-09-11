"""
VoiceGuard — Ensemble Scorer (Steps 93-95)

Combines scores from two detection models:
  - Model A: AASIST deep neural network (spoof probability)
  - Model B: XGBoost feature-based classifier (spoof probability)

into a single acoustic_score in [0, 100] range.

Supports:
  - Configurable weighted average (default: 0.6 × A + 0.4 × B)
  - Confidence-weighted fusion (low-confidence outputs get reduced weight)
  - Weights tunable via environment variables
"""

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default ensemble weights (tunable via env vars)
DEFAULT_WEIGHT_A = float(os.environ.get("ENSEMBLE_WEIGHT_A", "0.6"))
DEFAULT_WEIGHT_B = float(os.environ.get("ENSEMBLE_WEIGHT_B", "0.4"))

# Confidence threshold: scores within this distance of 0.5 are "low confidence"
CONFIDENCE_DEAD_ZONE = float(os.environ.get("ENSEMBLE_CONFIDENCE_DEAD_ZONE", "0.15"))


class EnsembleScorer:
    """
    Combines Model A (AASIST) and Model B (XGBoost) spoof probabilities
    into a single acoustic risk score.

    Usage:
        scorer = EnsembleScorer()

        # Basic weighted average
        score = scorer.combine(model_a_score=0.85, model_b_score=0.72)
        # score ≈ 80.2 (out of 100)

        # With detailed breakdown
        result = scorer.combine_with_details(0.85, 0.72)
        # {'acoustic_score': 80.2, 'model_a_score': 0.85, ...}
    """

    def __init__(
        self,
        weight_a: float = DEFAULT_WEIGHT_A,
        weight_b: float = DEFAULT_WEIGHT_B,
        confidence_dead_zone: float = CONFIDENCE_DEAD_ZONE,
        use_confidence_weighting: bool = True,
    ):
        """
        Args:
            weight_a: Base weight for Model A (AASIST). Default 0.6.
            weight_b: Base weight for Model B (XGBoost). Default 0.4.
            confidence_dead_zone: Distance from 0.5 below which a model's
                output is considered low-confidence. Default 0.15.
            use_confidence_weighting: If True, dynamically adjust weights
                based on model confidence.
        """
        # Normalize weights to sum to 1.0
        total = weight_a + weight_b
        self.weight_a = weight_a / total
        self.weight_b = weight_b / total
        self.confidence_dead_zone = confidence_dead_zone
        self.use_confidence_weighting = use_confidence_weighting

    def _compute_confidence(self, score: float) -> float:
        """
        Compute confidence level for a model's output.

        Scores near 0.5 have low confidence (model is uncertain).
        Scores near 0.0 or 1.0 have high confidence.

        Returns a value in [0.0, 1.0] where:
          - 1.0 = full confidence (score at 0 or 1)
          - 0.0 = no confidence (score at 0.5)
        """
        distance_from_center = abs(score - 0.5)
        # Normalize to [0, 1]: distance 0.5 → confidence 1.0
        confidence = min(distance_from_center / 0.5, 1.0)
        return confidence

    def _confidence_adjusted_weights(
        self,
        score_a: float,
        score_b: float,
    ) -> Tuple[float, float]:
        """
        Adjust ensemble weights based on individual model confidence.

        If one model is uncertain (score near 0.5), its weight is reduced
        and the other model's weight is increased proportionally.

        Returns:
            (adjusted_weight_a, adjusted_weight_b) normalized to sum to 1.0.
        """
        conf_a = self._compute_confidence(score_a)
        conf_b = self._compute_confidence(score_b)

        # If model output is within dead zone of 0.5, reduce its effective weight
        if abs(score_a - 0.5) < self.confidence_dead_zone:
            conf_a *= 0.3  # Drastically reduce weight for uncertain model
        if abs(score_b - 0.5) < self.confidence_dead_zone:
            conf_b *= 0.3

        # Apply confidence as a multiplier on base weights
        effective_a = self.weight_a * (0.5 + 0.5 * conf_a)
        effective_b = self.weight_b * (0.5 + 0.5 * conf_b)

        # Normalize
        total = effective_a + effective_b
        if total < 1e-10:
            return self.weight_a, self.weight_b

        return effective_a / total, effective_b / total

    def combine(
        self,
        model_a_score: float,
        model_b_score: Optional[float] = None,
    ) -> float:
        """
        Combine model scores into a single acoustic score.

        Args:
            model_a_score: AASIST spoof probability in [0, 1].
            model_b_score: XGBoost spoof probability in [0, 1].
                           If None, only Model A is used.

        Returns:
            Combined acoustic_score in [0, 100] range.
            0 = definitely bonafide, 100 = definitely spoofed.
        """
        # Clamp inputs
        model_a_score = max(0.0, min(1.0, model_a_score))

        if model_b_score is None:
            # Single-model mode: just scale to [0, 100]
            return model_a_score * 100.0

        model_b_score = max(0.0, min(1.0, model_b_score))

        # Get weights (with or without confidence adjustment)
        if self.use_confidence_weighting:
            w_a, w_b = self._confidence_adjusted_weights(model_a_score, model_b_score)
        else:
            w_a, w_b = self.weight_a, self.weight_b

        # Weighted average
        combined = (w_a * model_a_score) + (w_b * model_b_score)

        # Scale to [0, 100]
        acoustic_score = combined * 100.0
        return max(0.0, min(100.0, acoustic_score))

    def combine_with_details(
        self,
        model_a_score: float,
        model_b_score: Optional[float] = None,
    ) -> Dict:
        """
        Combine scores and return detailed breakdown.

        Returns:
            Dict with keys:
            - acoustic_score: float in [0, 100]
            - model_a_score: float in [0, 1]
            - model_b_score: float in [0, 1] or None
            - model_a_confidence: float in [0, 1]
            - model_b_confidence: float in [0, 1] or None
            - weight_a_effective: float
            - weight_b_effective: float
            - agreement: 'strong_agree', 'agree', 'disagree', 'strong_disagree'
        """
        acoustic_score = self.combine(model_a_score, model_b_score)

        result = {
            "acoustic_score": round(acoustic_score, 2),
            "model_a_score": round(model_a_score, 4),
            "model_b_score": round(model_b_score, 4) if model_b_score is not None else None,
            "model_a_confidence": round(self._compute_confidence(model_a_score), 4),
        }

        if model_b_score is not None:
            result["model_b_confidence"] = round(self._compute_confidence(model_b_score), 4)

            if self.use_confidence_weighting:
                w_a, w_b = self._confidence_adjusted_weights(model_a_score, model_b_score)
            else:
                w_a, w_b = self.weight_a, self.weight_b

            result["weight_a_effective"] = round(w_a, 4)
            result["weight_b_effective"] = round(w_b, 4)

            # Assess agreement
            both_high = model_a_score > 0.7 and model_b_score > 0.7
            both_low = model_a_score < 0.3 and model_b_score < 0.3
            diff = abs(model_a_score - model_b_score)

            if both_high or both_low:
                result["agreement"] = "strong_agree"
            elif diff < 0.2:
                result["agreement"] = "agree"
            elif diff < 0.4:
                result["agreement"] = "disagree"
            else:
                result["agreement"] = "strong_disagree"
        else:
            result["weight_a_effective"] = 1.0
            result["weight_b_effective"] = 0.0
            result["agreement"] = "single_model"

        return result

    def update_weights(self, weight_a: float, weight_b: float) -> None:
        """
        Update ensemble weights at runtime (for demo tuning).

        Args:
            weight_a: New weight for Model A.
            weight_b: New weight for Model B.
        """
        total = weight_a + weight_b
        self.weight_a = weight_a / total
        self.weight_b = weight_b / total
        logger.info(f"Ensemble weights updated: A={self.weight_a:.2f}, B={self.weight_b:.2f}")
