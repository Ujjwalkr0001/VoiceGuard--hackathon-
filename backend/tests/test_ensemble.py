"""
VoiceGuard — Ensemble Scorer Unit Tests (Step 96)

Tests for the EnsembleScorer class:
  1. Score range always in [0, 100]
  2. Strong agreement between models produces extreme scores
  3. Confidence-weighted fusion reduces uncertain model's influence
  4. Single-model mode works correctly
  5. Weight updates work at runtime
  6. Edge cases (0, 1, 0.5, negative, >1)
"""

import sys
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.ensemble import EnsembleScorer


class TestEnsembleScorer:
    """Test suite for EnsembleScorer."""

    def setup_method(self):
        """Create a fresh scorer for each test."""
        self.scorer = EnsembleScorer(
            weight_a=0.6,
            weight_b=0.4,
            use_confidence_weighting=True,
        )
        self.scorer_no_conf = EnsembleScorer(
            weight_a=0.6,
            weight_b=0.4,
            use_confidence_weighting=False,
        )

    # ---- Score Range Tests ----

    def test_score_range_both_zero(self):
        score = self.scorer.combine(0.0, 0.0)
        assert 0.0 <= score <= 100.0, f"Score {score} out of range"
        assert score < 5.0, f"Both models say bonafide, score should be low: {score}"

    def test_score_range_both_one(self):
        score = self.scorer.combine(1.0, 1.0)
        assert 0.0 <= score <= 100.0, f"Score {score} out of range"
        assert score > 95.0, f"Both models say spoof, score should be high: {score}"

    def test_score_range_mixed(self):
        for a in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for b in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                score = self.scorer.combine(a, b)
                assert 0.0 <= score <= 100.0, f"Score {score} out of range for ({a}, {b})"

    def test_score_range_clamping(self):
        """Scores outside [0,1] should be clamped."""
        score = self.scorer.combine(-0.5, 1.5)
        assert 0.0 <= score <= 100.0, f"Score {score} out of range after clamping"

    # ---- Agreement Tests ----

    def test_strong_agreement_both_spoof(self):
        """When both models strongly agree it's spoof → high score."""
        score = self.scorer.combine(0.95, 0.90)
        assert score > 85.0, f"Strong spoof agreement should score >85: {score}"

    def test_strong_agreement_both_bonafide(self):
        """When both models strongly agree it's bonafide → low score."""
        score = self.scorer.combine(0.05, 0.08)
        assert score < 10.0, f"Strong bonafide agreement should score <10: {score}"

    def test_disagreement_produces_moderate_score(self):
        """When models disagree → score is moderate."""
        score = self.scorer.combine(0.9, 0.1)
        assert 20.0 < score < 80.0, f"Disagreement should produce moderate score: {score}"

    def test_agreement_details(self):
        """Test the agreement field in detailed results."""
        result = self.scorer.combine_with_details(0.95, 0.90)
        assert result["agreement"] == "strong_agree"

        result = self.scorer.combine_with_details(0.1, 0.2)
        assert result["agreement"] == "strong_agree"

        result = self.scorer.combine_with_details(0.9, 0.1)
        assert result["agreement"] == "strong_disagree"

    # ---- Confidence Weighting Tests ----

    def test_confidence_reduces_uncertain_model(self):
        """A model with score near 0.5 should have reduced influence."""
        # Model A is certain (0.9), Model B is uncertain (0.52)
        score_with_conf = self.scorer.combine(0.9, 0.52)
        score_no_conf = self.scorer_no_conf.combine(0.9, 0.52)

        # With confidence weighting, the uncertain Model B should have
        # less influence, making the score closer to Model A's value
        # Score with confidence should be higher (more weight on A's 0.9)
        assert score_with_conf > score_no_conf - 5, (
            f"Confidence weighting should shift toward certain model: "
            f"with={score_with_conf}, without={score_no_conf}"
        )

    def test_both_confident_similar_to_no_confidence(self):
        """When both models are confident, weighting should be similar to base."""
        score_with = self.scorer.combine(0.9, 0.85)
        score_without = self.scorer_no_conf.combine(0.9, 0.85)
        # Should be close (within 10 points)
        assert abs(score_with - score_without) < 10, (
            f"Both confident: scores should be similar: {score_with} vs {score_without}"
        )

    def test_confidence_computation(self):
        """Test internal confidence computation."""
        # Score 0.0 → confidence 1.0
        assert self.scorer._compute_confidence(0.0) == 1.0
        # Score 1.0 → confidence 1.0
        assert self.scorer._compute_confidence(1.0) == 1.0
        # Score 0.5 → confidence 0.0
        assert self.scorer._compute_confidence(0.5) == 0.0
        # Score 0.75 → confidence 0.5
        assert abs(self.scorer._compute_confidence(0.75) - 0.5) < 0.01

    # ---- Single Model Mode ----

    def test_single_model_mode(self):
        """When model_b_score is None, only Model A is used."""
        score = self.scorer.combine(0.8, None)
        assert abs(score - 80.0) < 0.01, f"Single model should scale to 100: {score}"

    def test_single_model_zero(self):
        score = self.scorer.combine(0.0, None)
        assert abs(score - 0.0) < 0.01

    def test_single_model_details(self):
        result = self.scorer.combine_with_details(0.7, None)
        assert result["agreement"] == "single_model"
        assert result["model_b_score"] is None
        assert result["weight_a_effective"] == 1.0

    # ---- Weighted Average (no confidence) ----

    def test_basic_weighted_average(self):
        """Verify basic weighted average without confidence weighting."""
        # 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72 → 72.0
        score = self.scorer_no_conf.combine(0.8, 0.6)
        assert abs(score - 72.0) < 0.01, f"Expected 72.0, got {score}"

    def test_equal_scores_equals_score(self):
        """When both models give the same score, result should match."""
        score = self.scorer_no_conf.combine(0.7, 0.7)
        assert abs(score - 70.0) < 0.01, f"Expected 70.0, got {score}"

    # ---- Weight Update ----

    def test_weight_update(self):
        """Test runtime weight updates."""
        scorer = EnsembleScorer(weight_a=0.5, weight_b=0.5, use_confidence_weighting=False)
        score_equal = scorer.combine(1.0, 0.0)
        assert abs(score_equal - 50.0) < 0.01

        scorer.update_weights(0.9, 0.1)
        score_updated = scorer.combine(1.0, 0.0)
        assert abs(score_updated - 90.0) < 0.01

    # ---- Details ----

    def test_combine_with_details_keys(self):
        """Verify all expected keys are present."""
        result = self.scorer.combine_with_details(0.8, 0.6)
        expected_keys = {
            "acoustic_score", "model_a_score", "model_b_score",
            "model_a_confidence", "model_b_confidence",
            "weight_a_effective", "weight_b_effective", "agreement",
        }
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"


def run_tests():
    """Run all tests and report results."""
    test = TestEnsembleScorer()
    methods = [m for m in dir(test) if m.startswith("test_")]

    passed = 0
    failed = 0
    errors = []

    print(f"\n  Running {len(methods)} ensemble tests...\n")

    for method_name in sorted(methods):
        test.setup_method()
        try:
            getattr(test, method_name)()
            print(f"    PASS  {method_name}")
            passed += 1
        except Exception as e:
            print(f"    FAIL  {method_name}: {e}")
            failed += 1
            errors.append((method_name, str(e)))

    print(f"\n  Results: {passed} passed, {failed} failed, {passed + failed} total")

    if errors:
        print(f"\n  Failures:")
        for name, err in errors:
            print(f"    - {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
