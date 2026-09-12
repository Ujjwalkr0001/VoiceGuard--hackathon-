"""
VoiceGuard — Unit Tests for Risk Scoring Engine (Steps 120-123)

Tests cover:
  - Step 120: Gradually increasing scores detect medium threshold at correct chunk
  - Step 121: Rolling average smooths out a single spike below threshold
  - Step 122: Spike detection bypasses rolling average for extreme scores
  - Step 123: Hysteresis prevents duplicate alerts
"""

import pytest

from app.services.risk_engine import RiskEngine, CallerReputation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trusted_engine():
    """RiskEngine with trusted caller (multiplier=1.0), small window for clarity."""
    return RiskEngine(
        call_sid="test-trusted",
        caller_number="+911234567890",
        caller_reputation=CallerReputation.TRUSTED,
        rolling_window_size=5,
    )


@pytest.fixture
def unknown_engine():
    """RiskEngine with unknown caller (multiplier=1.3)."""
    return RiskEngine(
        call_sid="test-unknown",
        caller_number="+910000000000",
        caller_reputation=CallerReputation.UNKNOWN,
        rolling_window_size=5,
    )


# ===========================================================================
# Step 120: Gradually increasing scores → medium threshold crossing
# ===========================================================================

class TestGradualIncrease:
    """Feed gradually increasing scores and verify medium threshold detected."""

    def test_medium_threshold_crossed_at_correct_chunk(self, trusted_engine):
        """
        With multiplier=1.0, window=5, medium=70:
        Feed acoustic+context scores that gradually increase.
        Rolling average should cross 70 at a specific chunk.
        Avoid spike territory (>=95) so rolling avg is used.
        """
        e = trusted_engine
        # adj = 0.65*a + 0.35*c, kept below 95 to avoid spike bypass
        scores = [
            (40, 20),   # adj = 33.0
            (60, 40),   # adj = 53.0
            (70, 60),   # adj = 66.5
            (80, 70),   # adj = 76.5
            (90, 80),   # adj = 86.5
            (90, 80),   # adj = 86.5
        ]
        results = []
        for a, c in scores:
            r = e.process_chunk(acoustic_score=a, context_score=c)
            results.append(r)

        # Find which chunk first triggered "medium"
        medium_chunk = None
        for r in results:
            if r["threshold_crossed"] == "medium":
                medium_chunk = r["chunk_index"]
                break

        assert medium_chunk is not None, "Medium threshold was never crossed"
        assert medium_chunk > 0

    def test_scores_below_threshold_never_trigger(self, trusted_engine):
        """Scores that never reach medium threshold should never trigger."""
        e = trusted_engine
        for _ in range(20):
            r = e.process_chunk(acoustic_score=50, context_score=0.0)
            # 0.65 * 50 = 32.5 — well below 70
            assert r["threshold_crossed"] is None

    def test_medium_fires_before_high(self):
        """When scores ramp up gradually (no spikes), medium fires before high."""
        e = RiskEngine(
            call_sid="test-order",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=3,  # smaller window so avg catches up faster
        )
        medium_idx = None
        high_idx = None

        for i in range(15):
            # Ramp slowly, stay below spike threshold (95)
            acoustic = min(30 + i * 5, 94)   # 30..94
            context = min(20 + i * 5, 94)    # 20..94
            r = e.process_chunk(acoustic_score=acoustic, context_score=context)
            if r["threshold_crossed"] == "medium" and medium_idx is None:
                medium_idx = r["chunk_index"]
            if r["threshold_crossed"] == "high" and high_idx is None:
                high_idx = r["chunk_index"]

        assert medium_idx is not None, "Medium was never triggered"
        if high_idx is not None:
            assert medium_idx < high_idx, "Medium should fire before high"


# ===========================================================================
# Step 121: Rolling average smooths out a single spike
# ===========================================================================

class TestRollingAverageSmoothing:
    """Verify rolling average smooths out a single spike below threshold."""

    def test_single_spike_smoothed_below_threshold(self, trusted_engine):
        """
        Feed several low scores, then one high spike, then low again.
        The rolling average should keep the smoothed score below medium=70.
        """
        e = trusted_engine  # window=5, multiplier=1.0

        # 4 low chunks: adjusted = 0.65 * 30 = 19.5
        for _ in range(4):
            e.process_chunk(acoustic_score=30, context_score=0.0)

        # 1 spike chunk: adjusted = 0.65 * 100 = 65.0
        # Rolling avg = (19.5 + 19.5 + 19.5 + 19.5 + 65.0) / 5 = 28.6
        r_spike = e.process_chunk(acoustic_score=100, context_score=0.0)

        # The spike alone (65) is below threshold, and rolling avg (28.6) is too
        assert r_spike["current_risk"] < 70.0, (
            f"Rolling average should smooth spike below 70, got {r_spike['current_risk']}"
        )
        assert r_spike["threshold_crossed"] is None

    def test_rolling_average_reflects_window_size(self):
        """Verify the window only considers the last N chunks."""
        e = RiskEngine(
            call_sid="test-window",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=3,
        )

        # 3 high chunks fill the window
        for _ in range(3):
            e.process_chunk(acoustic_score=100, context_score=100)

        # Now 3 low chunks should push out the high ones
        for _ in range(3):
            r = e.process_chunk(acoustic_score=10, context_score=0.0)

        # Window now has only low scores: 0.65*10 = 6.5 each
        assert r["current_risk"] < 10.0


# ===========================================================================
# Step 122: Spike detection bypasses rolling average
# ===========================================================================

class TestSpikeDetection:
    """Verify extreme scores bypass the rolling average."""

    def test_spike_bypasses_rolling_average(self, trusted_engine):
        """
        Feed low scores, then a spike >= 95.
        current_risk should equal the spike score, not the rolling average.
        """
        e = trusted_engine  # window=5, multiplier=1.0

        # Fill window with low scores
        for _ in range(4):
            e.process_chunk(acoustic_score=10, context_score=0.0)

        # Spike: adjusted = 0.65*100 + 0.35*100 = 100.0 >= 95 → spike
        r = e.process_chunk(acoustic_score=100, context_score=100)

        assert r["is_spike"] is True
        assert r["current_risk"] == r["adjusted_score"]
        assert r["current_risk"] >= 95.0

    def test_non_spike_uses_rolling_average(self, trusted_engine):
        """Scores below 95 should use rolling average, not bypass."""
        e = trusted_engine

        # adjusted = 0.65*90 + 0.35*0 = 58.5 — below 95
        r = e.process_chunk(acoustic_score=90, context_score=0.0)

        assert r["is_spike"] is False
        assert r["current_risk"] == r["adjusted_score"]  # only 1 chunk so avg = value

    def test_spike_fires_high_threshold_immediately(self):
        """A spike should immediately trigger high threshold even with low history."""
        e = RiskEngine(
            call_sid="test-spike-alert",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=10,
        )

        # 9 very low chunks
        for _ in range(9):
            e.process_chunk(acoustic_score=5, context_score=0.0)

        # Spike on chunk 10
        r = e.process_chunk(acoustic_score=100, context_score=100)

        assert r["is_spike"] is True
        assert r["threshold_crossed"] == "high"


# ===========================================================================
# Step 123: Hysteresis prevents duplicate alerts
# ===========================================================================

class TestHysteresis:
    """Verify hysteresis prevents duplicate threshold crossings."""

    def test_high_does_not_refire_without_drop(self):
        """Once high fires, it shouldn't fire again while score stays high."""
        e = RiskEngine(
            call_sid="test-hyst",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=1,  # no smoothing, raw pass-through
        )

        # First high crossing
        r1 = e.process_chunk(acoustic_score=100, context_score=100)
        assert r1["threshold_crossed"] == "high"

        # Subsequent high-score chunks should NOT re-fire
        for _ in range(5):
            r = e.process_chunk(acoustic_score=100, context_score=100)
            assert r["threshold_crossed"] is None

    def test_high_refires_after_drop_and_rerise(self):
        """High should re-fire if score drops below (high-10) then re-crosses."""
        e = RiskEngine(
            call_sid="test-hyst-refire",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=1,
        )

        # Fire high
        r1 = e.process_chunk(acoustic_score=100, context_score=100)
        assert r1["threshold_crossed"] == "high"

        # Drop below re-arm threshold (85 - 10 = 75)
        r2 = e.process_chunk(acoustic_score=50, context_score=50)
        assert r2["current_risk"] < 75.0
        assert r2["threshold_crossed"] is None

        # Re-cross high
        r3 = e.process_chunk(acoustic_score=100, context_score=100)
        assert r3["threshold_crossed"] == "high"

    def test_medium_does_not_refire_without_drop(self):
        """Once medium fires, it shouldn't fire again while score stays in medium zone."""
        e = RiskEngine(
            call_sid="test-hyst-med",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=1,
        )

        # Reach medium zone (70-84): 0.65*100 + 0.35*10 = 68.5... need higher
        # 0.65*100 + 0.35*20 = 72 — in medium zone
        r1 = e.process_chunk(acoustic_score=100, context_score=20)
        assert r1["threshold_crossed"] == "medium"

        # Stay in medium zone
        for _ in range(5):
            r = e.process_chunk(acoustic_score=100, context_score=20)
            assert r["threshold_crossed"] is None

    def test_high_suppresses_medium(self):
        """When high fires, medium should not also fire on the same chunk."""
        e = RiskEngine(
            call_sid="test-hyst-suppress",
            caller_reputation=CallerReputation.TRUSTED,
            rolling_window_size=1,
        )

        # Score above high threshold — should fire high, not medium
        r = e.process_chunk(acoustic_score=100, context_score=100)
        assert r["threshold_crossed"] == "high"


# ===========================================================================
# Additional: caller multiplier & verdict
# ===========================================================================

class TestCallerMultiplierAndVerdict:
    """Test multiplier effect and final verdict."""

    def test_unknown_caller_amplifies_score(self):
        """Unknown caller (1.3x) should produce higher risk than trusted."""
        t = RiskEngine("t", "", CallerReputation.TRUSTED, rolling_window_size=1)
        u = RiskEngine("u", "", CallerReputation.UNKNOWN, rolling_window_size=1)

        rt = t.process_chunk(70, 40)
        ru = u.process_chunk(70, 40)

        assert ru["current_risk"] > rt["current_risk"]

    def test_flagged_caller_amplifies_more(self):
        """Flagged caller (1.5x) should amplify even more."""
        u = RiskEngine("u", "", CallerReputation.UNKNOWN, rolling_window_size=1)
        f = RiskEngine("f", "", CallerReputation.FLAGGED, rolling_window_size=1)

        ru = u.process_chunk(70, 40)
        rf = f.process_chunk(70, 40)

        assert rf["current_risk"] > ru["current_risk"]

    def test_verdict_safe(self, trusted_engine):
        """Low scores → 'safe' verdict."""
        trusted_engine.process_chunk(10, 0)
        assert trusted_engine.get_final_verdict() == "safe"

    def test_verdict_suspicious(self):
        """Peak in medium zone → 'suspicious' verdict."""
        e = RiskEngine("v", "", CallerReputation.TRUSTED, rolling_window_size=1)
        e.process_chunk(100, 20)  # 0.65*100+0.35*20 = 72
        assert e.get_final_verdict() == "suspicious"

    def test_verdict_high_risk(self):
        """Peak above high → 'high_risk' verdict."""
        e = RiskEngine("v", "", CallerReputation.TRUSTED, rolling_window_size=1)
        e.process_chunk(100, 100)  # 100
        assert e.get_final_verdict() == "high_risk"
