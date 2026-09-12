"""
VoiceGuard — Unit Tests for Context/Risk-Phrase Analyzer (Step 108)

Tests cover:
  - English risk phrase detection (critical, high, medium tiers)
  - Hindi risk phrase detection
  - Hinglish (code-mixed) risk phrase detection
  - Risk score computation (severity weighting, density, clamping)
  - Edge cases: empty text, no matches, very short text, overlapping spans
  - Full analyze() output structure and signal generation
"""

import pytest

from app.ml.context_analyzer import (
    ContextAnalyzer,
    RiskPhrase,
    PhraseMatch,
    RISK_PHRASES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Default ContextAnalyzer with built-in risk phrases."""
    return ContextAnalyzer()


@pytest.fixture
def analyzer_with_custom():
    """ContextAnalyzer with additional custom phrases."""
    custom = [
        RiskPhrase("test phrase", "critical", 1.0, "en", "credential"),
    ]
    return ContextAnalyzer(custom_phrases=custom)


# ===========================================================================
# 1. English phrase detection
# ===========================================================================

class TestEnglishPhraseDetection:
    """Test detection of English risk phrases across all severity tiers."""

    def test_critical_otp(self, analyzer):
        """Detect critical-tier 'OTP' keyword."""
        result = analyzer.detect_phrases("Please share your OTP to proceed")
        phrases = [m.phrase for m in result]
        assert "otp" in phrases or "share your otp" in phrases

    def test_critical_password(self, analyzer):
        """Detect critical-tier 'password' keyword."""
        result = analyzer.detect_phrases("Enter your password now")
        phrases = [m.phrase for m in result]
        assert "password" in phrases

    def test_critical_upi_pin(self, analyzer):
        """Detect critical-tier 'UPI PIN'."""
        result = analyzer.detect_phrases("Tell me your UPI PIN for verification")
        phrases = [m.phrase for m in result]
        assert "upi pin" in phrases

    def test_critical_cvv(self, analyzer):
        """Detect critical-tier 'CVV'."""
        result = analyzer.detect_phrases("What is the CVV on the back of your card?")
        phrases = [m.phrase for m in result]
        assert "cvv" in phrases

    def test_critical_aadhaar(self, analyzer):
        """Detect critical-tier 'Aadhaar number'."""
        result = analyzer.detect_phrases("We need your Aadhaar number for verification")
        phrases = [m.phrase for m in result]
        assert "aadhaar number" in phrases

    def test_high_transfer_money(self, analyzer):
        """Detect high-tier 'transfer money'."""
        result = analyzer.detect_phrases("You need to transfer money immediately")
        phrases = [m.phrase for m in result]
        assert "transfer money" in phrases

    def test_high_bank_account(self, analyzer):
        """Detect high-tier 'bank account'."""
        result = analyzer.detect_phrases("Share your bank account details")
        phrases = [m.phrase for m in result]
        assert "bank account" in phrases

    def test_high_neft(self, analyzer):
        """Detect high-tier 'NEFT'."""
        result = analyzer.detect_phrases("Please do NEFT transfer to this account")
        phrases = [m.phrase for m in result]
        assert "neft" in phrases

    def test_medium_urgent(self, analyzer):
        """Detect medium-tier 'urgent'."""
        result = analyzer.detect_phrases("This is urgent, act now")
        phrases = [m.phrase for m in result]
        assert "urgent" in phrases

    def test_medium_kyc(self, analyzer):
        """Detect medium-tier 'KYC'."""
        result = analyzer.detect_phrases("Your KYC verification is pending")
        phrases = [m.phrase for m in result]
        assert "kyc" in phrases or "kyc verification" in phrases

    def test_medium_block_account(self, analyzer):
        """Detect medium-tier 'block account' / 'suspend'."""
        result = analyzer.detect_phrases("We will block your account if you don't comply")
        matches = analyzer.detect_phrases("We will block your account if you don't comply")
        tiers = [m.tier for m in matches]
        assert "medium" in tiers

    def test_case_insensitive(self, analyzer):
        """English phrase matching should be case-insensitive."""
        result_upper = analyzer.detect_phrases("SHARE YOUR OTP NOW")
        result_lower = analyzer.detect_phrases("share your otp now")
        assert len(result_upper) > 0
        assert len(result_lower) > 0

    def test_multiple_phrases_in_one_text(self, analyzer):
        """Detect multiple distinct risk phrases in a single text."""
        text = "Share your OTP and transfer money to this bank account immediately"
        result = analyzer.detect_phrases(text)
        # Should find at least 3 different phrases
        assert len(result) >= 3
        tiers_found = set(m.tier for m in result)
        # Should span multiple tiers
        assert len(tiers_found) >= 2


# ===========================================================================
# 2. Hindi phrase detection
# ===========================================================================

class TestHindiPhraseDetection:
    """Test detection of Hindi risk phrases."""

    def test_hindi_otp(self, analyzer):
        """Detect Hindi 'ओटीपी' (OTP)."""
        result = analyzer.detect_phrases("कृपया अपना ओटीपी बताइए")
        phrases = [m.phrase for m in result]
        assert "ओटीपी" in phrases or "ओटीपी बताइए" in phrases

    def test_hindi_password(self, analyzer):
        """Detect Hindi 'पासवर्ड' (password)."""
        result = analyzer.detect_phrases("अपना पासवर्ड दर्ज करें")
        phrases = [m.phrase for m in result]
        assert "पासवर्ड" in phrases

    def test_hindi_upi_pin(self, analyzer):
        """Detect Hindi 'यूपीआई पिन' (UPI PIN)."""
        result = analyzer.detect_phrases("अपना यूपीआई पिन बताइए")
        phrases = [m.phrase for m in result]
        assert "यूपीआई पिन" in phrases

    def test_hindi_transfer_money(self, analyzer):
        """Detect Hindi financial phrases like 'पैसे भेजो' or 'पैसे ट्रांसफर'."""
        result = analyzer.detect_phrases("तुरंत पैसे भेजिए इस खाते में")
        phrases = [m.phrase for m in result]
        # Should detect at least one financial or urgency phrase
        assert len(result) >= 1
        # 'पैसे भेजिए' is in the phrase list (high-tier)
        assert "पैसे भेजिए" in phrases or any(m.tier in ("high", "medium") for m in result)

    def test_hindi_urgency(self, analyzer):
        """Detect Hindi urgency phrases 'तुरंत' / 'जल्दी'."""
        result = analyzer.detect_phrases("तुरंत भुगतान करें नहीं तो खाता बंद हो जाएगा")
        tiers = [m.tier for m in result]
        # Should find medium-tier urgency phrases
        assert "medium" in tiers or "high" in tiers

    def test_hindi_aadhaar(self, analyzer):
        """Detect Hindi 'आधार नंबर' (Aadhaar number)."""
        result = analyzer.detect_phrases("अपना आधार नंबर बताइए")
        phrases = [m.phrase for m in result]
        assert "आधार नंबर" in phrases

    def test_hindi_multiple_phrases(self, analyzer):
        """Detect multiple Hindi risk phrases."""
        text = "ओटीपी बताइए और पैसे भेजिए तुरंत"
        result = analyzer.detect_phrases(text)
        assert len(result) >= 2


# ===========================================================================
# 3. Hinglish (code-mixed) phrase detection
# ===========================================================================

class TestHinglishPhraseDetection:
    """Test detection in code-mixed Hindi-English text."""

    def test_hinglish_otp_in_hindi_sentence(self, analyzer):
        """Detect English 'OTP' embedded in Hindi sentence."""
        result = analyzer.detect_phrases("Aapka OTP kya hai? Please batayiye")
        phrases = [m.phrase for m in result]
        assert "otp" in phrases

    def test_hinglish_bank_account(self, analyzer):
        """Detect 'bank account' in Hinglish context."""
        result = analyzer.detect_phrases("Apna bank account details share karo jaldi")
        phrases = [m.phrase for m in result]
        assert "bank account" in phrases

    def test_hinglish_mixed_script(self, analyzer):
        """Detect phrases when text mixes Devanagari and Latin scripts."""
        text = "आपका password क्या है? KYC update करना urgent है"
        result = analyzer.detect_phrases(text)
        assert len(result) >= 2

    def test_hinglish_scam_scenario(self, analyzer):
        """Simulate a realistic Hinglish scam call transcript."""
        text = (
            "Hello sir, main RBI se bol raha hoon. "
            "Aapke account mein suspicious activity detect hui hai. "
            "Aapko immediately verify karna hoga. "
            "Please apna OTP share karein."
        )
        result = analyzer.detect_phrases(text)
        # Should find multiple risk signals
        assert len(result) >= 2
        tiers = set(m.tier for m in result)
        assert "critical" in tiers  # OTP


# ===========================================================================
# 4. Risk score computation
# ===========================================================================

class TestRiskScoreComputation:
    """Test the compute_risk_score method."""

    def test_no_phrases_returns_zero(self, analyzer):
        """Score should be 0 when no risk phrases are detected."""
        score = analyzer.compute_risk_score("Hello, how are you? The weather is nice.")
        assert score == 0.0

    def test_single_critical_phrase_scores_high(self, analyzer):
        """A single critical phrase should produce a meaningful score."""
        score = analyzer.compute_risk_score("Please share your OTP")
        assert score > 30.0  # Critical phrase should contribute significantly

    def test_multiple_critical_phrases_score_higher(self, analyzer):
        """Multiple critical phrases should produce a higher score."""
        score_single = analyzer.compute_risk_score("Share your OTP")
        score_multi = analyzer.compute_risk_score(
            "Share your OTP and password and CVV number"
        )
        assert score_multi > score_single

    def test_critical_scores_higher_than_medium(self, analyzer):
        """Critical phrases should produce higher scores than medium ones."""
        score_critical = analyzer.compute_risk_score("Enter your OTP now")
        score_medium = analyzer.compute_risk_score("This is urgent, act right now")
        assert score_critical > score_medium

    def test_score_capped_at_100(self, analyzer):
        """Score should never exceed 100."""
        text = (
            "OTP password CVV UPI PIN Aadhaar number card number "
            "transfer money bank account NEFT RTGS IMPS urgent "
            "immediately block account KYC penalty"
        )
        score = analyzer.compute_risk_score(text)
        assert score <= 100.0

    def test_score_always_non_negative(self, analyzer):
        """Score should never be negative."""
        score = analyzer.compute_risk_score("Just a normal conversation about nothing")
        assert score >= 0.0

    def test_empty_text_returns_zero(self, analyzer):
        """Empty text should return 0 score."""
        assert analyzer.compute_risk_score("") == 0.0
        assert analyzer.compute_risk_score("   ") == 0.0

    def test_density_affects_score(self, analyzer):
        """Higher phrase density (short text, many phrases) should score higher."""
        sparse = "OTP " + ("nothing to see here " * 20)
        dense = "Share OTP and password now"
        score_sparse = analyzer.compute_risk_score(sparse)
        score_dense = analyzer.compute_risk_score(dense)
        # Dense text with multiple phrases should score higher than sparse
        assert score_dense > score_sparse


# ===========================================================================
# 5. Edge cases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, analyzer):
        """Empty string returns no matches."""
        assert analyzer.detect_phrases("") == []

    def test_none_like_whitespace(self, analyzer):
        """Whitespace-only string returns no matches."""
        assert analyzer.detect_phrases("   \n\t  ") == []

    def test_no_risk_phrases_found(self, analyzer):
        """Benign text returns no matches."""
        result = analyzer.detect_phrases("The cat sat on the mat in the sun")
        assert len(result) == 0

    def test_partial_word_not_matched(self, analyzer):
        """'password' inside 'passwords' should match (word boundary)."""
        # Word boundary behavior depends on regex
        result = analyzer.detect_phrases("Your passwords are weak")
        # "password" may or may not match inside "passwords" depending on boundary
        # Just verify no crash
        assert isinstance(result, list)

    def test_very_long_text(self, analyzer):
        """Handle very long text without errors."""
        text = "This is a normal sentence. " * 1000 + "Share your OTP now."
        result = analyzer.detect_phrases(text)
        assert len(result) >= 1

    def test_special_characters_in_text(self, analyzer):
        """Handle special characters without regex errors."""
        text = "Share your OTP!!! @#$%^&*() <urgently> [now]"
        result = analyzer.detect_phrases(text)
        assert len(result) >= 1


# ===========================================================================
# 6. PhraseMatch structure
# ===========================================================================

class TestPhraseMatchStructure:
    """Test that PhraseMatch objects have correct fields."""

    def test_match_has_required_fields(self, analyzer):
        """Each PhraseMatch should have all required fields."""
        result = analyzer.detect_phrases("Share your OTP immediately")
        assert len(result) > 0
        match = result[0]
        assert isinstance(match.phrase, str)
        assert match.tier in ("critical", "high", "medium")
        assert isinstance(match.weight, float)
        assert match.category in ("credential", "financial", "urgency")
        assert match.language in ("en", "hi", "both")
        assert isinstance(match.start_pos, int)
        assert isinstance(match.end_pos, int)
        assert isinstance(match.context, str)
        assert match.start_pos < match.end_pos

    def test_matches_sorted_by_position(self, analyzer):
        """Matches should be sorted by their position in the text."""
        text = "Urgent! Share your OTP. Transfer money to bank account."
        result = analyzer.detect_phrases(text)
        if len(result) >= 2:
            for i in range(1, len(result)):
                assert result[i].start_pos >= result[i - 1].start_pos

    def test_context_snippet_present(self, analyzer):
        """Context snippet should contain surrounding text."""
        text = "Please share your OTP for verification purposes"
        result = analyzer.detect_phrases(text)
        assert len(result) > 0
        # Context should include characters around the match
        assert len(result[0].context) > len(result[0].phrase)


# ===========================================================================
# 7. Full analyze() method
# ===========================================================================

class TestAnalyzeMethod:
    """Test the full analyze() output."""

    def test_analyze_returns_required_keys(self, analyzer):
        """analyze() should return all documented keys."""
        result = analyzer.analyze("Share your OTP now")
        assert "context_risk_score" in result
        assert "detected_phrases" in result
        assert "tier_counts" in result
        assert "top_signals" in result
        assert "text_length" in result
        assert "timestamp_sec" in result

    def test_analyze_score_is_numeric(self, analyzer):
        """context_risk_score should be a number in [0, 100]."""
        result = analyzer.analyze("Share your OTP and password")
        score = result["context_risk_score"]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 100.0

    def test_analyze_tier_counts(self, analyzer):
        """tier_counts should have counts for all three tiers."""
        result = analyzer.analyze("OTP transfer money urgent")
        counts = result["tier_counts"]
        assert "critical" in counts
        assert "high" in counts
        assert "medium" in counts
        assert all(isinstance(v, int) for v in counts.values())

    def test_analyze_top_signals_are_strings(self, analyzer):
        """top_signals should be a list of human-readable strings."""
        result = analyzer.analyze("Share your OTP to bank account urgently")
        signals = result["top_signals"]
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_analyze_top_signals_capped_at_5(self, analyzer):
        """top_signals should have at most 5 entries."""
        text = (
            "OTP password CVV UPI PIN Aadhaar number card number "
            "transfer money bank account NEFT urgent KYC"
        )
        result = analyzer.analyze(text)
        assert len(result["top_signals"]) <= 5

    def test_analyze_timestamp_passed_through(self, analyzer):
        """timestamp_sec should be passed through from input."""
        result = analyzer.analyze("Hello", timestamp_sec=42.5)
        assert result["timestamp_sec"] == 42.5

    def test_analyze_detected_phrases_serializable(self, analyzer):
        """detected_phrases should be a list of dicts (not dataclass objects)."""
        result = analyzer.analyze("Share your OTP")
        phrases = result["detected_phrases"]
        assert isinstance(phrases, list)
        if len(phrases) > 0:
            assert isinstance(phrases[0], dict)
            assert "phrase" in phrases[0]
            assert "tier" in phrases[0]
            assert "weight" in phrases[0]

    def test_analyze_benign_text(self, analyzer):
        """Benign text should produce score 0 and no detected phrases."""
        result = analyzer.analyze("Hello, how are you? Nice weather today!")
        assert result["context_risk_score"] == 0.0
        assert len(result["detected_phrases"]) == 0
        assert result["tier_counts"]["critical"] == 0


# ===========================================================================
# 8. Custom phrases
# ===========================================================================

class TestCustomPhrases:
    """Test that custom phrases are supported."""

    def test_custom_phrase_detected(self, analyzer_with_custom):
        """Custom phrases should be detected alongside built-in ones."""
        result = analyzer_with_custom.detect_phrases("This is a test phrase in the text")
        phrases = [m.phrase for m in result]
        assert "test phrase" in phrases

    def test_custom_phrase_scored(self, analyzer_with_custom):
        """Custom phrases should contribute to the risk score."""
        score = analyzer_with_custom.compute_risk_score("This is a test phrase")
        assert score > 0.0


# ===========================================================================
# 9. Realistic scam scenarios
# ===========================================================================

class TestRealisticScenarios:
    """Test with realistic scam call transcripts."""

    def test_bank_fraud_scenario_english(self, analyzer):
        """Realistic English bank fraud call."""
        text = (
            "Hello sir, this is calling from State Bank. "
            "Your account has been compromised due to suspicious activity. "
            "We need to verify your identity immediately. "
            "Please share your OTP that you will receive on your phone."
        )
        result = analyzer.analyze(text)
        assert result["context_risk_score"] > 40.0
        assert result["tier_counts"]["critical"] >= 1  # OTP

    def test_kyc_scam_scenario_hindi(self, analyzer):
        """Realistic Hindi KYC scam call."""
        text = (
            "नमस्ते, मैं आरबीआई से बोल रहा हूं। "
            "आपका केवाईसी अपडेट नहीं हुआ है। "
            "तुरंत अपना आधार नंबर और ओटीपी बताइए "
            "नहीं तो आपका खाता बंद हो जाएगा।"
        )
        result = analyzer.analyze(text)
        assert result["context_risk_score"] > 30.0
        assert len(result["detected_phrases"]) >= 2

    def test_benign_customer_service_call(self, analyzer):
        """Benign customer service call should score low."""
        text = (
            "Hello, thank you for calling our support line. "
            "How can I help you today? "
            "Let me check your order status. "
            "Your package is scheduled for delivery tomorrow."
        )
        result = analyzer.analyze(text)
        assert result["context_risk_score"] < 20.0
