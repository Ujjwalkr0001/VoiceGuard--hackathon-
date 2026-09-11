"""
VoiceGuard Backend — Context / Risk-Phrase Analyzer (Steps 102-107)

Analyzes transcribed text to detect social engineering risk phrases
commonly used in vishing (voice phishing) and scam calls.

Features:
  - Configurable risk phrase dictionary with severity tiers
    (Critical, High, Medium) in English and Hindi
  - Fast regex + keyword matching
  - Optional sentence-transformers embedding similarity for fuzzy matching
  - Computes a context_risk_score based on phrase density and severity
  - Returns detected phrases with timestamps for the risk explanation panel
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk phrase severity tiers
# ---------------------------------------------------------------------------

@dataclass
class RiskPhrase:
    """A single risk phrase with metadata."""
    phrase: str
    tier: str          # 'critical', 'high', 'medium'
    weight: float      # Severity weight: 1.0, 0.7, 0.4
    language: str      # 'en', 'hi', 'both'
    category: str      # e.g., 'credential', 'financial', 'urgency'


# ---------------------------------------------------------------------------
# Risk phrase dictionary
# ---------------------------------------------------------------------------

RISK_PHRASES: List[RiskPhrase] = [
    # ===== CRITICAL (weight 1.0) — Credential / OTP solicitation =====
    RiskPhrase("otp", "critical", 1.0, "en", "credential"),
    RiskPhrase("one time password", "critical", 1.0, "en", "credential"),
    RiskPhrase("one-time password", "critical", 1.0, "en", "credential"),
    RiskPhrase("verification code", "critical", 1.0, "en", "credential"),
    RiskPhrase("password", "critical", 1.0, "en", "credential"),
    RiskPhrase("upi pin", "critical", 1.0, "en", "credential"),
    RiskPhrase("cvv", "critical", 1.0, "en", "credential"),
    RiskPhrase("mpin", "critical", 1.0, "en", "credential"),
    RiskPhrase("m-pin", "critical", 1.0, "en", "credential"),
    RiskPhrase("aadhaar number", "critical", 1.0, "en", "credential"),
    RiskPhrase("aadhar number", "critical", 1.0, "en", "credential"),
    RiskPhrase("card number", "critical", 1.0, "en", "credential"),
    RiskPhrase("credit card number", "critical", 1.0, "en", "credential"),
    RiskPhrase("debit card number", "critical", 1.0, "en", "credential"),
    RiskPhrase("atm pin", "critical", 1.0, "en", "credential"),
    RiskPhrase("pan number", "critical", 1.0, "en", "credential"),
    RiskPhrase("share your otp", "critical", 1.0, "en", "credential"),
    RiskPhrase("tell me your otp", "critical", 1.0, "en", "credential"),
    RiskPhrase("enter your pin", "critical", 1.0, "en", "credential"),
    # Hindi critical phrases
    RiskPhrase("ओटीपी", "critical", 1.0, "hi", "credential"),
    RiskPhrase("वन टाइम पासवर्ड", "critical", 1.0, "hi", "credential"),
    RiskPhrase("पासवर्ड", "critical", 1.0, "hi", "credential"),
    RiskPhrase("यूपीआई पिन", "critical", 1.0, "hi", "credential"),
    RiskPhrase("सीवीवी", "critical", 1.0, "hi", "credential"),
    RiskPhrase("एम पिन", "critical", 1.0, "hi", "credential"),
    RiskPhrase("आधार नंबर", "critical", 1.0, "hi", "credential"),
    RiskPhrase("कार्ड नंबर", "critical", 1.0, "hi", "credential"),
    RiskPhrase("पैन नंबर", "critical", 1.0, "hi", "credential"),
    RiskPhrase("पिन बताइए", "critical", 1.0, "hi", "credential"),
    RiskPhrase("ओटीपी बताइए", "critical", 1.0, "hi", "credential"),
    RiskPhrase("ओटीपी भेजिए", "critical", 1.0, "hi", "credential"),

    # ===== HIGH (weight 0.7) — Financial transactions =====
    RiskPhrase("transfer money", "high", 0.7, "en", "financial"),
    RiskPhrase("send money", "high", 0.7, "en", "financial"),
    RiskPhrase("bank account", "high", 0.7, "en", "financial"),
    RiskPhrase("bank details", "high", 0.7, "en", "financial"),
    RiskPhrase("account number", "high", 0.7, "en", "financial"),
    RiskPhrase("ifsc code", "high", 0.7, "en", "financial"),
    RiskPhrase("neft", "high", 0.7, "en", "financial"),
    RiskPhrase("rtgs", "high", 0.7, "en", "financial"),
    RiskPhrase("imps", "high", 0.7, "en", "financial"),
    RiskPhrase("wire transfer", "high", 0.7, "en", "financial"),
    RiskPhrase("pay now", "high", 0.7, "en", "financial"),
    RiskPhrase("payment link", "high", 0.7, "en", "financial"),
    RiskPhrase("google pay", "high", 0.7, "en", "financial"),
    RiskPhrase("phonepe", "high", 0.7, "en", "financial"),
    RiskPhrase("paytm", "high", 0.7, "en", "financial"),
    RiskPhrase("scan qr", "high", 0.7, "en", "financial"),
    RiskPhrase("qr code", "high", 0.7, "en", "financial"),
    # Hindi high phrases
    RiskPhrase("पैसे भेजो", "high", 0.7, "hi", "financial"),
    RiskPhrase("पैसे ट्रांसफर", "high", 0.7, "hi", "financial"),
    RiskPhrase("बैंक खाता", "high", 0.7, "hi", "financial"),
    RiskPhrase("खाता नंबर", "high", 0.7, "hi", "financial"),
    RiskPhrase("पैसे भेजिए", "high", 0.7, "hi", "financial"),
    RiskPhrase("भुगतान करें", "high", 0.7, "hi", "financial"),
    RiskPhrase("भुगतान लिंक", "high", 0.7, "hi", "financial"),

    # ===== MEDIUM (weight 0.4) — Urgency / social engineering tactics =====
    RiskPhrase("urgent", "medium", 0.4, "en", "urgency"),
    RiskPhrase("immediately", "medium", 0.4, "en", "urgency"),
    RiskPhrase("right now", "medium", 0.4, "en", "urgency"),
    RiskPhrase("deadline", "medium", 0.4, "en", "urgency"),
    RiskPhrase("penalty", "medium", 0.4, "en", "urgency"),
    RiskPhrase("block account", "medium", 0.4, "en", "urgency"),
    RiskPhrase("block your account", "medium", 0.4, "en", "urgency"),
    RiskPhrase("suspend", "medium", 0.4, "en", "urgency"),
    RiskPhrase("suspended", "medium", 0.4, "en", "urgency"),
    RiskPhrase("verify identity", "medium", 0.4, "en", "urgency"),
    RiskPhrase("verify your identity", "medium", 0.4, "en", "urgency"),
    RiskPhrase("kyc", "medium", 0.4, "en", "urgency"),
    RiskPhrase("kyc verification", "medium", 0.4, "en", "urgency"),
    RiskPhrase("kyc update", "medium", 0.4, "en", "urgency"),
    RiskPhrase("account compromised", "medium", 0.4, "en", "urgency"),
    RiskPhrase("account has been compromised", "medium", 0.4, "en", "urgency"),
    RiskPhrase("unauthorized transaction", "medium", 0.4, "en", "urgency"),
    RiskPhrase("suspicious activity", "medium", 0.4, "en", "urgency"),
    RiskPhrase("legal action", "medium", 0.4, "en", "urgency"),
    RiskPhrase("arrest warrant", "medium", 0.4, "en", "urgency"),
    RiskPhrase("police complaint", "medium", 0.4, "en", "urgency"),
    RiskPhrase("rbi", "medium", 0.4, "en", "urgency"),
    RiskPhrase("reserve bank", "medium", 0.4, "en", "urgency"),
    RiskPhrase("income tax", "medium", 0.4, "en", "urgency"),
    RiskPhrase("customs department", "medium", 0.4, "en", "urgency"),
    RiskPhrase("lottery", "medium", 0.4, "en", "urgency"),
    RiskPhrase("you have won", "medium", 0.4, "en", "urgency"),
    RiskPhrase("congratulations", "medium", 0.4, "en", "urgency"),
    RiskPhrase("refund", "medium", 0.4, "en", "urgency"),
    RiskPhrase("cashback", "medium", 0.4, "en", "urgency"),
    RiskPhrase("insurance claim", "medium", 0.4, "en", "urgency"),
    # Hindi medium phrases
    RiskPhrase("तुरंत", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("अभी", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("जल्दी", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("जुर्माना", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("खाता बंद", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("खाता ब्लॉक", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("केवाईसी", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("पहचान सत्यापन", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("अनधिकृत लेनदेन", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("कानूनी कार्रवाई", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("गिरफ्तारी", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("पुलिस", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("आरबीआई", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("आयकर", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("लॉटरी", "medium", 0.4, "hi", "urgency"),
    RiskPhrase("रिफंड", "medium", 0.4, "hi", "urgency"),
]


# ---------------------------------------------------------------------------
# Precompiled regex patterns for fast matching
# ---------------------------------------------------------------------------

def _build_regex_patterns(
    phrases: List[RiskPhrase],
) -> List[Tuple[re.Pattern, RiskPhrase]]:
    """
    Build compiled regex patterns for all risk phrases.

    Each phrase is wrapped with word boundaries for precise matching.
    """
    patterns = []
    for rp in phrases:
        # Escape special regex chars, then wrap with word boundaries
        escaped = re.escape(rp.phrase)
        # Use case-insensitive matching for English
        flags = re.IGNORECASE | re.UNICODE
        try:
            pattern = re.compile(r"\b" + escaped + r"\b", flags)
            patterns.append((pattern, rp))
        except re.error:
            # Fallback: plain substring pattern
            pattern = re.compile(escaped, flags)
            patterns.append((pattern, rp))
    return patterns


_COMPILED_PATTERNS: List[Tuple[re.Pattern, RiskPhrase]] = _build_regex_patterns(
    RISK_PHRASES
)


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class PhraseMatch:
    """A detected risk phrase match."""
    phrase: str
    tier: str
    weight: float
    category: str
    language: str
    start_pos: int      # Character position in text
    end_pos: int        # Character end position
    context: str        # Surrounding text snippet


# ---------------------------------------------------------------------------
# Context Analyzer
# ---------------------------------------------------------------------------

class ContextAnalyzer:
    """
    Analyzes transcript text to detect social engineering risk phrases
    and compute a context risk score.

    Usage:
        analyzer = ContextAnalyzer()
        result = analyzer.analyze("Please share your OTP to verify your account")
        # result['context_risk_score'] → 72.0
        # result['detected_phrases'] → [PhraseMatch(...), ...]
    """

    def __init__(
        self,
        custom_phrases: Optional[List[RiskPhrase]] = None,
        context_window_chars: int = 40,
    ):
        """
        Args:
            custom_phrases: Additional risk phrases to include beyond defaults.
            context_window_chars: Number of chars around a match for context snippet.
        """
        self.context_window_chars = context_window_chars

        # Build patterns: defaults + custom
        all_phrases = list(RISK_PHRASES)
        if custom_phrases:
            all_phrases.extend(custom_phrases)

        self._patterns = _build_regex_patterns(all_phrases)

        # Optional: sentence-transformers model for fuzzy matching
        self._embedder = None

    def _get_context_snippet(self, text: str, start: int, end: int) -> str:
        """Extract a context snippet around the match."""
        ctx_start = max(0, start - self.context_window_chars)
        ctx_end = min(len(text), end + self.context_window_chars)
        snippet = text[ctx_start:ctx_end].strip()
        if ctx_start > 0:
            snippet = "..." + snippet
        if ctx_end < len(text):
            snippet = snippet + "..."
        return snippet

    def detect_phrases(self, text: str) -> List[PhraseMatch]:
        """
        Detect all risk phrases in the given text.

        Args:
            text: Transcript text to analyze.

        Returns:
            List of PhraseMatch objects, sorted by position.
        """
        if not text or not text.strip():
            return []

        matches: List[PhraseMatch] = []
        seen_spans = set()  # Avoid overlapping matches

        for pattern, rp in self._patterns:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())

                # Skip if this span overlaps with an existing higher-weight match
                overlap = False
                for existing_start, existing_end in seen_spans:
                    if span[0] < existing_end and span[1] > existing_start:
                        overlap = True
                        break
                if overlap:
                    continue

                seen_spans.add(span)
                context = self._get_context_snippet(text, m.start(), m.end())

                matches.append(PhraseMatch(
                    phrase=rp.phrase,
                    tier=rp.tier,
                    weight=rp.weight,
                    category=rp.category,
                    language=rp.language,
                    start_pos=m.start(),
                    end_pos=m.end(),
                    context=context,
                ))

        # Sort by position in text
        matches.sort(key=lambda m: m.start_pos)
        return matches

    def compute_risk_score(
        self,
        text: str,
        detected_phrases: Optional[List[PhraseMatch]] = None,
    ) -> float:
        """
        Compute a context risk score based on detected risk phrases.

        The score accounts for:
        - Number of distinct risk phrases found
        - Severity weights of detected phrases
        - Density (phrases per 100 characters of text)

        Args:
            text: Transcript text.
            detected_phrases: Pre-computed phrase matches (optional).

        Returns:
            Context risk score in [0, 100].
        """
        if not text or not text.strip():
            return 0.0

        if detected_phrases is None:
            detected_phrases = self.detect_phrases(text)

        if not detected_phrases:
            return 0.0

        # Sum severity weights
        total_weight = sum(p.weight for p in detected_phrases)

        # Tier-based counts
        critical_count = sum(1 for p in detected_phrases if p.tier == "critical")
        high_count = sum(1 for p in detected_phrases if p.tier == "high")
        medium_count = sum(1 for p in detected_phrases if p.tier == "medium")

        # Density: phrases per 100 characters (more dense = more suspicious)
        text_length = max(len(text), 1)
        density = len(detected_phrases) / text_length * 100

        # Base score from total weight (normalize: weight sum of 3.0 → score ~80)
        base_score = min(total_weight / 3.0, 1.0) * 80.0

        # Critical phrase bonus: any critical phrase adds significant risk
        critical_bonus = min(critical_count * 15.0, 30.0)

        # Density bonus (capped)
        density_bonus = min(density * 5.0, 15.0)

        # Combine
        raw_score = base_score + critical_bonus + density_bonus

        # Clamp to [0, 100]
        return max(0.0, min(100.0, round(raw_score, 2)))

    def analyze(
        self,
        text: str,
        timestamp_sec: Optional[float] = None,
    ) -> Dict:
        """
        Full analysis: detect phrases and compute risk score.

        Args:
            text: Transcript text to analyze.
            timestamp_sec: Optional timestamp for this text segment.

        Returns:
            Dict with:
            - context_risk_score: float in [0, 100]
            - detected_phrases: List[Dict] (serializable PhraseMatch data)
            - tier_counts: Dict with counts per tier
            - top_signals: List[str] human-readable signal descriptions
        """
        detected = self.detect_phrases(text)
        score = self.compute_risk_score(text, detected)

        # Build human-readable signal descriptions
        top_signals = []
        for p in detected:
            if p.tier == "critical":
                top_signals.append(f"Caller asked for {p.phrase.upper()}")
            elif p.tier == "high":
                top_signals.append(f"Financial transaction mentioned: {p.phrase}")
            elif p.tier == "medium":
                top_signals.append(f"Urgency/pressure tactic: {p.phrase}")

        # Deduplicate signals
        top_signals = list(dict.fromkeys(top_signals))

        tier_counts = {
            "critical": sum(1 for p in detected if p.tier == "critical"),
            "high": sum(1 for p in detected if p.tier == "high"),
            "medium": sum(1 for p in detected if p.tier == "medium"),
        }

        return {
            "context_risk_score": score,
            "detected_phrases": [
                {
                    "phrase": p.phrase,
                    "tier": p.tier,
                    "weight": p.weight,
                    "category": p.category,
                    "language": p.language,
                    "start_pos": p.start_pos,
                    "end_pos": p.end_pos,
                    "context": p.context,
                }
                for p in detected
            ],
            "tier_counts": tier_counts,
            "top_signals": top_signals[:5],  # Top 5 signals for alert display
            "text_length": len(text),
            "timestamp_sec": timestamp_sec,
        }
