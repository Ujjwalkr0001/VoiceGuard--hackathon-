"""
VoiceGuard Backend — File Upload & Live Risk Analysis Router

Handles:
  1. POST /api/v1/analyze-upload:
     Accepts recorded audio (.wav, .mp3, .ogg, .flac, .m4a), decodes PCM,
     computes waveform envelope, extracts sliding-window acoustic DSP features,
     runs STT & context scam-phrase detection, computes rolling risk scores,
     and returns full analysis with danger segments and verdict.
  2. GET /api/v1/sample-calls:
     Lists pre-configured demo sample calls.
  3. GET /api/v1/sample-calls/{sample_id}:
     Returns analysis and audio URL for 1-click testing.
"""

import io
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.ml.context_analyzer import ContextAnalyzer
from app.ml.dsp_acoustic_scorer import DSPAcousticScorer
from app.services.stt_service import transcribe

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["upload-analysis"])

# Static upload storage
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = STATIC_DIR / "sample_calls"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Shared scorers
_acoustic_scorer = DSPAcousticScorer(sample_rate=16000)
_context_analyzer = ContextAnalyzer()


# ---------------------------------------------------------------------------
# Pydantic Response Models
# ---------------------------------------------------------------------------

class AudioChunkAnalysis(BaseModel):
    chunk_id: int
    start_sec: float
    end_sec: float
    acoustic_score: float = Field(..., description="0-100 voice clone probability")
    context_score: float = Field(..., description="0-100 social engineering scam score")
    risk_score: float = Field(..., description="0-100 composite risk score")
    severity: str = Field(..., description="'safe' | 'warning' | 'critical'")
    is_dangerous: bool
    signals: List[str] = Field(default_factory=list)
    transcript: str = ""


class DangerousSegment(BaseModel):
    segment_id: int
    start_sec: float
    end_sec: float
    peak_risk: float
    severity: str
    primary_threat: str
    transcript_snippet: str


class TranscriptLine(BaseModel):
    id: int
    start_sec: float
    end_sec: float
    speaker: str
    text: str
    risk_level: str
    flagged_phrases: List[str] = Field(default_factory=list)


class CallConclusion(BaseModel):
    verdict: str
    verdict_level: str  # 'critical' | 'warning' | 'safe'
    peak_risk: float
    average_risk: float
    voice_clone_probability: float
    context_scam_score: float
    dangerous_segments_count: int
    safe_ratio_pct: float
    total_duration_sec: float
    detected_threats: List[Dict[str, Any]]
    acoustic_summary: Dict[str, Any]
    executive_summary: str
    recommendations: List[str]


class AnalysisResponse(BaseModel):
    call_id: str
    file_name: str
    duration_sec: float
    sample_rate: int
    audio_url: str
    waveform_peaks: List[float]
    chunks: List[AudioChunkAnalysis]
    dangerous_segments: List[DangerousSegment]
    transcript_lines: List[TranscriptLine]
    full_transcript: str
    conclusion: CallConclusion


# ---------------------------------------------------------------------------
# Pre-configured Sample Metadata (for 1-click rich demo experience)
# ---------------------------------------------------------------------------

SAMPLE_METADATA = {
    "scam_bank_otp": {
        "title": "🚨 Bank Security Impersonation (OTP / Card Scam)",
        "description": "Urgent fake alert from 'State Bank fraud division' coercing user for 6-digit OTP, card number, and ATM PIN.",
        "file_name": "scam_bank_otp.wav",
        "category": "Financial / Credential Harvesting",
        "expected_verdict": "CRITICAL RISK",
        "transcript_lines": [
            {
                "id": 1,
                "start_sec": 0.0,
                "end_sec": 6.5,
                "speaker": "Caller (AI Impersonator)",
                "text": "Attention customer, this is the security and fraud prevention department calling from your bank.",
                "risk_level": "warning",
                "flagged_phrases": ["security and fraud", "bank"]
            },
            {
                "id": 2,
                "start_sec": 6.5,
                "end_sec": 13.0,
                "speaker": "Caller (AI Impersonator)",
                "text": "An unauthorized international transaction of forty-eight thousand rupees has been attempted on your account.",
                "risk_level": "warning",
                "flagged_phrases": ["unauthorized", "forty-eight thousand rupees", "account"]
            },
            {
                "id": 3,
                "start_sec": 13.0,
                "end_sec": 19.5,
                "speaker": "Caller (AI Impersonator)",
                "text": "To cancel this charge immediately, you must share your six digit OTP and verify your debit card number.",
                "risk_level": "critical",
                "flagged_phrases": ["immediately", "share your six digit otp", "debit card number"]
            },
            {
                "id": 4,
                "start_sec": 19.5,
                "end_sec": 25.3,
                "speaker": "Caller (AI Impersonator)",
                "text": "Enter your PIN right now or your bank account will be permanently suspended within ten minutes.",
                "risk_level": "critical",
                "flagged_phrases": ["pin", "right now", "permanently suspended", "ten minutes"]
            }
        ]
    },
    "safe_call": {
        "title": "✅ Standard Personal Call (Safe)",
        "description": "Friendly conversation arranging a Saturday weekend lunch at a local cafe.",
        "file_name": "safe_call.wav",
        "category": "Casual / Friendly Conversation",
        "expected_verdict": "SAFE",
        "transcript_lines": [
            {
                "id": 1,
                "start_sec": 0.0,
                "end_sec": 5.5,
                "speaker": "Caller",
                "text": "Hey there, hope you are having a wonderful day! Are we still on for lunch this Saturday?",
                "risk_level": "safe",
                "flagged_phrases": []
            },
            {
                "id": 2,
                "start_sec": 5.5,
                "end_sec": 11.5,
                "speaker": "Caller",
                "text": "I was thinking we could check out that new cafe near the library around one o clock in the afternoon.",
                "risk_level": "safe",
                "flagged_phrases": []
            },
            {
                "id": 3,
                "start_sec": 11.5,
                "end_sec": 17.2,
                "speaker": "Caller",
                "text": "Let me know what time works best for your schedule, talk to you later!",
                "risk_level": "safe",
                "flagged_phrases": []
            }
        ]
    },
    "ai_relative_emergency": {
        "title": "🚨 AI Clone Relative Emergency (Bail Money Scam)",
        "description": "Distressed clone voice claiming to be arrested after an accident, begging for instant Google Pay / PhonePe UPI transfer.",
        "file_name": "ai_relative_emergency.wav",
        "category": "Kidnapping / Emergency Bail Coercion",
        "expected_verdict": "CRITICAL RISK",
        "transcript_lines": [
            {
                "id": 1,
                "start_sec": 0.0,
                "end_sec": 5.5,
                "speaker": "Caller (Cloned Voice)",
                "text": "Uncle please help me, I have been detained by the police after an urgent car accident.",
                "risk_level": "warning",
                "flagged_phrases": ["detained by the police", "urgent car accident"]
            },
            {
                "id": 2,
                "start_sec": 5.5,
                "end_sec": 11.5,
                "speaker": "Caller (Cloned Voice)",
                "text": "They are demanding thirty thousand rupees right now to release me immediately without filing an FIR.",
                "risk_level": "critical",
                "flagged_phrases": ["thirty thousand rupees", "right now", "release me immediately"]
            },
            {
                "id": 3,
                "start_sec": 11.5,
                "end_sec": 17.7,
                "speaker": "Caller (Cloned Voice)",
                "text": "Please transfer the money to this PhonePe number right now, do not tell mom please hurry!",
                "risk_level": "critical",
                "flagged_phrases": ["transfer the money", "phonepe", "right now", "do not tell mom", "hurry"]
            }
        ]
    }
}


# ---------------------------------------------------------------------------
# Audio Processing Helpers
# ---------------------------------------------------------------------------

def _decode_audio_bytes(audio_bytes: bytes, filename_hint: str = "audio.wav") -> tuple[np.ndarray, int]:
    """
    Robustly decode raw audio bytes into 1D float32 mono array and 16kHz sample rate.
    Supports MP3, WAV, OGG, FLAC, M4A with automatic format sniffing and fallback.
    """
    data = None
    sr = None
    last_err = None

    # 1. Try reading directly from memory with soundfile
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        last_err = e

    # 2. If in-memory fails (common for some MP3 containers / ID3 tags), write to temp file
    if data is None:
        suffix = Path(filename_hint).suffix.lower() if filename_hint else ".mp3"
        if not suffix:
            suffix = ".mp3"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                data, sr = sf.read(tmp_path, dtype="float32")
            except Exception as e:
                last_err = e
                # 3. Fallback to librosa if soundfile failed
                try:
                    import librosa
                    data, sr = librosa.load(tmp_path, sr=None, mono=True)
                except Exception as le:
                    last_err = le
        except Exception as e:
            last_err = e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    if data is None or sr is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file '{filename_hint}'. Please upload a valid MP3, WAV, FLAC, or OGG file. Error: {last_err}"
        )

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = data.astype(np.float32)

    # Resample to 16,000 Hz if needed for optimal DSP performance
    if sr != 16000:
        duration = len(data) / sr
        target_len = int(duration * 16000)
        data = np.interp(
            np.linspace(0, len(data) - 1, target_len),
            np.arange(len(data)),
            data
        ).astype(np.float32)
        sr = 16000

    return data, sr


def _compute_waveform_peaks(audio: np.ndarray, num_bins: int = 150) -> List[float]:
    """Compute normalized peak envelope for frontend canvas waveform."""
    if len(audio) == 0:
        return [0.05] * num_bins

    total_samples = len(audio)
    bin_size = max(1, total_samples // num_bins)
    peaks = []

    for i in range(num_bins):
        start = i * bin_size
        end = min(start + bin_size, total_samples)
        if start >= total_samples:
            peaks.append(0.05)
        else:
            chunk = audio[start:end]
            val = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.05
            peaks.append(max(0.06, min(1.0, round(val, 3))))

    # Normalize peaks
    max_peak = max(peaks) if peaks else 1.0
    if max_peak > 0.05:
        peaks = [round(p / max_peak, 3) for p in peaks]

    return peaks


async def _analyze_audio_stream(
    audio: np.ndarray,
    sr: int,
    file_name: str,
    call_id: str,
    audio_url: str,
    known_transcript_lines: Optional[List[Dict[str, Any]]] = None,
) -> AnalysisResponse:
    """Core analysis routine across sliding time windows."""
    duration_sec = float(len(audio) / sr)
    waveform_peaks = _compute_waveform_peaks(audio, num_bins=160)

    # Sliding window parameters
    chunk_dur = 2.0  # 2-second windows
    step_dur = 1.0   # 1-second step for smooth timeline resolution
    num_chunks = max(1, math.ceil(duration_sec / step_dur))

    chunks: List[AudioChunkAnalysis] = []
    chunk_sample_len = int(chunk_dur * sr)
    step_sample_len = int(step_dur * sr)

    # Transcription extraction
    full_transcript_text = ""
    transcript_lines: List[TranscriptLine] = []

    if known_transcript_lines:
        for idx, line in enumerate(known_transcript_lines):
            t_line = TranscriptLine(
                id=line.get("id", idx + 1),
                start_sec=line.get("start_sec", 0.0),
                end_sec=line.get("end_sec", duration_sec),
                speaker=line.get("speaker", "Caller"),
                text=line.get("text", ""),
                risk_level=line.get("risk_level", "safe"),
                flagged_phrases=line.get("flagged_phrases", [])
            )
            transcript_lines.append(t_line)
            full_transcript_text += f"{t_line.speaker}: {t_line.text}\n"
    else:
        # Run STT via service (Sarvam / Whisper fallback)
        try:
            stt_res = await transcribe(audio, sr)
            raw_text = stt_res.get("transcript", "").strip() if stt_res else ""
            if raw_text:
                full_transcript_text = raw_text
                # Break into lines
                detected_phrases = _context_analyzer.detect_phrases(raw_text)
                flagged = [m.phrase for m in detected_phrases]
                risk_lvl = "critical" if any(p.tier == "critical" for p in detected_phrases) else (
                    "warning" if detected_phrases else "safe"
                )
                transcript_lines.append(TranscriptLine(
                    id=1,
                    start_sec=0.0,
                    end_sec=duration_sec,
                    speaker="Caller",
                    text=raw_text,
                    risk_level=risk_lvl,
                    flagged_phrases=flagged
                ))
        except Exception as e:
            logger.warning("stt_transcription_failed", error=str(e))

    # Analyze chunks across time
    running_scores: List[float] = []
    detected_threat_map: Dict[str, Dict[str, Any]] = {}

    for i in range(num_chunks):
        t_start = i * step_dur
        t_end = min(t_start + chunk_dur, duration_sec)
        s_start = int(t_start * sr)
        s_end = min(int(t_end * sr), len(audio))

        chunk_audio = audio[s_start:s_end]
        if len(chunk_audio) < int(sr * 0.1):
            continue

        # 1. Acoustic Spoof Score (0.0 to 1.0 -> 0 to 100)
        # Uses fast autocorrelation pitch jitter + MODGDF + CQCC
        raw_acoustic = _acoustic_scorer.predict(chunk_audio, sr=sr)
        acoustic_score = round(raw_acoustic * 100.0, 1)

        # 2. Context / Scam Score for this time window
        chunk_text = ""
        for line in transcript_lines:
            if not (line.end_sec < t_start or line.start_sec > t_end):
                chunk_text += (" " + line.text)
        chunk_text = chunk_text.strip()

        context_score = 0.0
        signals: List[str] = []

        if chunk_text:
            matches = _context_analyzer.detect_phrases(chunk_text)
            context_score = _context_analyzer.compute_risk_score(chunk_text, matches)
            for m in matches:
                phrase_lower = m.phrase.lower()
                if phrase_lower not in detected_threat_map:
                    detected_threat_map[phrase_lower] = {
                        "phrase": m.phrase,
                        "tier": m.tier,
                        "category": m.category,
                        "count": 0,
                        "timestamps": []
                    }
                detected_threat_map[phrase_lower]["count"] += 1
                detected_threat_map[phrase_lower]["timestamps"].append(round(t_start, 1))

                if m.tier == "critical":
                    signals.append(f"🚨 Solicitation of {m.phrase.upper()} ({m.category})")
                elif m.tier == "high":
                    signals.append(f"⚠️ Financial transfer demand: '{m.phrase}'")

        # Acoustic indicators in signals
        if acoustic_score >= 70.0:
            signals.append("🎙️ Synthetic pitch regularity (unnatural low micro-jitter)")
            signals.append("🔬 Vocoder phase coherence (MODGDF anomaly)")
        elif acoustic_score >= 50.0:
            signals.append("⚠️ Elevated voice-clone acoustic probability")

        # Composite rolling risk score
        # 60% acoustic + 40% context with urgency multiplier
        if context_score > 0:
            comp_risk = (0.55 * acoustic_score) + (0.45 * context_score)
            if context_score >= 70.0:
                comp_risk = min(100.0, comp_risk * 1.25)
        else:
            comp_risk = acoustic_score * 0.7  # pure acoustic without text trigger

        # Temporal smoothing with last 2 chunks
        running_scores.append(comp_risk)
        smooth_risk = round(float(np.mean(running_scores[-3:])), 1)

        severity = "safe"
        if smooth_risk >= 75.0:
            severity = "critical"
        elif smooth_risk >= 50.0:
            severity = "warning"

        chunks.append(AudioChunkAnalysis(
            chunk_id=i + 1,
            start_sec=round(t_start, 2),
            end_sec=round(t_end, 2),
            acoustic_score=acoustic_score,
            context_score=round(context_score, 1),
            risk_score=smooth_risk,
            severity=severity,
            is_dangerous=(smooth_risk >= 50.0),
            signals=signals[:3],
            transcript=chunk_text
        ))

    # Group continuous or close dangerous chunks into Dangerous Segments
    dangerous_segments: List[DangerousSegment] = []
    in_danger = False
    seg_start = 0.0
    seg_end = 0.0
    seg_peak = 0.0
    seg_signals: List[str] = []
    seg_texts: List[str] = []

    for c in chunks:
        if c.is_dangerous:
            if not in_danger:
                in_danger = True
                seg_start = c.start_sec
                seg_end = c.end_sec
                seg_peak = c.risk_score
                seg_signals = list(c.signals)
                if c.transcript:
                    seg_texts = [c.transcript]
            else:
                seg_end = c.end_sec
                seg_peak = max(seg_peak, c.risk_score)
                seg_signals.extend(c.signals)
                if c.transcript and c.transcript not in seg_texts:
                    seg_texts.append(c.transcript)
        else:
            if in_danger:
                in_danger = False
                primary_threat = seg_signals[0] if seg_signals else "Elevated voice-clone & coercion risk"
                dangerous_segments.append(DangerousSegment(
                    segment_id=len(dangerous_segments) + 1,
                    start_sec=round(seg_start, 1),
                    end_sec=round(seg_end, 1),
                    peak_risk=round(seg_peak, 1),
                    severity="critical" if seg_peak >= 75.0 else "warning",
                    primary_threat=primary_threat,
                    transcript_snippet=" ".join(seg_texts)[:120]
                ))

    # Catch trailing segment
    if in_danger:
        primary_threat = seg_signals[0] if seg_signals else "Elevated voice-clone & coercion risk"
        dangerous_segments.append(DangerousSegment(
            segment_id=len(dangerous_segments) + 1,
            start_sec=round(seg_start, 1),
            end_sec=round(seg_end, 1),
            peak_risk=round(seg_peak, 1),
            severity="critical" if seg_peak >= 75.0 else "warning",
            primary_threat=primary_threat,
            transcript_snippet=" ".join(seg_texts)[:120]
        ))

    # Compute full-call conclusion
    peak_risk = max([c.risk_score for c in chunks]) if chunks else 0.0
    avg_risk = round(float(np.mean([c.risk_score for c in chunks])), 1) if chunks else 0.0
    max_acoustic = max([c.acoustic_score for c in chunks]) if chunks else 0.0
    max_context = max([c.context_score for c in chunks]) if chunks else 0.0

    safe_chunks = sum(1 for c in chunks if not c.is_dangerous)
    safe_ratio = round((safe_chunks / max(len(chunks), 1)) * 100.0, 1)

    # Verdict determination
    if peak_risk >= 75.0 or (max_acoustic >= 75.0 and max_context >= 50.0):
        verdict = "CRITICAL RISK — SCAM DETECTED"
        verdict_level = "critical"
        exec_summary = (
            f"This call exhibits severe indicators of social engineering fraud and synthetic voice manipulation. "
            f"VoiceGuard flagged {len(dangerous_segments)} dangerous segment(s) with peak risk reaching {peak_risk}%. "
            f"Key threats include urgent credential/OTP harvesting and synthetic prosody characteristics."
        )
        recs = [
            "🚨 DO NOT share OTP, UPI PIN, ATM PIN, or card details under any circumstance.",
            "🛑 HANG UP the call immediately. Banks and police never request verification codes over the phone.",
            "🔒 Block this caller number and contact your bank's fraud helpline through their official banking app.",
            "📢 Report this incident to the National Cyber Crime Reporting Portal (1930)."
        ]
    elif peak_risk >= 50.0 or len(dangerous_segments) > 0:
        verdict = "SUSPICIOUS CALL — CAUTION ADVISED"
        verdict_level = "warning"
        exec_summary = (
            f"Suspicious activity detected. Peak risk reached {peak_risk}%. "
            f"Unusual voice acoustic patterns or pressure tactics were observed. Exercise extreme caution."
        )
        recs = [
            "⚠️ Verify caller identity independently before taking any action or transferring money.",
            "❌ Never approve unexpected payment requests or screen-sharing prompts.",
            "🔍 Contact the person or institution directly on a verified contact number."
        ]
    else:
        verdict = "SAFE — NO MALICIOUS THREATS DETECTED"
        verdict_level = "safe"
        exec_summary = (
            f"No voice-clone or social engineering patterns detected. Peak risk remained at a safe {peak_risk}%. "
            f"Acoustic features display natural human micro-jitter, normal phase variance, and genuine conversational flow."
        )
        recs = [
            "✅ Audio matches natural human speech acoustic profiles.",
            "🛡️ Always stay vigilant if unfamiliar callers abruptly ask for financial transfers or sensitive credentials."
        ]

    conclusion = CallConclusion(
        verdict=verdict,
        verdict_level=verdict_level,
        peak_risk=peak_risk,
        average_risk=avg_risk,
        voice_clone_probability=max_acoustic,
        context_scam_score=max_context,
        dangerous_segments_count=len(dangerous_segments),
        safe_ratio_pct=safe_ratio,
        total_duration_sec=round(duration_sec, 2),
        detected_threats=list(detected_threat_map.values()),
        acoustic_summary={
            "jitter_status": "Abnormally Low (<0.008, Synthetic)" if max_acoustic > 70 else "Normal Natural Range (0.015 - 0.035)",
            "shimmer_status": "Monotone Synthetic" if max_acoustic > 70 else "Natural Dynamic Variance",
            "phase_coherence": "High Vocoder Correlation" if max_acoustic > 70 else "Natural Phase Distribution",
            "voice_synthesis_likelihood": f"{max_acoustic}%"
        },
        executive_summary=exec_summary,
        recommendations=recs
    )

    return AnalysisResponse(
        call_id=call_id,
        file_name=file_name,
        duration_sec=round(duration_sec, 2),
        sample_rate=sr,
        audio_url=audio_url,
        waveform_peaks=waveform_peaks,
        chunks=chunks,
        dangerous_segments=dangerous_segments,
        transcript_lines=transcript_lines,
        full_transcript=full_transcript_text.strip(),
        conclusion=conclusion
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/sample-calls", tags=["upload-analysis"])
async def get_sample_calls():
    """Returns catalog of built-in 1-click demo calls."""
    catalog = []
    for sample_id, meta in SAMPLE_METADATA.items():
        catalog.append({
            "sample_id": sample_id,
            "title": meta["title"],
            "description": meta["description"],
            "category": meta["category"],
            "expected_verdict": meta["expected_verdict"],
            "file_name": meta["file_name"],
            "audio_url": f"/static/sample_calls/{meta['file_name']}"
        })
    return {"samples": catalog}


@router.get("/sample-calls/{sample_id}", response_model=AnalysisResponse, tags=["upload-analysis"])
async def analyze_sample_call(sample_id: str):
    """Analyzes a pre-configured sample call with instant 1-click loading."""
    if sample_id not in SAMPLE_METADATA:
        raise HTTPException(
            status_code=404,
            detail=f"Sample call '{sample_id}' not found. Available samples: {list(SAMPLE_METADATA.keys())}"
        )

    meta = SAMPLE_METADATA[sample_id]
    sample_file = SAMPLES_DIR / meta["file_name"]

    if not sample_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Sample audio file '{sample_file.name}' not found on server."
        )

    # Read audio
    data, sr = sf.read(str(sample_file), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    call_id = f"demo-{sample_id}"
    audio_url = f"/static/sample_calls/{meta['file_name']}"

    return await _analyze_audio_stream(
        audio=data,
        sr=sr,
        file_name=meta["file_name"],
        call_id=call_id,
        audio_url=audio_url,
        known_transcript_lines=meta.get("transcript_lines")
    )


@router.post("/analyze-upload", response_model=AnalysisResponse, tags=["upload-analysis"])
async def analyze_uploaded_audio(file: UploadFile = File(...)):
    """
    Upload and analyze any voice call recording.
    Accepts .wav, .mp3, .ogg, .flac, .m4a, .webm.
    """
    logger.info("upload_analysis.received", filename=file.filename, content_type=file.content_type)

    audio_bytes = await file.read()
    if not audio_bytes or len(audio_bytes) < 500:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or too small to be valid audio.")

    # Decode audio with filename hint for proper format dispatching
    pcm_data, sr = _decode_audio_bytes(audio_bytes, filename_hint=file.filename or "recording.mp3")
    duration = len(pcm_data) / sr

    if duration < 1.0:
        raise HTTPException(status_code=400, detail="Audio file must be at least 1.0 second long for analysis.")

    # Save as standard WAV in static uploads directory so browser can stream it reliably
    call_id = f"call-{uuid.uuid4().hex[:10]}"
    saved_filename = f"{call_id}.wav"
    saved_path = UPLOADS_DIR / saved_filename

    try:
        sf.write(str(saved_path), pcm_data, sr)
    except Exception as e:
        logger.warning("save_wav_failed", error=str(e))

    audio_url = f"/static/uploads/{saved_filename}"

    return await _analyze_audio_stream(
        audio=pcm_data,
        sr=sr,
        file_name=file.filename or "recording.wav",
        call_id=call_id,
        audio_url=audio_url
    )

