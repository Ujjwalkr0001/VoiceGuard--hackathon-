"""
VoiceGuard Backend — Speech-to-Text Service (Steps 97-101)

Provides real-time speech-to-text transcription for audio chunks via:
  1. Sarvam AI STT API (primary — supports Hindi, English, Hinglish)
  2. Local Whisper model (fallback — used when Sarvam fails or times out)

Features:
  - Async HTTP calls to Sarvam AI with configurable timeout
  - Automatic fallback to Whisper tiny/base on API failure
  - Hindi, English, and code-mixed (Hinglish) transcription
  - Sliding transcript window (last 30 seconds) per call session
"""

import asyncio
import io
import logging
import time
import wave
from collections import deque
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sarvam AI STT configuration
# ---------------------------------------------------------------------------
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_TIMEOUT_SEC = 2.0  # Max wait for Sarvam API response

# Sarvam retires old model versions server-side: a deprecated id makes the API
# reject the request with HTTP 400, not fall back. Bump this when that happens.
SARVAM_STT_MODEL = "saaras:v3"

# Supported language codes for Sarvam AI
SARVAM_LANGUAGES = {
    "hi-IN": "Hindi",
    "en-IN": "English (India)",
    "auto": "Auto-detect",
}

# ---------------------------------------------------------------------------
# Whisper fallback configuration
# ---------------------------------------------------------------------------
_whisper_model = None  # Lazy-loaded on first fallback
WHISPER_MODEL_SIZE = "tiny"  # Use 'tiny' for speed; 'base' for quality


def _get_whisper_model():
    """Lazy-load the Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper

            logger.info(
                "stt.whisper.loading",
                extra={"model_size": WHISPER_MODEL_SIZE},
            )
            _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info("stt.whisper.loaded")
        except ImportError:
            # NB: "msg" is a reserved LogRecord attribute — passing it via
            # `extra` makes logging raise KeyError, turning this graceful
            # degradation path into a hard crash.
            logger.error(
                "stt.whisper.import_error",
                extra={"detail": "openai-whisper not installed, fallback unavailable"},
            )
            return None
        except Exception as e:
            logger.error(
                "stt.whisper.load_error",
                extra={"error": str(e)},
            )
            return None
    return _whisper_model


def _pcm_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
    sample_width: int = 2,
) -> bytes:
    """
    Convert a float32 numpy audio array to WAV bytes in memory.

    Args:
        audio: Audio signal as float32 ndarray in [-1, 1].
        sample_rate: Sample rate in Hz.
        sample_width: Bytes per sample (2 = 16-bit PCM).

    Returns:
        WAV file content as bytes.
    """
    # Clamp and convert to int16
    audio_clamped = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (audio_clamped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sarvam AI STT Client
# ---------------------------------------------------------------------------

async def transcribe_sarvam(
    audio: np.ndarray,
    sample_rate: int,
    language_code: str = "auto",
) -> Optional[Dict]:
    """
    Transcribe audio using the Sarvam AI STT API.

    Args:
        audio: Audio signal as float32 ndarray in [-1, 1].
        sample_rate: Sample rate in Hz.
        language_code: Language hint — 'hi-IN', 'en-IN', or 'auto'.

    Returns:
        Dict with keys:
          - transcript (str): Transcribed text
          - language (str): Detected language code
          - confidence (float): Confidence score [0, 1]
          - source (str): 'sarvam'
        Or None if the API call fails.
    """
    api_key = settings.sarvam_api_key
    if not api_key:
        logger.warning("stt.sarvam.no_api_key")
        return None

    wav_bytes = _pcm_to_wav_bytes(audio, sample_rate)

    headers = {
        "api-subscription-key": api_key,
    }

    # Sarvam expects multipart form with the audio file
    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav"),
    }
    data = {
        "language_code": language_code,
        "model": SARVAM_STT_MODEL,
        "with_timestamps": "false",
    }

    try:
        async with httpx.AsyncClient(timeout=SARVAM_TIMEOUT_SEC) as client:
            t0 = time.monotonic()
            response = await client.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
            )
            latency_ms = (time.monotonic() - t0) * 1000

        if response.status_code != 200:
            logger.warning(
                "stt.sarvam.api_error",
                extra={
                    "status": response.status_code,
                    "body": response.text[:200],
                    "latency_ms": round(latency_ms, 1),
                },
            )
            return None

        result = response.json()

        transcript = result.get("transcript", "").strip()
        detected_lang = result.get("language_code", language_code)
        # saaras:v3 reports "language_probability"; older responses used
        # "confidence". Accept either so this survives another version bump.
        confidence = float(
            result.get("confidence", result.get("language_probability", 0.0)) or 0.0
        )

        logger.debug(
            "stt.sarvam.success",
            extra={
                "transcript_len": len(transcript),
                "language": detected_lang,
                "confidence": round(confidence, 3),
                "latency_ms": round(latency_ms, 1),
            },
        )

        return {
            "transcript": transcript,
            "language": detected_lang,
            "confidence": confidence,
            "source": "sarvam",
            "latency_ms": round(latency_ms, 1),
        }

    except httpx.TimeoutException:
        logger.warning(
            "stt.sarvam.timeout",
            extra={"timeout_sec": SARVAM_TIMEOUT_SEC},
        )
        return None
    except Exception as e:
        logger.error(
            "stt.sarvam.exception",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return None


# ---------------------------------------------------------------------------
# Whisper Fallback
# ---------------------------------------------------------------------------

async def transcribe_whisper(
    audio: np.ndarray,
    sample_rate: int,
) -> Optional[Dict]:
    """
    Transcribe audio using local Whisper model (fallback).

    Runs inference in a thread pool to avoid blocking the event loop.

    Args:
        audio: Audio signal as float32 ndarray in [-1, 1].
        sample_rate: Sample rate in Hz.

    Returns:
        Dict with keys: transcript, language, confidence, source
        Or None if Whisper is unavailable.
    """
    model = _get_whisper_model()
    if model is None:
        return None

    def _run_whisper() -> Dict:
        """Synchronous Whisper inference (runs in thread)."""
        import whisper

        t0 = time.monotonic()

        # Whisper expects 16 kHz float32
        if sample_rate != 16000:
            # Simple resampling via linear interpolation
            duration = len(audio) / sample_rate
            target_len = int(duration * 16000)
            audio_16k = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        else:
            audio_16k = audio.astype(np.float32)

        # Pad or trim to 30 seconds (Whisper's expected length)
        audio_padded = whisper.pad_or_trim(audio_16k)

        # Run with language detection
        result = model.transcribe(
            audio_padded,
            language=None,  # Auto-detect
            task="transcribe",
            fp16=False,  # CPU safety
            without_timestamps=True,
        )

        latency_ms = (time.monotonic() - t0) * 1000
        transcript = result.get("text", "").strip()
        language = result.get("language", "unknown")

        return {
            "transcript": transcript,
            "language": language,
            "confidence": 0.5,  # Whisper doesn't expose confidence directly
            "source": "whisper",
            "latency_ms": round(latency_ms, 1),
        }

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run_whisper)

        logger.debug(
            "stt.whisper.success",
            extra={
                "transcript_len": len(result["transcript"]),
                "language": result["language"],
                "latency_ms": result["latency_ms"],
            },
        )
        return result

    except Exception as e:
        logger.error(
            "stt.whisper.exception",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return None


# ---------------------------------------------------------------------------
# Unified STT function with automatic fallback
# ---------------------------------------------------------------------------

async def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    language_code: str = "auto",
    prefer_whisper: bool = False,
) -> Dict:
    """
    Transcribe audio with automatic fallback.

    Primary: Sarvam AI STT API
    Fallback: Local Whisper model (if Sarvam fails or times out)

    Args:
        audio: Audio signal as float32 ndarray in [-1, 1].
        sample_rate: Sample rate in Hz.
        language_code: Language hint for Sarvam ('hi-IN', 'en-IN', 'auto').
        prefer_whisper: If True, skip Sarvam and go straight to Whisper.

    Returns:
        Dict with keys: transcript, language, confidence, source, latency_ms
        Returns empty transcript if both methods fail.
    """
    result = None

    # Try Sarvam first (unless fallback-only mode)
    if not prefer_whisper:
        result = await transcribe_sarvam(audio, sample_rate, language_code)

    # Fallback to Whisper if Sarvam failed
    if result is None:
        logger.info("stt.falling_back_to_whisper")
        result = await transcribe_whisper(audio, sample_rate)

    # If both fail, return empty result
    if result is None:
        logger.warning("stt.all_methods_failed")
        return {
            "transcript": "",
            "language": "unknown",
            "confidence": 0.0,
            "source": "none",
            "latency_ms": 0.0,
        }

    return result


# ---------------------------------------------------------------------------
# Transcript Accumulator (per-call sliding window)
# ---------------------------------------------------------------------------

class TranscriptAccumulator:
    """
    Accumulates transcripts across chunks for a single call session.

    Maintains a sliding window of the last N seconds of transcript text,
    enabling the context analyzer to detect risk phrases across chunk
    boundaries.
    """

    def __init__(
        self,
        call_sid: str,
        window_duration_sec: float = 30.0,
        chunk_duration_sec: float = 2.0,
    ):
        """
        Args:
            call_sid: Twilio call SID for this session.
            window_duration_sec: How many seconds of transcript to keep.
            chunk_duration_sec: Duration of each audio chunk.
        """
        self.call_sid = call_sid
        self.window_duration_sec = window_duration_sec
        self.chunk_duration_sec = chunk_duration_sec

        # Max chunks to keep = window / chunk_duration
        max_chunks = max(1, int(window_duration_sec / chunk_duration_sec))
        self._entries: deque = deque(maxlen=max_chunks)

        self._total_chunks: int = 0
        self._languages_seen: Dict[str, int] = {}

    def add(
        self,
        transcript: str,
        timestamp_sec: float,
        language: str = "unknown",
        confidence: float = 0.0,
        source: str = "unknown",
    ) -> None:
        """
        Add a new transcript chunk to the accumulator.

        Args:
            transcript: Transcribed text for this chunk.
            timestamp_sec: Start time of the chunk in the call.
            language: Detected language code.
            confidence: Confidence score [0, 1].
            source: STT source ('sarvam' or 'whisper').
        """
        entry = {
            "transcript": transcript,
            "timestamp_sec": timestamp_sec,
            "language": language,
            "confidence": confidence,
            "source": source,
        }
        self._entries.append(entry)
        self._total_chunks += 1

        # Track language distribution
        if language != "unknown":
            self._languages_seen[language] = (
                self._languages_seen.get(language, 0) + 1
            )

    @property
    def full_text(self) -> str:
        """
        Get the concatenated transcript text within the sliding window.

        Returns:
            Single string of all transcript chunks joined with spaces.
        """
        return " ".join(
            entry["transcript"]
            for entry in self._entries
            if entry["transcript"]
        )

    @property
    def recent_entries(self) -> List[Dict]:
        """Get all transcript entries in the sliding window."""
        return list(self._entries)

    @property
    def dominant_language(self) -> str:
        """
        Get the most frequently detected language.

        Returns:
            Language code string, or 'unknown' if no transcripts yet.
        """
        if not self._languages_seen:
            return "unknown"
        return max(self._languages_seen, key=self._languages_seen.get)

    @property
    def stats(self) -> Dict:
        """Return accumulator statistics."""
        return {
            "call_sid": self.call_sid,
            "total_chunks_transcribed": self._total_chunks,
            "window_chunks": len(self._entries),
            "window_text_length": len(self.full_text),
            "dominant_language": self.dominant_language,
            "languages_seen": dict(self._languages_seen),
        }

    def clear(self) -> None:
        """Clear all accumulated transcripts."""
        self._entries.clear()
        self._total_chunks = 0
        self._languages_seen.clear()
