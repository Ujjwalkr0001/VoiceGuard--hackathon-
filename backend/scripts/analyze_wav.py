"""
VoiceGuard Backend — Offline audio-file analyzer.

Runs a WAV/FLAC/OGG file through the same processing stages the live call
pipeline uses, and prints what each stage produced. This is the quickest way
to test detection on a real audio sample without Twilio, ngrok, or a phone.

It mirrors CallPipelineCoordinator._process_chunk(), but reads from a file
instead of a Twilio Media Stream and skips alert dispatch / persistence.

Usage:
    cd backend
    python scripts/analyze_wav.py path/to/sample.wav
    python scripts/analyze_wav.py sample.wav --caller-reputation flagged
    python scripts/analyze_wav.py sample.wav --transcript "text to score instead of STT"

Notes:
  - Audio is resampled to 8 kHz to match Twilio's telephony sample rate.
  - Stages with no model checkpoint / no STT credentials report as UNAVAILABLE
    rather than silently scoring 0, so you can see what is actually running.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The DSP extractors log a debug line per feature family per chunk, which
# buries this script's own report. Keep warnings and errors. The modules are
# split between stdlib logging and structlog, so both need muting.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("app").setLevel(logging.WARNING)

import structlog  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

from app.config import settings  # noqa: E402
from app.ml.context_analyzer import ContextAnalyzer  # noqa: E402
from app.ml.dsp_features import extract_all_features  # noqa: E402
from app.ml.ensemble import EnsembleScorer  # noqa: E402
from app.pipeline.model_manager import model_manager  # noqa: E402
from app.services.risk_engine import CallerReputation, RiskEngine  # noqa: E402
from app.services.stt_service import transcribe  # noqa: E402

TWILIO_SR = 8000


def load_audio_8k(path: Path) -> np.ndarray:
    """Read an audio file, downmix to mono, resample to 8 kHz float32."""
    try:
        import soundfile as sf
    except ImportError:
        print("[ERROR] soundfile is required: pip install soundfile")
        raise SystemExit(1)

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TWILIO_SR:
        from scipy.signal import resample_poly

        from math import gcd

        g = gcd(int(sr), TWILIO_SR)
        audio = resample_poly(audio, TWILIO_SR // g, int(sr) // g)

    return np.ascontiguousarray(audio, dtype=np.float32)


def iter_chunks(audio: np.ndarray):
    """
    Split audio into overlapping chunks exactly like AudioBufferManager does,
    so offline results line up with what the live pipeline would see.
    """
    chunk_n = settings.audio_chunk_samples
    hop_n = chunk_n - settings.audio_overlap_samples
    idx = 0
    start = 0
    while start + chunk_n <= len(audio):
        yield idx, start / TWILIO_SR, audio[start : start + chunk_n]
        idx += 1
        start += hop_n


async def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze an audio file with the VoiceGuard pipeline.")
    ap.add_argument("audio_file", type=Path)
    ap.add_argument(
        "--caller-reputation",
        choices=[r.value for r in CallerReputation],
        default=CallerReputation.UNKNOWN.value,
        help="Caller reputation, which applies a risk multiplier.",
    )
    ap.add_argument(
        "--transcript",
        default=None,
        help="Score this text with the context analyzer instead of running STT. "
        "Useful when no STT credentials are configured.",
    )
    ap.add_argument("--no-stt", action="store_true", help="Skip speech-to-text entirely.")
    args = ap.parse_args()

    if not args.audio_file.exists():
        print(f"[ERROR] No such file: {args.audio_file}")
        return 1

    audio = load_audio_8k(args.audio_file)
    duration = len(audio) / TWILIO_SR
    print(f"[audio] {args.audio_file.name}  {duration:.2f}s @ {TWILIO_SR}Hz (mono)")

    # --- Models: report honestly whether each is actually available ---
    await model_manager.load_all()
    model_a = model_manager.model_a
    model_b = model_manager.model_b
    print(f"[model] Model A (AASIST):  {'loaded' if model_a else 'UNAVAILABLE - neutral 0.5'}")
    print(f"[model] Model B (XGBoost): {'loaded' if model_b else 'UNAVAILABLE - excluded'}")

    # --- Speech-to-text over the whole file (the live pipeline batches chunks) ---
    transcript = args.transcript or ""
    if args.transcript:
        print(f"[stt  ] using --transcript override: {transcript!r}")
    elif args.no_stt:
        print("[stt  ] skipped (--no-stt)")
    else:
        stt = await transcribe(audio=audio, sample_rate=TWILIO_SR)
        transcript = stt.get("transcript", "")
        if transcript:
            print(f"[stt  ] source={stt.get('source')} lang={stt.get('language')}: {transcript!r}")
        else:
            print(
                f"[stt  ] UNAVAILABLE (source={stt.get('source')}) - no transcript, so the "
                "context signal contributes 0. Pass --transcript to test context scoring."
            )

    context_score = 0.0
    if transcript.strip():
        ctx = ContextAnalyzer().analyze(text=transcript, timestamp_sec=0.0)
        context_score = ctx.get("context_risk_score", 0.0)
        print(f"[ctx  ] context_risk_score={context_score:.2f}")
        for sig in ctx.get("top_signals", []):
            print(f"         - {sig}")

    # --- Per-chunk acoustic analysis + risk fusion ---
    ensemble = EnsembleScorer()
    engine = RiskEngine(
        call_sid=f"OFFLINE-{args.audio_file.stem}",
        caller_number="+10000000000",
        caller_reputation=CallerReputation(args.caller_reputation),
    )

    print(f"\n{'chunk':>5} {'t(s)':>6} {'dsp_ms':>7} {'acoustic':>9} {'context':>8} {'risk':>7}  alert")
    print("-" * 60)

    any_chunk = False
    for idx, t_sec, chunk in iter_chunks(audio):
        any_chunk = True
        feats = extract_all_features(chunk, TWILIO_SR)

        score_a = float(model_a.predict(chunk)) if model_a else 0.5
        score_b = float(model_b(feats["features"])) if model_b else None

        ens = ensemble.combine_with_details(score_a, score_b)
        result = engine.process_chunk(
            acoustic_score=ens["acoustic_score"],
            context_score=context_score,
            timestamp_sec=t_sec,
        )
        print(
            f"{idx:>5} {t_sec:>6.1f} {feats['total_time_ms']:>7.0f} "
            f"{ens['acoustic_score']:>9.1f} {context_score:>8.1f} "
            f"{result['current_risk']:>7.1f}  {result['threshold_crossed'] or ''}"
        )

    if not any_chunk:
        print(
            f"(none - audio is {duration:.2f}s but a chunk needs "
            f"{settings.audio_chunk_duration_sec}s)"
        )
        return 1

    print("-" * 60)
    print(f"peak risk : {engine.get_peak_risk():.1f}/100")
    print(f"verdict   : {engine.get_final_verdict()}")
    if not model_a and not model_b:
        print(
            "\n[warning] No acoustic model is loaded, so the acoustic score is a fixed\n"
            "          neutral 0.5 and carries no detection signal. The verdict above\n"
            "          reflects only the context/transcript analysis and caller multiplier."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
