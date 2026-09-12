"""
VoiceGuard Backend — Call Pipeline Coordinator (Steps 143–147)

Ties all processing components together into a single per-call
orchestration pipeline:

    Audio Chunk (from AudioBufferManager)
        ├── 1. DSP Feature Extraction
        ├── 2. Model A: AASIST deep inference
        ├── 3. Model B: XGBoost on DSP features
        ├── 4. Ensemble: combine Model A + B
        ├── 5. STT: Sarvam AI / Whisper transcription
        ├── 6. Context Analyzer: risk phrase detection
        ├── 7. Risk Engine: composite score + smoothing
        ├── 8. Alert Dispatcher: threshold → push / SMS
        └── 9. DynamoDB: persist risk timeline

One CallPipelineCoordinator is instantiated per active Twilio Media
Stream connection and runs an async processing loop that consumes
chunks from the AudioBufferManager's queue.

Usage:
    coordinator = CallPipelineCoordinator(
        call_sid="CA123",
        caller_number="+919876543210",
        user_id="user_001",
        buffer_manager=audio_buffer,
    )
    await coordinator.start()   # begins async processing loop
    await coordinator.stop()    # graceful shutdown
"""

import asyncio
import logging
import time
from typing import Optional

from app.pipeline.metrics import pipeline_metrics, PipelineMetrics

import numpy as np

from app.config import settings
from app.ml.context_analyzer import ContextAnalyzer
from app.ml.dsp_features import extract_all_features
from app.ml.ensemble import EnsembleScorer
from app.services.alert_dispatcher import AlertDispatcher
from app.services.dynamo_client import dynamo_client
from app.services.redis_client import (
    store_risk_state,
    append_score_history,
    cleanup_call_state,
)
from app.services.risk_engine import CallerReputation, RiskEngine
from app.services.stt_service import TranscriptAccumulator, transcribe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker for Sarvam AI STT (Step 151)
# ---------------------------------------------------------------------------

class _STTCircuitBreaker:
    """
    Simple circuit breaker for Sarvam AI STT.

    After `failure_threshold` consecutive failures, switches to
    Whisper-only mode for `recovery_timeout_sec` seconds.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self._consecutive_failures = 0
        self._open_since: Optional[float] = None

    @property
    def is_open(self) -> bool:
        """True if circuit is open (Sarvam should be skipped)."""
        if self._open_since is None:
            return False
        elapsed = time.time() - self._open_since
        if elapsed >= self.recovery_timeout_sec:
            # Recovery period elapsed — close the circuit and retry
            self._open_since = None
            self._consecutive_failures = 0
            logger.info("stt_circuit_breaker.closed_after_recovery")
            return False
        return True

    def record_success(self) -> None:
        """Record a successful Sarvam API call."""
        self._consecutive_failures = 0
        if self._open_since is not None:
            self._open_since = None
            logger.info("stt_circuit_breaker.closed_on_success")

    def record_failure(self) -> None:
        """Record a failed Sarvam API call."""
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= self.failure_threshold
            and self._open_since is None
        ):
            self._open_since = time.time()
            logger.warning(
                "stt_circuit_breaker.opened",
                extra={
                    "consecutive_failures": self._consecutive_failures,
                    "recovery_sec": self.recovery_timeout_sec,
                },
            )


# ---------------------------------------------------------------------------
# Pipeline Coordinator (Steps 143–147)
# ---------------------------------------------------------------------------

class CallPipelineCoordinator:
    """
    Orchestrates the full processing pipeline for a single active call.

    Instantiated when a Twilio Media Stream connects. Runs an async
    loop consuming audio chunks from the AudioBufferManager and
    routing them through feature extraction, model inference, NLP
    analysis, risk scoring, alerting, and persistence.
    """

    # STT batching: accumulate audio before transcribing (Step 145.5)
    STT_BATCH_CHUNKS = 3  # ~5 seconds of audio at 2s chunks with overlap

    # DynamoDB write batching: persist every Nth chunk (Step 145.9)
    DYNAMO_WRITE_EVERY = 5

    def __init__(
        self,
        call_sid: str,
        caller_number: str,
        user_id: str,
        buffer_manager,  # AudioBufferManager instance
        caller_reputation: CallerReputation = CallerReputation.UNKNOWN,
        alert_dispatcher: Optional[AlertDispatcher] = None,
        ensemble_scorer: Optional[EnsembleScorer] = None,
        context_analyzer: Optional[ContextAnalyzer] = None,
    ):
        self.call_sid = call_sid
        self.caller_number = caller_number
        self.user_id = user_id
        self.buffer_manager = buffer_manager

        # --- Components (shared or per-call) ---
        self.risk_engine = RiskEngine(
            call_sid=call_sid,
            caller_number=caller_number,
            caller_reputation=caller_reputation,
        )
        self.ensemble = ensemble_scorer or EnsembleScorer()
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.alert_dispatcher = alert_dispatcher
        self.transcript_accumulator = TranscriptAccumulator(
            call_sid=call_sid,
        )
        self._stt_circuit_breaker = _STTCircuitBreaker()

        # --- STT audio accumulator (batch multiple chunks) ---
        self._stt_audio_buffer: list[np.ndarray] = []
        self._stt_chunk_count = 0

        # --- State ---
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._chunks_processed = 0
        self._start_time = time.time()

        # --- Placeholders for ML models (loaded at app startup, shared) ---
        # Model A (AASIST) and Model B (XGBoost) are expected to be set
        # via set_models() before the pipeline starts processing.
        self._model_a = None  # AASIST inference wrapper
        self._model_b = None  # XGBoost inference function

        logger.info(
            "pipeline.coordinator_created",
            extra={
                "call_sid": call_sid,
                "caller_number": caller_number,
                "user_id": user_id,
                "caller_reputation": caller_reputation.value,
            },
        )

    def set_models(self, model_a=None, model_b=None) -> None:
        """
        Inject shared ML model instances.

        Called by the app startup routine to share models across
        all active call coordinators (Step 149).

        Args:
            model_a: AASIST wrapper with `predict(audio) -> float`.
            model_b: XGBoost inference function `predict(features) -> float`.
        """
        self._model_a = model_a
        self._model_b = model_b

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def start(self) -> None:
        """Start the async processing loop."""
        if self._running:
            logger.warning(
                "pipeline.already_running",
                extra={"call_sid": self.call_sid},
            )
            return

        self._running = True

        # Create DynamoDB session record
        try:
            await dynamo_client.create_session(
                self.call_sid, self.caller_number, self.user_id,
            )
        except Exception as e:
            logger.error(
                "pipeline.dynamo_create_failed",
                extra={"call_sid": self.call_sid, "error": str(e)},
            )

        # Launch the processing loop as a background task
        self._task = asyncio.create_task(
            self._processing_loop(),
            name=f"pipeline-{self.call_sid}",
        )
        logger.info(
            "pipeline.started",
            extra={"call_sid": self.call_sid},
        )

    async def stop(self) -> None:
        """
        Graceful shutdown (Step 146).

        - Finalize DynamoDB session with verdict
        - Flush Redis state
        - Cancel the processing task
        """
        self._running = False

        # Cancel the processing task if still running
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Determine final verdict
        final_verdict = self.risk_engine.get_final_verdict()

        # Finalize DynamoDB session
        try:
            await dynamo_client.finalize_session(self.call_sid, final_verdict)
        except Exception as e:
            logger.error(
                "pipeline.dynamo_finalize_failed",
                extra={"call_sid": self.call_sid, "error": str(e)},
            )

        # Flush Redis state
        try:
            await cleanup_call_state(self.call_sid)
        except Exception as e:
            logger.error(
                "pipeline.redis_cleanup_failed",
                extra={"call_sid": self.call_sid, "error": str(e)},
            )

        elapsed = time.time() - self._start_time
        logger.info(
            "pipeline.stopped",
            extra={
                "call_sid": self.call_sid,
                "chunks_processed": self._chunks_processed,
                "duration_sec": round(elapsed, 1),
                "final_verdict": final_verdict,
                "peak_risk": self.risk_engine.get_peak_risk(),
                "transcript_stats": self.transcript_accumulator.stats,
            },
        )

    # -------------------------------------------------------------------
    # Main processing loop
    # -------------------------------------------------------------------

    async def _processing_loop(self) -> None:
        """
        Async loop that consumes chunks from the audio buffer queue
        and runs the full pipeline on each.
        """
        logger.info(
            "pipeline.loop_started",
            extra={"call_sid": self.call_sid},
        )

        try:
            while self._running:
                try:
                    # Wait for a chunk (with timeout to check _running flag)
                    chunk_data = await asyncio.wait_for(
                        self.buffer_manager.chunk_queue.get(),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    await self._process_chunk(chunk_data)
                except Exception as e:
                    logger.error(
                        "pipeline.chunk_processing_error",
                        extra={
                            "call_sid": self.call_sid,
                            "chunk_index": chunk_data.get("chunk_index", -1),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

        except asyncio.CancelledError:
            logger.info(
                "pipeline.loop_cancelled",
                extra={"call_sid": self.call_sid},
            )
        except Exception as e:
            logger.error(
                "pipeline.loop_error",
                extra={
                    "call_sid": self.call_sid,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # -------------------------------------------------------------------
    # Per-chunk processing (Step 145)
    # -------------------------------------------------------------------

    async def _process_chunk(self, chunk_data: dict) -> None:
        """
        Process a single audio chunk through the full pipeline.

        Steps:
            1. Extract DSP features
            2. Run Model A inference (AASIST)
            3. Run Model B inference (XGBoost on DSP features)
            4. Compute ensemble score
            5. Run STT (batched)
            6. Run context analysis on accumulated transcript
            7. Feed scores to RiskEngine
            8. Check threshold → dispatch alert if crossed
            9. Persist to Redis + DynamoDB (batched)
        """
        chunk_start = time.perf_counter()
        audio = chunk_data["audio"]  # np.ndarray float32 [-1, 1]
        sr = chunk_data["sample_rate"]
        chunk_index = chunk_data["chunk_index"]
        timestamp_sec = chunk_data["timestamp_sec"]

        # ------ Step 1: DSP Feature Extraction ------
        t0 = time.perf_counter()
        features_result = await asyncio.to_thread(
            extract_all_features, audio, sr,
        )
        feature_vector = features_result["features"]
        feature_time_ms = (time.perf_counter() - t0) * 1000
        pipeline_metrics.record(
            PipelineMetrics.FEATURE_EXTRACTION, feature_time_ms, self.call_sid,
        )

        # ------ Steps 2 & 3: Model Inference (concurrent, Step 147) ------
        t0 = time.perf_counter()
        model_a_score, model_b_score = await self._run_model_inference(
            audio, sr, feature_vector,
        )
        inference_time_ms = (time.perf_counter() - t0) * 1000
        pipeline_metrics.record(
            PipelineMetrics.MODEL_A_INFERENCE, inference_time_ms, self.call_sid,
        )

        # ------ Step 4: Ensemble Score ------
        t0 = time.perf_counter()
        ensemble_result = self.ensemble.combine_with_details(
            model_a_score, model_b_score,
        )
        acoustic_score = ensemble_result["acoustic_score"]
        pipeline_metrics.record(
            PipelineMetrics.ENSEMBLE,
            (time.perf_counter() - t0) * 1000, self.call_sid,
        )

        # ------ Step 5: STT (batched, every STT_BATCH_CHUNKS) ------
        self._stt_audio_buffer.append(audio)
        self._stt_chunk_count += 1

        context_score = 0.0
        context_signals: list[str] = []

        if self._stt_chunk_count >= self.STT_BATCH_CHUNKS:
            t0 = time.perf_counter()
            context_result = await self._run_stt_and_context(
                timestamp_sec,
            )
            context_score = context_result.get("context_risk_score", 0.0)
            context_signals = context_result.get("top_signals", [])
            pipeline_metrics.record(
                PipelineMetrics.STT_TRANSCRIPTION,
                (time.perf_counter() - t0) * 1000, self.call_sid,
            )

        # ------ Step 7: Risk Engine ------
        t0 = time.perf_counter()
        risk_result = self.risk_engine.process_chunk(
            acoustic_score=acoustic_score,
            context_score=context_score,
            timestamp_sec=timestamp_sec,
        )
        current_risk = risk_result["current_risk"]
        threshold_crossed = risk_result["threshold_crossed"]
        pipeline_metrics.record(
            PipelineMetrics.RISK_ENGINE,
            (time.perf_counter() - t0) * 1000, self.call_sid,
        )

        # ------ Step 8: Alert Dispatch ------
        if threshold_crossed and self.alert_dispatcher:
            # Build signal descriptions
            signals = []
            if acoustic_score > 60:
                signals.append(
                    f"AI-generated voice patterns detected "
                    f"(acoustic score: {acoustic_score:.0f})"
                )
            if ensemble_result.get("agreement") == "strong_agree":
                signals.append("Both detection models agree on synthetic voice")
            signals.extend(context_signals)
            if not signals:
                signals.append(f"Risk score elevated: {current_risk:.0f}/100")

            t0 = time.perf_counter()
            try:
                alert_id = await self.alert_dispatcher.dispatch_alert(
                    call_sid=self.call_sid,
                    caller_number=self.caller_number,
                    risk_score=current_risk,
                    risk_level=threshold_crossed,
                    signals=signals,
                )
                if alert_id:
                    pipeline_metrics.record_alert()
                    # Track in DynamoDB
                    await dynamo_client.append_alert(
                        self.call_sid,
                        threshold_crossed,
                        "fcm+sms" if threshold_crossed == "high" else "fcm",
                    )
            except Exception as e:
                logger.error(
                    "pipeline.alert_dispatch_failed",
                    extra={
                        "call_sid": self.call_sid,
                        "error": str(e),
                    },
                )
            pipeline_metrics.record(
                PipelineMetrics.ALERT_DISPATCH,
                (time.perf_counter() - t0) * 1000, self.call_sid,
            )

        # ------ Step 9: Persist to Redis + DynamoDB (batched) ------
        t0 = time.perf_counter()
        await self._persist_state(
            chunk_index=chunk_index,
            timestamp_sec=timestamp_sec,
            current_risk=current_risk,
            acoustic_score=acoustic_score,
            context_score=context_score,
        )
        pipeline_metrics.record(
            PipelineMetrics.REDIS_PERSIST,
            (time.perf_counter() - t0) * 1000, self.call_sid,
        )

        # ------ Step 148/152: Record total pipeline latency ------
        self._chunks_processed += 1
        total_time_ms = (time.perf_counter() - chunk_start) * 1000
        pipeline_metrics.record(
            PipelineMetrics.TOTAL_PIPELINE, total_time_ms, self.call_sid,
        )
        pipeline_metrics.record_chunk()

        logger.debug(
            "pipeline.chunk_complete",
            extra={
                "call_sid": self.call_sid,
                "chunk_index": chunk_index,
                "acoustic_score": round(acoustic_score, 1),
                "context_score": round(context_score, 1),
                "current_risk": round(current_risk, 1),
                "threshold_crossed": threshold_crossed,
                "feature_ms": round(feature_time_ms, 1),
                "inference_ms": round(inference_time_ms, 1),
                "total_ms": round(total_time_ms, 1),
            },
        )

    # -------------------------------------------------------------------
    # Model inference (Steps 2-3, concurrent via Step 147)
    # -------------------------------------------------------------------

    async def _run_model_inference(
        self,
        audio: np.ndarray,
        sr: int,
        feature_vector: np.ndarray,
    ) -> tuple[float, Optional[float]]:
        """
        Run Model A and Model B concurrently using asyncio.gather.

        Returns:
            (model_a_score, model_b_score) — both in [0, 1].
            model_b_score is None if XGBoost is not loaded.
        """
        async def _infer_model_a() -> float:
            if self._model_a is None:
                return 0.5  # Neutral fallback
            try:
                score = await asyncio.to_thread(
                    self._model_a.predict, audio,
                )
                return float(score)
            except Exception as e:
                logger.error(
                    "pipeline.model_a_error",
                    extra={"call_sid": self.call_sid, "error": str(e)},
                )
                return 0.5

        async def _infer_model_b() -> Optional[float]:
            if self._model_b is None:
                return None
            try:
                score = await asyncio.to_thread(
                    self._model_b, feature_vector,
                )
                return float(score)
            except Exception as e:
                logger.error(
                    "pipeline.model_b_error",
                    extra={"call_sid": self.call_sid, "error": str(e)},
                )
                return None

        # Run both concurrently
        model_a_score, model_b_score = await asyncio.gather(
            _infer_model_a(),
            _infer_model_b(),
        )

        return model_a_score, model_b_score

    # -------------------------------------------------------------------
    # STT + Context Analysis (Steps 5-6)
    # -------------------------------------------------------------------

    async def _run_stt_and_context(
        self,
        timestamp_sec: float,
    ) -> dict:
        """
        Batch-transcribe accumulated audio and run context analysis.

        Concatenates buffered audio chunks, transcribes, adds to the
        sliding transcript window, and runs phrase detection.
        """
        # Concatenate buffered audio
        if not self._stt_audio_buffer:
            return {"context_risk_score": 0.0, "top_signals": []}

        combined_audio = np.concatenate(self._stt_audio_buffer)
        self._stt_audio_buffer.clear()
        self._stt_chunk_count = 0

        # Determine whether to skip Sarvam (circuit breaker)
        prefer_whisper = self._stt_circuit_breaker.is_open

        # Transcribe
        try:
            stt_result = await transcribe(
                audio=combined_audio,
                sample_rate=8000,  # Twilio sample rate
                prefer_whisper=prefer_whisper,
            )

            transcript = stt_result.get("transcript", "")
            source = stt_result.get("source", "none")

            # Update circuit breaker
            if source == "sarvam":
                self._stt_circuit_breaker.record_success()
            elif source == "none" and not prefer_whisper:
                self._stt_circuit_breaker.record_failure()

            # Add to accumulator
            if transcript:
                self.transcript_accumulator.add(
                    transcript=transcript,
                    timestamp_sec=timestamp_sec,
                    language=stt_result.get("language", "unknown"),
                    confidence=stt_result.get("confidence", 0.0),
                    source=source,
                )

        except Exception as e:
            logger.error(
                "pipeline.stt_error",
                extra={"call_sid": self.call_sid, "error": str(e)},
            )
            if not prefer_whisper:
                self._stt_circuit_breaker.record_failure()
            return {"context_risk_score": 0.0, "top_signals": []}

        # Run context analysis on the accumulated transcript window
        full_text = self.transcript_accumulator.full_text
        if not full_text.strip():
            return {"context_risk_score": 0.0, "top_signals": []}

        try:
            context_result = self.context_analyzer.analyze(
                text=full_text,
                timestamp_sec=timestamp_sec,
            )
            return context_result
        except Exception as e:
            logger.error(
                "pipeline.context_analysis_error",
                extra={"call_sid": self.call_sid, "error": str(e)},
            )
            return {"context_risk_score": 0.0, "top_signals": []}

    # -------------------------------------------------------------------
    # Persistence (Step 9)
    # -------------------------------------------------------------------

    async def _persist_state(
        self,
        chunk_index: int,
        timestamp_sec: float,
        current_risk: float,
        acoustic_score: float,
        context_score: float,
    ) -> None:
        """
        Persist risk state to Redis (every chunk) and DynamoDB (batched).
        """
        # Always update Redis (hot path)
        try:
            await store_risk_state(
                call_sid=self.call_sid,
                current_score=current_risk,
                peak_score=self.risk_engine.get_peak_risk(),
                chunk_count=self._chunks_processed + 1,
            )
            await append_score_history(
                call_sid=self.call_sid,
                timestamp=time.time(),
                score=current_risk,
            )
        except Exception as e:
            logger.error(
                "pipeline.redis_persist_failed",
                extra={"call_sid": self.call_sid, "error": str(e)},
            )

        # Batch DynamoDB writes: only every Nth chunk (Step 145.9)
        if (chunk_index + 1) % self.DYNAMO_WRITE_EVERY == 0:
            try:
                await dynamo_client.append_risk_score(
                    call_sid=self.call_sid,
                    timestamp=timestamp_sec,
                    score=current_risk,
                    signals={
                        "acoustic": round(acoustic_score, 2),
                        "context": round(context_score, 2),
                    },
                )
            except Exception as e:
                logger.error(
                    "pipeline.dynamo_persist_failed",
                    extra={"call_sid": self.call_sid, "error": str(e)},
                )
