"""
VoiceGuard Backend — Pipeline Metrics Collector (Steps 148, 152)

Collects per-chunk performance metrics for profiling and monitoring:
    - End-to-end pipeline latency per chunk
    - Model A (AASIST) inference time
    - Model B (XGBoost) inference time
    - DSP feature extraction time
    - STT transcription time
    - Context analysis time

Metrics are logged as structured JSON (picked up by CloudWatch in
production) and exposed via a summary endpoint for real-time monitoring.

Usage:
    from app.pipeline.metrics import pipeline_metrics

    # Record a timing
    pipeline_metrics.record("feature_extraction", 42.5, call_sid="CA123")

    # Get summary stats
    stats = pipeline_metrics.summary()
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Keep the last N measurements per metric for rolling stats
_WINDOW_SIZE = 200


@dataclass
class _MetricWindow:
    """Rolling window of timing measurements for a single metric."""
    values: deque = field(default_factory=lambda: deque(maxlen=_WINDOW_SIZE))
    total_count: int = 0
    total_sum: float = 0.0

    def record(self, value_ms: float) -> None:
        self.values.append(value_ms)
        self.total_count += 1
        self.total_sum += value_ms

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    @property
    def max(self) -> float:
        if not self.values:
            return 0.0
        return max(self.values)

    @property
    def min(self) -> float:
        if not self.values:
            return 0.0
        return min(self.values)

    def _percentile(self, pct: int) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        idx = int(len(sorted_vals) * pct / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def to_dict(self) -> dict:
        return {
            "count": self.total_count,
            "window_size": self.count,
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
        }


class PipelineMetrics:
    """
    Collects and reports pipeline performance metrics.

    Thread-safe for concurrent call sessions since each metric
    window is append-only from the async event loop.
    """

    # Standard metric names
    TOTAL_PIPELINE = "total_pipeline"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_A_INFERENCE = "model_a_inference"
    MODEL_B_INFERENCE = "model_b_inference"
    ENSEMBLE = "ensemble"
    STT_TRANSCRIPTION = "stt_transcription"
    CONTEXT_ANALYSIS = "context_analysis"
    RISK_ENGINE = "risk_engine"
    ALERT_DISPATCH = "alert_dispatch"
    REDIS_PERSIST = "redis_persist"
    DYNAMO_PERSIST = "dynamo_persist"

    def __init__(self):
        self._metrics: dict[str, _MetricWindow] = defaultdict(_MetricWindow)
        self._start_time = time.time()
        self._chunks_total = 0
        self._alerts_total = 0
        self._stt_failures = 0

    def record(
        self,
        metric_name: str,
        value_ms: float,
        call_sid: Optional[str] = None,
    ) -> None:
        """
        Record a single timing measurement.

        Args:
            metric_name: Name of the metric (use class constants).
            value_ms: Duration in milliseconds.
            call_sid: Optional call SID for per-call logging.
        """
        self._metrics[metric_name].record(value_ms)

        # Log slow operations
        if metric_name == self.TOTAL_PIPELINE and value_ms > 500:
            logger.warning(
                "metrics.slow_chunk",
                extra={
                    "metric": metric_name,
                    "latency_ms": round(value_ms, 1),
                    "call_sid": call_sid,
                    "target_ms": 500,
                },
            )

    def record_chunk(self) -> None:
        """Increment the total chunks processed counter."""
        self._chunks_total += 1

    def record_alert(self) -> None:
        """Increment the total alerts dispatched counter."""
        self._alerts_total += 1

    def record_stt_failure(self) -> None:
        """Increment the STT failure counter."""
        self._stt_failures += 1

    def summary(self) -> dict:
        """
        Return a summary of all collected metrics.

        Suitable for the /health or /metrics endpoint.
        """
        uptime = time.time() - self._start_time
        result = {
            "uptime_sec": round(uptime, 1),
            "chunks_total": self._chunks_total,
            "alerts_total": self._alerts_total,
            "stt_failures": self._stt_failures,
            "timings": {},
        }

        for name, window in sorted(self._metrics.items()):
            result["timings"][name] = window.to_dict()

        # Add pipeline health assessment
        pipeline = self._metrics.get(self.TOTAL_PIPELINE)
        if pipeline and pipeline.count > 0:
            result["pipeline_health"] = {
                "target_ms": 500,
                "p50_ok": pipeline.p50 < 500,
                "p95_ok": pipeline.p95 < 500,
                "p50_ms": round(pipeline.p50, 1),
                "p95_ms": round(pipeline.p95, 1),
            }

        return result

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self._metrics.clear()
        self._start_time = time.time()
        self._chunks_total = 0
        self._alerts_total = 0
        self._stt_failures = 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

pipeline_metrics = PipelineMetrics()
