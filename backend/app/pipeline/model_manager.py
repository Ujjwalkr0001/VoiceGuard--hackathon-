"""
VoiceGuard Backend — Model Manager (Steps 149–150)

Handles loading, caching, and warm-up of ML models at application startup.
Models are loaded once and shared across all active call pipelines to
avoid redundant memory usage and model-load latency.

Models managed:
    - Model A: AASIST (or AASIST-L) deep neural network for raw audio
    - Model B: XGBoost classifier on DSP feature vectors

Usage:
    from app.pipeline.model_manager import model_manager

    # At app startup (in lifespan)
    await model_manager.load_all()

    # In pipeline coordinator
    coordinator.set_models(
        model_a=model_manager.model_a,
        model_b=model_manager.model_b_predict,
    )

    # At app shutdown
    model_manager.unload_all()
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model paths (relative to backend/ directory)
_DEFAULT_AASIST_CHECKPOINT = "app/ml/models/aasist_finetuned.pt"
_DEFAULT_XGBOOST_MODEL = "app/ml/models/xgboost_spoof.joblib"


class ModelManager:
    """
    Singleton manager for ML model lifecycle.

    Handles lazy/eager loading, warm-up inference, and provides
    references for the pipeline coordinator to use.
    """

    def __init__(self):
        self.model_a = None           # AASIST wrapper instance
        self.model_b = None           # XGBoost model object
        self._loaded = False
        self._load_times: dict = {}   # Model name → load time in ms

    # -------------------------------------------------------------------
    # Step 149: Load models once at startup
    # -------------------------------------------------------------------

    async def load_all(
        self,
        aasist_checkpoint: Optional[str] = None,
        xgboost_model_path: Optional[str] = None,
    ) -> dict:
        """
        Load all ML models into memory.

        Runs in a thread pool to avoid blocking the event loop during
        large model deserialization.

        Args:
            aasist_checkpoint: Path to AASIST TorchScript checkpoint.
            xgboost_model_path: Path to XGBoost joblib file.

        Returns:
            Dict with load status and timing for each model.
        """
        results = {}

        # --- Model A: AASIST ---
        results["model_a"] = await self._load_model_a(aasist_checkpoint)

        # --- Model B: XGBoost ---
        results["model_b"] = await self._load_model_b(xgboost_model_path)

        self._loaded = True
        logger.info(
            "model_manager.all_loaded",
            extra={"results": results, "load_times_ms": self._load_times},
        )
        return results

    async def _load_model_a(self, checkpoint_path: Optional[str] = None) -> dict:
        """Load AASIST model."""
        path = checkpoint_path or _DEFAULT_AASIST_CHECKPOINT
        result = {"model": "aasist", "status": "not_found", "path": path}

        if not Path(path).exists():
            logger.warning(
                "model_manager.aasist_not_found",
                extra={"path": path, "detail": "AASIST checkpoint not found - Model A will return neutral scores"},
            )
            return result

        def _load():
            t0 = time.perf_counter()
            try:
                # Try importing the AASIST wrapper from ml/models
                import sys
                sys.path.insert(0, str(Path("..").resolve()))
                from ml.models.aasist_wrapper import AASISTWrapper
                wrapper = AASISTWrapper(checkpoint_path=path)
                elapsed = (time.perf_counter() - t0) * 1000
                return wrapper, elapsed
            except ImportError as ie:
                try:
                    # Fallback: try loading as a TorchScript model
                    import torch
                    model = torch.jit.load(path, map_location="cpu")
                    model.eval()
                    elapsed = (time.perf_counter() - t0) * 1000
                    return model, elapsed
                except ImportError:
                    # torch not installed — skip gracefully
                    raise ImportError(
                        "torch is not installed — skipping AASIST model loading"
                    ) from ie

        try:
            self.model_a, load_ms = await asyncio.to_thread(_load)
            self._load_times["model_a"] = round(load_ms, 1)
            result["status"] = "loaded"
            result["load_ms"] = round(load_ms, 1)
            logger.info(
                "model_manager.aasist_loaded",
                extra={"path": path, "load_ms": round(load_ms, 1)},
            )
        except ImportError as e:
            result["status"] = "skipped"
            result["reason"] = str(e)
            logger.warning(
                "model_manager.aasist_skipped",
                extra={"path": path, "reason": str(e)},
            )
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(
                "model_manager.aasist_load_failed",
                extra={"path": path, "error": str(e)},
            )

        return result

    async def _load_model_b(self, model_path: Optional[str] = None) -> dict:
        """Load XGBoost model."""
        path = model_path or _DEFAULT_XGBOOST_MODEL
        result = {"model": "xgboost", "status": "not_found", "path": path}

        if not Path(path).exists():
            logger.warning(
                "model_manager.xgboost_not_found",
                extra={"path": path, "detail": "XGBoost model not found - Model B will be disabled"},
            )
            return result

        def _load():
            import joblib
            t0 = time.perf_counter()
            model = joblib.load(path)
            elapsed = (time.perf_counter() - t0) * 1000
            return model, elapsed

        try:
            self.model_b, load_ms = await asyncio.to_thread(_load)
            self._load_times["model_b"] = round(load_ms, 1)
            result["status"] = "loaded"
            result["load_ms"] = round(load_ms, 1)
            logger.info(
                "model_manager.xgboost_loaded",
                extra={"path": path, "load_ms": round(load_ms, 1)},
            )
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(
                "model_manager.xgboost_load_failed",
                extra={"path": path, "error": str(e)},
            )

        return result

    # -------------------------------------------------------------------
    # Step 150: Warm-up inference
    # -------------------------------------------------------------------

    async def warm_up(self) -> dict:
        """
        Run dummy inference on each loaded model to JIT-compile and warm caches.

        This ensures the first real inference doesn't have cold-start latency.

        Returns:
            Dict with warm-up timing for each model.
        """
        results = {}

        # Warm up Model A (AASIST)
        if self.model_a is not None:
            results["model_a"] = await self._warm_up_model_a()

        # Warm up Model B (XGBoost)
        if self.model_b is not None:
            results["model_b"] = await self._warm_up_model_b()

        logger.info(
            "model_manager.warm_up_complete",
            extra={"results": results},
        )
        return results

    async def _warm_up_model_a(self) -> dict:
        """Run dummy AASIST inference."""
        # Generate 2 seconds of random audio at 8kHz
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.01

        def _infer():
            t0 = time.perf_counter()
            try:
                if hasattr(self.model_a, "predict"):
                    self.model_a.predict(dummy_audio)
                else:
                    # TorchScript model
                    import torch
                    with torch.no_grad():
                        tensor = torch.from_numpy(dummy_audio).unsqueeze(0)
                        self.model_a(tensor)
                return (time.perf_counter() - t0) * 1000
            except Exception as e:
                logger.warning(
                    "model_manager.warm_up_model_a_error",
                    extra={"error": str(e)},
                )
                return -1.0

        try:
            latency_ms = await asyncio.to_thread(_infer)
            return {"status": "warmed", "latency_ms": round(latency_ms, 1)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _warm_up_model_b(self) -> dict:
        """Run dummy XGBoost inference."""
        # 96-dim feature vector (matches extract_all_features output)
        dummy_features = np.random.randn(1, 96).astype(np.float32)

        def _infer():
            t0 = time.perf_counter()
            try:
                if hasattr(self.model_b, "predict_proba"):
                    self.model_b.predict_proba(dummy_features)
                elif hasattr(self.model_b, "predict"):
                    self.model_b.predict(dummy_features)
                return (time.perf_counter() - t0) * 1000
            except Exception as e:
                logger.warning(
                    "model_manager.warm_up_model_b_error",
                    extra={"error": str(e)},
                )
                return -1.0

        try:
            latency_ms = await asyncio.to_thread(_infer)
            return {"status": "warmed", "latency_ms": round(latency_ms, 1)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # -------------------------------------------------------------------
    # XGBoost prediction helper
    # -------------------------------------------------------------------

    def model_b_predict(self, feature_vector: np.ndarray) -> float:
        """
        XGBoost inference wrapper compatible with the pipeline coordinator.

        Args:
            feature_vector: 1-D numpy array of DSP features (96-dim).

        Returns:
            Spoof probability in [0, 1].
        """
        if self.model_b is None:
            return 0.5  # Neutral fallback

        features_2d = feature_vector.reshape(1, -1)

        if hasattr(self.model_b, "predict_proba"):
            # XGBoost sklearn API: predict_proba returns [[P(bon), P(spoof)]]
            proba = self.model_b.predict_proba(features_2d)
            return float(proba[0][1])  # P(spoof)
        else:
            # Raw XGBoost: predict returns margin/probability directly
            pred = self.model_b.predict(features_2d)
            return float(pred[0])

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def unload_all(self) -> None:
        """Release all model references and free memory."""
        self.model_a = None
        self.model_b = None
        self._loaded = False
        logger.info("model_manager.unloaded")

    @property
    def status(self) -> dict:
        """Return current model loading status."""
        return {
            "loaded": self._loaded,
            "model_a": "loaded" if self.model_a is not None else "not_loaded",
            "model_b": "loaded" if self.model_b is not None else "not_loaded",
            "load_times_ms": self._load_times,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

model_manager = ModelManager()
