"""
VoiceGuard — AASIST Inference Wrapper

Wraps the pretrained AASIST (Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks) model for inference.

Provides a clean interface for the VoiceGuard pipeline to get
spoof probability scores from raw audio chunks.

Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
    Spectro-Temporal Graph Attention Networks", 2021.
    https://github.com/clovaai/aasist
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add the AASIST repo to sys.path so we can import the Model class
AASIST_REPO_DIR = Path(__file__).parent / "aasist_repo"
if str(AASIST_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_REPO_DIR))

logger = logging.getLogger(__name__)

# Default model configuration (matches AASIST.conf)
DEFAULT_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,       # ~4.04 seconds at 16 kHz
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# Lightweight AASIST-L configuration (85K params, faster inference)
AASIST_L_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

# Default checkpoint paths (relative to this file)
DEFAULT_CHECKPOINT = AASIST_REPO_DIR / "models" / "weights" / "AASIST.pth"
AASIST_L_CHECKPOINT = AASIST_REPO_DIR / "models" / "weights" / "AASIST-L.pth"

# Expected sample rate
EXPECTED_SR = 16000


class AASISTWrapper:
    """
    Inference wrapper for the AASIST anti-spoofing model.

    Usage:
        wrapper = AASISTWrapper()
        wrapper.load_model()  # Uses default pretrained checkpoint

        # Single inference
        score = wrapper.predict(audio_16khz)
        # score ∈ [0.0, 1.0], where higher = more likely spoofed

        # Batch inference
        scores = wrapper.predict_batch([audio1, audio2, ...])
    """

    def __init__(
        self,
        model_config: Optional[dict] = None,
        device: Optional[str] = None,
        use_lightweight: bool = False,
    ):
        """
        Initialize the AASIST wrapper.

        Args:
            model_config: Model hyperparameters dict. If None, uses default.
            device: 'cuda', 'cpu', or None (auto-detect).
            use_lightweight: If True, use AASIST-L (85K params, faster).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_lightweight = use_lightweight

        if model_config is None:
            self.model_config = AASIST_L_MODEL_CONFIG if use_lightweight else DEFAULT_MODEL_CONFIG
        else:
            self.model_config = model_config

        self.model = None
        self.nb_samp = self.model_config.get("nb_samp", 64600)
        self._is_loaded = False

    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Load the AASIST model and pretrained weights.

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
                If None, uses the default pretrained checkpoint
                from the cloned AASIST repository.
        """
        if checkpoint_path is None:
            checkpoint_path = str(
                AASIST_L_CHECKPOINT if self.use_lightweight else DEFAULT_CHECKPOINT
            )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"AASIST checkpoint not found: {checkpoint_path}\n"
                f"Make sure the AASIST repo is cloned at: {AASIST_REPO_DIR}"
            )

        logger.info(f"Loading AASIST model from {checkpoint_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model variant: {'AASIST-L (lightweight)' if self.use_lightweight else 'AASIST (full)'}")

        # Import the Model class from the AASIST repo
        from models.AASIST import Model

        # Instantiate model
        self.model = Model(self.model_config)

        # Load checkpoint
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Load weights
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._is_loaded = True

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"AASIST model loaded: {n_params:,} parameters")

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded before inference."""
        if not self._is_loaded or self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        Pad (with repetition) or truncate audio to match the expected
        input length (nb_samp).

        AASIST expects a fixed-length input (64600 samples = ~4.04s at 16kHz).
        For shorter chunks (e.g., 2-second VoiceGuard windows), we repeat-pad.
        For longer audio, we truncate to nb_samp.

        Args:
            audio: 1D numpy array of audio samples.

        Returns:
            1D numpy array of exactly nb_samp samples.
        """
        length = len(audio)

        if length >= self.nb_samp:
            # Truncate
            return audio[:self.nb_samp]

        # Repeat-pad to fill nb_samp
        repeats = (self.nb_samp // length) + 1
        padded = np.tile(audio, repeats)
        return padded[:self.nb_samp]

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int = EXPECTED_SR) -> float:
        """
        Run inference on a single audio sample and return spoof probability.

        Args:
            audio: 1D numpy array of audio samples (mono, float32).
                   Expected at 16 kHz. If a different sample rate is
                   provided, the caller should resample first.
            sr: Sample rate of the input audio (default: 16000).

        Returns:
            Spoof probability in [0.0, 1.0].
            - 0.0 → very likely bonafide (genuine)
            - 1.0 → very likely spoofed (AI-generated)
        """
        self._ensure_loaded()

        if sr != EXPECTED_SR:
            logger.warning(
                f"Input audio is at {sr} Hz, but AASIST expects {EXPECTED_SR} Hz. "
                f"Consider resampling before calling predict()."
            )

        # Ensure float32
        audio = audio.astype(np.float32)

        # Pad or truncate to expected length
        audio = self._pad_or_truncate(audio)

        # Convert to tensor: shape (1, nb_samp)
        x = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        # Forward pass
        _, output = self.model(x)

        # output shape: (1, 2) — logits for [bonafide, spoof]
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)

        # Return spoof probability (index 1)
        spoof_prob = probs[0, 1].item()

        return spoof_prob

    @torch.no_grad()
    def predict_batch(
        self,
        audio_list: list,
        sr: int = EXPECTED_SR,
    ) -> list:
        """
        Run inference on a batch of audio samples.

        Args:
            audio_list: List of 1D numpy arrays (mono, float32, 16kHz).
            sr: Sample rate.

        Returns:
            List of spoof probabilities in [0.0, 1.0].
        """
        self._ensure_loaded()

        if not audio_list:
            return []

        # Prepare batch tensor
        batch = []
        for audio in audio_list:
            audio = audio.astype(np.float32)
            audio = self._pad_or_truncate(audio)
            batch.append(torch.FloatTensor(audio))

        x = torch.stack(batch, dim=0).to(self.device)

        # Forward pass
        _, output = self.model(x)

        # Softmax → spoof probabilities
        probs = F.softmax(output, dim=1)
        spoof_probs = probs[:, 1].cpu().numpy().tolist()

        return spoof_probs

    def predict_with_details(
        self,
        audio: np.ndarray,
        sr: int = EXPECTED_SR,
    ) -> dict:
        """
        Run inference and return detailed results including timing.

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sample rate.

        Returns:
            Dict with keys:
            - spoof_probability: float in [0, 1]
            - bonafide_probability: float in [0, 1]
            - is_spoofed: bool (threshold at 0.5)
            - inference_time_ms: float
            - input_duration_sec: float
            - model_variant: str
        """
        self._ensure_loaded()

        input_duration = len(audio) / sr

        start_time = time.perf_counter()
        audio = audio.astype(np.float32)
        audio = self._pad_or_truncate(audio)

        x = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        _, output = self.model(x)
        probs = F.softmax(output, dim=1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        spoof_prob = probs[0, 1].item()
        bonafide_prob = probs[0, 0].item()

        return {
            "spoof_probability": spoof_prob,
            "bonafide_probability": bonafide_prob,
            "is_spoofed": spoof_prob > 0.5,
            "inference_time_ms": round(elapsed_ms, 2),
            "input_duration_sec": round(input_duration, 3),
            "model_variant": "AASIST-L" if self.use_lightweight else "AASIST",
        }

    def warmup(self, n_runs: int = 3) -> float:
        """
        Run dummy inferences to warm up the model (JIT compilation, cache priming).

        Args:
            n_runs: Number of warm-up forward passes.

        Returns:
            Average inference time in milliseconds after warm-up.
        """
        self._ensure_loaded()

        dummy_audio = np.random.randn(self.nb_samp).astype(np.float32)
        times = []

        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(dummy_audio)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        logger.info(f"AASIST warm-up complete: avg {avg_ms:.1f}ms per inference ({n_runs} runs)")

        return avg_ms


# ---------------------------------------------------------------------------
# Module-level singleton for use across the backend
# ---------------------------------------------------------------------------

_global_wrapper: Optional[AASISTWrapper] = None


def get_aasist_model(
    use_lightweight: bool = False,
    checkpoint_path: Optional[str] = None,
) -> AASISTWrapper:
    """
    Get or initialize the global AASIST model singleton.

    Thread-safe for reads after initialization (model is in eval mode
    with no gradient computation).

    Args:
        use_lightweight: Use AASIST-L (faster, fewer params).
        checkpoint_path: Optional path to a custom checkpoint.

    Returns:
        Initialized AASISTWrapper ready for inference.
    """
    global _global_wrapper

    if _global_wrapper is None:
        _global_wrapper = AASISTWrapper(use_lightweight=use_lightweight)
        _global_wrapper.load_model(checkpoint_path)
        _global_wrapper.warmup()

    return _global_wrapper
