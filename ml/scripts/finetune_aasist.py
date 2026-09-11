"""
VoiceGuard — AASIST Fine-Tuning Script (Step 76)

Fine-tunes a pretrained AASIST model on the combined VoiceGuard dataset
(ASVspoof 2019 + Sarvam TTS + In-the-Wild + Other TTS).

Features:
  - Custom PyTorch Dataset loading from manifest CSV or directory structure
  - Data augmentation pipeline:
      * Additive noise (SNR 10-30 dB)
      * Room impulse response (RIR) convolution
      * Telephone band-pass filter (300-3400 Hz, simulating Twilio 8kHz mu-law)
  - Adam optimizer with cosine annealing LR scheduler
  - Validation after each epoch: EER, precision, recall, F1
  - Early stopping on EER (patience = 5 epochs)
  - Checkpoint saving (best + periodic)
  - TorchScript export of best model

Usage:
    python scripts/finetune_aasist.py
    python scripts/finetune_aasist.py --epochs 30 --batch-size 24 --lr 0.0001
    python scripts/finetune_aasist.py --use-lightweight --freeze-encoder
    python scripts/finetune_aasist.py --resume checkpoints/epoch_010.pth
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add paths
ML_DIR = Path(__file__).parent.parent
MODELS_DIR = ML_DIR / "models"
DATA_DIR = ML_DIR / "data"
AASIST_REPO = MODELS_DIR / "aasist_repo"

sys.path.insert(0, str(AASIST_REPO))

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


# ===========================================================================
# Data Augmentation
# ===========================================================================

class AudioAugmentor:
    """
    Audio augmentation pipeline for anti-spoofing training.

    Augmentations (applied randomly per sample):
      1. Additive Gaussian noise at random SNR (10-30 dB)
      2. Room impulse response (synthetic RIR) convolution
      3. Telephone band-pass filter (300-3400 Hz, simulating Twilio)
    """

    def __init__(
        self,
        sr: int = 16000,
        noise_snr_range: Tuple[float, float] = (10.0, 30.0),
        noise_prob: float = 0.5,
        rir_prob: float = 0.3,
        telephone_prob: float = 0.4,
        enable: bool = True,
    ):
        self.sr = sr
        self.noise_snr_range = noise_snr_range
        self.noise_prob = noise_prob
        self.rir_prob = rir_prob
        self.telephone_prob = telephone_prob
        self.enable = enable

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if not self.enable:
            return audio

        # Apply augmentations randomly
        if np.random.random() < self.noise_prob:
            audio = self._add_noise(audio)

        if np.random.random() < self.rir_prob:
            audio = self._apply_rir(audio)

        if np.random.random() < self.telephone_prob:
            audio = self._telephone_filter(audio)

        return audio

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at a random SNR between snr_range."""
        snr_db = np.random.uniform(*self.noise_snr_range)
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-10:
            return audio
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return (audio + noise).astype(np.float32)

    def _apply_rir(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply synthetic room impulse response via convolution.
        Uses a simple exponentially decaying impulse to simulate
        room reverb without requiring external RIR files.
        """
        # Generate synthetic RIR: exponential decay with random params
        rt60 = np.random.uniform(0.1, 0.6)  # Reverberation time
        rir_length = int(rt60 * self.sr)
        if rir_length < 10:
            return audio

        t = np.arange(rir_length) / self.sr
        decay = np.exp(-6.9 * t / rt60)  # -60dB at RT60

        # Random early reflections
        rir = np.zeros(rir_length, dtype=np.float32)
        rir[0] = 1.0  # Direct path
        n_reflections = np.random.randint(3, 8)
        for _ in range(n_reflections):
            delay = np.random.randint(1, rir_length)
            amplitude = np.random.uniform(0.1, 0.5) * decay[delay]
            rir[delay] += amplitude * np.random.choice([-1, 1])

        # Apply decay envelope + add diffuse tail
        diffuse = np.random.normal(0, 0.02, rir_length).astype(np.float32) * decay
        rir += diffuse

        # Normalize RIR
        rir = rir / (np.max(np.abs(rir)) + 1e-8)

        # Convolve (keep original length)
        convolved = np.convolve(audio, rir, mode="full")[:len(audio)]

        # Normalize to original RMS
        orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        conv_rms = np.sqrt(np.mean(convolved ** 2)) + 1e-8
        convolved = convolved * (orig_rms / conv_rms)

        return convolved.astype(np.float32)

    def _telephone_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply telephone-band band-pass filter (300-3400 Hz).
        Simulates Twilio's 8 kHz mu-law codec frequency response.

        Uses a simple FIR filter designed via windowed sinc method.
        """
        from scipy.signal import firwin, lfilter

        # Design band-pass filter: 300-3400 Hz
        nyq = self.sr / 2.0
        low = 300.0 / nyq
        high = min(3400.0 / nyq, 0.99)  # Clamp below Nyquist

        # FIR filter with 101 taps
        numtaps = 101
        try:
            coeffs = firwin(numtaps, [low, high], pass_zero=False)
            filtered = lfilter(coeffs, 1.0, audio)
        except Exception:
            # Fallback: skip filtering if scipy has issues
            return audio

        # Normalize
        orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        filt_rms = np.sqrt(np.mean(filtered ** 2)) + 1e-8
        filtered = filtered * (orig_rms / filt_rms)

        return filtered.astype(np.float32)


# ===========================================================================
# Dataset
# ===========================================================================

def pad_or_truncate(x: np.ndarray, max_len: int = 64600, random_crop: bool = True) -> np.ndarray:
    """
    Pad (repeat) or truncate audio to fixed length.

    Args:
        x: Input audio array.
        max_len: Target length (64600 = ~4.04s at 16kHz).
        random_crop: If True, take a random crop from longer audio (training).
                     If False, take from the start (evaluation).
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        if random_crop:
            start = np.random.randint(0, x_len - max_len + 1)
            return x[start:start + max_len]
        return x[:max_len]

    # Repeat-pad
    num_repeats = (max_len // x_len) + 1
    padded = np.tile(x, num_repeats)[:max_len]
    return padded


class VoiceGuardDataset(Dataset):
    """
    PyTorch Dataset for VoiceGuard anti-spoofing training.

    Loads audio files from a manifest CSV or from directory structure:
      data/{train,eval}/{bonafide,spoof}/*.wav

    Returns (waveform_tensor, label) pairs where:
      - waveform: float32 tensor of shape (nb_samp,)
      - label: 1 = bonafide, 0 = spoof
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        manifest_path: Optional[Path] = None,
        nb_samp: int = 64600,
        augmentor: Optional[AudioAugmentor] = None,
        max_files: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Root data directory (ml/data/).
            split: 'train' or 'eval'.
            manifest_path: Path to manifest.csv. If None, scan directories.
            nb_samp: Target audio length in samples.
            augmentor: AudioAugmentor instance for training augmentation.
            max_files: Limit number of files (for debugging).
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.nb_samp = nb_samp
        self.augmentor = augmentor
        self.is_train = (split == "train")

        self.file_list: List[Path] = []
        self.labels: List[int] = []
        self.sources: List[str] = []

        if manifest_path and Path(manifest_path).exists():
            self._load_from_manifest(manifest_path, split)
        else:
            self._load_from_directories(split)

        if max_files and len(self.file_list) > max_files:
            indices = np.random.RandomState(42).choice(
                len(self.file_list), max_files, replace=False
            )
            self.file_list = [self.file_list[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.sources = [self.sources[i] for i in indices]

        n_bonafide = sum(1 for l in self.labels if l == 1)
        n_spoof = sum(1 for l in self.labels if l == 0)
        print(f"  [{split}] Loaded {len(self.file_list)} files "
              f"({n_bonafide} bonafide, {n_spoof} spoof)")

    def _load_from_manifest(self, manifest_path: Path, split: str):
        """Load file list from manifest CSV."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"]
                label_str = row["label"]
                source = row.get("source", "unknown")

                # Filter by split: manifest paths start with train/ or eval/
                if not path.startswith(f"{split}/"):
                    continue

                full_path = self.data_dir / path
                if not full_path.exists():
                    continue

                label = 1 if label_str == "bonafide" else 0
                self.file_list.append(full_path)
                self.labels.append(label)
                self.sources.append(source)

    def _load_from_directories(self, split: str):
        """Scan directory structure for audio files."""
        split_dir = self.data_dir / split

        for label_name, label_int in [("bonafide", 1), ("spoof", 0)]:
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue

            for f in sorted(label_dir.rglob("*")):
                if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file():
                    self.file_list.append(f)
                    self.labels.append(label_int)
                    self.sources.append("directory")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        filepath = self.file_list[index]
        label = self.labels[index]

        # Load audio
        try:
            import soundfile as sf
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception:
            try:
                import librosa
                audio, sr = librosa.load(str(filepath), sr=16000, mono=True)
            except Exception:
                # Return silence on failure
                audio = np.zeros(self.nb_samp, dtype=np.float32)
                return torch.FloatTensor(audio), label

        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        audio = audio.astype(np.float32)

        # Apply augmentation (training only)
        if self.augmentor is not None and self.is_train:
            audio = self.augmentor(audio)

        # Pad or truncate to fixed length
        audio = pad_or_truncate(audio, self.nb_samp, random_crop=self.is_train)

        return torch.FloatTensor(audio), label


# ===========================================================================
# Metrics: EER Computation
# ===========================================================================

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    Args:
        scores: Array of spoof probability scores.
        labels: Array of labels (1 = bonafide, 0 = spoof).

    Returns:
        (eer, threshold) where eer is in [0, 1] and threshold is the
        operating point where FAR == FRR.
    """
    # Sort by score
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    n_bonafide = np.sum(labels == 1)
    n_spoof = np.sum(labels == 0)

    if n_bonafide == 0 or n_spoof == 0:
        return 0.5, 0.5  # Undefined

    # Compute FAR and FRR at each threshold
    fars = []
    frrs = []
    thresholds = np.unique(sorted_scores)

    # Add boundary thresholds
    thresholds = np.concatenate([[sorted_scores[0] - 0.01], thresholds, [sorted_scores[-1] + 0.01]])

    for threshold in thresholds:
        # Scores >= threshold → classified as spoof
        predicted_spoof = scores >= threshold

        # FAR: bonafide samples incorrectly classified as spoof
        far = np.sum(predicted_spoof & (labels == 1)) / n_bonafide

        # FRR: spoof samples incorrectly classified as bonafide
        frr = np.sum(~predicted_spoof & (labels == 0)) / n_spoof

        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)

    # Find intersection point (EER)
    diffs = fars - frrs
    # Find where FAR crosses FRR
    idx = np.argmin(np.abs(diffs))
    eer = (fars[idx] + frrs[idx]) / 2
    eer_threshold = thresholds[idx]

    return float(eer), float(eer_threshold)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute classification metrics at a given threshold.

    Returns dict with: eer, threshold, precision, recall, f1, accuracy
    """
    eer, eer_thresh = compute_eer(scores, labels)

    # Binary predictions at the given threshold
    predicted_spoof = scores >= threshold
    actual_spoof = labels == 0

    tp = np.sum(predicted_spoof & actual_spoof)
    fp = np.sum(predicted_spoof & ~actual_spoof)
    fn = np.sum(~predicted_spoof & actual_spoof)
    tn = np.sum(~predicted_spoof & ~actual_spoof)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    return {
        "eer": eer,
        "eer_threshold": eer_thresh,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ===========================================================================
# Training Loop
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (waveforms, labels) in enumerate(dataloader):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # AASIST output: (hidden, logits) where logits shape is (batch, 2)
        # Label convention: 1 = bonafide, 0 = spoof
        _, logits = model(waveforms)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / n_batches
            print(f"    Batch [{batch_idx+1}/{len(dataloader)}] loss={avg:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict]:
    """
    Run validation and compute metrics.

    Returns (avg_loss, metrics_dict).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0
    all_scores = []
    all_labels = []

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels_dev = labels.to(device)

        _, logits = model(waveforms)
        loss = criterion(logits, labels_dev)

        # Get spoof probability (softmax → index 0 is spoof score)
        probs = F.softmax(logits, dim=1)
        spoof_scores = probs[:, 0].cpu().numpy()  # Index 0 = spoof class

        all_scores.extend(spoof_scores.tolist())
        all_labels.extend(labels.numpy().tolist())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(np.array(all_scores), np.array(all_labels))

    return avg_loss, metrics


# ===========================================================================
# Main Fine-Tuning Pipeline
# ===========================================================================

def finetune(args: argparse.Namespace) -> None:
    """Main fine-tuning pipeline."""

    print("\n" + "=" * 65)
    print("  VoiceGuard — AASIST Fine-Tuning")
    print("=" * 65)

    # ---- Device ----
    device = torch.device(args.device if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    print(f"  Device: {device}")

    # ---- Model ----
    variant = "AASIST-L" if args.use_lightweight else "AASIST"
    print(f"  Model: {variant}")

    from models.AASIST import Model

    if args.use_lightweight:
        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
            "gat_dims": [24, 32],
            "pool_ratios": [0.4, 0.5, 0.7, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }
        pretrained_path = AASIST_REPO / "models" / "weights" / "AASIST-L.pth"
    else:
        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }
        pretrained_path = AASIST_REPO / "models" / "weights" / "AASIST.pth"

    model = Model(model_config)
    nb_samp = model_config["nb_samp"]

    # Load pretrained weights
    if args.resume:
        checkpoint_path = Path(args.resume)
        print(f"  Resuming from: {checkpoint_path}")
    else:
        checkpoint_path = pretrained_path
        print(f"  Loading pretrained: {checkpoint_path}")

    if checkpoint_path.exists():
        state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"  Weights loaded successfully")
    else:
        print(f"  WARNING: Checkpoint not found, training from scratch!")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ---- Optionally freeze encoder ----
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name or "conv_time" in name or "first_bn" in name:
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Encoder frozen. Trainable params: {n_trainable:,}")

    # ---- Data ----
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.csv"

    print(f"\n  Data directory: {data_dir}")
    print(f"  Manifest: {manifest_path if manifest_path.exists() else 'not found (scanning dirs)'}")

    augmentor = AudioAugmentor(
        sr=16000,
        noise_prob=0.5,
        rir_prob=0.3,
        telephone_prob=0.4,
        enable=True,
    )

    train_dataset = VoiceGuardDataset(
        data_dir=data_dir,
        split="train",
        manifest_path=manifest_path if manifest_path.exists() else None,
        nb_samp=nb_samp,
        augmentor=augmentor,
        max_files=args.max_files,
    )

    eval_dataset = VoiceGuardDataset(
        data_dir=data_dir,
        split="eval",
        manifest_path=manifest_path if manifest_path.exists() else None,
        nb_samp=nb_samp,
        augmentor=None,  # No augmentation for eval
        max_files=args.max_files,
    )

    if len(train_dataset) == 0:
        print("\n  ERROR: No training data found!")
        print("  Run: python scripts/prepare_dataset.py first")
        print("  Or download datasets: python scripts/download_datasets.py --instructions")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    ) if len(eval_dataset) > 0 else None

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr_min,
    )

    # ---- Output directory ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Save config
    config = {
        "model_config": model_config,
        "variant": variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "weight_decay": args.weight_decay,
        "freeze_encoder": args.freeze_encoder,
        "patience": args.patience,
        "train_files": len(train_dataset),
        "eval_files": len(eval_dataset) if eval_dataset else 0,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Output: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} -> {args.lr_min}")
    print(f"  Early stopping patience: {args.patience}")

    # ---- Training Loop ----
    best_eer = 1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    print(f"\n{'=' * 65}")
    print(f"  Starting training...")
    print(f"{'=' * 65}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        print(f"  Epoch [{epoch}/{args.epochs}]")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss = 0.0
        metrics = {}
        if eval_loader is not None:
            val_loss, metrics = validate(model, eval_loader, device)
            eer = metrics["eer"]
        else:
            eer = None

        elapsed = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]["lr"]

        # Log
        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "eer": eer,
            "lr": lr_current,
            "elapsed_sec": elapsed,
            **metrics,
        }
        history.append(epoch_info)

        eer_str = f"{eer:.4f}" if eer is not None else "N/A"
        print(f"    Loss: train={train_loss:.4f} val={val_loss:.4f} | "
              f"EER={eer_str} | F1={metrics.get('f1', 0):.3f} | "
              f"LR={lr_current:.6f} | {elapsed:.1f}s")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            path = checkpoints_dir / f"epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), str(path))
            print(f"    Checkpoint saved: {path.name}")

        # Early stopping on EER
        if eer is not None:
            if eer < best_eer:
                best_eer = eer
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                best_path = checkpoints_dir / "best.pth"
                torch.save(model.state_dict(), str(best_path))
                print(f"    ** New best EER: {best_eer:.4f} (epoch {epoch}) **")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n  Early stopping triggered (patience={args.patience})")
                    print(f"  Best EER: {best_eer:.4f} at epoch {best_epoch}")
                    break

    # ---- Save final model + history ----
    final_path = checkpoints_dir / "final.pth"
    torch.save(model.state_dict(), str(final_path))

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- Export best model as TorchScript ----
    if args.export_torchscript:
        print(f"\n  Exporting best model as TorchScript...")
        best_path = checkpoints_dir / "best.pth"
        if best_path.exists():
            model.load_state_dict(torch.load(str(best_path), map_location=device, weights_only=False))
        model.eval()

        try:
            # Use tracing (AASIST has dynamic shapes that make scripting difficult)
            dummy_input = torch.randn(1, nb_samp).to(device)
            traced = torch.jit.trace(model, dummy_input)
            export_path = output_dir / "aasist_finetuned.pt"
            traced.save(str(export_path))
            print(f"  TorchScript exported: {export_path}")
        except Exception as e:
            print(f"  WARNING: TorchScript export failed: {e}")
            print(f"  Saving as state_dict instead.")

    # ---- Final summary ----
    print(f"\n{'=' * 65}")
    print(f"  Fine-tuning complete!")
    print(f"  Best EER:      {best_eer:.4f} (epoch {best_epoch})")
    print(f"  Total epochs:  {len(history)}")
    print(f"  Checkpoints:   {checkpoints_dir}")
    print(f"  History:       {output_dir / 'training_history.json'}")
    print(f"{'=' * 65}\n")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VoiceGuard — Fine-tune AASIST anti-spoofing model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--use-lightweight", action="store_true",
                        help="Use AASIST-L (85K params, faster)")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers, only train GAT + output")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Data
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Path to data directory")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit files per split (for testing)")

    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Batch size (default: 24)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--lr-min", type=float, default=0.000005,
                        help="Minimum learning rate for cosine annealing")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="Weight decay (default: 0.0001)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience on EER (default: 5)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=str(ML_DIR / "outputs" / "finetune"),
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--export-torchscript", action="store_true",
                        help="Export best model as TorchScript after training")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu', 'cuda', or None (auto)")

    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
