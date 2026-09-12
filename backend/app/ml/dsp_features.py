"""
VoiceGuard Backend — DSP Feature Extraction Module

Extracts acoustic features from raw PCM audio chunks for voice clone detection.
Each function takes a numpy audio array and sample rate, and returns:
  1. A fixed-length numpy feature vector (for ML model input)
  2. A human-readable summary dict (for the risk explanation panel)

Feature families:
  - Group Delay:   Phase-based features from modified group delay function (MODGDF)
  - CQCC:          Constant-Q Cepstral Coefficients with deltas
  - Pitch Jitter:  F0 micro-prosody and perturbation measures
  - Pause/Rhythm:  VAD-based speech/silence statistics

Usage:
    from app.ml.dsp_features import extract_all_features
    features = extract_all_features(audio_array, sample_rate=8000)
"""

import numpy as np
import structlog
from scipy.fft import dct
from scipy.signal import get_window

logger = structlog.get_logger(__name__)

# Default STFT parameters (tuned for 8 kHz Twilio audio, 2-second chunks)
_DEFAULT_N_FFT = 512
_DEFAULT_HOP_LENGTH = 160  # 20 ms hop at 8 kHz
_DEFAULT_WIN_LENGTH = 400  # 50 ms window at 8 kHz


def _stft(audio: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform.

    Returns:
        Complex STFT matrix of shape (n_frames, n_fft // 2 + 1).
    """
    window = get_window("hann", win_length, fftbins=True)

    # Pad audio so we get complete frames
    pad_length = n_fft // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode="reflect")

    # Calculate number of frames
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    n_bins = n_fft // 2 + 1

    stft_matrix = np.zeros((n_frames, n_bins), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio_padded[start : start + n_fft]
        # Apply window (zero-pad if win_length < n_fft)
        windowed = np.zeros(n_fft)
        windowed[:win_length] = frame[:win_length] * window
        stft_matrix[i] = np.fft.rfft(windowed)

    return stft_matrix


def _cepstral_smoothing(magnitude_spectrum: np.ndarray, n_ceps: int = 30) -> np.ndarray:
    """
    Apply cepstral smoothing to a magnitude spectrum to get the spectral envelope.

    This suppresses fine spectral detail (zeros) that cause group delay instability.
    """
    log_spec = np.log(magnitude_spectrum + 1e-10)
    cepstrum = np.fft.irfft(log_spec)
    # Lifter: keep only first n_ceps cepstral coefficients
    liftered = np.zeros_like(cepstrum)
    liftered[:n_ceps] = cepstrum[:n_ceps]
    smoothed = np.exp(np.fft.rfft(liftered).real)
    return smoothed


def group_delay_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = _DEFAULT_N_FFT,
    hop_length: int = _DEFAULT_HOP_LENGTH,
    win_length: int = _DEFAULT_WIN_LENGTH,
    alpha: float = 0.4,
    gamma: float = 0.9,
    n_ceps_smooth: int = 30,
) -> tuple[np.ndarray, dict]:
    """
    Compute Modified Group Delay Function (MODGDF) features per Hegde et al.

    The group delay is the negative derivative of the phase spectrum w.r.t.
    frequency. The *modified* version normalizes by the cepstrally-smoothed
    magnitude spectrum to suppress instabilities near spectral zeros.

    MODGDF(ω) = sign(τ(ω)) × (|τ(ω)| / |S_cep(ω)|^{2γ})^α

    Args:
        audio: 1-D float32 audio array, normalized to [-1, 1].
        sr: Sample rate in Hz.
        n_fft: FFT size.
        hop_length: Hop length in samples.
        win_length: Analysis window length in samples.
        alpha: Compression exponent (controls dynamic range).
        gamma: Smoothing exponent (controls spectral-zero suppression).
        n_ceps_smooth: Number of cepstral coefficients for smoothing.

    Returns:
        features: 20-dim numpy vector (mean and std of 10 sub-band group delay statistics).
        summary: Human-readable dict with key metrics.
    """
    # Guard: very short or silent audio
    if len(audio) < n_fft:
        logger.warning("group_delay.audio_too_short", length=len(audio), n_fft=n_fft)
        return np.zeros(20, dtype=np.float32), {
            "group_delay_deviation": 0.0,
            "phase_coherence": 0.0,
            "status": "audio_too_short",
        }

    # --- Step 1: Compute STFT of x[n] and n*x[n] ---
    n_samples = len(audio)
    ramp = np.arange(n_samples, dtype=np.float64)
    audio_ramped = audio.astype(np.float64) * ramp

    X = _stft(audio.astype(np.float64), n_fft, hop_length, win_length)    # DFT of x[n]
    X_r = _stft(audio_ramped, n_fft, hop_length, win_length)               # DFT of n·x[n]

    n_frames, n_bins = X.shape

    # --- Step 2: Raw group delay per frame ---
    mag_sq = np.abs(X) ** 2 + 1e-10  # avoid division by zero
    tau_raw = (X_r.real * X.real + X_r.imag * X.imag) / mag_sq

    # --- Step 3: Cepstrally-smoothed magnitude spectrum ---
    mag = np.abs(X) + 1e-10
    S_cep = np.zeros_like(mag)
    for i in range(n_frames):
        S_cep[i] = _cepstral_smoothing(mag[i], n_ceps=n_ceps_smooth)

    # --- Step 4: Modified Group Delay Function ---
    denominator = S_cep ** (2 * gamma) + 1e-10
    modgdf = np.sign(tau_raw) * (np.abs(tau_raw) / denominator) ** alpha

    # --- Step 5: Extract statistical features from 10 sub-bands ---
    n_sub_bands = 10
    band_size = n_bins // n_sub_bands
    features = []

    for b in range(n_sub_bands):
        start_bin = b * band_size
        end_bin = start_bin + band_size if b < n_sub_bands - 1 else n_bins
        band_data = modgdf[:, start_bin:end_bin]

        # Mean and std across all frames and bins in this sub-band
        features.append(np.mean(band_data))
        features.append(np.std(band_data))

    feature_vector = np.array(features, dtype=np.float32)

    # --- Summary for risk explanation panel ---
    overall_deviation = float(np.std(modgdf))
    # Phase coherence: lower deviation → more coherent → more likely synthetic
    # Normalize to [0, 1] with a sigmoid-like mapping
    phase_coherence = float(1.0 / (1.0 + overall_deviation))

    summary = {
        "group_delay_deviation": round(overall_deviation, 4),
        "phase_coherence": round(phase_coherence, 4),
        "n_frames_analyzed": int(n_frames),
    }

    logger.debug(
        "group_delay.extracted",
        feature_dim=len(feature_vector),
        **summary,
    )

    return feature_vector, summary


# ---------------------------------------------------------------------------
# 2. Constant-Q Cepstral Coefficients (Step 50)
# ---------------------------------------------------------------------------

def _compute_deltas(coeffs: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Compute delta (first-order difference) features for a coefficient matrix.

    Args:
        coeffs: Matrix of shape (n_frames, n_coeffs).
        width: Number of frames on each side for regression.

    Returns:
        Delta matrix of the same shape.
    """
    n_frames, n_coeffs = coeffs.shape
    deltas = np.zeros_like(coeffs)
    denominator = 2 * sum(i * i for i in range(1, width + 1))

    for t in range(n_frames):
        delta = np.zeros(n_coeffs)
        for i in range(1, width + 1):
            t_prev = max(0, t - i)
            t_next = min(n_frames - 1, t + i)
            delta += i * (coeffs[t_next] - coeffs[t_prev])
        deltas[t] = delta / (denominator + 1e-10)

    return deltas


def cqcc_features(
    audio: np.ndarray,
    sr: int,
    n_cqcc: int = 20,
    n_bins_per_octave: int = 12,
    n_octaves: int = 4,
    hop_length: int = _DEFAULT_HOP_LENGTH,
) -> tuple[np.ndarray, dict]:
    """
    Compute Constant-Q Cepstral Coefficients (CQCC) with deltas.

    CQCCs are effective for anti-spoofing because the constant-Q transform
    provides better frequency resolution at low frequencies (where vocal
    tract resonances live), making it sensitive to vocoder artifacts.

    Pipeline:
        1. Constant-Q Transform (CQT) via librosa
        2. Log-power spectrum
        3. DCT → cepstral coefficients (first n_cqcc)
        4. Append delta and double-delta coefficients

    Args:
        audio: 1-D float32 audio array, normalized to [-1, 1].
        sr: Sample rate in Hz.
        n_cqcc: Number of cepstral coefficients to keep.
        n_bins_per_octave: Frequency bins per octave in CQT.
        n_octaves: Number of octaves to span.
        hop_length: Hop length for CQT.

    Returns:
        features: 60-dim numpy vector (20 CQCCs + 20 deltas + 20 double-deltas,
                  each averaged across frames).
        summary: Human-readable dict with key metrics.
    """
    import librosa

    n_bins = n_bins_per_octave * n_octaves  # total CQT frequency bins

    # Guard: very short or silent audio
    min_length = int(sr * 0.1)  # need at least 100ms
    if len(audio) < min_length:
        logger.warning("cqcc.audio_too_short", length=len(audio))
        return np.zeros(n_cqcc * 3, dtype=np.float32), {
            "cqcc_mean_energy": 0.0,
            "cqcc_spectral_variability": 0.0,
            "status": "audio_too_short",
        }

    # --- Step 1: Constant-Q Transform ---
    fmin = librosa.note_to_hz("C2")  # ~65 Hz — covers male F0
    cqt_matrix = librosa.cqt(
        y=audio.astype(np.float64),
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=n_bins_per_octave,
    )
    # cqt_matrix shape: (n_bins, n_frames), complex

    # --- Step 2: Log-power spectrum ---
    power_spectrum = np.abs(cqt_matrix) ** 2
    log_power = np.log(power_spectrum + 1e-10)  # (n_bins, n_frames)

    # --- Step 3: DCT to get cepstral coefficients ---
    # Transpose to (n_frames, n_bins) for DCT along frequency axis
    log_power_T = log_power.T  # (n_frames, n_bins)
    n_frames = log_power_T.shape[0]

    # Apply Type-II DCT along frequency axis, keep first n_cqcc coefficients
    cqcc_matrix = dct(log_power_T, type=2, axis=1, norm="ortho")[:, :n_cqcc]
    # cqcc_matrix shape: (n_frames, n_cqcc)

    # --- Step 4: Delta and double-delta ---
    delta_cqcc = _compute_deltas(cqcc_matrix)
    double_delta_cqcc = _compute_deltas(delta_cqcc)

    # --- Step 5: Aggregate across frames (mean) → fixed-length vector ---
    cqcc_mean = np.mean(cqcc_matrix, axis=0)       # (n_cqcc,)
    delta_mean = np.mean(delta_cqcc, axis=0)        # (n_cqcc,)
    dd_mean = np.mean(double_delta_cqcc, axis=0)    # (n_cqcc,)

    feature_vector = np.concatenate([cqcc_mean, delta_mean, dd_mean]).astype(np.float32)

    # --- Summary ---
    mean_energy = float(np.mean(np.abs(cqcc_mean)))
    spectral_variability = float(np.std(cqcc_matrix))

    summary = {
        "cqcc_mean_energy": round(mean_energy, 4),
        "cqcc_spectral_variability": round(spectral_variability, 4),
        "n_frames_analyzed": int(n_frames),
    }

    logger.debug(
        "cqcc.extracted",
        feature_dim=len(feature_vector),
        **summary,
    )

    return feature_vector, summary


# ---------------------------------------------------------------------------
# 3. Pitch Jitter & Micro-Prosody (Step 53)
# ---------------------------------------------------------------------------

def pitch_jitter(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 10.0,
    voicing_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Extract pitch (F0) jitter, shimmer, and micro-prosody features.

    Uses CREPE (via torchcrepe) for robust F0 estimation, then computes
    perturbation measures that distinguish natural speech (higher jitter)
    from synthetic speech (near-zero jitter).

    Features (8-dim):
        0. F0 mean (Hz)
        1. F0 std (Hz)
        2. F0 range (max - min, Hz)
        3. F0 slope (linear regression slope over time)
        4. Jitter absolute (mean |F0[i] - F0[i-1]|, Hz)
        5. Jitter relative (jitter_abs / F0_mean, dimensionless)
        6. Shimmer (mean |A[i] - A[i-1]| / mean(A), dimensionless)
        7. Voiced fraction (proportion of voiced frames)

    Args:
        audio: 1-D float32 array, normalized to [-1, 1].
        sr: Sample rate in Hz.
        hop_ms: F0 estimation hop size in milliseconds.
        voicing_threshold: CREPE confidence threshold for voiced/unvoiced.

    Returns:
        features: 8-dim numpy vector.
        summary: Human-readable dict with key metrics.
    """
    import torch
    import torchcrepe

    n_features = 8

    # Guard: very short audio
    min_length = int(sr * 0.05)  # need at least 50ms
    if len(audio) < min_length:
        logger.warning("pitch_jitter.audio_too_short", length=len(audio))
        return np.zeros(n_features, dtype=np.float32), {
            "f0_mean_hz": 0.0,
            "jitter_relative": 0.0,
            "shimmer": 0.0,
            "voiced_fraction": 0.0,
            "status": "audio_too_short",
        }

    # --- Step 1: Prepare audio tensor for torchcrepe ---
    # torchcrepe expects (batch, samples) tensor at the original sample rate
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    # Determine hop_length in samples
    hop_length = int(sr * hop_ms / 1000.0)

    # --- Step 2: Run CREPE pitch estimation ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # torchcrepe.predict returns (pitch, periodicity) tensors
        pitch, periodicity = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length=hop_length,
            fmin=50.0,     # Hz — below typical male F0
            fmax=550.0,    # Hz — above typical female F0
            model="tiny",  # fastest model, good enough for jitter
            batch_size=512,
            device=device,
            return_periodicity=True,
        )

        f0 = pitch.squeeze().cpu().numpy()             # (n_frames,)
        confidence = periodicity.squeeze().cpu().numpy()  # (n_frames,)

    except Exception as e:
        logger.error("pitch_jitter.crepe_failed", error=str(e))
        return np.zeros(n_features, dtype=np.float32), {
            "f0_mean_hz": 0.0,
            "jitter_relative": 0.0,
            "shimmer": 0.0,
            "voiced_fraction": 0.0,
            "status": "crepe_error",
        }

    # --- Step 3: Identify voiced frames ---
    voiced_mask = confidence >= voicing_threshold
    voiced_fraction = float(np.mean(voiced_mask))

    # Handle edge case: no voiced frames detected
    if voiced_fraction < 0.05 or np.sum(voiced_mask) < 3:
        logger.debug("pitch_jitter.mostly_unvoiced", voiced_fraction=voiced_fraction)
        features = np.zeros(n_features, dtype=np.float32)
        features[7] = voiced_fraction
        return features, {
            "f0_mean_hz": 0.0,
            "jitter_relative": 0.0,
            "shimmer": 0.0,
            "voiced_fraction": round(voiced_fraction, 4),
            "status": "mostly_unvoiced",
        }

    # Extract voiced F0 values
    f0_voiced = f0[voiced_mask]

    # --- Step 4: F0 statistics ---
    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_range = float(np.max(f0_voiced) - np.min(f0_voiced))

    # F0 slope via linear regression over voiced frames
    voiced_indices = np.where(voiced_mask)[0].astype(np.float64)
    if len(voiced_indices) >= 2:
        coeffs = np.polyfit(voiced_indices, f0_voiced, 1)
        f0_slope = float(coeffs[0])  # Hz per frame
    else:
        f0_slope = 0.0

    # --- Step 5: Jitter (pitch perturbation) ---
    f0_diffs = np.abs(np.diff(f0_voiced))
    jitter_abs = float(np.mean(f0_diffs)) if len(f0_diffs) > 0 else 0.0
    jitter_rel = jitter_abs / (f0_mean + 1e-10)

    # --- Step 6: Shimmer (amplitude perturbation) ---
    # Compute frame amplitudes from the original audio
    n_frames = len(f0)
    frame_amplitudes = np.zeros(n_frames)
    for i in range(n_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + hop_length, len(audio))
        if start_sample < len(audio):
            frame_amplitudes[i] = np.sqrt(np.mean(audio[start_sample:end_sample] ** 2))

    amp_voiced = frame_amplitudes[voiced_mask]
    if len(amp_voiced) >= 2:
        amp_diffs = np.abs(np.diff(amp_voiced))
        shimmer = float(np.mean(amp_diffs) / (np.mean(amp_voiced) + 1e-10))
    else:
        shimmer = 0.0

    # --- Step 7: Assemble feature vector ---
    feature_vector = np.array([
        f0_mean,
        f0_std,
        f0_range,
        f0_slope,
        jitter_abs,
        jitter_rel,
        shimmer,
        voiced_fraction,
    ], dtype=np.float32)

    summary = {
        "f0_mean_hz": round(f0_mean, 2),
        "f0_std_hz": round(f0_std, 2),
        "jitter_relative": round(jitter_rel, 6),
        "shimmer": round(shimmer, 6),
        "voiced_fraction": round(voiced_fraction, 4),
    }

    logger.debug(
        "pitch_jitter.extracted",
        feature_dim=len(feature_vector),
        **summary,
    )

    return feature_vector, summary


def fast_pitch_jitter(
    audio: np.ndarray,
    sr: int,
    hop_ms: float = 15.0,
    win_ms: float = 35.0,
    fmin: float = 50.0,
    fmax: float = 550.0,
) -> tuple[np.ndarray, dict]:
    """
    Fast autocorrelation-based pitch (F0) jitter, shimmer, and micro-prosody extraction.
    Computes exact same 8-dim feature representation as pitch_jitter() in <5ms on CPU.

    Features (8-dim):
        0. F0 mean (Hz)
        1. F0 std (Hz)
        2. F0 range (max - min, Hz)
        3. F0 slope (linear regression slope over time)
        4. Jitter absolute (mean |F0[i] - F0[i-1]|, Hz)
        5. Jitter relative (jitter_abs / F0_mean, dimensionless)
        6. Shimmer (mean |A[i] - A[i-1]| / mean(A), dimensionless)
        7. Voiced fraction (proportion of voiced frames)
    """
    n_features = 8
    min_length = int(sr * 0.05)
    if len(audio) < min_length:
        return np.zeros(n_features, dtype=np.float32), {
            "f0_mean_hz": 0.0,
            "jitter_relative": 0.0,
            "shimmer": 0.0,
            "voiced_fraction": 0.0,
            "status": "audio_too_short",
        }

    win_len = int(sr * win_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    min_lag = max(1, int(sr / fmax))
    max_lag = min(int(sr / fmin), win_len - 1)

    pitches = []
    amplitudes = []

    for start in range(0, len(audio) - win_len + 1, hop_len):
        frame = audio[start : start + win_len]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        amplitudes.append(rms)

        if rms < 1e-4:
            pitches.append(0.0)
            continue

        frame_d = frame - np.mean(frame)
        r = np.correlate(frame_d, frame_d, mode="full")
        r = r[len(frame_d) - 1 :]

        if r[0] < 1e-8 or min_lag >= len(r):
            pitches.append(0.0)
            continue

        search_end = min(max_lag, len(r))
        if search_end <= min_lag:
            pitches.append(0.0)
            continue

        peak_offset = int(np.argmax(r[min_lag:search_end]))
        peak_idx = min_lag + peak_offset
        norm_val = float(r[peak_idx] / (r[0] + 1e-10))

        if norm_val > 0.30 and peak_idx > 0:
            pitches.append(float(sr / peak_idx))
        else:
            pitches.append(0.0)

    pitches = np.array(pitches, dtype=np.float32)
    amplitudes = np.array(amplitudes, dtype=np.float32)
    voiced_mask = pitches > 0
    voiced_fraction = float(np.mean(voiced_mask)) if len(pitches) > 0 else 0.0

    if voiced_fraction < 0.05 or np.sum(voiced_mask) < 3:
        features = np.zeros(n_features, dtype=np.float32)
        features[7] = voiced_fraction
        return features, {
            "f0_mean_hz": 0.0,
            "jitter_relative": 0.0,
            "shimmer": 0.0,
            "voiced_fraction": round(voiced_fraction, 4),
            "status": "mostly_unvoiced",
        }

    f0_voiced = pitches[voiced_mask]
    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_range = float(np.max(f0_voiced) - np.min(f0_voiced))

    voiced_indices = np.where(voiced_mask)[0].astype(np.float64)
    if len(voiced_indices) >= 2:
        coeffs = np.polyfit(voiced_indices, f0_voiced, 1)
        f0_slope = float(coeffs[0])
    else:
        f0_slope = 0.0

    f0_diffs = np.abs(np.diff(f0_voiced))
    jitter_abs = float(np.mean(f0_diffs)) if len(f0_diffs) > 0 else 0.0
    jitter_rel = float(jitter_abs / (f0_mean + 1e-10))

    amp_voiced = amplitudes[voiced_mask]
    if len(amp_voiced) >= 2:
        amp_diffs = np.abs(np.diff(amp_voiced))
        shimmer = float(np.mean(amp_diffs) / (np.mean(amp_voiced) + 1e-10))
    else:
        shimmer = 0.0

    feature_vector = np.array([
        f0_mean,
        f0_std,
        f0_range,
        f0_slope,
        jitter_abs,
        jitter_rel,
        shimmer,
        voiced_fraction,
    ], dtype=np.float32)

    summary = {
        "f0_mean_hz": round(f0_mean, 2),
        "f0_std_hz": round(f0_std, 2),
        "jitter_relative": round(jitter_rel, 6),
        "shimmer": round(shimmer, 6),
        "voiced_fraction": round(voiced_fraction, 4),
    }

    return feature_vector, summary



# ---------------------------------------------------------------------------
# 4. Pause/Rhythm Statistics (Step 56)
# ---------------------------------------------------------------------------

# Global cache for Silero VAD model (loaded once, reused across calls)
_silero_vad_model = None
_silero_vad_utils = None


def _get_silero_vad():
    """Load and cache the Silero VAD model globally."""
    global _silero_vad_model, _silero_vad_utils

    if _silero_vad_model is None:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _silero_vad_model = model
        _silero_vad_utils = utils
        logger.info("silero_vad.loaded")

    return _silero_vad_model, _silero_vad_utils


def pause_rhythm_stats(
    audio: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    speech_threshold: float = 0.5,
    min_speech_duration_ms: int = 100,
    min_silence_duration_ms: int = 100,
) -> tuple[np.ndarray, dict]:
    """
    Compute pause and rhythm statistics using Silero VAD.

    Segments the audio into speech and silence regions, then extracts
    temporal features that differ between natural speech (varied pauses,
    irregular rhythm) and synthetic speech (uniform timing).

    Features (8-dim):
        0. Number of pauses
        1. Mean pause duration (seconds)
        2. Max pause duration (seconds)
        3. Pause duration variance (seconds²)
        4. Speech rate (proportion of voiced frames)
        5. Mean speech segment duration (seconds)
        6. Speech-to-silence ratio
        7. Rhythm regularity (std of speech segment durations — lower = more regular)

    Args:
        audio: 1-D float32 array, normalized to [-1, 1].
        sr: Sample rate in Hz.
        target_sr: Silero VAD requires 16 kHz; audio will be resampled if needed.
        speech_threshold: VAD probability threshold for speech detection.
        min_speech_duration_ms: Minimum speech segment length to keep.
        min_silence_duration_ms: Minimum silence segment length to count as a pause.

    Returns:
        features: 8-dim numpy vector.
        summary: Human-readable dict with key metrics.
    """
    import torch

    n_features = 8

    # Guard: very short audio
    min_length = int(sr * 0.2)  # need at least 200ms
    if len(audio) < min_length:
        logger.warning("pause_rhythm.audio_too_short", length=len(audio))
        return np.zeros(n_features, dtype=np.float32), {
            "num_pauses": 0,
            "mean_pause_sec": 0.0,
            "speech_rate": 0.0,
            "rhythm_regularity": 0.0,
            "status": "audio_too_short",
        }

    # --- Step 1: Resample to 16 kHz if needed (Silero VAD requirement) ---
    if sr != target_sr:
        import librosa
        audio_16k = librosa.resample(
            audio.astype(np.float64), orig_sr=sr, target_sr=target_sr
        ).astype(np.float32)
    else:
        audio_16k = audio.copy()

    total_duration_sec = len(audio_16k) / target_sr

    # --- Step 2: Run Silero VAD ---
    model, utils = _get_silero_vad()
    get_speech_timestamps = utils[0]

    audio_tensor = torch.tensor(audio_16k, dtype=torch.float32)

    try:
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=target_sr,
            threshold=speech_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )
    except Exception as e:
        logger.error("pause_rhythm.vad_failed", error=str(e))
        return np.zeros(n_features, dtype=np.float32), {
            "num_pauses": 0,
            "mean_pause_sec": 0.0,
            "speech_rate": 0.0,
            "rhythm_regularity": 0.0,
            "status": "vad_error",
        }

    # --- Step 3: Compute speech and pause segments ---
    total_samples = len(audio_16k)

    # Speech segments (in seconds)
    speech_durations = []
    for seg in speech_timestamps:
        start_sec = seg["start"] / target_sr
        end_sec = seg["end"] / target_sr
        speech_durations.append(end_sec - start_sec)

    # Pause segments (gaps between speech segments)
    pause_durations = []
    if len(speech_timestamps) == 0:
        # Entire audio is silence
        pause_durations.append(total_duration_sec)
    else:
        # Pause before first speech
        first_start = speech_timestamps[0]["start"] / target_sr
        if first_start > 0.01:
            pause_durations.append(first_start)

        # Pauses between speech segments
        for i in range(1, len(speech_timestamps)):
            prev_end = speech_timestamps[i - 1]["end"] / target_sr
            curr_start = speech_timestamps[i]["start"] / target_sr
            gap = curr_start - prev_end
            if gap > 0.01:
                pause_durations.append(gap)

        # Pause after last speech
        last_end = speech_timestamps[-1]["end"] / target_sr
        trailing = total_duration_sec - last_end
        if trailing > 0.01:
            pause_durations.append(trailing)

    # --- Step 4: Extract statistics ---
    num_pauses = len(pause_durations)
    total_speech_sec = sum(speech_durations) if speech_durations else 0.0
    speech_rate = total_speech_sec / (total_duration_sec + 1e-10)

    if num_pauses > 0:
        mean_pause = float(np.mean(pause_durations))
        max_pause = float(np.max(pause_durations))
        pause_variance = float(np.var(pause_durations))
    else:
        mean_pause = 0.0
        max_pause = 0.0
        pause_variance = 0.0

    if len(speech_durations) > 0:
        mean_speech_dur = float(np.mean(speech_durations))
        speech_silence_ratio = total_speech_sec / (
            total_duration_sec - total_speech_sec + 1e-10
        )
        # Rhythm regularity: lower std = more regular (synthetic)
        rhythm_regularity = float(np.std(speech_durations)) if len(speech_durations) > 1 else 0.0
    else:
        mean_speech_dur = 0.0
        speech_silence_ratio = 0.0
        rhythm_regularity = 0.0

    # --- Step 5: Assemble feature vector ---
    feature_vector = np.array([
        float(num_pauses),
        mean_pause,
        max_pause,
        pause_variance,
        speech_rate,
        mean_speech_dur,
        speech_silence_ratio,
        rhythm_regularity,
    ], dtype=np.float32)

    summary = {
        "num_pauses": num_pauses,
        "mean_pause_sec": round(mean_pause, 4),
        "max_pause_sec": round(max_pause, 4),
        "speech_rate": round(speech_rate, 4),
        "mean_speech_segment_sec": round(mean_speech_dur, 4),
        "rhythm_regularity": round(rhythm_regularity, 4),
    }

    logger.debug(
        "pause_rhythm.extracted",
        feature_dim=len(feature_vector),
        **summary,
    )

    return feature_vector, summary


# ---------------------------------------------------------------------------
# 5. Unified Feature Pipeline (Step 59)
# ---------------------------------------------------------------------------

# Feature dimensions for reference:
#   Group Delay:    20-dim
#   CQCC:           60-dim (20 + 20Δ + 20ΔΔ)
#   Pitch Jitter:    8-dim
#   Pause/Rhythm:    8-dim
#   TOTAL:          96-dim
TOTAL_FEATURE_DIM = 96


class FeatureNormalizer:
    """
    Z-score normalizer for DSP feature vectors.

    Applies: normalized = (features - mean) / (std + eps)

    Ships with identity defaults (mean=0, std=1) so features pass through
    unchanged until real statistics are computed from training data (Phase 4).
    Call `load_stats()` or `set_stats()` once training stats are available.
    """

    def __init__(self, feature_dim: int = TOTAL_FEATURE_DIM, eps: float = 1e-8):
        self.feature_dim = feature_dim
        self.eps = eps
        # Identity defaults — no-op until real stats are loaded
        self._mean = np.zeros(feature_dim, dtype=np.float32)
        self._std = np.ones(feature_dim, dtype=np.float32)
        self._is_fitted = False

    def set_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Set normalization statistics directly."""
        assert mean.shape == (self.feature_dim,), f"Mean shape mismatch: {mean.shape}"
        assert std.shape == (self.feature_dim,), f"Std shape mismatch: {std.shape}"
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)
        self._is_fitted = True
        logger.info("feature_normalizer.stats_set", feature_dim=self.feature_dim)

    def load_stats(self, path: str) -> None:
        """Load precomputed mean/std from a .npz file."""
        data = np.load(path)
        self.set_stats(data["mean"], data["std"])
        logger.info("feature_normalizer.stats_loaded", path=path)

    def save_stats(self, path: str) -> None:
        """Save current mean/std to a .npz file."""
        np.savez(path, mean=self._mean, std=self._std)
        logger.info("feature_normalizer.stats_saved", path=path)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to a feature vector."""
        return ((features - self._mean) / (self._std + self.eps)).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        """Whether real training stats have been loaded."""
        return self._is_fitted


# Global normalizer instance (loaded at startup, shared across calls)
_feature_normalizer = FeatureNormalizer()


def get_feature_normalizer() -> FeatureNormalizer:
    """Get the global feature normalizer instance."""
    return _feature_normalizer


def extract_all_features(
    audio: np.ndarray,
    sr: int,
    fast_mode: bool = True,
) -> dict:
    """
    Run the full DSP feature extraction pipeline on an audio chunk.

    Calls all four feature families, concatenates their vectors into a
    single fixed-length feature vector, and collects per-family summaries
    with timing information.

    Args:
        audio: 1-D float32 array, normalized to [-1, 1].
        sr: Sample rate in Hz.
        fast_mode: When True (default), uses fast autocorrelation pitch tracking (<5ms)
                   instead of CREPE on CPU (~4000ms).

    Returns:
        dict with keys:
            - "features": np.ndarray of shape (96,), dtype float32
            - "summaries": dict of per-family summaries
            - "timing_ms": dict of per-family extraction times in milliseconds
            - "total_time_ms": float, total extraction time
    """
    import time

    summaries = {}
    timing_ms = {}
    feature_parts = []

    # --- 1. Group Delay Features (20-dim) ---
    t0 = time.perf_counter()
    try:
        gd_features, gd_summary = group_delay_features(audio, sr)
        summaries["group_delay"] = gd_summary
    except Exception as e:
        logger.error("extract_all.group_delay_failed", error=str(e))
        gd_features = np.zeros(20, dtype=np.float32)
        summaries["group_delay"] = {"status": "error", "error": str(e)}
    timing_ms["group_delay"] = (time.perf_counter() - t0) * 1000
    feature_parts.append(gd_features)

    # --- 2. CQCC Features (60-dim) ---
    t0 = time.perf_counter()
    try:
        cqcc_feat, cqcc_summary = cqcc_features(audio, sr)
        summaries["cqcc"] = cqcc_summary
    except Exception as e:
        logger.error("extract_all.cqcc_failed", error=str(e))
        cqcc_feat = np.zeros(60, dtype=np.float32)
        summaries["cqcc"] = {"status": "error", "error": str(e)}
    timing_ms["cqcc"] = (time.perf_counter() - t0) * 1000
    feature_parts.append(cqcc_feat)

    # --- 3. Pitch Jitter (8-dim) ---
    t0 = time.perf_counter()
    try:
        if fast_mode:
            pj_features, pj_summary = fast_pitch_jitter(audio, sr)
        else:
            pj_features, pj_summary = pitch_jitter(audio, sr)
        summaries["pitch_jitter"] = pj_summary
    except Exception as e:
        logger.error("extract_all.pitch_jitter_failed", error=str(e))
        pj_features = np.zeros(8, dtype=np.float32)
        summaries["pitch_jitter"] = {"status": "error", "error": str(e)}
    timing_ms["pitch_jitter"] = (time.perf_counter() - t0) * 1000
    feature_parts.append(pj_features)

    # --- 4. Pause/Rhythm Stats (8-dim) ---
    t0 = time.perf_counter()
    try:
        pr_features, pr_summary = pause_rhythm_stats(audio, sr)
        summaries["pause_rhythm"] = pr_summary
    except Exception as e:
        logger.error("extract_all.pause_rhythm_failed", error=str(e))
        pr_features = np.zeros(8, dtype=np.float32)
        summaries["pause_rhythm"] = {"status": "error", "error": str(e)}
    timing_ms["pause_rhythm"] = (time.perf_counter() - t0) * 1000
    feature_parts.append(pr_features)

    # --- Concatenate all features ---
    feature_vector = np.concatenate(feature_parts).astype(np.float32)
    total_time = sum(timing_ms.values())

    assert feature_vector.shape == (TOTAL_FEATURE_DIM,), (
        f"Feature vector shape mismatch: expected ({TOTAL_FEATURE_DIM},), "
        f"got {feature_vector.shape}"
    )

    logger.info(
        "extract_all.complete",
        feature_dim=len(feature_vector),
        total_time_ms=round(total_time, 1),
        group_delay_ms=round(timing_ms["group_delay"], 1),
        cqcc_ms=round(timing_ms["cqcc"], 1),
        pitch_jitter_ms=round(timing_ms["pitch_jitter"], 1),
        pause_rhythm_ms=round(timing_ms["pause_rhythm"], 1),
    )

    # --- Z-score normalization ---
    normalized_vector = _feature_normalizer.normalize(feature_vector)

    return {
        "features": feature_vector,
        "features_normalized": normalized_vector,
        "summaries": summaries,
        "timing_ms": timing_ms,
        "total_time_ms": round(total_time, 2),
        "normalizer_fitted": _feature_normalizer.is_fitted,
    }
