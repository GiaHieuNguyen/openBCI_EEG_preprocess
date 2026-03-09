"""
EEG Preprocessing Engine
========================
Signal processing functions for OpenBCI Cyton+Daisy EEG data.

Functions:
  - bandpass_filter: Butterworth bandpass
  - notch_filter: Remove powerline noise
  - resample_data: Change sampling rate
  - detect_artifacts: Mark bad segments
  - compute_psd: Welch power spectral density
  - robust_normalize: Per-channel robust scaling
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch, resample_poly
from math import gcd


def bandpass_filter(data: np.ndarray, sfreq: float,
                    l_freq: float = 0.5, h_freq: float = 45.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to each channel.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    sfreq : float, sampling frequency in Hz
    l_freq : float, low cutoff frequency
    h_freq : float, high cutoff frequency
    order : int, filter order

    Returns
    -------
    filtered : ndarray, same shape as data
    """
    nyq = sfreq / 2.0
    low = l_freq / nyq
    high = h_freq / nyq
    # Clamp to valid range
    low = max(low, 1e-5)
    high = min(high, 1.0 - 1e-5)

    b, a = butter(order, [low, high], btype="band")
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered


def notch_filter(data: np.ndarray, sfreq: float,
                 freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove powerline noise.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    sfreq : float, sampling frequency
    freq : float, frequency to notch out (50 or 60 Hz)
    quality : float, quality factor (higher = narrower notch)

    Returns
    -------
    filtered : ndarray, same shape as data
    """
    b, a = iirnotch(freq, quality, sfreq)
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = filtfilt(b, a, data[ch])
    return filtered


def resample_data(data: np.ndarray, orig_sfreq: float,
                  target_sfreq: float) -> np.ndarray:
    """
    Resample data to a new sampling rate using polyphase filtering.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    orig_sfreq : float, original sampling frequency
    target_sfreq : float, target sampling frequency

    Returns
    -------
    resampled : ndarray, shape (n_channels, new_n_samples)
    """
    if orig_sfreq == target_sfreq:
        return data.copy()

    up = int(target_sfreq)
    down = int(orig_sfreq)
    g = gcd(up, down)
    up //= g
    down //= g

    resampled = np.zeros((data.shape[0], int(data.shape[1] * up / down)), dtype=data.dtype)
    for ch in range(data.shape[0]):
        resampled[ch] = resample_poly(data[ch], up, down)
    return resampled


def detect_artifacts(data: np.ndarray, sfreq: float,
                     window_sec: float = 1.0,
                     max_abs_uv: float = 500.0,
                     flat_thresh: float = 0.5) -> dict:
    """
    Detect artifact segments in the EEG data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    sfreq : float, sampling frequency
    window_sec : float, window size in seconds for artifact checking
    max_abs_uv : float, maximum absolute amplitude threshold (uV)
    flat_thresh : float, minimum std deviation threshold (uV) for flatline detection

    Returns
    -------
    dict with:
        - 'bad_segments': list of (start_sample, end_sample, reason) tuples
        - 'n_bad': int, number of bad segments
        - 'n_total': int, total number of segments
        - 'bad_mask': ndarray, boolean mask of bad samples
    """
    window_samples = int(sfreq * window_sec)
    n_samples = data.shape[1]
    n_windows = max(0, (n_samples - window_samples) // window_samples + 1)

    bad_segments = []
    bad_mask = np.zeros(n_samples, dtype=bool)

    for wi in range(n_windows):
        start = wi * window_samples
        end = min(start + window_samples, n_samples)
        seg = data[:, start:end]

        # Check for NaN/Inf
        if np.any(np.isnan(seg)) or np.any(np.isinf(seg)):
            bad_segments.append((start, end, "nan_inf"))
            bad_mask[start:end] = True
            continue

        # Check amplitude threshold
        if np.max(np.abs(seg)) > max_abs_uv:
            bad_segments.append((start, end, "high_amplitude"))
            bad_mask[start:end] = True
            continue

        # Check for flatline (very low variance)
        channel_stds = np.std(seg, axis=1)
        if np.any(channel_stds < flat_thresh):
            bad_segments.append((start, end, "flatline"))
            bad_mask[start:end] = True
            continue

    return {
        "bad_segments": bad_segments,
        "n_bad": len(bad_segments),
        "n_total": n_windows,
        "bad_mask": bad_mask.tolist(),
        "pct_bad": (len(bad_segments) / max(n_windows, 1)) * 100,
    }


def compute_psd(data: np.ndarray, sfreq: float,
                n_fft: int = 256, n_overlap: int = 128) -> dict:
    """
    Compute power spectral density using Welch's method.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    sfreq : float, sampling frequency
    n_fft : int, FFT length
    n_overlap : int, overlap between segments

    Returns
    -------
    dict with 'freqs' and 'psd' (shape: n_channels x n_freqs)
    """
    n_channels = data.shape[0]
    freqs = None
    psd_all = []

    for ch in range(n_channels):
        f, pxx = welch(data[ch], fs=sfreq, nperseg=n_fft, noverlap=n_overlap)
        if freqs is None:
            freqs = f
        psd_all.append(pxx)

    return {
        "freqs": freqs.tolist(),
        "psd": np.array(psd_all).tolist(),
    }


def robust_normalize(data: np.ndarray, scale: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """
    Robust normalization with gain: X = scale * X / (q95(|X|) + eps) per channel.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    scale : float, output gain after normalization
    eps : float, small constant for numerical stability

    Returns
    -------
    normalized : ndarray, same shape as data
    """
    q95 = np.quantile(np.abs(data), q=0.95, axis=-1, keepdims=True)
    return scale * data / (q95 + eps)


def moving_average_smooth(data: np.ndarray, window_samples: int = 5) -> np.ndarray:
    """
    Smooth each channel with a centered moving average.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    window_samples : int, odd window length in samples

    Returns
    -------
    smoothed : ndarray, same shape as data
    """
    w = max(1, int(window_samples))
    if w == 1:
        return data.copy()
    if w % 2 == 0:
        w += 1

    kernel = np.ones(w, dtype=np.float64) / float(w)
    smoothed = np.zeros_like(data)
    for ch in range(data.shape[0]):
        smoothed[ch] = np.convolve(data[ch], kernel, mode="same")
    return smoothed


def compute_statistics(data: np.ndarray, ch_names: list) -> list:
    """
    Compute per-channel statistics.

    Returns list of dicts with: channel, mean, std, min, max, median, rms
    """
    stats = []
    for i, ch in enumerate(ch_names):
        ch_data = data[i]
        stats.append({
            "channel": ch,
            "mean": float(np.mean(ch_data)),
            "std": float(np.std(ch_data)),
            "min": float(np.min(ch_data)),
            "max": float(np.max(ch_data)),
            "median": float(np.median(ch_data)),
            "rms": float(np.sqrt(np.mean(ch_data ** 2))),
        })
    return stats
