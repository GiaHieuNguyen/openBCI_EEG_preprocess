from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, firwin, freqz, welch


def load_eeg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Sample Index and EXG Channel 0 from OpenBCI export text."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1))
    sample_index = data[:, 0]
    signal = data[:, 1]
    mask = np.isfinite(sample_index) & np.isfinite(signal)
    return signal[mask], sample_index[mask]


def unwrap_sample_index(sample_index: np.ndarray) -> np.ndarray:
    """Unwrap cyclic sample counter (e.g., 0..255 rollover) into monotonic index."""
    idx = np.rint(sample_index).astype(int)
    modulus = int(idx.max() + 1)
    if modulus <= 1:
        return idx.astype(float)

    unwrapped = np.empty_like(idx, dtype=float)
    offset = 0
    unwrapped[0] = float(idx[0])
    threshold = modulus // 2

    for i in range(1, len(idx)):
        if idx[i] < idx[i - 1] - threshold:
            offset += modulus
        unwrapped[i] = float(idx[i] + offset)
    return unwrapped


def maybe_show() -> None:
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "raw_data" / "s01_ex01_s01.txt"
    if not data_path.exists():
        raise FileNotFoundError(f"EEG file not found: {data_path}")

    raw_signal, sample_index = load_eeg(data_path)
    sample_index_unwrapped = unwrap_sample_index(sample_index)
    fs_design = 200.0  # OpenBCI target sampling rate for filter design.

    lowcut = 0.5
    highcut = 40.0
    numtaps = 401
    windows = ["boxcar", "hamming", "hann", "blackman"]

    print(f"Loaded: {data_path.name}")
    print(f"Samples: {len(raw_signal)}")
    print(f"Sample index raw range: {sample_index[0]:.0f} -> {sample_index[-1]:.0f}")
    print(
        "Sample index unwrapped range: "
        f"{sample_index_unwrapped[0]:.0f} -> {sample_index_unwrapped[-1]:.0f}"
    )
    print(f"Approx duration from samples: {len(raw_signal) / fs_design:.2f} s")
    print(f"Filter design fs: {fs_design:.1f} Hz")
    print(f"Raw mean/std: {raw_signal.mean():.3f} / {raw_signal.std():.3f} uV")

    responses = {}
    taps_by_window = {}
    filtered_by_window = {}
    metrics = {}

    for window in windows:
        taps = firwin(
            numtaps,
            [lowcut, highcut],
            pass_zero=False,
            window=window,
            fs=fs_design,
        )
        w, h = freqz(taps, worN=8192, fs=fs_design)
        filtered = filtfilt(taps, [1.0], raw_signal)
        responses[window] = (w, h)
        taps_by_window[window] = taps
        filtered_by_window[window] = filtered

        # Spectral metrics from filtered signal.
        f_psd, p_psd = welch(filtered, fs=fs_design, nperseg=2048)
        p50 = np.interp(50.0, f_psd, p_psd)
        p50_db = 10 * np.log10(max(p50, 1e-20))

        sig_mask = (f_psd >= lowcut) & (f_psd <= highcut)
        noise_mask = f_psd > highcut
        signal_power = np.trapezoid(p_psd[sig_mask], f_psd[sig_mask])
        noise_power = np.trapezoid(p_psd[noise_mask], f_psd[noise_mask])
        snr_db = 10 * np.log10(max(signal_power, 1e-20) / max(noise_power, 1e-20))

        metrics[window] = {
            "power_50hz_db": p50_db,
            "snr_db": snr_db,
        }

    # --- Plot 1: Magnitude response (dB) ---
    fig_mag = plt.figure(figsize=(10, 6))
    for window in windows:
        freq, h = responses[window]
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        plt.plot(freq, mag_db, linewidth=1.5, label=window)
    plt.title("FIR Bandpass Magnitude Response: Window Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, fs_design / 2)
    plt.ylim(-140, 5)
    plt.axvline(lowcut, color="gray", linestyle="--", linewidth=1)
    plt.axvline(highcut, color="gray", linestyle="--", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Window")
    plt.tight_layout()

    # --- Plot 2: Impulse response (coefficients) ---
    n = np.arange(numtaps)
    fig_imp = plt.figure(figsize=(10, 6))
    for window in windows:
        plt.plot(n, taps_by_window[window], linewidth=1.3, label=window)
    plt.title("FIR Impulse Response: Window Comparison")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Window")
    plt.tight_layout()

    # --- Plot 3: Raw vs filtered time segment ---
    seg_seconds = 10
    seg_n = min(int(seg_seconds * fs_design), len(raw_signal))
    t = sample_index_unwrapped[:seg_n] / fs_design
    fig_time = plt.figure(figsize=(12, 6))
    plt.plot(t, raw_signal[:seg_n], color="0.55", linewidth=1.0, label="Raw (EXG ch0)")
    for window in windows:
        plt.plot(t, filtered_by_window[window][:seg_n], linewidth=1.2, label=f"{window} filtered")
    plt.title(f"Raw vs FIR-Filtered EEG (First {seg_n / fs_design:.1f} s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    # --- Plot 5: Quantitative metric comparison ---
    metric_windows = windows
    power_50_list = [metrics[w]["power_50hz_db"] for w in metric_windows]
    snr_list = [metrics[w]["snr_db"] for w in metric_windows]
    x = np.arange(len(metric_windows))
    width = 0.38

    fig_metrics = plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, power_50_list, width=width, label="50 Hz power (dB)")
    plt.bar(x + width / 2, snr_list, width=width, label="SNR (dB)")
    plt.xticks(x, metric_windows)
    plt.ylabel("dB")
    plt.title("FIR Window Metrics: 50 Hz Stop-Band Power and SNR")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # --- Plot 4: PSD comparison ---
    fig_psd = plt.figure(figsize=(12, 6))
    f_raw, p_raw = welch(raw_signal, fs=fs_design, nperseg=1024)
    plt.semilogy(f_raw, p_raw, color="0.55", linewidth=1.3, label="Raw (EXG ch0)")
    for window in windows:
        f, p = welch(filtered_by_window[window], fs=fs_design, nperseg=1024)
        plt.semilogy(f, p, linewidth=1.3, label=f"{window} filtered")
    plt.title("PSD Comparison: Raw vs FIR Windowed Filters")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.xlim(0, 60)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    out_files = {
        "fir_window_magnitude_response_real_eeg.png": fig_mag,
        "fir_window_impulse_response_real_eeg.png": fig_imp,
        "fir_window_time_comparison_real_eeg.png": fig_time,
        "fir_window_psd_comparison_real_eeg.png": fig_psd,
        "fir_window_metrics_real_eeg.png": fig_metrics,
    }
    for out_path, fig in out_files.items():
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

    print("\nWindow comparison metrics")
    print("window\t50Hz_power_dB\tSNR_dB")
    for window in windows:
        print(
            f"{window}\t"
            f"{metrics[window]['power_50hz_db']:.2f}\t\t"
            f"{metrics[window]['snr_db']:.2f}"
        )

    maybe_show()


if __name__ == "__main__":
    main()
