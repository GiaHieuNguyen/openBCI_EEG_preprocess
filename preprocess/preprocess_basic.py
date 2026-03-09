import argparse
import os
import sys
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

# Add project root to path for shared imports
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from common import apply_brainflow_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Read EEG stream from OpenBCI Cyton via BrainFlow.")
    parser.add_argument(
        "--serial-port",
        default="/dev/ttyUSB0",
        help="Serial port for Cyton on Linux (default: /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--board-id",
        type=int,
        default=BoardIds.CYTON_DAISY_BOARD.value,
        help=(
            f"BrainFlow board id (default: {BoardIds.CYTON_DAISY_BOARD.value} "
            "for CYTON_DAISY_BOARD / 16 channels)."
        ),
    )
    parser.add_argument("--l-freq", type=float, default=0.5, help="Bandpass low cutoff in Hz.")
    parser.add_argument("--h-freq", type=float, default=40.0, help="Bandpass high cutoff in Hz.")
    parser.add_argument("--bandpass-order", type=int, default=4, help="Bandpass filter order.")
    parser.add_argument(
        "--notch",
        type=float,
        default=50.0,
        help="Mains notch frequency in Hz: 50, 60, or 0 to disable.",
    )
    return parser.parse_args()


def apply_preprocessing(eeg_data: np.ndarray, sampling_rate: int, args) -> np.ndarray:
    """Convenience wrapper that unpacks argparse args into the shared function."""
    return apply_brainflow_preprocessing(
        eeg_data, sampling_rate,
        l_freq=args.l_freq, h_freq=args.h_freq,
        bandpass_order=args.bandpass_order, notch=args.notch,
    )


# ── MNE-based preprocessing (used by preprocess.py) ────────────────────


def preprocess_raw(raw, notch=60.0, l_freq=0.5, h_freq=70.0, resample=256.0):
    """
    Apply standard MNE preprocessing pipeline to an MNE Raw object.

    Steps: notch filter → bandpass filter → resample.
    """
    import mne

    if notch:
        raw.notch_filter(freqs=notch, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if resample and resample != raw.info["sfreq"]:
        raw.resample(sfreq=resample, verbose=False)
    return raw


def sliding_windows(data, sfreq, win_sec=2.0, step_sec=1.0):
    """
    Split multi-channel data into overlapping windows.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_samples)
    sfreq : float
    win_sec : float, window size in seconds
    step_sec : float, step size in seconds

    Returns
    -------
    windows : ndarray, shape (n_windows, n_channels, win_samples)
    starts : ndarray, start sample index for each window
    """
    win_samples = int(sfreq * win_sec)
    step_samples = int(sfreq * step_sec)
    n_samples = data.shape[1]

    starts = np.arange(0, n_samples - win_samples + 1, step_samples)
    windows = np.array([data[:, s : s + win_samples] for s in starts])
    return windows, starts


def label_windows(starts, sfreq, win_sec=2.0, seizure_intervals_sec=None):
    """
    Label windows as seizure (1) or non-seizure (0).

    A window is labelled 1 if more than 50% of its duration overlaps
    with any seizure interval.

    Parameters
    ----------
    starts : ndarray, start sample index for each window
    sfreq : float
    win_sec : float
    seizure_intervals_sec : list of (start_sec, end_sec) tuples

    Returns
    -------
    labels : ndarray of int, shape (n_windows,)
    """
    if seizure_intervals_sec is None:
        seizure_intervals_sec = []

    win_samples = int(sfreq * win_sec)
    labels = np.zeros(len(starts), dtype=int)

    for i, s in enumerate(starts):
        win_start_sec = s / sfreq
        win_end_sec = (s + win_samples) / sfreq
        win_dur = win_end_sec - win_start_sec

        overlap = 0.0
        for sz_start, sz_end in seizure_intervals_sec:
            ov_start = max(win_start_sec, sz_start)
            ov_end = min(win_end_sec, sz_end)
            if ov_end > ov_start:
                overlap += ov_end - ov_start

        if overlap / win_dur > 0.5:
            labels[i] = 1

    return labels


def main():
    args = parse_args()
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    board_id = args.board_id

    if not os.path.exists(params.serial_port):
        raise FileNotFoundError(
            f"Serial port '{params.serial_port}' was not found. "
            "Check `ls /dev/ttyUSB*` and udev permissions."
        )

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    print(
        f"Streaming started from {params.serial_port} (board_id={board_id}). "
        "Press Ctrl+C to stop."
    )
    print(
        f"Preprocessing: bandpass {args.l_freq}-{args.h_freq} Hz "
        f"(order={args.bandpass_order}), notch={args.notch} Hz, fs={sampling_rate} Hz."
    )
    if len(eeg_channels) != 16:
        print(f"Warning: expected 16 EEG channels for Daisy, detected {len(eeg_channels)}.")

    try:
        while True:
            time.sleep(1)
            data = board.get_board_data()

            num_samples = data.shape[1]
            if num_samples > 0:
                eeg_data = data[eeg_channels]
                eeg_filtered = apply_preprocessing(eeg_data, sampling_rate, args)
                latest_sample = eeg_filtered[:, -1]

                print(
                    f"[{num_samples} new samples] Latest preprocessed EEG "
                    f"({len(eeg_channels)} channels):"
                )
                print(latest_sample)
                print("-" * 50)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        board.stop_stream()
        board.release_session()
        print("Done.")


if __name__ == "__main__":
    main()
