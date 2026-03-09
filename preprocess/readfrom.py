import argparse
import os
import sys
import time
from pathlib import Path

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
    parser.add_argument(
        "--recording-dir",
        default="recording",
        help="Directory for output window_XX.hex files.",
    )
    parser.add_argument(
        "--hex-bit-width",
        type=int,
        default=4,
        help="Fixed-point bit width for hex output (e.g., 16 or 32).",
    )
    parser.add_argument(
        "--hex-fractional-bits",
        type=int,
        default=2,
        help="Number of fractional bits in fixed-point format.",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=1280,
        help="Number of sample lines per window file.",
    )
    parser.add_argument(
        "--window-prefix",
        default="window",
        help="Window file prefix. Output format: <prefix>_XX.hex",
    )
    parser.add_argument(
        "--window-start-index",
        type=int,
        default=1,
        help="First window index to try (auto-increments if file exists).",
    )
    return parser.parse_args()


def apply_preprocessing(eeg_data: np.ndarray, sampling_rate: int, args) -> np.ndarray:
    """Convenience wrapper that unpacks argparse args into the shared function."""
    return apply_brainflow_preprocessing(
        eeg_data, sampling_rate,
        l_freq=args.l_freq, h_freq=args.h_freq,
        bandpass_order=args.bandpass_order, notch=args.notch,
    )


def float_to_fixed_hex(value: float, bit_width: int, fractional_bits: int) -> str:
    """Convert one float value to two's-complement fixed-point hex with saturation."""
    scale_factor = 1 << fractional_bits
    max_val = (1 << (bit_width - 1)) - 1
    min_val = -(1 << (bit_width - 1))
    bit_mask = (1 << bit_width) - 1
    hex_chars = bit_width // 4

    int_val = int(round(value * scale_factor))
    if int_val > max_val:
        int_val = max_val
    elif int_val < min_val:
        int_val = min_val

    if int_val < 0:
        int_val = (1 << bit_width) + int_val

    return format(int_val & bit_mask, f"0{hex_chars}X")


def resolve_recording_dir(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def find_first_available_index(recording_dir: Path, prefix: str, start_index: int) -> int:
    idx = start_index
    while (recording_dir / f"{prefix}_{idx:02d}.hex").exists():
        idx += 1
    return idx


class HexWindowWriter:
    def __init__(
        self,
        recording_dir: Path,
        window_prefix: str,
        start_index: int,
        samples_per_window: int,
        bit_width: int,
        fractional_bits: int,
    ) -> None:
        self.recording_dir = recording_dir
        self.window_prefix = window_prefix
        self.samples_per_window = samples_per_window
        self.bit_width = bit_width
        self.fractional_bits = fractional_bits
        self.recording_dir.mkdir(parents=True, exist_ok=True)

        self.window_idx = find_first_available_index(recording_dir, window_prefix, start_index)
        self.lines_in_window = 0
        self.file = None
        self.current_path = None
        self._open_window()

    def _open_window(self) -> None:
        self.current_path = self.recording_dir / f"{self.window_prefix}_{self.window_idx:02d}.hex"
        self.file = open(self.current_path, "w", encoding="ascii")
        self.lines_in_window = 0
        print(f"Started new hex window: {self.current_path}")

    def _roll_window(self) -> None:
        if self.file:
            self.file.close()
            print(f"Completed hex window: {self.current_path} ({self.samples_per_window} lines)")
        self.window_idx += 1
        self._open_window()

    def write_samples(self, samples: np.ndarray) -> None:
        """
        samples shape: [n_samples, n_channels]
        Each line stores one sample across all channels.
        """
        for row in samples:
            if self.lines_in_window >= self.samples_per_window:
                self._roll_window()
            hex_values = [
                float_to_fixed_hex(float(val), self.bit_width, self.fractional_bits) for val in row
            ]
            self.file.write("".join(hex_values) + "\n")
            self.lines_in_window += 1

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None


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
    recording_dir = resolve_recording_dir(args.recording_dir)
    writer = HexWindowWriter(
        recording_dir=recording_dir,
        window_prefix=args.window_prefix,
        start_index=args.window_start_index,
        samples_per_window=args.window_samples,
        bit_width=args.hex_bit_width,
        fractional_bits=args.hex_fractional_bits,
    )
    print(
        f"Hex output dir: {recording_dir} "
        f"(Q{args.hex_bit_width - args.hex_fractional_bits}.{args.hex_fractional_bits}, "
        f"{args.window_samples} lines/window)."
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
                writer.write_samples(eeg_filtered.T)

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
        writer.close()
        print("Done.")


if __name__ == "__main__":
    main()
