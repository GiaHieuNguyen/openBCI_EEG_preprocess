"""
Data Loader for OpenBCI Cyton+Daisy Raw EEG Files
==================================================
Supports:
  - OpenBCI GUI .txt format (comma-delimited, with 4-line header)
  - BrainFlow .csv format (tab-delimited, no header)

Both formats contain 16 EXG channels at 125 Hz.
"""

import os
import numpy as np
import pandas as pd


# OpenBCI Cyton+Daisy default channel labels (10-20 system mapping)
DEFAULT_CH_NAMES = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",   # Cyton (Ch 1-8)
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",       # Daisy (Ch 9-16)
]

DEFAULT_SFREQ = 125.0  # Hz


def load_openbci_txt(filepath: str) -> dict:
    """
    Load OpenBCI GUI .txt format.

    Format:
      - Lines 1-4: metadata (% comments + column header)
      - Data: comma-separated, 32+ columns
      - Columns: Sample Index, EXG Ch 0-15, Accel 0-2, ..., Timestamp, Marker, Timestamp(Formatted)
    """
    # Parse metadata from header
    metadata = {}
    header_lines = 0
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("%"):
                if "Number of channels" in line:
                    metadata["n_channels"] = int(line.split("=")[1].strip())
                elif "Sample Rate" in line:
                    metadata["sample_rate"] = float(line.split("=")[1].strip().split()[0])
                elif "Board" in line:
                    metadata["board"] = line.split("=")[1].strip()
                header_lines += 1
            elif header_lines > 0 and not line.startswith("%"):
                # This is the column header line
                header_lines += 1
                break

    sfreq = metadata.get("sample_rate", DEFAULT_SFREQ)

    # Read data, skipping header lines
    df = pd.read_csv(filepath, skiprows=header_lines, header=None)

    # Columns 1-16 are EXG channels (0-indexed: columns 1 to 16)
    eeg_data = df.iloc[:, 1:17].values.T  # Shape: (16, n_samples)

    # Timestamp column (index 30)
    timestamps = df.iloc[:, 30].values if df.shape[1] > 30 else np.arange(eeg_data.shape[1]) / sfreq

    # Convert from uV (OpenBCI GUI exports in uV by default)
    return {
        "data": eeg_data.astype(np.float64),
        "sfreq": sfreq,
        "ch_names": DEFAULT_CH_NAMES[:eeg_data.shape[0]],
        "timestamps": timestamps,
        "n_samples": eeg_data.shape[1],
        "duration_sec": eeg_data.shape[1] / sfreq,
        "source_format": "openbci_txt",
        "metadata": metadata,
    }


def load_brainflow_csv(filepath: str) -> dict:
    """
    Load BrainFlow .csv format.

    Format:
      - No header
      - Tab-delimited
      - Columns: Sample Index (0), EXG Ch 1-16 (1-16), Accel (17-19), ..., Timestamp (30)
    """
    df = pd.read_csv(filepath, sep="\t", header=None)

    # Columns 1-16 are EXG channels
    eeg_data = df.iloc[:, 1:17].values.T  # Shape: (16, n_samples)

    # Timestamp column (index 30)
    timestamps = df.iloc[:, 30].values if df.shape[1] > 30 else np.arange(eeg_data.shape[1]) / DEFAULT_SFREQ

    return {
        "data": eeg_data.astype(np.float64),
        "sfreq": DEFAULT_SFREQ,
        "ch_names": DEFAULT_CH_NAMES[:eeg_data.shape[0]],
        "timestamps": timestamps,
        "n_samples": eeg_data.shape[1],
        "duration_sec": eeg_data.shape[1] / DEFAULT_SFREQ,
        "source_format": "brainflow_csv",
        "metadata": {"board": "BrainFlow"},
    }


def auto_detect_and_load(filepath: str) -> dict:
    """
    Auto-detect file format by extension and load accordingly.
    
    .txt  -> OpenBCI GUI format
    .csv  -> BrainFlow format
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        return load_openbci_txt(filepath)
    elif ext == ".csv":
        return load_brainflow_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .txt or .csv")
