"""
Unit tests for EEG preprocessing and utility functions.

Run: pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))


# ── hex_convert tests ────────────────────────────────────────────────────


class TestHexConvert:
    """Tests for preprocess/hex_convert.py functions."""

    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocess"))
        from preprocess.hex_convert import preprocess_eeg, float_to_configurable_hex, apply_notch_filter, apply_bandpass_filter
        self.preprocess_eeg = preprocess_eeg
        self.float_to_configurable_hex = float_to_configurable_hex
        self.apply_notch_filter = apply_notch_filter
        self.apply_bandpass_filter = apply_bandpass_filter

    def test_preprocess_eeg_removes_dc(self):
        """Preprocessing should remove DC offset (center around zero)."""
        data = [100.0, 105.0, 95.0, 100.0, 110.0, 90.0] * 50  # 300 samples
        result = self.preprocess_eeg(data, fs=250.0)
        assert abs(np.mean(result)) < abs(np.mean(data)), "DC offset should be reduced"

    def test_preprocess_eeg_output_length(self):
        """Output length should match input length."""
        data = list(np.random.randn(500))
        result = self.preprocess_eeg(data, fs=250.0)
        assert len(result) == len(data)

    def test_hex_conversion_writes_file(self, tmp_path):
        """float_to_configurable_hex should write a valid hex file."""
        data = [1.5, -1.5, 0.0, 3.0, -3.0]
        outfile = str(tmp_path / "test.hex")
        self.float_to_configurable_hex(data, bit_width=16, fractional_bits=5, filename=outfile)
        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 5

    def test_hex_conversion_saturation(self, tmp_path):
        """Values exceeding range should be saturated, not wrapped."""
        data = [1e10, -1e10]  # Way beyond any 16-bit range
        outfile = str(tmp_path / "sat.hex")
        self.float_to_configurable_hex(data, bit_width=16, fractional_bits=0, filename=outfile)
        with open(outfile) as f:
            lines = f.read().strip().split("\n")
        # Max positive 16-bit signed is 7FFF, max negative is 8000
        assert lines[0] == "7FFF"
        assert lines[1] == "8000"

    def test_hex_conversion_zero(self, tmp_path):
        """Zero should convert to all-zero hex."""
        outfile = str(tmp_path / "zero.hex")
        self.float_to_configurable_hex([0.0], bit_width=16, fractional_bits=5, filename=outfile)
        with open(outfile) as f:
            content = f.read().strip()
        assert content == "0000"

    def test_hex_32bit_format(self, tmp_path):
        """32-bit output should have 8 hex characters per line."""
        outfile = str(tmp_path / "wide.hex")
        self.float_to_configurable_hex([1.0, -1.0], bit_width=32, fractional_bits=13, filename=outfile)
        with open(outfile) as f:
            lines = f.read().strip().split("\n")
        for line in lines:
            assert len(line) == 8, f"Expected 8 hex chars, got {len(line)}: {line}"


# ── preprocessing (software/) tests ──────────────────────────────────────


class TestPreprocessing:
    """Tests for software/preprocessing.py functions."""

    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "software"))
        from software.preprocessing import (
            bandpass_filter, notch_filter, resample_data,
            detect_artifacts, compute_psd, robust_normalize,
            moving_average_smooth, compute_statistics,
        )
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.resample_data = resample_data
        self.detect_artifacts = detect_artifacts
        self.compute_psd = compute_psd
        self.robust_normalize = robust_normalize
        self.moving_average_smooth = moving_average_smooth
        self.compute_statistics = compute_statistics

    def _make_signal(self, n_channels=4, n_samples=1000, sfreq=250.0):
        """Generate synthetic multi-channel EEG-like data."""
        rng = np.random.default_rng(42)
        t = np.arange(n_samples) / sfreq
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Mix of alpha (10 Hz) + noise
            data[ch] = 20 * np.sin(2 * np.pi * 10 * t + ch) + rng.normal(0, 5, n_samples)
        return data, sfreq

    def test_bandpass_preserves_shape(self):
        data, sfreq = self._make_signal()
        result = self.bandpass_filter(data, sfreq, 1.0, 40.0)
        assert result.shape == data.shape

    def test_bandpass_removes_dc(self):
        data, sfreq = self._make_signal()
        data += 10000  # Add big DC offset
        result = self.bandpass_filter(data, sfreq, 1.0, 40.0)
        assert abs(np.mean(result)) < 100, "Bandpass should remove DC"

    def test_notch_preserves_shape(self):
        data, sfreq = self._make_signal()
        result = self.notch_filter(data, sfreq, 50.0)
        assert result.shape == data.shape

    def test_resample_changes_length(self):
        data, _ = self._make_signal(sfreq=250.0)
        result = self.resample_data(data, 250.0, 125.0)
        assert result.shape[0] == data.shape[0]  # Same channels
        assert result.shape[1] == data.shape[1] // 2  # Half samples

    def test_resample_identity(self):
        data, _ = self._make_signal()
        result = self.resample_data(data, 250.0, 250.0)
        np.testing.assert_array_equal(result, data)

    def test_detect_artifacts_clean_data(self):
        data, sfreq = self._make_signal()
        result = self.detect_artifacts(data, sfreq, max_abs_uv=10000)
        assert result["n_bad"] == 0

    def test_detect_artifacts_high_amplitude(self):
        data, sfreq = self._make_signal()
        data[0, 100:200] = 99999  # Inject artifact
        result = self.detect_artifacts(data, sfreq, max_abs_uv=500)
        assert result["n_bad"] > 0

    def test_detect_artifacts_nan(self):
        data, sfreq = self._make_signal()
        data[0, 50] = np.nan
        result = self.detect_artifacts(data, sfreq)
        assert result["n_bad"] > 0

    def test_compute_psd_output_structure(self):
        data, sfreq = self._make_signal()
        result = self.compute_psd(data, sfreq)
        assert "freqs" in result
        assert "psd" in result
        assert len(result["psd"]) == data.shape[0]

    def test_robust_normalize_range(self):
        data, _ = self._make_signal()
        result = self.robust_normalize(data, scale=1.0)
        # After normalization, most values should be near [-1, 1]
        assert np.quantile(np.abs(result), 0.95) <= 1.5

    def test_moving_average_smooth_preserves_shape(self):
        data, _ = self._make_signal()
        result = self.moving_average_smooth(data, window_samples=5)
        assert result.shape == data.shape

    def test_moving_average_smooth_identity(self):
        data, _ = self._make_signal()
        result = self.moving_average_smooth(data, window_samples=1)
        np.testing.assert_array_almost_equal(result, data)

    def test_compute_statistics_all_channels(self):
        data, _ = self._make_signal(n_channels=4)
        ch_names = ["Ch1", "Ch2", "Ch3", "Ch4"]
        stats = self.compute_statistics(data, ch_names)
        assert len(stats) == 4
        for s in stats:
            assert all(k in s for k in ["channel", "mean", "std", "min", "max", "rms"])


# ── data_loader tests ────────────────────────────────────────────────────


class TestDataLoader:
    """Tests for software/data_loader.py functions."""

    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "software"))
        from software.data_loader import auto_detect_and_load

        self.auto_detect_and_load = auto_detect_and_load

    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("some data")
        with pytest.raises(ValueError, match="Unsupported"):
            self.auto_detect_and_load(str(f))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.auto_detect_and_load("/nonexistent/path.csv")


# ── sliding_windows / label_windows tests ────────────────────────────────


class TestWindowFunctions:
    """Tests for preprocess_basic.py windowing and labeling."""

    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocess"))
        from preprocess.preprocess_basic import sliding_windows, label_windows
        self.sliding_windows = sliding_windows
        self.label_windows = label_windows

    def test_sliding_windows_shape(self):
        data = np.random.randn(4, 1000)
        wins, starts = self.sliding_windows(data, sfreq=250.0, win_sec=1.0, step_sec=0.5)
        assert wins.shape[1] == 4  # n_channels
        assert wins.shape[2] == 250  # win_samples = 1.0 * 250
        assert len(starts) == wins.shape[0]

    def test_sliding_windows_no_overlap(self):
        data = np.random.randn(2, 500)
        wins, starts = self.sliding_windows(data, sfreq=250.0, win_sec=1.0, step_sec=1.0)
        assert wins.shape[0] == 2  # 500 / 250 = 2 windows

    def test_label_windows_all_non_seizure(self):
        starts = np.array([0, 250, 500])
        labels = self.label_windows(starts, sfreq=250.0, win_sec=1.0, seizure_intervals_sec=[])
        np.testing.assert_array_equal(labels, [0, 0, 0])

    def test_label_windows_with_seizure(self):
        starts = np.array([0, 250, 500, 750])
        # Seizure from 1.0s to 3.0s — windows at 1.0s and 2.0s should be labeled
        labels = self.label_windows(
            starts, sfreq=250.0, win_sec=1.0,
            seizure_intervals_sec=[(1.0, 3.0)],
        )
        assert labels[0] == 0  # 0-1s: no overlap
        assert labels[1] == 1  # 1-2s: full overlap
        assert labels[2] == 1  # 2-3s: full overlap
        assert labels[3] == 0  # 3-4s: no overlap
