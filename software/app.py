"""
EEG Visualization & Preprocessing Web Application
===================================================
Flask backend serving REST API for EEG data loading,
preprocessing, and visualization.

⚠ SINGLE-USER DESKTOP APPLICATION ⚠
This app stores all data in a global in-memory dict protected by a
threading lock.  It is designed for a single user on a local machine.
Running behind a multi-worker WSGI server (gunicorn, etc.) or serving
multiple concurrent browser sessions will cause data corruption.

Usage:
    python app.py
    Open http://localhost:5000
"""

import os
import sys
import threading
import time
import traceback
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

# Add project root to path for shared imports
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import auto_detect_and_load, DEFAULT_CH_NAMES
from preprocessing import (
    bandpass_filter, notch_filter, resample_data,
    detect_artifacts, compute_psd, robust_normalize,
    moving_average_smooth, compute_statistics,
)

try:
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    from brainflow.data_filter import (
        DataFilter,
        DetrendOperations,
        FilterTypes,
        NoiseTypes,
    )
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BoardIds = None
    BoardShim = None
    BrainFlowInputParams = None
    DataFilter = None
    DetrendOperations = None
    FilterTypes = None
    NoiseTypes = None
    BRAINFLOW_AVAILABLE = False

app = Flask(__name__)

from typing import Any, Dict, List, Optional

# ── Global State ────────────────────────────────────────────
# In-memory store for loaded data (single-user desktop app)
state: Dict[str, Any] = {
    "raw": None,         # Original loaded data dict
    "processed": None,   # Current processed data (ndarray)
    "sfreq": None,
    "ch_names": None,
    "filepath": None,
    "time_axis": None,
    "history": [],       # Processing step history
    "mode": "file",      # file | live
    "live_total_samples": 0,
}
state_lock = threading.Lock()

live_state: Dict[str, Any] = {
    "running": False,
    "error": None,
    "thread": None,
    "stop_event": None,
    "config": {},
}
live_state_lock = threading.Lock()

RAW_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "raw_data")
)
PROJECT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..")
)


def _default_serial_port() -> str:
    return "COM3" if os.name == "nt" else "/dev/ttyUSB0"


def _snapshot_state():
    with state_lock:
        data = state["processed"]
        if data is None:
            return None
        return {
            "processed": data.copy(),
            "sfreq": state["sfreq"],
            "ch_names": list(state["ch_names"]) if state["ch_names"] else [],
            "mode": state["mode"],
            "live_total_samples": int(state["live_total_samples"] or 0),
            "history": list(state["history"]),
            "raw": state["raw"],
            "time_axis": None if state["time_axis"] is None else state["time_axis"].copy(),
        }


def _normalize_timestamps_to_seconds(timestamps, n_samples: int, sfreq: float) -> np.ndarray:
    """
    Convert source timestamps to a monotonic seconds timeline starting at 0.
    Falls back to uniform sampling timeline when timestamps are missing/invalid.
    """
    fallback = np.arange(n_samples, dtype=np.float64) / float(sfreq)
    if timestamps is None:
        return fallback

    try:
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    except Exception:
        return fallback

    if t.size != n_samples or t.size == 0:
        return fallback
    if not np.all(np.isfinite(t)):
        return fallback

    t = t - t[0]
    if t.size > 1:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return fallback
        med_dt = float(np.median(dt))
        expected_dt = 1.0 / float(sfreq)
        ratio = med_dt / expected_dt if expected_dt > 0 else 1.0
        # Heuristic: timestamps may be in ms or us.
        if ratio > 10000.0:
            t = t / 1e6
        elif ratio > 10.0:
            t = t / 1e3

    if t.size > 1 and np.any(np.diff(t) <= 0):
        return fallback
    return t


def _apply_live_preprocessing(eeg_data: np.ndarray, sampling_rate: int, config: dict) -> np.ndarray:
    """Apply channel-wise preprocessing using the shared BrainFlow preprocessing module."""
    from common import apply_brainflow_preprocessing
    return apply_brainflow_preprocessing(
        eeg_data, sampling_rate,
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
        bandpass_order=config["order"],
        notch=config["notch"],
    )


def _append_live_chunk(chunk: np.ndarray, sampling_rate: int, config: dict):
    """Append processed live chunk and keep only trailing buffer window."""
    max_samples = max(1, int(config["buffer_seconds"] * sampling_rate))

    with state_lock:
        if state["processed"] is None:
            state["processed"] = np.zeros((chunk.shape[0], 0), dtype=np.float64)

        prev_total = int(state["live_total_samples"] or 0)
        new_times = (np.arange(chunk.shape[1], dtype=np.float64) + prev_total) / float(sampling_rate)

        state["processed"] = np.concatenate([state["processed"], chunk], axis=1)
        if state["time_axis"] is None:
            state["time_axis"] = np.array([], dtype=np.float64)
        state["time_axis"] = np.concatenate([state["time_axis"], new_times], axis=0)
        if state["processed"].shape[1] > max_samples:
            state["processed"] = state["processed"][:, -max_samples:]
            state["time_axis"] = state["time_axis"][-max_samples:]

        state["live_total_samples"] += chunk.shape[1]
        state["raw"] = {
            "data": state["processed"].copy(),
            "sfreq": float(sampling_rate),
            "ch_names": list(state["ch_names"]) if state["ch_names"] else [],
            "timestamps": state["time_axis"].copy(),
        }


def _live_stream_worker(config: dict, stop_event: threading.Event):
    board = None
    try:
        params = BrainFlowInputParams()
        params.serial_port = config["serial_port"]
        board = BoardShim(config["board_id"], params)
        board.prepare_session()
        board.start_stream()

        eeg_channels = BoardShim.get_eeg_channels(config["board_id"])
        sampling_rate = BoardShim.get_sampling_rate(config["board_id"])
        ch_names = DEFAULT_CH_NAMES[:len(eeg_channels)]
        if len(ch_names) < len(eeg_channels):
            ch_names.extend([f"Ch{i + 1}" for i in range(len(ch_names), len(eeg_channels))])

        with state_lock:
            time_axis = np.array([], dtype=np.float64)
            state["raw"] = {
                "data": np.zeros((len(eeg_channels), 0), dtype=np.float64),
                "sfreq": float(sampling_rate),
                "ch_names": ch_names,
                "timestamps": time_axis.copy(),
            }
            state["processed"] = np.zeros((len(eeg_channels), 0), dtype=np.float64)
            state["sfreq"] = float(sampling_rate)
            state["ch_names"] = ch_names
            state["filepath"] = "LIVE_STREAM"
            state["time_axis"] = time_axis
            state["history"] = [
                (
                    f"Live stream started ({config['serial_port']}, board_id={config['board_id']}) "
                    f"bandpass {config['l_freq']}-{config['h_freq']} Hz, notch {config['notch']} Hz"
                )
            ]
            state["mode"] = "live"
            state["live_total_samples"] = 0

        with live_state_lock:
            live_state["running"] = True
            live_state["error"] = None

        while not stop_event.is_set():
            time.sleep(0.2)
            data = board.get_board_data()
            if data.shape[1] <= 0:
                continue
            eeg_data = data[eeg_channels]
            eeg_filtered = _apply_live_preprocessing(eeg_data, sampling_rate, config)
            _append_live_chunk(eeg_filtered, sampling_rate, config)
    except Exception as e:
        traceback.print_exc()
        with live_state_lock:
            live_state["error"] = str(e)
    finally:
        if board is not None:
            try:
                board.stop_stream()
            except Exception:
                pass
            try:
                board.release_session()
            except Exception:
                pass
        with live_state_lock:
            live_state["running"] = False
            live_state["thread"] = None
            live_state["stop_event"] = None


def _stop_live_stream():
    with live_state_lock:
        thread = live_state["thread"]
        stop_event = live_state["stop_event"]
        is_running = thread is not None

    if is_running and stop_event is not None:
        stop_event.set()
        thread.join(timeout=3.0)

    with live_state_lock:
        live_state["running"] = False
        live_state["thread"] = None
        live_state["stop_event"] = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/files", methods=["GET"])
def list_files():
    """List available raw data files."""
    files = []
    seen = set()
    scan_dirs = [RAW_DATA_DIR, PROJECT_DATA_DIR]
    for data_dir in scan_dirs:
        if not os.path.isdir(data_dir):
            continue
        for f in sorted(os.listdir(data_dir)):
            if not f.lower().endswith((".csv", ".txt")):
                continue
            if f.lower() == "requirements.txt":
                continue
            fpath = os.path.join(data_dir, f)
            if not os.path.isfile(fpath):
                continue
            if fpath in seen:
                continue
            seen.add(fpath)
            files.append({
                "name": f if data_dir == RAW_DATA_DIR else f"[project] {f}",
                "path": fpath,
                "size_mb": round(os.path.getsize(fpath) / 1e6, 2),
            })
    return jsonify({"files": files})


@app.route("/api/load", methods=["POST"])
def load_file():
    """Load a data file."""
    try:
        _stop_live_stream()
        body = request.get_json()
        filepath = body.get("filepath", "")

        if not filepath or not os.path.isfile(filepath):
            return jsonify({"error": "File not found"}), 400

        result = auto_detect_and_load(filepath)

        with state_lock:
            state["raw"] = result
            state["processed"] = result["data"].copy()
            state["sfreq"] = result["sfreq"]
            state["ch_names"] = result["ch_names"]
            state["filepath"] = filepath
            state["time_axis"] = _normalize_timestamps_to_seconds(
                result.get("timestamps"), result["data"].shape[1], result["sfreq"]
            )
            state["history"] = ["Loaded raw data"]
            state["mode"] = "file"
            state["live_total_samples"] = 0

        return jsonify({
            "success": True,
            "info": {
                "n_channels": result["data"].shape[0],
                "n_samples": result["n_samples"],
                "sfreq": result["sfreq"],
                "duration_sec": round(result["duration_sec"], 2),
                "source_format": result["source_format"],
                "ch_names": result["ch_names"],
                "metadata": result.get("metadata", {}),
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/channels", methods=["GET"])
def get_channels():
    """Get channel data for plotting. Supports downsampling for performance."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    # Optional time range (in seconds)
    t_start = float(request.args.get("t_start", 0))
    t_end = float(request.args.get("t_end", -1))
    max_points = int(request.args.get("max_points", 5000))

    data = snapshot["processed"]
    sfreq = snapshot["sfreq"]
    timeline = snapshot["time_axis"]
    n_samples = data.shape[1]

    if timeline is None or timeline.shape[0] != n_samples:
        timeline = np.arange(n_samples, dtype=np.float64) / float(sfreq)

    start_idx = int(np.searchsorted(timeline, t_start, side="left"))
    if t_end > 0:
        end_idx = int(np.searchsorted(timeline, t_end, side="right"))
    else:
        end_idx = n_samples
    start_idx = max(0, min(start_idx, n_samples))
    end_idx = max(start_idx, min(end_idx, n_samples))

    segment = data[:, start_idx:end_idx]
    segment_time = timeline[start_idx:end_idx]

    # Downsample for visualization if too many points
    actual_points = segment.shape[1]
    if actual_points > max_points:
        step = actual_points // max_points
        segment = segment[:, ::step]
        time_axis = segment_time[::step]
    else:
        time_axis = segment_time

    channels = {}
    for i, ch_name in enumerate(snapshot["ch_names"]):
        channels[ch_name] = segment[i].tolist()

    return jsonify({
        "time": time_axis.tolist(),
        "channels": channels,
        "sfreq": sfreq,
        "total_duration": float(timeline[-1]) if n_samples > 0 else 0.0,
        "mode": snapshot["mode"],
    })


@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    """Apply preprocessing to the data."""
    snapshot = _snapshot_state()
    if snapshot is None or snapshot["raw"] is None:
        return jsonify({"error": "No data loaded"}), 400

    try:
        body = request.get_json()
        action = body.get("action", "")

        if snapshot["mode"] == "live":
            return jsonify({
                "error": "Manual preprocess actions are disabled during live streaming. "
                         "Use live stream filter parameters when starting the stream."
            }), 400

        data = snapshot["processed"]
        sfreq = snapshot["sfreq"]

        if action == "bandpass":
            l_freq = float(body.get("l_freq", 0.5))
            h_freq = float(body.get("h_freq", 45.0))
            order = int(body.get("order", 4))
            processed = bandpass_filter(data, sfreq, l_freq, h_freq, order)
            with state_lock:
                state["processed"] = processed
                state["history"].append(f"Bandpass {l_freq}-{h_freq} Hz (order {order})")

        elif action == "notch":
            freq = float(body.get("freq", 50.0))
            quality = float(body.get("quality", 30.0))
            processed = notch_filter(data, sfreq, freq, quality)
            with state_lock:
                state["processed"] = processed
                state["history"].append(f"Notch {freq} Hz (Q={quality})")

        elif action == "resample":
            target_sfreq = float(body.get("target_sfreq", 128.0))
            processed = resample_data(data, sfreq, target_sfreq)
            old_time = snapshot["time_axis"]
            n_new = processed.shape[1]
            if old_time is not None and old_time.shape[0] > 1:
                new_time = np.linspace(float(old_time[0]), float(old_time[-1]), n_new, dtype=np.float64)
            else:
                new_time = np.arange(n_new, dtype=np.float64) / target_sfreq
            with state_lock:
                state["processed"] = processed
                state["sfreq"] = target_sfreq
                state["time_axis"] = new_time
                state["history"].append(f"Resampled to {target_sfreq} Hz")

        elif action == "normalize":
            normalize_scale = float(body.get("normalize_scale", 10.0))
            if normalize_scale <= 0:
                return jsonify({"error": "normalize_scale must be > 0"}), 400
            processed = robust_normalize(data, scale=normalize_scale)
            with state_lock:
                state["processed"] = processed
                state["history"].append(f"Robust normalization (q95, gain={normalize_scale})")

        elif action == "smooth":
            window_samples = int(body.get("window_samples", 5))
            if window_samples < 1:
                return jsonify({"error": "window_samples must be >= 1"}), 400
            processed = moving_average_smooth(data, window_samples=window_samples)
            with state_lock:
                state["processed"] = processed
                state["history"].append(f"Moving average smoothing (window={window_samples} samples)")

        elif action == "reset":
            with state_lock:
                state["processed"] = state["raw"]["data"].copy()
                state["sfreq"] = state["raw"]["sfreq"]
                state["time_axis"] = _normalize_timestamps_to_seconds(
                    state["raw"].get("timestamps"),
                    state["raw"]["data"].shape[1],
                    state["raw"]["sfreq"],
                )
                state["history"] = ["Loaded raw data", "Reset to raw"]

        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

        current = _snapshot_state()

        return jsonify({
            "success": True,
            "shape": list(current["processed"].shape),
            "sfreq": current["sfreq"],
            "history": current["history"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/psd", methods=["GET"])
def get_psd():
    """Compute and return PSD."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    n_fft = int(request.args.get("n_fft", 256))
    result = compute_psd(snapshot["processed"], snapshot["sfreq"], n_fft=n_fft)
    result["ch_names"] = snapshot["ch_names"]
    return jsonify(result)


@app.route("/api/artifacts", methods=["GET"])
def get_artifacts():
    """Detect artifacts in current data."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    max_abs = float(request.args.get("max_abs_uv", 500.0))
    flat_thresh = float(request.args.get("flat_thresh", 0.5))
    window_sec = float(request.args.get("window_sec", 1.0))

    result = detect_artifacts(
        snapshot["processed"], snapshot["sfreq"],
        window_sec=window_sec,
        max_abs_uv=max_abs,
        flat_thresh=flat_thresh,
    )
    # Don't send the full mask in the summary, just segments
    result_summary = {
        "bad_segments": [
            {"start": s, "end": e, "reason": r,
             "start_sec": round(s / snapshot["sfreq"], 2),
             "end_sec": round(e / snapshot["sfreq"], 2)}
            for s, e, r in result["bad_segments"]
        ],
        "n_bad": result["n_bad"],
        "n_total": result["n_total"],
        "pct_bad": round(result["pct_bad"], 1),
    }
    return jsonify(result_summary)


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """Get per-channel statistics."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    stats = compute_statistics(snapshot["processed"], snapshot["ch_names"])
    return jsonify({"statistics": stats})


@app.route("/api/export", methods=["POST"])
def export_data():
    """Export processed data to CSV."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    try:
        export_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data")
        os.makedirs(export_dir, exist_ok=True)

        filename = "processed_eeg.csv"
        export_path = os.path.join(export_dir, filename)

        data = snapshot["processed"]
        sfreq = snapshot["sfreq"]
        n_samples = data.shape[1]

        # Build DataFrame
        df = pd.DataFrame(data.T, columns=snapshot["ch_names"])
        df.insert(0, "Time_s", np.arange(n_samples) / sfreq)

        df.to_csv(export_path, index=False)

        return jsonify({
            "success": True,
            "filepath": export_path,
            "filename": filename,
            "n_channels": data.shape[0],
            "n_samples": n_samples,
            "sfreq": sfreq,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/start", methods=["POST"])
def start_live_stream():
    if not BRAINFLOW_AVAILABLE:
        return jsonify({
            "error": "BrainFlow is not installed. Install with `pip install brainflow` to use live streaming."
        }), 400

    body = request.get_json(silent=True) or {}
    serial_port = body.get("serial_port", _default_serial_port())
    board_id = int(body.get("board_id", BoardIds.CYTON_DAISY_BOARD.value))
    l_freq = float(body.get("l_freq", 0.5))
    h_freq = float(body.get("h_freq", 40.0))
    order = int(body.get("order", 4))
    notch = float(body.get("notch", 50.0))
    buffer_seconds = float(body.get("buffer_seconds", 30.0))

    if notch not in (0.0, 50.0, 60.0):
        return jsonify({"error": "Notch must be 0, 50, or 60"}), 400
    if h_freq <= l_freq:
        return jsonify({"error": "h_freq must be > l_freq"}), 400
    if buffer_seconds <= 0:
        return jsonify({"error": "buffer_seconds must be > 0"}), 400

    _stop_live_stream()

    config = {
        "serial_port": serial_port,
        "board_id": board_id,
        "l_freq": l_freq,
        "h_freq": h_freq,
        "order": order,
        "notch": notch,
        "buffer_seconds": buffer_seconds,
    }
    stop_event = threading.Event()
    thread = threading.Thread(target=_live_stream_worker, args=(config, stop_event), daemon=True)

    with live_state_lock:
        live_state["config"] = config
        live_state["error"] = None
        live_state["stop_event"] = stop_event
        live_state["thread"] = thread

    thread.start()
    time.sleep(0.6)

    with live_state_lock:
        err = live_state["error"]
        running = live_state["running"]

    if err:
        return jsonify({"error": f"Failed to start live stream: {err}"}), 500

    snapshot = _snapshot_state()
    return jsonify({
        "success": True,
        "running": running,
        "config": config,
        "info": {
            "n_channels": 0 if snapshot is None else len(snapshot["ch_names"]),
            "sfreq": None if snapshot is None else snapshot["sfreq"],
            "ch_names": [] if snapshot is None else snapshot["ch_names"],
            "mode": "live",
        }
    })


@app.route("/api/live/stop", methods=["POST"])
def stop_live_stream():
    _stop_live_stream()
    with state_lock:
        state["mode"] = "file"
    return jsonify({"success": True, "running": False})


@app.route("/api/live/status", methods=["GET"])
def live_status():
    snapshot = _snapshot_state()
    with live_state_lock:
        running = live_state["running"]
        err = live_state["error"]
        config = dict(live_state["config"])

    return jsonify({
        "brainflow_available": BRAINFLOW_AVAILABLE,
        "running": running,
        "error": err,
        "config": config,
        "mode": None if snapshot is None else snapshot["mode"],
        "n_channels": 0 if snapshot is None else len(snapshot["ch_names"]),
        "n_samples_buffered": 0 if snapshot is None else int(snapshot["processed"].shape[1]),
        "live_total_samples": 0 if snapshot is None else snapshot["live_total_samples"],
        "sfreq": None if snapshot is None else snapshot["sfreq"],
    })


@app.route("/api/bandpower", methods=["GET"])
def get_bandpower():
    """Compute per-channel band power for standard EEG bands."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    from scipy.signal import welch as sp_welch

    data = snapshot["processed"]
    sfreq = snapshot["sfreq"]
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    n_ch = data.shape[0]
    result = {band: [] for band in bands}

    for ch in range(n_ch):
        freqs, psd = sp_welch(data[ch], fs=sfreq, nperseg=min(data.shape[1], int(2 * sfreq)))
        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            power = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
            result[band_name].append(power)

    return jsonify({
        "bands": result,
        "band_names": list(bands.keys()),
        "ch_names": snapshot["ch_names"],
    })


@app.route("/api/ica", methods=["POST"])
def apply_ica():
    """Apply ICA for artifact removal."""
    snapshot = _snapshot_state()
    if snapshot is None or snapshot["raw"] is None:
        return jsonify({"error": "No data loaded"}), 400

    if snapshot["mode"] == "live":
        return jsonify({"error": "ICA is not available during live streaming."}), 400

    try:
        from sklearn.decomposition import FastICA

        body = request.get_json(silent=True) or {}
        n_components = int(body.get("n_components", 0))
        exclude_components = body.get("exclude", [])

        data = snapshot["processed"]
        n_ch = data.shape[0]

        if n_components <= 0:
            n_components = n_ch

        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        sources = ica.fit_transform(data.T)

        if exclude_components:
            for idx in exclude_components:
                if 0 <= idx < sources.shape[1]:
                    sources[:, idx] = 0.0

            reconstructed = ica.inverse_transform(sources).T
            with state_lock:
                state["processed"] = reconstructed
                state["history"].append(f"ICA: removed components {exclude_components}")

            return jsonify({
                "success": True,
                "action": "reconstructed",
                "excluded": exclude_components,
                "n_components": n_components,
            })
        else:
            mixing = ica.mixing_.tolist()
            comp_rms = [float(np.sqrt(np.mean(sources[:, i] ** 2))) for i in range(sources.shape[1])]

            return jsonify({
                "success": True,
                "action": "decomposed",
                "n_components": n_components,
                "mixing_matrix": mixing,
                "component_rms": comp_rms,
            })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/topomap", methods=["GET"])
def get_topomap():
    """Return 2D scalp coordinates and per-channel power for topomap visualization."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    pos_10_20 = {
        "Fp1": (-0.22, 0.90), "Fp2": (0.22, 0.90),
        "F7":  (-0.70, 0.50), "F3":  (-0.33, 0.50),
        "F4":  (0.33, 0.50),  "F8":  (0.70, 0.50),
        "T7":  (-0.90, 0.00), "C3":  (-0.40, 0.00),
        "C4":  (0.40, 0.00),  "T8":  (0.90, 0.00),
        "P7":  (-0.70, -0.50), "P3": (-0.33, -0.50),
        "P4":  (0.33, -0.50),  "P8": (0.70, -0.50),
        "O1":  (-0.22, -0.90), "O2": (0.22, -0.90),
    }

    data = snapshot["processed"]
    ch_names = snapshot["ch_names"]

    channels = []
    for i, ch in enumerate(ch_names):
        pos = pos_10_20.get(ch, (0.0, 0.0))
        rms = float(np.sqrt(np.mean(data[i] ** 2)))
        channels.append({"name": ch, "x": pos[0], "y": pos[1], "power": rms})

    return jsonify({"channels": channels})


recording_state = {"active": False, "filepath": None, "samples_written": 0}
recording_lock = threading.Lock()


@app.route("/api/record/start", methods=["POST"])
def start_recording():
    """Start recording processed data to a CSV file."""
    snapshot = _snapshot_state()
    if snapshot is None:
        return jsonify({"error": "No data loaded"}), 400

    with recording_lock:
        if recording_state["active"]:
            return jsonify({"error": "Already recording"}), 400

        import datetime
        rec_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data")
        os.makedirs(rec_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{ts}.csv"
        filepath = os.path.join(rec_dir, filename)

        data = snapshot["processed"]
        ch_names = snapshot["ch_names"]
        sfreq = snapshot["sfreq"]
        n_samples = data.shape[1]

        df = pd.DataFrame(data.T, columns=ch_names)
        df.insert(0, "Time_s", np.arange(n_samples) / sfreq)
        df.to_csv(filepath, index=False)

        recording_state["active"] = True
        recording_state["filepath"] = filepath
        recording_state["samples_written"] = n_samples

    return jsonify({
        "success": True, "recording": True,
        "filepath": filepath, "filename": filename,
        "samples_written": n_samples,
    })


@app.route("/api/record/stop", methods=["POST"])
def stop_recording():
    """Stop recording."""
    with recording_lock:
        recording_state["active"] = False
        filepath = recording_state["filepath"]
        samples = recording_state["samples_written"]
        recording_state["filepath"] = None
        recording_state["samples_written"] = 0
    return jsonify({"success": True, "recording": False, "filepath": filepath, "samples_written": samples})


@app.route("/api/record/status", methods=["GET"])
def recording_status():
    """Get recording status."""
    with recording_lock:
        return jsonify({
            "active": recording_state["active"],
            "filepath": recording_state["filepath"],
            "samples_written": recording_state["samples_written"],
        })


if __name__ == "__main__":
    print("=" * 60)
    print("  EEG Visualization & Preprocessing Software")
    print("  OpenBCI Cyton+Daisy | 16 Channels | 125 Hz")
    print("=" * 60)
    print(f"  Raw data directory: {RAW_DATA_DIR}")
    print(f"  Open in browser: http://localhost:5000")
    print(f"  BrainFlow available: {BRAINFLOW_AVAILABLE}")
    if BRAINFLOW_AVAILABLE:
        print(f"  Default serial port: {_default_serial_port()}")
    print("=" * 60)
    app.run(debug=True, port=5000)
