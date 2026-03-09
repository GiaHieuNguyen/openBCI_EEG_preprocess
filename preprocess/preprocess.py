import argparse
import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mne
import pandas as pd

from preprocess_basic import preprocess_raw, sliding_windows, label_windows


BANDS = [
    (0.5, 4.0),   # delta
    (4.0, 8.0),   # theta
    (8.0, 12.0),  # alpha
    (12.0, 30.0), # beta
    (30.0, 70.0), # gamma (capped by h_freq)
]


def parse_intervals(text):
    """Parse "start,end;start,end" into list of (float, float)."""
    if not text:
        return []
    intervals = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(",")
        intervals.append((float(a), float(b)))
    return intervals


def load_muse_csv(csv_path):
    """
    Load Muse CSV and return an MNE RawArray using RAW_* columns.
    Time column is assumed to be epoch seconds.
    """
    df = pd.read_csv(csv_path)

    raw_cols = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing RAW columns in Muse CSV: {missing}")
    if "time" not in df.columns:
        raise ValueError("Muse CSV must include a 'time' column.")

    # Drop rows with NaN time, sort, and remove duplicate timestamps
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df[~df["time"].duplicated(keep="first")]

    t = df["time"].to_numpy()
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Not enough valid time samples to estimate sampling rate.")
    sfreq = 1.0 / np.median(dt)

    data = df[raw_cols].to_numpy().T  # [n_ch, n_samples]

    info = mne.create_info(ch_names=raw_cols, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def bandpower_features(windows, sfreq, bands):
    """
    windows: [n_win, n_ch, n_samples]
    returns: [n_win, n_ch * n_bands]
    """
    n_win, n_ch, n_samp = windows.shape
    feats = np.zeros((n_win, n_ch * len(bands)), dtype=np.float32)

    for i in range(n_win):
        for ch in range(n_ch):
            freqs, psd = signal.welch(windows[i, ch], sfreq, nperseg=min(n_samp, int(2 * sfreq)))
            for bi, (f0, f1) in enumerate(bands):
                mask = (freqs >= f0) & (freqs <= f1)
                feats[i, ch * len(bands) + bi] = np.trapezoid(psd[mask], freqs[mask])
    return feats


def run_pipeline(raw, name, win_sec, step_sec, seizure_intervals_sec, max_windows, seed):
    data = raw.get_data()
    sfreq = raw.info["sfreq"]

    X, starts = sliding_windows(data, sfreq, win_sec=win_sec, step_sec=step_sec)
    if X.shape[0] == 0:
        raise ValueError(f"No windows created for {name}. Check window size vs recording length.")

    y = label_windows(starts, sfreq, win_sec=win_sec, seizure_intervals_sec=seizure_intervals_sec)

    # Optional downsample of windows for speed
    if max_windows and X.shape[0] > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_windows, replace=False)
        X = X[idx]
        y = y[idx]

    # Ensure both classes exist
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{name}: only one class present. Provide correct seizure intervals or use a file with seizures."
        )

    feats = bandpower_features(X, sfreq, BANDS)

    X_train, X_test, y_train, y_test = train_test_split(
        feats, y, test_size=0.25, random_state=seed, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, n_jobs=None))
    ])
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return {
        "name": name,
        "acc": acc,
        "n_win": int(len(y)),
        "n_pos": pos,
        "n_neg": neg,
        "sfreq": sfreq,
    }


def main():
    ap = argparse.ArgumentParser(description="Quick CPU baseline on CHB-MIT window labels.")
    ap.add_argument("--edf", default="chb01_03.edf")
    ap.add_argument("--muse-csv", default="", help="Path to Muse CSV (uses RAW_* columns).")
    ap.add_argument("--seizures", default="", help='e.g. "2996,3036;1234,1250" (seconds)')
    ap.add_argument("--notch", type=float, default=60.0)
    ap.add_argument("--l-freq", type=float, default=0.5)
    ap.add_argument("--h-freq", type=float, default=70.0)
    ap.add_argument("--resample", type=float, default=256.0)
    ap.add_argument("--win-sec", type=float, default=2.0)
    ap.add_argument("--step-sec", type=float, default=1.0)
    ap.add_argument("--max-windows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    seizure_intervals_sec = parse_intervals(args.seizures)
    if not seizure_intervals_sec:
        raise SystemExit(
            "No seizure intervals provided. Use --seizures with CHB-MIT summary for this file."
        )

    if args.muse_csv:
        raw_unprocessed = load_muse_csv(args.muse_csv)
    else:
        raw_unprocessed = mne.io.read_raw_edf(args.edf, preload=True, verbose=False)
    raw_preprocessed = preprocess_raw(
        raw_unprocessed.copy(),
        notch=args.notch,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        resample=args.resample,
    )

    res_raw = run_pipeline(
        raw_unprocessed,
        "raw",
        args.win_sec,
        args.step_sec,
        seizure_intervals_sec,
        args.max_windows,
        args.seed,
    )
    res_pre = run_pipeline(
        raw_preprocessed,
        "preprocessed",
        args.win_sec,
        args.step_sec,
        seizure_intervals_sec,
        args.max_windows,
        args.seed,
    )

    print("Results")
    for r in (res_raw, res_pre):
        print(
            f"- {r['name']}: acc={r['acc']:.3f} windows={r['n_win']} pos={r['n_pos']} neg={r['n_neg']} sfreq={r['sfreq']}"
        )


if __name__ == "__main__":
    main()
