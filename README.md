# 🧠 EEG Studio — OpenBCI Cyton+Daisy Signal Processing & Visualization

Complete EEG pipeline: **signal acquisition → preprocessing → hex conversion (FPGA) → interactive web visualization**.

## Features

- **Live streaming** from OpenBCI Cyton+Daisy via BrainFlow (16 channels, 125 Hz)
- **Preprocessing**: bandpass/notch filtering, resampling, normalization, smoothing
- **Artifact detection**: amplitude threshold + flatline detection
- **Web UI**: real-time time-series plots (Plotly.js), PSD with EEG band markers, per-channel statistics
- **Hex conversion**: fixed-point output for FPGA/hardware integration
- **FIR filter analysis**: window comparison (Boxcar, Hamming, Hann, Blackman)
- **Seizure detection**: band-power feature extraction + logistic regression (CHB-MIT / Muse)

## Directory Structure

```
EEG/
├── common/              # Shared preprocessing module
├── raw_data/            # Raw .txt and .csv EEG data files
├── preprocess/          # Offline processing scripts
│   ├── preprocess.py        # Seizure detection pipeline (CHB-MIT / Muse)
│   ├── preprocess_basic.py  # BrainFlow real-time preprocessing + MNE utils
│   ├── readfrom.py          # Live stream → hex window recording
│   ├── hex_convert.py       # Float → fixed-point hex conversion
│   └── testing.py           # FIR filter comparison & visualization
├── software/            # Flask web application
│   ├── app.py               # Backend (REST API)
│   ├── data_loader.py       # OpenBCI/BrainFlow file parsers
│   ├── preprocessing.py     # Scipy-based signal processing engine
│   ├── stream_data.py       # Standalone BrainFlow streaming test
│   ├── run_webapp.sh        # Launch script (auto-opens browser)
│   ├── templates/           # Jinja2 HTML templates
│   └── static/              # CSS + JavaScript frontend
├── recording/           # Hex window data files
├── tests/               # Unit tests (pytest)
├── requirements.txt     # Python dependencies
└── .gitignore
```

## Setup

```bash
# Create virtual environment
python3 -m venv eeg_env
source eeg_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Application
```bash
cd software
python app.py
# Open http://localhost:5000
```
Or use the launcher script:
```bash
bash software/run_webapp.sh
```

### Live Recording (CLI)
```bash
python preprocess/readfrom.py --serial-port /dev/ttyUSB0
```

### Seizure Detection Pipeline
```bash
python preprocess/preprocess.py --edf chb01_03.edf --seizures "2996,3036"
```

### FIR Filter Analysis
```bash
python preprocess/testing.py
```

## Hardware

- **Board**: OpenBCI Cyton + Daisy (16 channels EEG)
- **Sampling rate**: 125 Hz
- **Connection**: USB serial (`/dev/ttyUSB0` on Linux, `COM3` on Windows)
- **Channel mapping**: 10-20 system (Fp1, Fp2, C3, C4, P7, P8, O1, O2, F7, F8, F3, F4, T7, T8, P3, P4)

## ⚠ Note

The web application is a **single-user desktop tool**. Do not deploy behind a multi-worker WSGI server.
