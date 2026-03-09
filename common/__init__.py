# Shared BrainFlow preprocessing functions
# Used by: preprocess/preprocess_basic.py, preprocess/readfrom.py, software/app.py

import numpy as np
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, NoiseTypes


def apply_brainflow_preprocessing(eeg_data: np.ndarray, sampling_rate: int,
                                   l_freq: float = 0.5, h_freq: float = 40.0,
                                   bandpass_order: int = 4,
                                   notch: float = 50.0) -> np.ndarray:
    """
    Apply basic real-time preprocessing to each EEG channel using BrainFlow.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_samples)
    sampling_rate : int
    l_freq : float, bandpass low cutoff Hz
    h_freq : float, bandpass high cutoff Hz
    bandpass_order : int, filter order
    notch : float, mains notch frequency (50, 60, or 0 to disable)

    Returns
    -------
    filtered : ndarray, same shape as eeg_data
    """
    filtered = np.ascontiguousarray(eeg_data.copy(), dtype=np.float64)

    for ch in range(filtered.shape[0]):
        channel = filtered[ch]
        DataFilter.detrend(channel, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(
            channel,
            sampling_rate,
            l_freq,
            h_freq,
            bandpass_order,
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0.0,
        )
        if notch == 50.0:
            DataFilter.remove_environmental_noise(channel, sampling_rate, NoiseTypes.FIFTY.value)
        elif notch == 60.0:
            DataFilter.remove_environmental_noise(channel, sampling_rate, NoiseTypes.SIXTY.value)

    return filtered
