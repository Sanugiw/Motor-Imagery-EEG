"""
Preprocessing module for Motor Imagery EEG Data.
Implements the exact methodology from Zhang et al., Sensors 2020:
1. 6th-order Butterworth 8-30 Hz bandpass filter (zero-phase via sosfiltfilt).
2. Filter bank decomposition: 10 sub-bands (4 Hz bandwidth, 2 Hz overlap) in 8-30 Hz.
3. Baseline removal (demeaning).
4. Continuous filtering prior to trial segmentation to eliminate boundary edge artifacts.
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy import signal


def design_butter_bandpass(
    lowcut: float = 8.0,
    highcut: float = 30.0,
    fs: float = 250.0,
    order: int = 6
) -> np.ndarray:
    """
    Design Butterworth bandpass filter in SOS (second-order sections) format.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz (8.0 Hz in paper).
    highcut : float
        Upper cutoff frequency in Hz (30.0 Hz in paper).
    fs : float
        Sampling frequency in Hz (250.0 Hz for Dataset 3).
    order : int
        Filter order (6th order as specified in Section 2.3 of the paper).

    Returns
    -------
    sos : np.ndarray
        Second-order sections filter representation.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if low <= 0 or high >= 1:
        raise ValueError(f"Cutoff frequencies [{lowcut}, {highcut}] must be within (0, Nyquist={nyq}).")

    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    return sos


def apply_butter_bandpass(
    data: np.ndarray,
    lowcut: float = 8.0,
    highcut: float = 30.0,
    fs: float = 250.0,
    order: int = 6,
    axis: int = -1
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter along the specified axis.
    """
    sos = design_butter_bandpass(lowcut, highcut, fs, order)
    filtered = signal.sosfiltfilt(sos, data, axis=axis)
    return filtered


def remove_baseline(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Remove DC offset by subtracting the mean across the time axis.
    """
    mean_val = np.mean(data, axis=axis, keepdims=True)
    return data - mean_val


def get_filter_bank_subbands(
    start_freq: float = 8.0,
    end_freq: float = 30.0,
    bandwidth: float = 4.0,
    step: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Generate sub-bands for CSP-FB as defined in Section 2.4.2 of the paper:
    10 sub-bands in the range 8-30 Hz with 4 Hz bandwidth and 2 Hz overlap:
    [8-12, 10-14, 12-16, 14-18, 16-20, 18-22, 20-24, 22-26, 24-28, 26-30 Hz].
    """
    bands = []
    current_low = start_freq
    while current_low + bandwidth <= end_freq + 1e-6:
        bands.append((float(current_low), float(current_low + bandwidth)))
        current_low += step
    return bands


def filter_bank_decomposition(
    data: np.ndarray,
    fs: float = 250.0,
    bands: Optional[List[Tuple[float, float]]] = None,
    order: int = 6,
    axis: int = -1
) -> np.ndarray:
    """
    Decompose multi-channel signals into sub-bands using 6th-order Butterworth filters.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples) or (n_trials, n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    bands : list of (float, float), optional
        Sub-bands to filter. Defaults to 10 sub-bands in 8-30 Hz.
    order : int
        Butterworth order (default 6).
    axis : int
        Time axis.

    Returns
    -------
    np.ndarray
        Sub-band signals array with shape (n_subbands, ...).
    """
    if bands is None:
        bands = get_filter_bank_subbands()

    subband_filtered = []
    for low, high in bands:
        filt = apply_butter_bandpass(data, lowcut=low, highcut=high, fs=fs, order=order, axis=axis)
        subband_filtered.append(filt)

    return np.array(subband_filtered, dtype=np.float64)


def preprocess_and_segment_pipeline(
    continuous_raw: np.ndarray,
    cue_indices: np.ndarray,
    fs: float = 250.0,
    t_min: float = 0.5,
    t_max: float = 2.5,
    lowcut: float = 8.0,
    highcut: float = 30.0,
    filter_order: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete end-to-end preprocessing pipeline for Dataset 3:
    1. Filter continuous signals with 6th-order Butterworth 8-30 Hz (avoids edge ringing).
    2. Filter continuous signals into 10 sub-bands for CSP-FB.
    3. Segment single trials [t_min, t_max] = [0.5, 2.5] s relative to cue onset.
    4. Remove trial baseline.

    Returns
    -------
    trials_raw : np.ndarray of shape (n_trials, n_channels, n_samples)
    trials_filt : np.ndarray of shape (n_trials, n_channels, n_samples)
    trials_fb : np.ndarray of shape (10, n_trials, n_channels, n_samples)
    time_vector : np.ndarray of shape (n_samples,)
    """
    # 1. 8-30 Hz Bandpass on continuous data
    continuous_filt = apply_butter_bandpass(
        continuous_raw, lowcut=lowcut, highcut=highcut, fs=fs, order=filter_order, axis=-1
    )

    # 2. 10 Sub-band filter bank on continuous data
    subbands = get_filter_bank_subbands()
    continuous_fb = []
    for low, high in subbands:
        f = apply_butter_bandpass(continuous_raw, lowcut=low, highcut=high, fs=fs, order=filter_order, axis=-1)
        continuous_fb.append(f)
    continuous_fb = np.array(continuous_fb, dtype=np.float64)  # shape: (10, n_channels, n_time)

    # 3. Segment single trials: [0.5, 2.5] s
    start_samp = int(round(t_min * fs))
    end_samp = int(round(t_max * fs))
    n_samples = end_samp - start_samp

    trials_raw = []
    trials_filt = []
    trials_fb = [[] for _ in range(len(subbands))]

    for onset in cue_indices:
        s = onset + start_samp
        e = onset + end_samp
        trials_raw.append(continuous_raw[:, s:e])
        trials_filt.append(continuous_filt[:, s:e])
        for b_idx in range(len(subbands)):
            trials_fb[b_idx].append(continuous_fb[b_idx, :, s:e])

    trials_raw = np.array(trials_raw, dtype=np.float64)
    trials_filt = np.array(trials_filt, dtype=np.float64)
    trials_fb = np.array(trials_fb, dtype=np.float64)  # (10, n_trials, n_channels, n_samples)

    # 4. Remove DC offset from trials
    trials_raw = remove_baseline(trials_raw, axis=-1)
    trials_filt = remove_baseline(trials_filt, axis=-1)
    for b_idx in range(len(subbands)):
        trials_fb[b_idx] = remove_baseline(trials_fb[b_idx], axis=-1)

    time_vector = np.linspace(t_min, t_max, n_samples, endpoint=False)
    return trials_raw, trials_filt, trials_fb, time_vector
