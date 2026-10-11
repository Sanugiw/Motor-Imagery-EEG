"""
Segmentation module for Motor Imagery trials.
Supports:
1. Segmenting trials directly from continuous data using cue onset markers and time windows.
2. Slicing pre-extracted trials to custom temporal sub-windows.
"""

from typing import Tuple, Optional, List
import numpy as np


def extract_trials_from_continuous(
    continuous_data: np.ndarray,
    cue_indices: np.ndarray,
    fs: float = 250.0,
    t_min: float = 0.5,
    t_max: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract single-trial segments from continuous multi-channel signal.

    Parameters
    ----------
    continuous_data : np.ndarray
        Shape (n_channels, n_timepoints).
    cue_indices : np.ndarray
        Sample indices corresponding to motor imagery cue onsets.
    fs : float
        Sampling frequency in Hz (250 Hz for Dataset 3).
    t_min : float
        Start time in seconds relative to cue onset (0.5 s in paper).
    t_max : float
        End time in seconds relative to cue onset (2.5 s in paper).

    Returns
    -------
    trials : np.ndarray
        Shape (n_trials, n_channels, n_samples).
    time_vector : np.ndarray
        Time points in seconds corresponding to the trial window.
    """
    start_offset = int(round(t_min * fs))
    end_offset = int(round(t_max * fs))
    n_samples = end_offset - start_offset

    trials = []
    for onset in cue_indices:
        s = onset + start_offset
        e = onset + end_offset
        if s < 0 or e > continuous_data.shape[-1]:
            raise IndexError(f"Trial window [{s}, {e}] exceeds continuous data bounds [0, {continuous_data.shape[-1]}].")
        trials.append(continuous_data[:, s:e])

    trials_arr = np.array(trials, dtype=np.float64)
    time_vec = np.linspace(t_min, t_max, n_samples, endpoint=False)
    return trials_arr, time_vec


def slice_trial_window(
    trials: np.ndarray,
    time_in_sec: np.ndarray,
    t_start: float,
    t_end: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice pre-extracted trials along the time axis.
    """
    tol = 1e-6
    mask = (time_in_sec >= t_start - tol) & (time_in_sec <= t_end + tol)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No samples found within [{t_start}, {t_end}] s.")
    return trials[:, :, idx], time_in_sec[idx]
