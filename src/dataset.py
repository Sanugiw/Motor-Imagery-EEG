"""
Dataset Loader for BCI Competition IV Dataset IIb (Dataset 3 in the paper).
Loads GDF files (B0103T.gdf ... B0903T.gdf), extracts EEG channels, cue events, and single trials.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import mne


EEG_CHANNELS = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
CUE_LEFT_CODE = '769'   # Class 1: Left hand
CUE_RIGHT_CODE = '770'  # Class 2: Right hand
TRIAL_START_CODE = '768'


def load_bci_comp4_dataset_2b(
    gdf_path: str,
    eeg_channels: Optional[List[str]] = None,
    t_min: float = 0.5,
    t_max: float = 2.5,
    t_epoch_min: float = -1.0,
    t_epoch_max: float = 4.0
) -> Dict[str, Any]:
    """
    Load a BCI Competition IV Dataset IIb session GDF file (e.g. B0103T.gdf).
    Extracts raw continuous data, events, and single trials segmented according to the paper.

    Parameters
    ----------
    gdf_path : str
        Path to the .gdf file.
    eeg_channels : list of str, optional
        EEG channels to keep (defaults to ['EEG:C3', 'EEG:Cz', 'EEG:C4']).
    t_min : float
        Start time in seconds relative to cue onset (default 0.5 s, from paper).
    t_max : float
        End time in seconds relative to cue onset (default 2.5 s, from paper).
    t_epoch_min : float
        Start time for full inspection epoch relative to cue (default -1.0 s).
    t_epoch_max : float
        End time for full inspection epoch relative to cue (default 4.0 s).

    Returns
    -------
    dict
        Structured dictionary containing continuous data, events, segmented trials,
        and train/test splits.
    """
    if not os.path.exists(gdf_path):
        raise FileNotFoundError(f"GDF file not found: {gdf_path}")

    if eeg_channels is None:
        eeg_channels = EEG_CHANNELS

    # Load raw GDF file
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    available_chs = [ch for ch in eeg_channels if ch in raw.ch_names]
    if len(available_chs) == 0:
        # Fallback to first 3 channels if names differ
        available_chs = raw.ch_names[:3]

    raw.pick(available_chs)
    fs = float(raw.info['sfreq'])
    continuous_raw = raw.get_data()  # shape: (n_channels, n_samples)

    # Extract annotations and events
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    cue_left_id = event_id.get(CUE_LEFT_CODE)
    cue_right_id = event_id.get(CUE_RIGHT_CODE)

    if cue_left_id is None or cue_right_id is None:
        raise ValueError(f"Could not find cue annotations in {gdf_path}. Found: {list(event_id.keys())}")

    # Extract cue events chronologically
    cue_mask = (events[:, 2] == cue_left_id) | (events[:, 2] == cue_right_id)
    cue_events = events[cue_mask]

    # Convert to standard labels: 1 for Left Hand, 2 for Right Hand
    labels = np.array([1 if e[2] == cue_left_id else 2 for e in cue_events], dtype=int)

    # Sample offsets relative to cue onset
    start_samp = int(round(t_min * fs))
    end_samp = int(round(t_max * fs))
    epoch_start_samp = int(round(t_epoch_min * fs))
    epoch_end_samp = int(round(t_epoch_max * fs))

    trials_segmented_raw = []
    trials_epoch_raw = []

    for onset, _, _ in cue_events:
        # Segmented task window: [0.5, 2.5] s
        s_idx = onset + start_samp
        e_idx = onset + end_samp
        trials_segmented_raw.append(continuous_raw[:, s_idx:e_idx])

        # Full inspection epoch: [-1.0, 4.0] s
        es_idx = onset + epoch_start_samp
        ee_idx = onset + epoch_end_samp
        trials_epoch_raw.append(continuous_raw[:, es_idx:ee_idx])

    trials_segmented_raw = np.array(trials_segmented_raw, dtype=np.float64)  # (160, 3, 500)
    trials_epoch_raw = np.array(trials_epoch_raw, dtype=np.float64)          # (160, 3, 1250)

    # Time vectors relative to cue
    time_segmented = np.linspace(t_min, t_max, trials_segmented_raw.shape[-1], endpoint=False)
    time_epoch = np.linspace(t_epoch_min, t_epoch_max, trials_epoch_raw.shape[-1], endpoint=False)

    # Clean channel names (e.g. 'EEG:C3' -> 'C3')
    clean_channel_names = [ch.replace('EEG:', '').strip() for ch in available_chs]
    subject_id = os.path.basename(gdf_path)[:3]

    # Train / Test split: 80 trials for training, 80 trials for testing (as stated in the paper)
    n_train = 80
    X_train_raw = trials_segmented_raw[:n_train]
    y_train = labels[:n_train]
    X_test_raw = trials_segmented_raw[n_train:]
    y_test = labels[n_train:]

    return {
        'subject_id': subject_id,
        'gdf_filename': os.path.basename(gdf_path),
        'sample_rate': fs,
        'channel_names': clean_channel_names,
        'raw_continuous': continuous_raw,
        'cue_sample_indices': cue_events[:, 0],
        'labels_all': labels,
        'class_names': ['Left Hand', 'Right Hand'],
        'time_segmented': time_segmented,
        'time_epoch': time_epoch,
        'trials_segmented_raw': trials_segmented_raw,
        'trials_epoch_raw': trials_epoch_raw,
        'X_train_raw': X_train_raw,
        'y_train': y_train,
        'X_test_raw': X_test_raw,
        'y_test': y_test
    }
