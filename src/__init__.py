"""
Motor Imagery EEG Preprocessing & Classification Package.
Based on:
'The CSP-Based New Features Plus Non-Convex Log Sparse Feature Selection
for Motor Imagery EEG Classification', Zhang et al., Sensors 2020.
"""

from .dataset3 import load_bci_comp4_dataset_2b
from .segmentation import extract_trials_from_continuous, slice_trial_window
from .preprocess_dataset3 import (
    apply_butter_bandpass,
    design_butter_bandpass,
    remove_baseline,
    filter_bank_decomposition,
    get_filter_bank_subbands,
    preprocess_and_segment_pipeline
)

__all__ = [
    'load_bci_comp4_dataset_2b',
    'extract_trials_from_continuous',
    'slice_trial_window',
    'apply_butter_bandpass',
    'design_butter_bandpass',
    'remove_baseline',
    'filter_bank_decomposition',
    'get_filter_bank_subbands',
    'preprocess_and_segment_pipeline',
]
