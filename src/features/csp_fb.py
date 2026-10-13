"""Proposed CSP-FB feature extraction (CSP projection then 10-band bank)."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .csp_features import CSPFeatures


def bands_8_30():
    return [(float(lo), float(lo + 4)) for lo in range(8, 27, 2)]


def extract_csp_fb_features(projected: np.ndarray, fs: float):
    result = []
    for low, high in bands_8_30():
        if high >= fs / 2:
            raise ValueError(f"Filter band {low}-{high} Hz exceeds Nyquist at {fs} Hz.")
        sos = butter(6, [low, high], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, projected, axis=-1)
        var = np.var(filtered, axis=-1, ddof=1)
        result.append(np.log(np.maximum(var, np.finfo(float).tiny)))
    return np.concatenate(result, axis=1)


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    csp = CSPFeatures(n_pairs).fit(X_train, y_train)
    return (extract_csp_fb_features(csp.project(X_train), fs),
            extract_csp_fb_features(csp.project(X_test), fs), csp)
