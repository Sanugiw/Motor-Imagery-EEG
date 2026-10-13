"""Proposed CSP-Wavelet feature extraction (CSP first, then db4 DWT)."""

import numpy as np
import pywt

from .csp_features import CSPFeatures


def mostly_inside(lo: float, hi: float, low: float = 8.0, high: float = 30.0) -> bool:
    """True if more than half of the band [lo, hi] lies within [low, high].

    Reproduces the paper's sub-band choice (Figs. 3-4): at 100 Hz the DWT keeps
    D3/D2 (6.25-25 Hz) but not D1 (25-50 Hz), and WPD keeps S(3,1)-S(3,4); at
    250/256 Hz the DWT keeps D4/D3 and WPD keeps S(4,1)-S(4,3).
    """
    return (min(hi, high) - max(lo, low)) > 0.5 * (hi - lo)


def selected_dwt_indices(fs: float, level: int, low: float = 8.0, high: float = 30.0):
    """Return wavedec coefficient indices of the sub-bands used for features."""
    bands = [(0.0, fs / (2 ** (level + 1)))]
    bands.extend((fs / (2 ** (j + 1)), fs / (2**j)) for j in range(level, 0, -1))
    return [i for i, (lo, hi) in enumerate(bands) if mostly_inside(lo, hi, low, high)]


def extract_csp_wavelet_features(projected: np.ndarray, fs: float, level: int):
    selected = selected_dwt_indices(fs, level)
    rows = []
    for trial in projected:
        features = []
        for channel in trial:
            coeffs = pywt.wavedec(channel, "db4", level=level, mode="symmetric")
            for i in selected:
                c = coeffs[i]
                features.extend((float(np.sum(np.square(c))), float(np.std(c, ddof=1))))
        rows.append(features)
    return np.asarray(rows, dtype=np.float64)


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    level = 3 if fs <= 100 else 4
    csp = CSPFeatures(n_pairs).fit(X_train, y_train)
    return (extract_csp_wavelet_features(csp.project(X_train), fs, level),
            extract_csp_wavelet_features(csp.project(X_test), fs, level), csp)
