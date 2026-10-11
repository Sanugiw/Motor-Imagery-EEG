"""Traditional Wavelet-CSP: db4 DWT on each EEG channel before CSP."""

import numpy as np
import pywt

from .csp_features import CSPFeatures
from .csp_wavelet import selected_dwt_indices


def reconstruct_selected_wavelet_channels(X, fs: float):
    level = 3 if fs <= 100 else 4
    selected = selected_dwt_indices(fs, level)
    output = []
    for trial in np.asarray(X, dtype=float):
        channels = []
        for signal in trial:
            coeffs = pywt.wavedec(signal, "db4", level=level, mode="symmetric")
            for idx in selected:
                keep = [np.zeros_like(c) for c in coeffs]
                keep[idx] = coeffs[idx]
                reconstructed = pywt.waverec(keep, "db4", mode="symmetric")[:signal.size]
                channels.append(reconstructed)
        output.append(channels)
    return np.asarray(output)


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    train_channels = reconstruct_selected_wavelet_channels(X_train, fs)
    test_channels = reconstruct_selected_wavelet_channels(X_test, fs)
    model = CSPFeatures(n_pairs).fit(train_channels, y_train)
    return model.transform(train_channels), model.transform(test_channels), model
