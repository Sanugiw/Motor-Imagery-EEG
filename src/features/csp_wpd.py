"""Proposed CSP-WPD feature extraction (CSP first, then db4 WPD)."""

import numpy as np
import pywt

from .csp_features import CSPFeatures
from .csp_wavelet import mostly_inside


def selected_packet_indices(fs: float, level: int, low: float = 8.0, high: float = 30.0):
    width = fs / (2 ** (level + 1))
    return [i for i in range(2**level)
            if mostly_inside(i * width, (i + 1) * width, low, high)]


def extract_csp_wpd_features(projected: np.ndarray, fs: float, level: int):
    selected = selected_packet_indices(fs, level)
    rows = []
    for trial in projected:
        features = []
        for channel in trial:
            packet = pywt.WaveletPacket(data=channel, wavelet="db4", mode="symmetric", maxlevel=level)
            for index, node in enumerate(packet.get_level(level, order="freq")):
                if index in selected:
                    coefficients = node.data
                    features.extend((float(np.sum(np.square(coefficients))),
                                     float(np.std(coefficients, ddof=1))))
        rows.append(features)
    return np.asarray(rows, dtype=np.float64)


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    level = 3 if fs <= 100 else 4
    csp = CSPFeatures(n_pairs).fit(X_train, y_train)
    return (extract_csp_wpd_features(csp.project(X_train), fs, level),
            extract_csp_wpd_features(csp.project(X_test), fs, level), csp)
