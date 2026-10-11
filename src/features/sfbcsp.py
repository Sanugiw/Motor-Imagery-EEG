"""SFBCSP benchmark: 17 raw-signal subbands, CSP and LASSO selection."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .csp_features import CSPFeatures
from ..feature_selection.lasso_selection import LassoFeatureSelector


def bands_4_40():
    return [(float(lo), float(lo + 4)) for lo in range(4, 37, 2)]


def extract_filterbank_csp_features(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    train_blocks, test_blocks = [], []
    for low, high in bands_4_40():
        sos = butter(6, [low, high], btype="bandpass", fs=fs, output="sos")
        tr = sosfiltfilt(sos, X_train, axis=-1)
        te = sosfiltfilt(sos, X_test, axis=-1)
        csp = CSPFeatures(n_pairs).fit(tr, y_train)
        train_blocks.append(csp.transform(tr))
        test_blocks.append(csp.transform(te))
    return np.concatenate(train_blocks, axis=1), np.concatenate(test_blocks, axis=1)


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    xtr, xte = extract_filterbank_csp_features(X_train, y_train, X_test, fs, n_pairs)
    selector = LassoFeatureSelector().fit(xtr, y_train)
    return selector.transform(xtr), selector.transform(xte), selector
