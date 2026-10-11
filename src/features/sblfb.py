"""SBLFB benchmark: 17 raw-signal subbands, CSP and sparse Bayesian selection."""

from .sfbcsp import extract_filterbank_csp_features
from ..feature_selection.sbl_selection import SparseBayesianSelector


def extract_train_test(X_train, y_train, X_test, fs: float, n_pairs: int = 3):
    train, test = extract_filterbank_csp_features(X_train, y_train, X_test, fs, n_pairs)
    selector = SparseBayesianSelector().fit(train, y_train)
    return selector.transform(train), selector.transform(test), selector
