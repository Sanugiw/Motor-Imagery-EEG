"""Paper-style binary CSP fitting and traditional log-variance features."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


class CSPFeatures:
    """CSP using normalized per-trial covariance (Zhang et al., Eq. 2-5).

    Input trials have shape ``(trials, channels, samples)``. The paper uses
    three filters from each end of the generalized eigenvalue spectrum.
    """

    def __init__(self, n_pairs: int = 3):
        self.n_pairs = n_pairs
        self.classes_: np.ndarray | None = None
        self.filters_: np.ndarray | None = None

    @staticmethod
    def _mean_normalized_covariance(trials: np.ndarray) -> np.ndarray:
        covariances = []
        for trial in trials:
            covariance = trial @ trial.T
            trace = float(np.trace(covariance))
            if trace <= np.finfo(float).eps:
                raise ValueError("CSP received a zero-energy trial.")
            covariances.append(covariance / trace)
        return np.mean(covariances, axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatures":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if X.ndim != 3 or len(X) != len(y):
            raise ValueError("X must be trials x channels x samples and match y.")
        if len(classes) != 2:
            raise ValueError(f"Binary CSP needs two classes, found {classes.tolist()}.")
        c1 = self._mean_normalized_covariance(X[y == classes[0]])
        c2 = self._mean_normalized_covariance(X[y == classes[1]])
        if self.n_pairs > X.shape[1]:
            raise ValueError(f"CSP cannot select {self.n_pairs} filters from {X.shape[1]} channels.")
        # The symmetric generalized eigenproblem is numerically stable and
        # equivalent to C2^-1 C1 for the positive-semidefinite covariances.
        vals, vecs = eigh(c1, c1 + c2, check_finite=False)
        order = np.argsort(vals)[::-1]
        selected = np.r_[order[: self.n_pairs], order[-self.n_pairs :]]
        self.filters_ = vecs[:, selected].T
        self.classes_ = classes
        return self

    def project(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("Call fit before projecting trials.")
        return np.einsum("fc,nct->nft", self.filters_, np.asarray(X, dtype=float))

    def transform(self, X: np.ndarray) -> np.ndarray:
        projected = self.project(X)
        variance = np.var(projected, axis=-1, ddof=1)
        normalized = variance / np.maximum(variance.sum(axis=1, keepdims=True), np.finfo(float).tiny)
        return np.log(np.maximum(normalized, np.finfo(float).tiny))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


def traditional_csp_features(X_train, y_train, X_test, n_pairs: int = 3):
    model = CSPFeatures(n_pairs=n_pairs).fit(X_train, y_train)
    return model.transform(X_train), model.transform(X_test), model
