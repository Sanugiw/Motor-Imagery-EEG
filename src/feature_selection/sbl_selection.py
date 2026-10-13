"""Sparse Bayesian learning (ARD regression) weights for the SBLFB baseline."""

import numpy as np
from sklearn.preprocessing import StandardScaler


class SparseBayesianSelector:
    """Type-II maximum-likelihood ARD regression for binary class targets."""

    def __init__(self, max_iter: int = 500, tol: float = 1e-5, prune_alpha: float = 1e5):
        self.max_iter, self.tol = max_iter, tol
        # Pruned features settle near alpha ~ 1e7-1e8 and kept ones below ~1e3, so the
        # cut-off must sit in that gap; 1e8 kept almost every feature.
        self.prune_alpha = prune_alpha
        self.scaler_ = StandardScaler()
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(y).reshape(-1)
        target = np.where(labels == np.unique(labels)[0], -1.0, 1.0)
        self.scaler_.fit(X)
        X = self.scaler_.transform(X)
        n, p = X.shape
        alpha = np.ones(p)
        beta = 1.0 / max(float(np.var(target)), 1e-8)
        xtx, xty = X.T @ X, X.T @ target
        for _ in range(self.max_iter):
            precision = np.diag(alpha) + beta * xtx
            covariance = np.linalg.pinv(precision, hermitian=True)
            mean = beta * covariance @ xty
            gamma = np.clip(1.0 - alpha * np.diag(covariance), 0.0, 1.0)
            next_alpha = gamma / np.maximum(mean**2, 1e-12)
            residual = target - X @ mean
            next_beta = max((n - gamma.sum()) / max(float(residual @ residual), 1e-12), 1e-8)
            if np.max(np.abs(np.log(next_alpha + 1e-12) - np.log(alpha + 1e-12))) < self.tol:
                alpha, beta = next_alpha, next_beta
                break
            alpha, beta = np.clip(next_alpha, 1e-12, 1e12), next_beta
        self.coef_ = mean
        self.alpha_ = alpha
        return self

    def transform(self, X):
        if self.coef_ is None:
            raise RuntimeError("Call fit before transform.")
        return self.scaler_.transform(np.asarray(X, dtype=float))

    def get_support(self):
        return self.alpha_ < self.prune_alpha
