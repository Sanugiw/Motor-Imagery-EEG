"""Non-convex LOG selector using the paper's iterative log-thresholding."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .lasso_selection import lambda_grid


def log_thresholding(X, y, lam: float, a: float = 0.001,
                     max_iter: int = 5000, tol: float = 1e-7):
    """Solve 0.5||y-Xw||^2 + lam*log(1+||w||_1/a), as Eq. 12-17."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    lipschitz = float(np.linalg.norm(X.T @ X, ord=2))
    if lipschitz <= np.finfo(float).eps:
        return np.zeros(X.shape[1], dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    xty = X.T @ y
    xtx = X.T @ X
    for _ in range(max_iter):
        v = w - (xtx @ w - xty) / lipschitz
        av = np.abs(v)
        radicand = np.maximum((a - av) ** 2 + 4.0 * np.maximum(a * av - lam / lipschitz, 0.0), 0.0)
        next_w = np.sign(v) * 0.5 * (av - a + np.sqrt(radicand))
        if np.linalg.norm(next_w - w) <= tol * max(1.0, np.linalg.norm(w)):
            w = next_w
            break
        w = next_w
    return w


class LogFeatureSelector:
    def __init__(self, n_splits: int = 10, random_state: int = 42,
                 a: float = 0.001, max_iter: int = 5000):
        self.n_splits, self.random_state = n_splits, random_state
        self.a, self.max_iter = a, max_iter
        self.scaler_ = StandardScaler()
        self.coef_ = None
        self.best_lambda_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        labels = np.asarray(y).reshape(-1)
        encoded = np.where(labels == np.unique(labels)[0], -1.0, 1.0)
        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        errors = []
        for lam in lambda_grid():
            losses = []
            for tr, va in folds.split(X, labels):
                scaler = StandardScaler().fit(X[tr])
                xtr = scaler.transform(X[tr])
                # Fit-fold class coding follows the global binary label mapping.
                w = log_thresholding(xtr, encoded[tr], lam, self.a, self.max_iter)
                residual = encoded[va] - scaler.transform(X[va]) @ w
                losses.append(float(np.mean(residual**2)))
            errors.append(np.mean(losses))
        self.best_lambda_ = float(lambda_grid()[int(np.argmin(errors))])
        scaled = self.scaler_.fit_transform(X)
        self.coef_ = log_thresholding(scaled, encoded, self.best_lambda_, self.a, self.max_iter)
        return self

    def transform(self, X):
        if self.coef_ is None:
            raise RuntimeError("Call fit before transform.")
        return self.scaler_.transform(np.asarray(X, dtype=float))

    def get_support(self):
        return np.abs(self.coef_) > np.finfo(float).eps
