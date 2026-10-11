"""LASSO feature weights with the paper's 10-fold lambda grid."""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def lambda_grid():
    return 2.0 ** np.round(np.arange(-5.0, 5.0001, 0.2), 10)


class LassoFeatureSelector:
    def __init__(
        self,
        n_splits: int = 10,
        random_state: int = 42,
        max_iter: int = 1_000_000,
        tol: float = 1e-3,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.scaler_ = StandardScaler()
        self.coef_ = None
        self.best_lambda_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.where(np.asarray(y).reshape(-1) == np.unique(y)[0], -1.0, 1.0)
        splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        lambdas = lambda_grid()
        scores = np.zeros(len(lambdas), dtype=float)
        for train, valid in splitter.split(X, y):
            scaler = StandardScaler().fit(X[train])
            x_train = scaler.transform(X[train])
            x_valid = scaler.transform(X[valid])
            # Traverse the regularization path high-to-low, reusing the prior
            # solution as a warm start for each smaller lambda.
            model = Lasso(
                alpha=lambdas[-1] / len(train),
                max_iter=self.max_iter,
                tol=self.tol,
                warm_start=True,
                selection="random",
                random_state=self.random_state,
            )
            for index in range(len(lambdas) - 1, -1, -1):
                lam = lambdas[index]
                # sklearn uses SSE/(2*n) + alpha*L1; the paper grid is lambda.
                model.alpha = lam / len(train)
                model.fit(x_train, y[train])
                residual = y[valid] - model.predict(x_valid)
                scores[index] += float(np.mean(residual**2))
        scores /= self.n_splits
        self.best_lambda_ = float(lambdas[int(np.argmin(scores))])
        scaled = self.scaler_.fit_transform(X)
        self.model_ = Lasso(
            alpha=self.best_lambda_ / len(X),
            max_iter=self.max_iter,
            tol=self.tol,
            selection="random",
            random_state=self.random_state,
        )
        self.model_.fit(scaled, y)
        self.coef_ = np.asarray(self.model_.coef_)
        return self

    def transform(self, X):
        if self.coef_ is None:
            raise RuntimeError("Call fit before transform.")
        return self.scaler_.transform(np.asarray(X, dtype=float))

    def get_support(self):
        return np.abs(self.coef_) > np.finfo(float).eps
