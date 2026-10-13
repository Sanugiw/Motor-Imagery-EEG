"""Comparison feature selectors for the paper's Table 8: F-score, GA and BPSO.

F-score: features ranked by Fisher score; the subset size is chosen by 10-fold
cross-validated FLDA accuracy on the training set (a wrapper, as in the paper).
GA / BPSO: wrapper search with the settings of Too et al. 2019 (paper ref. [67]):
population 10, 100 iterations, KNN (k=5) on a 20 % hold-out split, fitness
0.99 * error + 0.01 * (selected / total). Each selector exposes ``get_support()``.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def fisher_scores(X, y):
    classes = np.unique(y)
    mean = X.mean(axis=0)
    between = sum(np.sum(y == c) * (X[y == c].mean(axis=0) - mean) ** 2 for c in classes)
    within = sum(np.sum(y == c) * X[y == c].var(axis=0) for c in classes)
    return between / np.maximum(within, np.finfo(float).eps)


class FScoreSelector:
    def __init__(self, n_splits: int = 10, random_state: int = 42):
        self.n_splits, self.random_state = n_splits, random_state

    def fit(self, X, y):
        X = StandardScaler().fit_transform(np.asarray(X, float))
        y = np.asarray(y).reshape(-1)
        order = np.argsort(fisher_scores(X, y))[::-1]
        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        scores = [cross_val_score(LinearDiscriminantAnalysis(solver="svd"), X[:, order[:k]], y, cv=folds).mean()
                  for k in range(1, X.shape[1] + 1)]
        self.support_ = np.zeros(X.shape[1], bool)
        self.support_[order[: int(np.argmax(scores)) + 1]] = True
        return self

    def get_support(self):
        return self.support_


class _WrapperSearch:
    """Shared KNN hold-out fitness for the GA and BPSO searches."""

    def __init__(self, n_agents: int = 10, n_iter: int = 100, k: int = 5,
                 holdout: float = 0.2, alpha: float = 0.99, random_state: int = 42):
        self.n_agents, self.n_iter, self.k = n_agents, n_iter, k
        self.holdout, self.alpha, self.random_state = holdout, alpha, random_state

    def _prepare(self, X, y):
        X = StandardScaler().fit_transform(np.asarray(X, float))
        y = np.asarray(y).reshape(-1)
        self._split = train_test_split(X, y, test_size=self.holdout, stratify=y,
                                       random_state=self.random_state)
        self._cache = {}
        return np.random.default_rng(self.random_state), X.shape[1]

    def _fitness(self, mask):
        key = mask.tobytes()
        if key not in self._cache:
            if not mask.any():
                self._cache[key] = 1.0
            else:
                xtr, xva, ytr, yva = self._split
                knn = KNeighborsClassifier(self.k).fit(xtr[:, mask], ytr)
                error = 1.0 - knn.score(xva[:, mask], yva)
                self._cache[key] = self.alpha * error + (1 - self.alpha) * mask.mean()
        return self._cache[key]

    def get_support(self):
        return self.support_


class GASelector(_WrapperSearch):
    """Binary genetic algorithm: roulette selection, one-point crossover, bit-flip mutation."""

    def __init__(self, crossover: float = 0.8, mutation: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.crossover, self.mutation = crossover, mutation

    def fit(self, X, y):
        rng, dim = self._prepare(X, y)
        pop = rng.random((self.n_agents, dim)) > 0.5
        fit = np.array([self._fitness(p) for p in pop])
        best, best_fit = pop[np.argmin(fit)].copy(), fit.min()
        for _ in range(self.n_iter):
            weights = 1.0 / (1.0 + fit)
            prob = weights / weights.sum()
            children = []
            for _ in range(self.n_agents // 2):
                a, b = pop[rng.choice(self.n_agents, p=prob)], pop[rng.choice(self.n_agents, p=prob)]
                if rng.random() < self.crossover and dim > 1:
                    cut = rng.integers(1, dim)
                    a, b = np.r_[a[:cut], b[cut:]], np.r_[b[:cut], a[cut:]]
                children += [a ^ (rng.random(dim) < self.mutation), b ^ (rng.random(dim) < self.mutation)]
            merged = np.vstack([pop, children])
            merged_fit = np.r_[fit, [self._fitness(c) for c in children]]
            keep = np.argsort(merged_fit)[: self.n_agents]  # elitist replacement
            pop, fit = merged[keep], merged_fit[keep]
            if fit[0] < best_fit:
                best, best_fit = pop[0].copy(), fit[0]
        self.support_ = best
        return self


class BPSOSelector(_WrapperSearch):
    """Binary particle swarm optimisation with a sigmoid transfer function."""

    def __init__(self, c1: float = 2.0, c2: float = 2.0, w_max: float = 0.9, w_min: float = 0.4,
                 v_max: float = 6.0, **kwargs):
        super().__init__(**kwargs)
        self.c1, self.c2, self.w_max, self.w_min, self.v_max = c1, c2, w_max, w_min, v_max

    def fit(self, X, y):
        rng, dim = self._prepare(X, y)
        pos = rng.random((self.n_agents, dim)) > 0.5
        vel = np.zeros((self.n_agents, dim))
        fit = np.array([self._fitness(p) for p in pos])
        pbest, pbest_fit = pos.copy(), fit.copy()
        g = int(np.argmin(fit))
        gbest, gbest_fit = pos[g].copy(), fit[g]
        for t in range(self.n_iter):
            inertia = self.w_max - (self.w_max - self.w_min) * t / max(self.n_iter - 1, 1)
            r1, r2 = rng.random((2, self.n_agents, dim))
            vel = (inertia * vel + self.c1 * r1 * (pbest.astype(float) - pos)
                   + self.c2 * r2 * (gbest.astype(float) - pos))
            vel = np.clip(vel, -self.v_max, self.v_max)
            pos = rng.random((self.n_agents, dim)) < 1.0 / (1.0 + np.exp(-vel))
            fit = np.array([self._fitness(p) for p in pos])
            improved = fit < pbest_fit
            pbest[improved], pbest_fit[improved] = pos[improved], fit[improved]
            g = int(np.argmin(pbest_fit))
            if pbest_fit[g] < gbest_fit:
                gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]
        self.support_ = gbest
        return self
