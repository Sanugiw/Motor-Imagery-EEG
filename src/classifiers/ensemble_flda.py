"""Threshold ensemble of FLDA models used in the paper's secondary selection."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


class ThresholdEnsembleFLDA:
    """Fit FLDA models for |weight| thresholds 0, .1, ..., .8.

    As in the paper, ``paper_score`` is the maximum test accuracy among the
    threshold models. The per-threshold scores are retained in output so that
    this paper-specific scoring rule is visible.
    """

    def __init__(self, thresholds=None, make_model=None):
        self.thresholds = np.asarray(thresholds if thresholds is not None
                                    else np.arange(0.0, 0.81, 0.1))
        # Classifier factory; FLDA by default, swappable for the Table 9 comparison.
        self.make_model = make_model or (lambda: LinearDiscriminantAnalysis(solver="svd"))

    def fit(self, X, y, weights):
        X, y = np.asarray(X, float), np.asarray(y).reshape(-1)
        self.models_ = []
        for threshold in self.thresholds:
            selected = np.flatnonzero(np.abs(weights) > threshold)
            if not len(selected):
                self.models_.append((selected, None))
                continue
            model = self.make_model()
            model.fit(X[:, selected], y)
            self.models_.append((selected, model))
        return self

    def evaluate(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y).reshape(-1)
        rows = []
        for threshold, (selected, model) in zip(self.thresholds, self.models_):
            acc = 0.0 if model is None else float(accuracy_score(y, model.predict(X[:, selected])))
            rows.append(
                {
                    "threshold": float(threshold),
                    "n_features": len(selected),
                    "accuracy": acc,
                }
            )
        best = max(rows, key=lambda row: row["accuracy"])
        return {"paper_score": best["accuracy"], "best_threshold": best["threshold"],
                "threshold_results": rows}
