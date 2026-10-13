"""Bayesian linear discriminant analysis (Hoffmann et al. 2008, paper ref. [69]).

BLDA is Bayesian linear regression onto +/-1 class targets, with the prior and
noise precisions estimated from the training data by evidence maximisation;
scikit-learn's BayesianRidge implements exactly that regression.
"""

import numpy as np
from sklearn.linear_model import BayesianRidge


class BLDA:
    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        target = np.where(y == self.classes_[1], 1.0, -1.0)
        self.model_ = BayesianRidge().fit(np.asarray(X, float), target)
        return self

    def predict(self, X):
        return np.where(self.model_.predict(np.asarray(X, float)) > 0, self.classes_[1], self.classes_[0])
