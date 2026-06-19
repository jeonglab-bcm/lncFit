"""Simple model wrappers for benchmarking against XGBoost."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, Ridge


class MeanPredictor:
    """Null baseline: always predicts the training-set mean."""

    _mean: float = 0.0

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._mean, dtype=np.float32)


class RidgeModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._model: Ridge | None = None

    def fit(self, X, y):
        self._model = Ridge(alpha=self.alpha, max_iter=10000)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.float32)


class LassoModel:
    def __init__(self, alpha: float = 0.001):
        self.alpha = alpha
        self._model: Lasso | None = None

    def fit(self, X, y):
        self._model = Lasso(alpha=self.alpha, max_iter=10000)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X).astype(np.float32)
