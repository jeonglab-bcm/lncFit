"""L2-regularized logistic regression baseline (fast linear model)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("logreg")
class LogRegClassifier(ClassifierModel):
    """L2 logistic regression with balanced class weights.

    A linear reference point for what the k-mer + cell-line features can support
    without a nonlinear model. ``class_weight="balanced"`` handles the ~5%
    positive rate (the linear analogue of XGBoost's scale_pos_weight).
    """

    model_type = "logreg"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 42, **params) -> None:
        super().__init__(C=C, max_iter=max_iter, seed=seed, **params)
        self._model: LogisticRegression | None = None

    def fit(self, X, y) -> "LogRegClassifier":
        self._model = LogisticRegression(
            C=self.params["C"],
            max_iter=self.params["max_iter"],
            class_weight="balanced",
            random_state=self.params["seed"],
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        return self._model.predict_proba(X)[:, 1]
