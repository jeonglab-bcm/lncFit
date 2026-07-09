"""Null baseline: predicts the training-set positive rate for every row."""
from __future__ import annotations

import numpy as np

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("null")
class NullClassifier(ClassifierModel):
    """Constant-prior baseline.

    Predicts P(positive) = training-set base rate for every test row. Gives
    AUROC 0.5 and AUPRC == base rate by construction — the floor any real model
    must clear (mirrors MeanPredictor in lncfit.models for the regression side).
    """

    model_type = "null"

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._rate: float = 0.0

    def fit(self, X, y) -> "NullClassifier":
        self._rate = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.full(X.shape[0], self._rate, dtype=np.float64)
