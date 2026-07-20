"""k-nearest-neighbors classifier baseline (no training, distance-weighted vote)."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("knn")
class KNNClassifier(ClassifierModel):
    """Distance-weighted k-nearest-neighbors.

    No parametric decision boundary and no notion of class imbalance to correct for
    directly, so ``weights="distance"`` is used instead of a balanced-class knob --
    it lets close neighbors dominate the vote, which matters when positives are rare
    and mostly outvoted under uniform weighting.
    """

    model_type = "knn"

    def __init__(self, n_neighbors: int = 25, n_jobs: int = -1, **params) -> None:
        super().__init__(n_neighbors=n_neighbors, n_jobs=n_jobs, **params)
        self._model: SKKNeighborsClassifier | None = None

    def fit(self, X, y) -> "KNNClassifier":
        self._model = SKKNeighborsClassifier(
            n_neighbors=self.params["n_neighbors"],
            weights="distance",
            n_jobs=self.params["n_jobs"],
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        return self._model.predict_proba(X)[:, 1]
