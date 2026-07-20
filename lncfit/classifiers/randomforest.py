"""Random forest classifier baseline (bagged trees, nonlinear, no boosting)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("randomforest")
class RandomForestClassifier(ClassifierModel):
    """Bagged decision trees with balanced-subsample class weighting.

    A nonlinear reference point that, unlike XGBoost, doesn't boost on residuals --
    each tree sees an independent bootstrap sample. ``class_weight="balanced_subsample"``
    reweights within each bootstrap draw for the ~5% positive rate.
    """

    model_type = "randomforest"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        seed: int = 42,
        n_jobs: int = -1,
        **params,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
            n_jobs=n_jobs,
            **params,
        )
        self._model: SKRandomForestClassifier | None = None

    def fit(self, X, y) -> "RandomForestClassifier":
        self._model = SKRandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=self.params["seed"],
            n_jobs=self.params["n_jobs"],
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        return self._model.predict_proba(X)[:, 1]
