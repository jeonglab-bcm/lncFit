"""Histogram-based gradient boosting with built-in balanced class weighting."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as SKHistGradientBoosting

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("histgb")
class HistGradientBoostingClassifier(ClassifierModel):
    """sklearn's histogram-based GBM, class-weighted for the ~4.5% positive rate.

    Aimed at the imbalance specifically: ``class_weight="balanced"`` reweights the
    loss by inverse class frequency, so the minority class isn't drowned out the
    way it is in an unweighted fit. (The project's XGBoost configs leave
    ``scale_pos_weight=1.0``; a direct XGBoost reweighting experiment at the
    natural ratio ~21.35 hurt AUROC/AUPRC noticeably, so "reweighting helps" is
    not a given here -- this wrapper exists to test whether a *different* boosting
    implementation's handling of the same problem lands differently.)

    Also has built-in early stopping on an internal validation split, unlike the
    project's XGBoost wrapper which fits a flat ``n_estimators``.

    Accepts dense input only -- sklearn's HistGradientBoosting does not support
    sparse X, so k-mer features must be built with ``sparse=False`` (which is what
    ``lncfit.pipeline`` already does).
    """

    model_type = "histgb"

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int | None = None,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
        class_weight: str | None = "balanced",
        seed: int = 42,
        **params,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            seed=seed,
            **params,
        )
        self._model: SKHistGradientBoosting | None = None

    def fit(self, X, y) -> "HistGradientBoostingClassifier":
        p = self.params
        self._model = SKHistGradientBoosting(
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            max_iter=p["max_iter"],
            early_stopping=p["early_stopping"],
            validation_fraction=p["validation_fraction"],
            n_iter_no_change=p["n_iter_no_change"],
            class_weight=p["class_weight"],
            random_state=p["seed"],
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        return self._model.predict_proba(X)[:, 1]
