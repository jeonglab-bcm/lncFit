"""XGBoost gradient-boosted-trees classifier wrapper."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("xgboost")
class XGBoostClassifier(ClassifierModel):
    """XGBoost binary classifier with automatic scale_pos_weight.

    Defaults match scripts/train_lncrna_xgboost.py (hist tree method,
    binary:logistic, aucpr eval metric). scale_pos_weight is computed from the
    training label balance inside fit() unless passed explicitly, so callers
    never have to precompute it.
    """

    model_type = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        seed: int = 42,
        nthread: int = -1,
        **params,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            seed=seed,
            nthread=nthread,
            **params,
        )
        self._model: xgb.XGBClassifier | None = None

    def fit(self, X, y) -> "XGBoostClassifier":
        spw = self.params["scale_pos_weight"]
        if spw is None:
            n_pos = int(np.sum(y))
            n_neg = len(y) - n_pos
            spw = n_neg / n_pos if n_pos > 0 else 1.0
        self._model = xgb.XGBClassifier(
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            max_depth=self.params["max_depth"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            nthread=self.params["nthread"],
            random_state=self.params["seed"],
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        return self._model.predict_proba(X)[:, 1]
