"""Balanced bagging: an ensemble of trees each fit on a class-balanced subsample."""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("balanced_bagging")
class BalancedBaggingClassifier(ClassifierModel):
    """Bag of trees, each fit on all positives + an equal-size majority subsample.

    A different family of imbalance handling than reweighting the loss (which is
    what ``randomforest``, ``logreg``, and ``histgb`` do here): instead of telling
    one model to care more about rare positives, this builds many models that each
    see a *balanced* problem, then averages them. Every estimator gets all n_pos
    positives plus ``sampling_ratio * n_pos`` majority rows drawn without
    replacement, so across enough estimators most of the majority class is still
    seen -- unlike a single undersampled fit, which throws ~95% of the data away
    and keeps only one draw of it.

    This is the classic RUSBoost/EasyEnsemble idea (Liu et al. 2009), implemented
    directly on sklearn primitives rather than pulling in imbalanced-learn for one
    estimator.

    sampling_ratio: majority:minority ratio per estimator. 1.0 = fully balanced;
    higher keeps more of the majority class per fit (less aggressive undersampling,
    each estimator closer to the real base rate).
    """

    model_type = "balanced_bagging"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 9,
        min_samples_leaf: int = 2,
        sampling_ratio: float = 1.0,
        max_features: str | float | None = "sqrt",
        seed: int = 42,
        **params,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            sampling_ratio=sampling_ratio,
            max_features=max_features,
            seed=seed,
            **params,
        )
        self._models: list[DecisionTreeClassifier] = []

    def fit(self, X, y) -> "BalancedBaggingClassifier":
        p = self.params
        y = np.asarray(y).astype(int)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_majority = min(len(neg_idx), int(round(p["sampling_ratio"] * len(pos_idx))))

        rng = np.random.default_rng(p["seed"])
        self._models = []
        for i in range(p["n_estimators"]):
            sampled_neg = rng.choice(neg_idx, size=n_majority, replace=False)
            rows = np.concatenate([pos_idx, sampled_neg])
            tree = DecisionTreeClassifier(
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                max_features=p["max_features"],
                # Vary the tree's own randomness per estimator too, so estimators
                # differ by more than just which majority rows they happened to see.
                random_state=p["seed"] + i,
            )
            tree.fit(X[rows], y[rows])
            self._models.append(tree)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._models, "call fit() before predict_proba()"
        # Mean of per-tree positive-class probability -- the usual bagging average.
        return np.mean([m.predict_proba(X)[:, 1] for m in self._models], axis=0)
