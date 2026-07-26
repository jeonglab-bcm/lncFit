"""Support vector machine classifier -- max-margin, kernel-based.

A genuinely different model family from the rest of the registry, which is
otherwise all trees (xgboost / randomforest / histgb / balanced_bagging) plus
one linear model (logreg) and one small net (mlp).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


@register_classifier("svm")
class SVMClassifier(ClassifierModel):
    """SVM with ``class_weight="balanced"`` and mandatory feature standardization.

    **Standardization is not optional here.** This project's feature matrices mix
    wildly different scales -- on the dnabert2 + celligner70 + distance recipe the
    per-column standard deviations run ~0.019 for embedding dims, ~17 for Celligner
    PCA columns, and ~197,000 for distance-to-gene. An unscaled RBF kernel is a
    function of Euclidean distance, so it would be almost entirely determined by
    that one distance column. A StandardScaler is therefore baked into the wrapper
    rather than left to the caller. (Sparse input is scaled with ``with_mean=False``
    so it isn't densified.)

    Probability output: by default ``predict_proba`` returns
    ``sigmoid(decision_function)``, which is a *monotone* transform of the SVM's
    margin. That leaves AUROC/AUPRC exactly unchanged (both are rank-based) and
    conveniently maps the SVM's own decision boundary, ``decision_function == 0``,
    to precisely 0.5 -- so a 0.5-threshold F1 means "on the positive side of the
    margin". These are NOT calibrated probabilities. Pass ``calibrate=True`` for
    real Platt/isotonic-style calibration via cross-validation, at roughly 5x the
    fit cost.

    Cost note: exact ``kernel="rbf"`` is ~quadratic in rows -- measured ~198s for a
    single fit on the full 25,010 x 844 recipe, which makes CV folds and tuning
    trials painful. Two ways out, both measured on that recipe:

    ``kernel_approx=N`` (recommended) replaces the exact kernel with an
    N-component Nystroem feature map plus a linear SVM, which is ~linear rather
    than quadratic in rows. At N=1000 this fit in **3.8s vs 198s (52x faster) and
    scored *better* than the exact kernel** (AUPRC 0.1733 vs 0.1581) -- the
    low-rank approximation evidently regularizes a kernel that was overfitting
    844 correlated dimensions. There is no accuracy/speed tradeoff to agonize over
    here; the approximation simply won.

    ``features.embedding_pca`` cuts the column count instead, which helps exact
    RBF (PCA-64: 31s, AUPRC 0.1610) but is dominated by Nystroem on both axes.

    ``kernel="linear"`` routes to LinearSVC (same objective, liblinear solver,
    ~14s) but performed poorly here (AUPRC 0.0983) -- this problem needs the
    nonlinearity.
    """

    model_type = "svm"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel_approx: int = 0,
        class_weight: str | None = "balanced",
        calibrate: bool = False,
        max_iter: int = -1,
        seed: int = 42,
        **params,
    ) -> None:
        super().__init__(
            kernel=kernel,
            C=C,
            gamma=gamma,
            kernel_approx=kernel_approx,
            class_weight=class_weight,
            calibrate=calibrate,
            max_iter=max_iter,
            seed=seed,
            **params,
        )
        self._model = None
        self._calibrated = False

    def _make_estimator(self):
        p = self.params
        if p["kernel"] != "linear" and p["kernel_approx"]:
            # Nystroem feature map + linear SVM: approximates the kernel in
            # ~linear rather than quadratic time in rows. Returned as a 2-step
            # pipeline so fit()'s scaler still sits in front of it.
            return make_pipeline(
                Nystroem(
                    kernel=p["kernel"],
                    gamma=None if p["gamma"] == "scale" else p["gamma"],
                    n_components=int(p["kernel_approx"]),
                    random_state=p["seed"],
                ),
                LinearSVC(
                    C=p["C"],
                    class_weight=p["class_weight"],
                    dual="auto",
                    max_iter=3000 if p["max_iter"] == -1 else p["max_iter"],
                    random_state=p["seed"],
                ),
            )
        if p["kernel"] == "linear":
            # Same max-margin objective, much faster solver at this row count.
            return LinearSVC(
                C=p["C"],
                class_weight=p["class_weight"],
                dual="auto",
                max_iter=3000 if p["max_iter"] == -1 else p["max_iter"],
                random_state=p["seed"],
            )
        return SVC(
            kernel=p["kernel"],
            C=p["C"],
            gamma=p["gamma"],
            class_weight=p["class_weight"],
            max_iter=p["max_iter"],
            # `probability` is left at its default False -- we read
            # decision_function directly, and the calibrated path wraps this
            # estimator in CalibratedClassifierCV. (Passing probability=True is
            # deprecated as of sklearn 1.9 for exactly that reason.)
            random_state=p["seed"],
        )

    def fit(self, X, y) -> "SVMClassifier":
        # with_mean=False keeps sparse input sparse (centering would densify it).
        scaler = StandardScaler(with_mean=not issparse(X))
        estimator = self._make_estimator()
        self._calibrated = bool(self.params["calibrate"])
        if self._calibrated:
            estimator = CalibratedClassifierCV(estimator, cv=3, method="sigmoid")
        self._model = make_pipeline(scaler, estimator)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        if self._calibrated:
            return self._model.predict_proba(X)[:, 1]
        # Monotone squash of the margin: preserves AUROC/AUPRC exactly and puts
        # the SVM's own boundary (decision_function == 0) at 0.5.
        margin = self._model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-margin))
