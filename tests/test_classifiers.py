import numpy as np
from scipy.sparse import csr_matrix

from lncfit.classifiers import CLASSIFIER_REGISTRY, build_classifier


def _toy_data(seed=0, n_train=200, n_test=60, n_feat=8, pos_rate=0.1):
    rng = np.random.default_rng(seed)
    X_tr = rng.random((n_train, n_feat))
    y_tr = (rng.random(n_train) < pos_rate).astype(int)
    X_te = rng.random((n_test, n_feat))
    return X_tr, y_tr, X_te


def test_classifier_wrappers_share_the_fit_predict_contract():
    X_tr, y_tr, X_te = _toy_data()
    for name in ["null", "logreg", "xgboost", "randomforest", "knn", "mlp",
                 "histgb", "balanced_bagging"]:
        # xgboost's scale_pos_weight is left at its default (None -> auto-computed
        # from y inside fit()), so this covers that path too.
        model = build_classifier(name)
        assert model.fit(X_tr, y_tr) is model
        proba = model.predict_proba(X_te)
        assert proba.shape == (X_te.shape[0],)
        assert proba.min() >= 0.0 and proba.max() <= 1.0
        if name == "null":
            assert np.allclose(proba, y_tr.mean())  # predicts exactly the training base rate

    # The one proven bug class in this codebase: XGBoost treats sparse implicit
    # zeros as *missing*, so dense/sparse must give identical shaped output.
    dense = build_classifier("xgboost").fit(X_tr, y_tr).predict_proba(X_te)
    sparse = build_classifier("xgboost").fit(csr_matrix(X_tr), y_tr).predict_proba(csr_matrix(X_te))
    assert dense.shape == sparse.shape

    # A mismatch here wouldn't crash anything -- it would silently mislabel
    # "model" in every run_info.json this project's tuning scripts write.
    for name, cls in CLASSIFIER_REGISTRY.items():
        assert cls.model_type == name


def test_imbalance_aware_models_actually_predict_positives():
    """The point of histgb/balanced_bagging: unlike the unweighted xgboost config,
    they should cross the 0.5 threshold on a rare-positive problem rather than
    predicting everything negative (which is what makes xgboost's F1 exactly 0)."""
    X_tr, y_tr, X_te = _toy_data(pos_rate=0.05, n_train=400)
    for name in ["histgb", "balanced_bagging"]:
        proba = build_classifier(name).fit(X_tr, y_tr).predict_proba(X_te)
        assert (proba >= 0.5).any(), f"{name} never predicts the positive class"


def test_balanced_bagging_sampling_ratio_controls_majority_subsample():
    # sampling_ratio is the knob that makes this "balanced" -- verify it's wired
    # through and that a larger ratio still fits (doesn't over-request rows).
    X_tr, y_tr, X_te = _toy_data(pos_rate=0.05, n_train=400)
    for ratio in [1.0, 3.0]:
        model = build_classifier("balanced_bagging", n_estimators=5, sampling_ratio=ratio)
        proba = model.fit(X_tr, y_tr).predict_proba(X_te)
        assert proba.shape == (X_te.shape[0],)
        assert model.params["sampling_ratio"] == ratio
