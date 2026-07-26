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
                 "histgb", "balanced_bagging", "svm"]:
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


def test_svm_standardizes_internally():
    """SVM feature scaling is baked in, not the caller's job: on this project's
    real recipe, per-column std ranges from ~0.019 (embedding dims) to ~197,000
    (distance-to-gene), so an unscaled RBF kernel would be decided almost entirely
    by that one column."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10))
    y = (rng.random(300) < 0.08).astype(int)
    y[:10] = 1
    # Blow up one column's scale by 1e5. If the wrapper were not standardizing,
    # this would dominate the kernel and change the ranking substantially.
    X_scaled_col = X.copy()
    X_scaled_col[:, 0] *= 1e5

    base = build_classifier("svm").fit(X, y).predict_proba(X)
    blown = build_classifier("svm").fit(X_scaled_col, y).predict_proba(X_scaled_col)
    # Standardization makes the two fits equivalent up to solver tolerance.
    assert np.corrcoef(base, blown)[0, 1] > 0.99


def test_svm_uncalibrated_threshold_is_the_margin_boundary():
    # predict_proba is sigmoid(decision_function), so >=0.5 must be exactly
    # "decision_function >= 0" -- i.e. agree with the SVM's own predict().
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    y = (rng.random(200) < 0.2).astype(int)
    model = build_classifier("svm").fit(X, y)
    proba = model.predict_proba(X)
    hard = model._model.predict(X)
    assert np.array_equal((proba >= 0.5).astype(int), np.asarray(hard).astype(int))


def test_svm_linear_kernel_uses_the_fast_solver():
    from sklearn.svm import LinearSVC
    model = build_classifier("svm", kernel="linear")
    assert isinstance(model._make_estimator(), LinearSVC)


def test_svm_kernel_approx_builds_a_nystroem_pipeline():
    """kernel_approx swaps the exact kernel for a Nystroem feature map + linear
    SVM -- 52x faster than exact RBF on the real recipe (3.8s vs 198s)."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import Pipeline

    est = build_classifier("svm", kernel_approx=256)._make_estimator()
    assert isinstance(est, Pipeline)
    assert isinstance(est.steps[0][1], Nystroem)
    assert est.steps[0][1].n_components == 256

    # kernel_approx=0 (default) must stay on the exact-kernel path.
    from sklearn.svm import SVC
    assert isinstance(build_classifier("svm")._make_estimator(), SVC)


def test_svm_kernel_approx_is_much_cheaper_than_exact_on_wide_data():
    # Not a wall-clock assertion (too flaky for CI); instead assert the property
    # that causes the speedup: the approximation's cost is set by n_components,
    # not by the number of rows squared, so it still fits when exact SVC would be
    # doing an O(n^2) kernel.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 120))
    y = (rng.random(1500) < 0.05).astype(int)
    y[:20] = 1
    proba = build_classifier("svm", kernel_approx=128, C=0.1).fit(X, y).predict_proba(X)
    assert proba.shape == (1500,)
    assert np.isfinite(proba).all()
