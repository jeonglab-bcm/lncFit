import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lncfit.classifiers import CLASSIFIER_REGISTRY, build_classifier


def _toy_data(seed=0, n_train=200, n_test=60, n_feat=8, pos_rate=0.1):
    rng = np.random.default_rng(seed)
    X_tr = rng.random((n_train, n_feat))
    y_tr = (rng.random(n_train) < pos_rate).astype(int)
    X_te = rng.random((n_test, n_feat))
    return X_tr, y_tr, X_te


@pytest.mark.parametrize("name", ["null", "logreg", "xgboost"])
def test_wrapper_fit_returns_self_and_proba_shape_range(name):
    # xgboost is fit here with scale_pos_weight left at its default (None ->
    # auto-computed from y inside fit()), so this also covers that path --
    # a separate standalone test for it would just be re-running this case.
    X_tr, y_tr, X_te = _toy_data()
    model = build_classifier(name)
    assert model.fit(X_tr, y_tr) is model
    proba = model.predict_proba(X_te)
    assert proba.shape == (X_te.shape[0],)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    if name == "null":
        assert np.allclose(proba, y_tr.mean())  # predicts exactly the training base rate


def test_xgboost_accepts_sparse_and_dense():
    # The one proven bug class in this codebase (XGBoost treats sparse implicit
    # zeros as *missing*) -- logreg has no such special-casing, so it's not
    # tested here too.
    X_tr, y_tr, X_te = _toy_data()
    dense = build_classifier("xgboost").fit(X_tr, y_tr).predict_proba(X_te)
    sparse = build_classifier("xgboost").fit(csr_matrix(X_tr), y_tr).predict_proba(csr_matrix(X_te))
    assert dense.shape == sparse.shape == (X_te.shape[0],)


def test_model_type_matches_registry_key():
    # A mismatch here wouldn't crash anything -- it would silently mislabel
    # "model" in every run_info.json this project's tuning scripts write, which
    # is exactly the field this whole session's result tables were built from.
    for name, cls in CLASSIFIER_REGISTRY.items():
        assert cls.model_type == name
