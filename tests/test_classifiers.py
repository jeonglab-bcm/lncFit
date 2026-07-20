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
    X_tr, y_tr, X_te = _toy_data()
    model = build_classifier(name)
    assert model.fit(X_tr, y_tr) is model
    proba = model.predict_proba(X_te)
    assert proba.shape == (X_te.shape[0],)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


@pytest.mark.parametrize("name", ["logreg", "xgboost"])
def test_wrapper_accepts_sparse_and_dense(name):
    X_tr, y_tr, X_te = _toy_data()
    dense = build_classifier(name).fit(X_tr, y_tr).predict_proba(X_te)
    sparse = build_classifier(name).fit(csr_matrix(X_tr), y_tr).predict_proba(csr_matrix(X_te))
    assert dense.shape == sparse.shape == (X_te.shape[0],)


def test_null_predicts_training_base_rate():
    X_tr, y_tr, X_te = _toy_data(pos_rate=0.2)
    model = build_classifier("null").fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    assert np.allclose(proba, y_tr.mean())


def test_model_type_matches_registry_key():
    # A mismatch here wouldn't crash anything -- it would silently mislabel
    # "model" in every run_info.json this project's tuning scripts write, which
    # is exactly the field this whole session's result tables were built from.
    for name, cls in CLASSIFIER_REGISTRY.items():
        assert cls.model_type == name


def test_xgboost_auto_scale_pos_weight_runs():
    # scale_pos_weight left None -> computed from y inside fit(); should not error.
    X_tr, y_tr, X_te = _toy_data(pos_rate=0.05)
    model = build_classifier("xgboost", n_estimators=10)
    proba = model.fit(X_tr, y_tr).predict_proba(X_te)
    assert proba.shape == (X_te.shape[0],)
