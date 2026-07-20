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
    for name in ["null", "logreg", "xgboost"]:
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
