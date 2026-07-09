import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tune_lncrna_xgboost import _classifier_kwargs, _natural_ratio


def test_natural_ratio_matches_neg_over_pos():
    y = np.array([0, 0, 0, 1])
    assert _natural_ratio(y) == 3.0


def test_natural_ratio_no_positives_falls_back_to_one():
    y = np.array([0, 0, 0])
    assert _natural_ratio(y) == 1.0


def test_classifier_kwargs_includes_scale_pos_weight_and_binary_objective():
    params = {
        "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0,
    }
    kwargs = _classifier_kwargs(params, scale_pos_weight=12.5, nthread=-1, seed=42, n_estimators=2000)
    assert kwargs["scale_pos_weight"] == 12.5
    assert kwargs["objective"] == "binary:logistic"
    assert kwargs["eval_metric"] == "aucpr"
    assert kwargs["n_estimators"] == 2000
    assert kwargs["max_depth"] == 6
