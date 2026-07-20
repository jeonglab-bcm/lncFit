import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tune_lncrna_xgboost import _classifier_kwargs, _natural_ratio


def test_natural_ratio_and_classifier_kwargs():
    assert _natural_ratio(np.array([0, 0, 0, 1])) == 3.0  # n_neg/n_pos
    assert _natural_ratio(np.array([0, 0, 0])) == 1.0  # zero positives -> fallback, not ZeroDivisionError

    params = {
        "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1, "reg_alpha": 0.0, "reg_lambda": 1.0,
    }
    kwargs = _classifier_kwargs(params, scale_pos_weight=12.5, nthread=-1, seed=42, n_estimators=2000)
    assert kwargs["scale_pos_weight"] == 12.5
    assert kwargs["objective"] == "binary:logistic"
    assert kwargs["eval_metric"] == "aucpr"
