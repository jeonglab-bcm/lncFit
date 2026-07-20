import math
import pytest
import numpy as np
from lncfit.metrics import compute_classification_metrics, compute_metrics


def test_compute_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_metrics("test", y, y)
    assert set(result.keys()) == {"split", "n", "pearson_r", "spearman_rho", "rmse", "mae", "r2"}
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)

    # custom (non-sklearn-wrapper) formulas against hand-computed values
    assert compute_metrics("test", np.array([0.0, 0.0]), np.array([1.0, 1.0]))["rmse"] == pytest.approx(1.0)
    assert compute_metrics("test", np.array([0.0, 2.0]), np.array([1.0, 1.0]))["mae"] == pytest.approx(1.0)

    # degenerate constant-target input must report NaN, not crash or misreport 0/1
    result = compute_metrics("test", np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    assert math.isnan(result["r2"])


def test_compute_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_classification_metrics("test", y_true, y_proba)
    assert set(result.keys()) == {
        "split", "n", "n_pos", "pos_rate", "auroc", "auprc", "f1", "precision", "recall", "accuracy",
    }
    assert result["auroc"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)

    # raising the threshold must change recall -- guards against a silently-ignored parameter
    high_thresh = compute_classification_metrics("test", y_true, y_proba, threshold=0.95)
    assert high_thresh["recall"] < result["recall"]

    # all-one-class y_true -> NaN, not a crash -- the real, recurring scenario
    # in this project where a cell-line/chromosome slice has zero positives
    degenerate = compute_classification_metrics("test", np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert math.isnan(degenerate["auroc"]) and math.isnan(degenerate["auprc"])
