import math
import pytest
import numpy as np
from lncfit.metrics import compute_classification_metrics, compute_metrics


def test_perfect_predictions_and_correct_keys():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_metrics("test", y, y)
    assert set(result.keys()) == {"split", "n", "pearson_r", "spearman_rho", "rmse", "mae", "r2"}
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_rmse_known_value():
    result = compute_metrics("test", np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert result["rmse"] == pytest.approx(1.0)


def test_mae_known_value():
    result = compute_metrics("test", np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert result["mae"] == pytest.approx(1.0)


def test_constant_target_r2_is_nan():
    # A degenerate constant-target input must report NaN, not crash or return a
    # misleading 0/1 -- pearsonr/spearmanr are undefined here.
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = compute_metrics("test", y_true, y_pred)
    assert math.isnan(result["r2"])


def test_classification_perfect_predictions_and_correct_keys():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_classification_metrics("test", y_true, y_proba)
    assert set(result.keys()) == {
        "split", "n", "n_pos", "pos_rate", "auroc", "auprc", "f1", "precision", "recall", "accuracy",
    }
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_classification_single_class_auc_is_nan():
    # Real, recurring scenario in this project: a cell-line/chromosome slice with
    # zero positives makes AUROC/AUPRC mathematically undefined. Must report NaN,
    # not crash the whole evaluation run.
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    result = compute_classification_metrics("test", y_true, y_proba)
    assert math.isnan(result["auroc"])
    assert math.isnan(result["auprc"])
    assert result["accuracy"] == pytest.approx(1.0)


def test_classification_threshold_applied():
    y_true = np.array([0, 1])
    y_proba = np.array([0.4, 0.6])
    high_thresh = compute_classification_metrics("test", y_true, y_proba, threshold=0.9)
    assert high_thresh["recall"] == pytest.approx(0.0)
