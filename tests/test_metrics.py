import math
import pytest
import numpy as np
from lncfit.metrics import compute_metrics


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_metrics("test", y, y)
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_returns_correct_keys():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_metrics("test", y, y)
    assert set(result.keys()) == {"split", "n", "pearson_r", "spearman_rho", "rmse", "mae", "r2"}


def test_label_stored_in_split():
    y = np.array([1.0, 2.0, 3.0])
    result = compute_metrics("MyLabel", y, y)
    assert result["split"] == "MyLabel"


def test_n_matches_input_length():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_metrics("test", y, y)
    assert result["n"] == 5


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    result = compute_metrics("test", y_true, y_pred)
    assert result["rmse"] == pytest.approx(1.0)


def test_mae_known_value():
    y_true = np.array([0.0, 2.0])
    y_pred = np.array([1.0, 1.0])
    result = compute_metrics("test", y_true, y_pred)
    assert result["mae"] == pytest.approx(1.0)


def test_constant_target_r2_is_nan():
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    result = compute_metrics("test", y_true, y_pred)
    assert math.isnan(result["r2"])
