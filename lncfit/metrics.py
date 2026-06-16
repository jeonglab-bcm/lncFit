from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for a set of predictions.

    Returns a dict with keys: split, n, pearson_r, spearman_rho, rmse, mae, r2.
    Also prints a formatted line to stdout.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    r, _   = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"  {label:<22}  n={n:>8,}  r={r:.4f}  ρ={rho:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    return {
        "split": label,
        "n": n,
        "pearson_r": round(float(r), 4),
        "spearman_rho": round(float(rho), 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }
