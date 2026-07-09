from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)


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


def compute_classification_metrics(
    label: str, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute binary classification metrics for a set of predicted hit probabilities.

    Returns a dict with keys: split, n, n_pos, pos_rate, auroc, auprc, f1, precision,
    recall, accuracy. AUROC/AUPRC are undefined (nan) when y_true has a single class,
    e.g. a fold with zero positive lncRNAs. Also prints a formatted line to stdout.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)
    y_pred = (y_pred_proba >= threshold).astype(int)
    n = len(y_true)
    n_pos = int(y_true.sum())

    if n_pos == 0 or n_pos == n:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_pred_proba))
        auprc = float(average_precision_score(y_true, y_pred_proba))

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    print(
        f"  {label:<22}  n={n:>8,}  pos={n_pos:>5} ({n_pos / n:.1%})  "
        f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}"
    )

    return {
        "split": label,
        "n": n,
        "n_pos": n_pos,
        "pos_rate": round(n_pos / n, 4) if n > 0 else float("nan"),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
    }
