"""Shared XGBoost construction and group evaluation utilities."""
from __future__ import annotations

import numpy as np

from lncfit.constants import CELL_LINES, DAYS
from lncfit.metrics import compute_classification_metrics, compute_metrics


def build_xgb_params(
    params: dict,
    objective: str,
    nthread: int = -1,
    seed: int = 42,
    n_estimators: int = 5000,
) -> dict:
    """Return XGBRegressor kwargs built from a hyperparameter dict.

    params must contain: learning_rate, max_depth, subsample, colsample_bytree,
    min_child_weight, reg_alpha, reg_lambda.
    """
    return dict(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        objective=objective,
        tree_method="hist",
        nthread=nthread,
        random_state=seed,
    )


def evaluate_by_group(
    records,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_lines: list[str] | None = None,
    days: list[int] | None = None,
    cross_terms: bool = True,
) -> list[dict]:
    """Compute metrics overall, per cell line, per day, and optionally per cell line × day.

    Returns a list of metric dicts suitable for writing to a CSV via polars or pandas.
    """
    if cell_lines is None:
        cell_lines = CELL_LINES
    if days is None:
        days = DAYS

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows = [compute_metrics("Overall", y_true, y_pred)]

    for cl in cell_lines:
        mask = np.array([r.cell_line == cl for r in records])
        if mask.sum() == 0:
            continue
        rows.append(compute_metrics(cl, y_true[mask], y_pred[mask]))

    for day in days:
        mask = np.array([r.day == day for r in records])
        if mask.sum() == 0:
            continue
        rows.append(compute_metrics(f"Day {day}", y_true[mask], y_pred[mask]))

    if cross_terms:
        for cl in cell_lines:
            for day in days:
                mask = np.array([r.cell_line == cl and r.day == day for r in records])
                if mask.sum() == 0:
                    continue
                rows.append(compute_metrics(f"{cl} Day {day}", y_true[mask], y_pred[mask]))

    return rows


def evaluate_lncrna_by_group(
    records,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    cell_lines: list[str] | None = None,
) -> list[dict]:
    """Compute classification metrics overall and per cell line for lncRNA-level records.

    No day breakdown: LncRnaRecord datasets are built for a single day (Day 14).
    """
    if cell_lines is None:
        cell_lines = CELL_LINES

    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    rows = [compute_classification_metrics("Overall", y_true, y_pred_proba)]

    for cl in cell_lines:
        mask = np.array([r.cell_line == cl for r in records])
        if mask.sum() == 0:
            continue
        rows.append(compute_classification_metrics(cl, y_true[mask], y_pred_proba[mask]))

    return rows
