"""Shared plotting utilities for XGBoost evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from lncfit.constants import CELL_LINES, DAYS


def scatter_panel(ax, y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    r, _ = pearsonr(y_true, y_pred)
    ax.scatter(y_true, y_pred, s=2, alpha=0.2, color="#2166ac", linewidths=0, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("Observed log2FC", fontsize=8)
    ax.set_ylabel("Predicted log2FC", fontsize=8)
    ax.set_title(f"{label}\nr={r:.3f}  n={len(y_true):,}", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_scatter_grid(
    preds_df,
    out_path: Path,
    k: int,
    cell_lines: list[str] | None = None,
    days: list[int] | None = None,
) -> None:
    """Two-row scatter grid: row 1 = overall + per cell line, row 2 = per day.

    Accepts a polars or pandas DataFrame with columns y_true, y_pred, cell_line, day.
    """
    if cell_lines is None:
        cell_lines = CELL_LINES
    if days is None:
        days = DAYS

    # Normalise to numpy-accessible interface (supports both polars and pandas)
    def _col(name):
        col = preds_df[name]
        return col.to_numpy() if hasattr(col, "to_numpy") else col.values

    def _eq(name, val):
        col = preds_df[name]
        if hasattr(preds_df, "filter"):
            import polars as pl
            return preds_df.filter(pl.col(name) == val)
        return preds_df[preds_df[name] == val]

    present_cls = [cl for cl in cell_lines if (preds_df["cell_line"] == cl).any()]
    present_days = [d for d in days if (preds_df["day"] == d).any()]
    n_cols = max(1 + len(present_cls), len(present_days))

    fig = plt.figure(figsize=(4 * n_cols, 8))
    gs = gridspec.GridSpec(2, n_cols, figure=fig, wspace=0.4, hspace=0.5)

    scatter_panel(fig.add_subplot(gs[0, 0]), _col("y_true"), _col("y_pred"), "Overall")
    for i, cl in enumerate(present_cls):
        sub = _eq("cell_line", cl)
        scatter_panel(fig.add_subplot(gs[0, i + 1]),
                      sub["y_true"].to_numpy() if hasattr(sub["y_true"], "to_numpy") else sub["y_true"].values,
                      sub["y_pred"].to_numpy() if hasattr(sub["y_pred"], "to_numpy") else sub["y_pred"].values,
                      cl)
    for i, day in enumerate(present_days):
        sub = _eq("day", day)
        scatter_panel(fig.add_subplot(gs[1, i]),
                      sub["y_true"].to_numpy() if hasattr(sub["y_true"], "to_numpy") else sub["y_true"].values,
                      sub["y_pred"].to_numpy() if hasattr(sub["y_pred"], "to_numpy") else sub["y_pred"].values,
                      f"Day {day}")

    fig.suptitle(f"Predicted vs. Observed log2FC  (k={k})", fontsize=11, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
