"""Standalone evaluation script for a saved XGBoost model.

Loads a trained model (.ubj / .json) and a test JSONL split, runs inference,
and writes a timestamped results bundle to results/eval_<YYYYMMDD_HHMMSS>/.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_jsonl
from lncfit.features import build_features, all_kmers
from lncfit.metrics import compute_metrics

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_DAYS = [7, 14]


def _scatter_panel(ax, y_true, y_pred, label):
    from scipy.stats import pearsonr
    r, _ = pearsonr(y_true, y_pred)
    ax.scatter(y_true, y_pred, s=2, alpha=0.2, color="#2166ac", linewidths=0, rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("Observed log2FC", fontsize=8)
    ax.set_ylabel("Predicted log2FC", fontsize=8)
    ax.set_title(f"{label}\nr={r:.3f}  n={len(y_true):,}", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_scatter(preds_df: pd.DataFrame, cell_lines: list, out_path: Path, k: int) -> None:
    present = [cl for cl in cell_lines if (preds_df["cell_line"] == cl).any()]
    n_panels = 1 + len(present)
    fig = plt.figure(figsize=(4 * n_panels, 4))
    gs = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.4)

    _scatter_panel(fig.add_subplot(gs[0, 0]),
                   preds_df["y_true"].values, preds_df["y_pred"].values, "Overall")
    for i, cl in enumerate(present):
        sub = preds_df[preds_df["cell_line"] == cl]
        _scatter_panel(fig.add_subplot(gs[0, i + 1]),
                       sub["y_true"].values, sub["y_pred"].values, cl)

    fig.suptitle(f"Predicted vs. Observed log2FC  (k={k})", fontsize=11, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved XGBoost log2FC model.")
    parser.add_argument("--model", required=True, help="Path to saved model (.ubj or .json)")
    parser.add_argument("--test", default="data/processed/test_chrom1.jsonl.gz")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=6)
    parser.add_argument("--vocab", default=None,
                        help="Path to vocab JSON sidecar. Auto-detected from model path if omitted.")
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model = xgb.XGBRegressor()
    model.load_model(args.model)

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test)
    print(f"  {len(test_records):,} records")

    vocab_path = Path(args.vocab) if args.vocab else Path(args.model).with_name(
        Path(args.model).stem + "_vocab.json"
    )
    vocab = None
    if vocab_path.exists():
        with open(vocab_path) as fh:
            vocab = json.load(fh)
        print(f"Loaded vocab from {vocab_path} ({len(vocab)} k-mers)")
    else:
        print(f"No vocab sidecar found at {vocab_path}; using full {4**args.k}-kmer vocabulary")

    print(f"Building features (k={args.k}, include_distance={args.include_distance}) ...")
    X_test, y_test, _ = build_features(
        test_records, k=args.k, include_distance=args.include_distance, vocab=vocab,
    )
    print(f"  Feature matrix: {X_test.shape[1]} columns")

    print("\nRunning inference ...")
    y_pred = model.predict(X_test)

    # --- Metrics ---
    print("\nMetrics:")
    metrics_rows = [compute_metrics("Overall", y_test, y_pred)]

    for cl in _CELL_LINES:
        mask = np.array([r.cell_line == cl for r in test_records])
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(cl, y_test[mask], y_pred[mask]))

    for day in _DAYS:
        mask = np.array([r.day == day for r in test_records])
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(f"day_{day}", y_test[mask], y_pred[mask]))

    # --- Write outputs ---
    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"\nMetrics CSV      -> {metrics_path}")

    preds_rows = []
    for rec, y_t, y_p in zip(test_records, y_test.values, y_pred):
        preds_rows.append({
            "target": rec.target,
            "cell_line": rec.cell_line,
            "day": rec.day,
            "replicate": rec.replicate,
            "y_true": float(y_t),
            "y_pred": float(y_p),
        })
    preds_path = out_dir / "predictions.csv"
    pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
    print(f"Predictions CSV  -> {preds_path}")

    run_info = {
        "model_path": str(args.model),
        "test_file": str(args.test),
        "k": args.k,
        "include_distance": args.include_distance,
        "n_test_records": len(test_records),
        "timestamp": timestamp,
        "git_commit": _git_commit(),
    }
    run_info_path = out_dir / "run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2)
    print(f"Run info JSON    -> {run_info_path}")

    preds_df = pd.DataFrame(preds_rows)
    scatter_path = out_dir / "scatter.png"
    plot_scatter(preds_df, _CELL_LINES, scatter_path, k=args.k)
    print(f"Scatter plot     -> {scatter_path}")


if __name__ == "__main__":
    main()
