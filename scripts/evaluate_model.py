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

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_jsonl
from lncfit.features import build_features, all_kmers
from lncfit.metrics import compute_metrics

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_DAYS = [7, 14]


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
    parser.add_argument("--k", type=int, choices=[3, 6], default=6)
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

    print(f"Building features (k={args.k}, include_distance={args.include_distance}) ...")
    X_test, y_test = build_features(test_records, k=args.k, include_distance=args.include_distance)
    print(f"  Feature matrix: {X_test.shape[1]} columns")

    print("\nRunning inference ...")
    y_pred = model.predict(X_test)

    test_df = X_test.copy()
    test_df["_y_true"] = y_test.values
    test_df["_y_pred"] = y_pred

    # --- Metrics ---
    print("\nMetrics:")
    metrics_rows = [compute_metrics("Overall", y_test.values, y_pred)]

    for cl in _CELL_LINES:
        col = f"cell_{cl}"
        if col not in test_df.columns:
            continue
        mask = test_df[col] == 1
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(cl,
                                            test_df.loc[mask, "_y_true"].values,
                                            test_df.loc[mask, "_y_pred"].values))

    for day in _DAYS:
        col = f"day_{day}"
        if col not in test_df.columns:
            continue
        mask = test_df[col] == 1
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(f"day_{day}",
                                            test_df.loc[mask, "_y_true"].values,
                                            test_df.loc[mask, "_y_pred"].values))

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


if __name__ == "__main__":
    main()
