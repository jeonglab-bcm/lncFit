"""Standalone evaluation script for a saved XGBoost model.

Loads a trained model (.ubj / .json) and a test JSONL split, runs inference,
and writes a timestamped results bundle to results/eval_<YYYYMMDD_HHMMSS>/.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.features import build_features
from lncfit.io import git_commit
from lncfit.plotting import plot_scatter_grid
from lncfit.screen_data import load_jsonl
from lncfit.xgboost_model import evaluate_by_group


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
    metrics_rows = evaluate_by_group(test_records, y_test, y_pred, cross_terms=False)

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
        "git_commit": git_commit(),
    }
    run_info_path = out_dir / "run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2)
    print(f"Run info JSON    -> {run_info_path}")

    preds_df = pd.DataFrame(preds_rows)
    scatter_path = out_dir / "scatter.png"
    plot_scatter_grid(preds_df, scatter_path, k=args.k)
    print(f"Scatter plot     -> {scatter_path}")


if __name__ == "__main__":
    main()
