"""Train an XGBoost model to predict CRISPR-screen log2 fold-change from k-mer and cell-context features."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_jsonl
from lncfit.features import build_features, fit_vocab
from lncfit.metrics import compute_metrics

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost log2FC predictor.")
    parser.add_argument("--train", default="data/processed/train_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_chrom1.jsonl.gz")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=6)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/model")
    parser.add_argument("--output-model", default=None)
    parser.add_argument("--output-metrics", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"xgboost_k{args.k}"
    model_path = Path(args.output_model) if args.output_model else out_dir / f"{tag}.ubj"
    params_path = out_dir / f"{tag}_params.json"

    print(f"Loading train records from {args.train} ...")
    train_records = load_jsonl(args.train)
    print(f"  {len(train_records):,} records")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test)
    print(f"  {len(test_records):,} records")

    print(f"\nFitting k={args.k} vocabulary on training sequences ...")
    vocab = fit_vocab([r.target_sequence for r in train_records], args.k)
    print(f"  {len(vocab)} / {4**args.k} k-mers observed")

    print(f"Building features (k={args.k}, include_distance={args.include_distance}) ...")
    X_train, y_train, train_cols = build_features(
        train_records, k=args.k, include_distance=args.include_distance, vocab=vocab,
    )
    X_test, y_test, _ = build_features(
        test_records, k=args.k, include_distance=args.include_distance, vocab=vocab,
    )
    print(f"  Feature matrix: {X_train.shape[1]} columns")

    print("\nFitting XGBoost ...")
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating ...")
    y_pred = model.predict(X_test)

    metrics_rows = [compute_metrics("Overall", y_test, y_pred)]
    for cl in _CELL_LINES:
        mask = np.array([r.cell_line == cl for r in test_records])
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(cl, y_test[mask], y_pred[mask]))

    model.save_model(str(model_path))
    print(f"\nModel saved      -> {model_path}")

    vocab_path = out_dir / f"{tag}_vocab.json"
    with open(vocab_path, "w") as fh:
        json.dump(vocab, fh)
    print(f"Vocab saved      -> {vocab_path}")

    params_dict = {
        "k": args.k,
        "include_distance": args.include_distance,
        "n_estimators": args.n_estimators,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": args.seed,
        "feature_columns": train_cols,
        "n_features": X_train.shape[1],
        "train_file": str(args.train),
        "test_file": str(args.test),
    }
    with open(params_path, "w") as fh:
        json.dump(params_dict, fh, indent=2)
    print(f"Params saved     -> {params_path}")

    if args.output_metrics:
        import pandas as pd
        pd.DataFrame(metrics_rows).to_csv(args.output_metrics, index=False)
        print(f"Metrics saved    -> {args.output_metrics}")


if __name__ == "__main__":
    main()
