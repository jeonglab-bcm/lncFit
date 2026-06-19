"""Benchmark null, Ridge, Lasso, and XGBoost on the same LOCO-CV folds.

XGBoost is not retrained — supply predictions from a previous run via --xgb-predictions.
Ridge and Lasso are evaluated with a regularization sweep; the best alpha per model is
selected by mean CV Spearman rho. The null baseline always predicts the training-set mean.

Usage (from project root):
  uv run python scripts/benchmark_models.py \\
    --xgb-predictions results/final_eval_20260616_221720/predictions.csv \\
    --body-sequences data/processed/body_sequences.json \\
    --k 3

Outputs (under --output-dir/results/benchmark_<timestamp>/):
  metrics_summary.csv   all models × all splits (model column prepended)
  cv_scores.csv         per-fold Spearman rho for null, Ridge (best alpha), Lasso (best alpha)
  run_info.json
"""

import argparse
import json
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.cv import build_folds
from lncfit.features import build_features, fit_vocab
from lncfit.io import git_commit
from lncfit.models import LassoModel, MeanPredictor, RidgeModel
from lncfit.screen_data import load_jsonl
from lncfit.xgboost_model import evaluate_by_group


_RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
_LASSO_ALPHAS = [1e-4, 1e-3, 0.01, 0.1, 1.0]


def _run_cv(model_factory, fold_data, cv_chroms):
    """Run LOCO-CV for one model and return (mean_rho, fold_rows).

    model_factory is a zero-arg callable that returns a fresh model instance.
    """
    fold_rows = []
    for val_chrom in cv_chroms:
        X_tr, y_tr, X_val, y_val, _X_es, _y_es = fold_data[val_chrom]
        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        rho, _ = spearmanr(y_val, y_pred)
        fold_rows.append({"chromosome": val_chrom, "n_val": len(y_val), "spearman_rho": float(rho)})
    mean_rho = float(np.mean([r["spearman_rho"] for r in fold_rows]))
    return mean_rho, fold_rows


def _sweep(model_cls, alphas, fold_data, cv_chroms, label):
    """Sweep alphas, return (best_alpha, best_mean_rho, best_fold_rows)."""
    best_alpha, best_rho, best_fold_rows = None, -np.inf, []
    for alpha in alphas:
        mean_rho, fold_rows = _run_cv(lambda a=alpha: model_cls(a), fold_data, cv_chroms)
        print(f"  {label} alpha={alpha:.4g}  CV rho={mean_rho:.4f}")
        if mean_rho > best_rho:
            best_rho, best_alpha, best_fold_rows = mean_rho, alpha, fold_rows
    return best_alpha, best_rho, best_fold_rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark null/Ridge/Lasso vs XGBoost on the same LOCO-CV folds."
    )
    parser.add_argument("--train", default="data/processed/train_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_chrom1.jsonl.gz")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument(
        "--body-sequences", default=None,
        help="Body sequences JSON used by the XGBoost reference run.",
    )
    parser.add_argument(
        "--xgb-predictions", required=True,
        help="predictions.csv from the XGBoost reference run.",
    )
    parser.add_argument("--signed-overlap", action="store_true",
                        help="Use signed overlap k-mer features (requires --body-sequences).")
    parser.add_argument("--output-dir", default=".",
                        help="Root directory for outputs (default: current directory).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # ── Load records ──────────────────────────────────────────────────────────
    print(f"Loading training records from {args.train} ...")
    train_records = load_jsonl(args.train)
    print(f"  {len(train_records):,} records")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test)
    print(f"  {len(test_records):,} records")

    body_sequences: dict | None = None
    if args.body_sequences:
        body_seq_path = Path(args.body_sequences)
        print(f"\nLoading body sequences from {body_seq_path} ...")
        with open(body_seq_path) as fh:
            _raw = json.load(fh)
        body_sequences = {k: tuple(v) for k, v in _raw.items()}
        print(f"  {len(body_sequences):,} genes with body sequences")

    # ── Build CV folds ────────────────────────────────────────────────────────
    print()
    cv_chroms, fold_data, _ = build_folds(
        train_records, k=args.k,
        include_distance=args.include_distance,
        body_sequences=body_sequences,
        signed_overlap=args.signed_overlap,
    )
    print(f"  {len(cv_chroms)} CV folds")

    # ── Null baseline ─────────────────────────────────────────────────────────
    print("\nNull baseline (mean predictor) CV ...")
    null_rho, null_fold_rows = _run_cv(MeanPredictor, fold_data, cv_chroms)
    print(f"  CV mean Spearman rho = {null_rho:.4f}")

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print("\nRidge regularization sweep ...")
    ridge_alpha, ridge_rho, ridge_fold_rows = _sweep(
        RidgeModel, _RIDGE_ALPHAS, fold_data, cv_chroms, "Ridge"
    )
    print(f"  Best alpha={ridge_alpha}  CV rho={ridge_rho:.4f}")

    # ── Lasso ─────────────────────────────────────────────────────────────────
    print("\nLasso regularization sweep ...")
    lasso_alpha, lasso_rho, lasso_fold_rows = _sweep(
        LassoModel, _LASSO_ALPHAS, fold_data, cv_chroms, "Lasso"
    )
    print(f"  Best alpha={lasso_alpha}  CV rho={lasso_rho:.4f}")

    # ── Build full-training features for final fits ───────────────────────────
    print("\nBuilding final training/test features ...")
    guide_seqs = [r.target_sequence for r in train_records]
    if body_sequences is not None:
        seen_targets = {r.target for r in train_records}
        body_seqs_for_vocab = [
            seq for t in seen_targets for seq in body_sequences.get(t, ())
        ]
    else:
        body_seqs_for_vocab = []
    final_vocab = fit_vocab(guide_seqs + body_seqs_for_vocab, args.k)
    X_train, y_train, _ = build_features(
        train_records, k=args.k, include_distance=args.include_distance,
        sparse=True, vocab=final_vocab, body_sequences=body_sequences,
        signed_overlap=args.signed_overlap,
    )
    X_test, y_test, _ = build_features(
        test_records, k=args.k, include_distance=args.include_distance,
        sparse=True, vocab=final_vocab, body_sequences=body_sequences,
        signed_overlap=args.signed_overlap,
    )
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Final fits + test evaluation ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = out_dir / "results" / f"benchmark_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    cv_rows: list[dict] = []

    linear_models = [
        ("null",                MeanPredictor(),          null_fold_rows),
        (f"ridge_a{ridge_alpha}", RidgeModel(ridge_alpha),  ridge_fold_rows),
        (f"lasso_a{lasso_alpha}", LassoModel(lasso_alpha),  lasso_fold_rows),
    ]

    for model_name, model, fold_rows in linear_models:
        print(f"\nFinal fit + test evaluation: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for row in evaluate_by_group(test_records, y_test, y_pred, cross_terms=True):
            all_metrics.append({"model": model_name, **row})
        for fold_row in fold_rows:
            cv_rows.append({"model": model_name, **fold_row})

    # ── XGBoost reference ─────────────────────────────────────────────────────
    print(f"\nLoading XGBoost reference predictions from {args.xgb_predictions} ...")
    xgb_df = pl.read_csv(args.xgb_predictions)
    print(f"  {len(xgb_df):,} predictions")

    _Rec = namedtuple("_Rec", ["cell_line", "day"])
    xgb_records = [_Rec(r["cell_line"], r["day"]) for r in xgb_df.iter_rows(named=True)]
    y_true_xgb = xgb_df["y_true"].to_numpy()
    y_pred_xgb = xgb_df["y_pred"].to_numpy()

    print("  Test set metrics (XGBoost reference):")
    for row in evaluate_by_group(xgb_records, y_true_xgb, y_pred_xgb, cross_terms=True):
        all_metrics.append({"model": "xgboost_body_kmer", **row})

    # ── Write outputs ─────────────────────────────────────────────────────────
    metrics_path = eval_dir / "metrics_summary.csv"
    pl.DataFrame(all_metrics).write_csv(metrics_path)
    print(f"\nMetrics summary  -> {metrics_path}")

    cv_path = eval_dir / "cv_scores.csv"
    pl.DataFrame(cv_rows).write_csv(cv_path)
    print(f"CV scores        -> {cv_path}")

    run_info = {
        "train_file": str(args.train),
        "test_file": str(args.test),
        "k": args.k,
        "include_distance": args.include_distance,
        "body_sequences_file": args.body_sequences,
        "xgb_predictions": str(args.xgb_predictions),
        "n_cv_folds": len(cv_chroms),
        "cv_chroms": cv_chroms,
        "null_cv_mean_rho": null_rho,
        "ridge_best_alpha": ridge_alpha,
        "ridge_cv_mean_rho": ridge_rho,
        "lasso_best_alpha": lasso_alpha,
        "lasso_cv_mean_rho": lasso_rho,
        "n_test_records": len(test_records),
        "timestamp": timestamp,
        "git_commit": git_commit(),
    }
    run_info_path = eval_dir / "run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2)
        fh.write("\n")
    print(f"Run info JSON    -> {run_info_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
