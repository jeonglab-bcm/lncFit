"""Sweep log2FC clipping quantile to assess impact on CV performance (issue #51).

Uses fixed hyperparameters from an existing best_params JSON so the only variable
is the clipping setting. Runs chromosome LOCO-CV for each quantile and reports:
  quantile | clip_limit | pct_clipped | cv_rho_mean | cv_rho_std

Usage:
    uv run python scripts/sweep_clip_quantile.py \\
        --best-params results/transcript_overlap/data/model/xgboost_best_params_k3_mse.json \\
        --body-sequences data/processed/body_sequences_transcript.json \\
        --output-dir results/clip_sweep
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.cv import build_folds
from lncfit.preprocessing import symmetric_quantile_clip
from lncfit.screen_data import load_jsonl
from lncfit.xgboost_model import build_xgb_params

QUANTILES = [1.0, 0.99, 0.975, 0.95, 0.90]


def run_cv_for_quantile(fold_data, cv_chroms, params, objective, nthread, seed, clip_limit):
    """Run one pass of LOCO-CV with the given clip_limit applied to y arrays."""
    fold_rhos = []
    n_estimators = params.get("n_estimators_cv_median", 500)
    es_rounds = max(50, int(0.5 / params["learning_rate"]))

    for val_chrom in cv_chroms:
        X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[val_chrom]

        # Apply clipping using pre-computed limit from training set
        y_tr_c,  _, _ = symmetric_quantile_clip(y_tr,  clip_limit=clip_limit)
        y_val_c, _, _ = symmetric_quantile_clip(y_val, clip_limit=clip_limit)
        y_es_c,  _, _ = symmetric_quantile_clip(y_es,  clip_limit=clip_limit)

        xgb_params = build_xgb_params(params, objective, nthread, seed)
        xgb_params["n_estimators"] = n_estimators

        model = xgb.XGBRegressor(
            **xgb_params,
            callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
        )
        model.fit(X_tr, y_tr_c, eval_set=[(X_es, y_es_c)], verbose=False)

        y_pred = model.predict(X_val)
        rho, _ = spearmanr(y_val_c, y_pred)
        fold_rhos.append(float(rho))
        del model
        gc.collect()

    return float(np.mean(fold_rhos)), float(np.std(fold_rhos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-params",
        default="results/transcript_overlap/data/model/xgboost_best_params_k3_mse.json",
        help="Path to best_params JSON from a completed tune_xgboost run.",
    )
    parser.add_argument("--train", default="data/processed/train_chrom1.jsonl.gz")
    parser.add_argument(
        "--body-sequences",
        default="data/processed/body_sequences_transcript.json",
    )
    parser.add_argument("--output-dir", default="results/clip_sweep")
    parser.add_argument("--nthread", type=int, default=-1)
    parser.add_argument(
        "--quantiles", nargs="+", type=float, default=QUANTILES,
        help=f"Quantiles to sweep (default: {QUANTILES}).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.best_params) as fh:
        params = json.load(fh)

    k = params["k"]
    objective = params["objective"]
    seed = params.get("seed", 42)

    print(f"Loaded hyperparams from {args.best_params}")
    print(f"  k={k}, objective={objective}, lr={params['learning_rate']:.5f}, "
          f"n_est={params.get('n_estimators_cv_median', '?')}")

    print(f"\nLoading training records from {args.train} ...")
    train_records = load_jsonl(args.train)
    print(f"  {len(train_records):,} records")

    body_sequences = None
    if args.body_sequences:
        bs_path = Path(args.body_sequences)
        if bs_path.exists():
            print(f"\nLoading body sequences from {bs_path} ...")
            with open(bs_path) as fh:
                _raw = json.load(fh)
            body_sequences = {k_: tuple(v) for k_, v in _raw.items()}
            print(f"  {len(body_sequences):,} genes")

    print("\nBuilding CV folds ...")
    cv_chroms, fold_data, _ = build_folds(
        train_records, k=k, body_sequences=body_sequences,
    )
    print(f"  {len(cv_chroms)} folds: {cv_chroms}")

    # Compute clip limits from ALL training y values (once, before sweep)
    all_train_y = np.concatenate([fold_data[c][1] for c in cv_chroms])
    print(f"\n  training y: min={all_train_y.min():.3f}, max={all_train_y.max():.3f}, "
          f"median |y|={np.median(np.abs(all_train_y)):.3f}")

    rows = []
    for quantile in sorted(args.quantiles, reverse=True):
        _, clip_limit, pct_clipped = symmetric_quantile_clip(all_train_y, quantile)
        print(f"\n── quantile={quantile:.3f}  limit={clip_limit:.4f}  "
              f"pct_clipped={pct_clipped*100:.2f}%", flush=True)

        cv_rho_mean, cv_rho_std = run_cv_for_quantile(
            fold_data, cv_chroms, params, objective, args.nthread, seed, clip_limit,
        )
        print(f"   CV Spearman ρ = {cv_rho_mean:.4f} ± {cv_rho_std:.4f}")

        rows.append({
            "quantile": quantile,
            "clip_limit": round(clip_limit, 4),
            "pct_clipped": round(pct_clipped * 100, 3),
            "cv_rho_mean": round(cv_rho_mean, 4),
            "cv_rho_std": round(cv_rho_std, 4),
        })

    df = pl.DataFrame(rows).sort("quantile", descending=True)

    csv_path = out_dir / "clip_quantile_sweep.csv"
    md_path  = out_dir / "clip_quantile_sweep.md"
    df.write_csv(csv_path)
    print(f"\nResults saved -> {csv_path}")

    md_lines = [
        "# log2FC Clipping Quantile Sweep (issue #51)\n",
        f"Config: k={k}, objective={objective}, "
        f"body_sequences={args.body_sequences}\n",
        "Fixed hyperparameters from: " + args.best_params + "\n\n",
        "| quantile | clip_limit | pct_clipped | cv_rho_mean | cv_rho_std |",
        "|----------|-----------|-------------|-------------|------------|",
    ]
    for row in df.iter_rows(named=True):
        md_lines.append(
            f"| {row['quantile']:.3f} | {row['clip_limit']:.4f} | "
            f"{row['pct_clipped']:.3f}% | {row['cv_rho_mean']:.4f} | "
            f"{row['cv_rho_std']:.4f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown table  -> {md_path}\n")
    print("\n".join(md_lines[3:]))


if __name__ == "__main__":
    main()
