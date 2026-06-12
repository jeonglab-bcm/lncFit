"""Hyperparameter tuning for the XGBoost lncRNA fitness predictor.

Uses Optuna (TPE sampler) with chromosome leave-one-out cross-validation (LOCO-CV)
to find optimal hyperparameters, then retrains a final model on all training data and
evaluates on the held-out test chromosome.

CV structure (per Optuna trial):
  For each chromosome c in training chromosomes:
    val set        = chromosome c
    early-stop set = chromosome (c+1) rotating — never the same as val
    train set      = all remaining training records
    → fit XGBoost with early stopping, evaluate Spearman rho on val set
  trial score = mean Spearman rho across all folds

Optimization notes:
  - Lower learning rates (0.005–0.05) with more trees and greater early-stopping
    patience consistently outperform higher rates; the model takes smaller steps and
    avoids overshooting the optimum.
  - The default training objective (reg:squarederror) minimises MSE, but the
    evaluation metric is Spearman rank correlation — a rank-based measure.  This
    mismatch means the model does not directly optimise what we measure.  Use
    --objective reg:pseudohubererror for a robust alternative that down-weights
    outliers and can improve rank correlation.

Outputs:
  data/model/xgboost_best_params_k<K>.json   best hyperparameter configuration
  results/cv/cv_scores.csv                   per-chromosome rho for every trial
  results/final_eval_<timestamp>/            final held-out test evaluation bundle
"""

import argparse
import gc
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_jsonl
from lncfit.features import build_features, fit_vocab
from lncfit.metrics import compute_metrics

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_DAYS = [7, 14]
_MIN_FOLD_RECORDS = 500


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


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


def main():
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters via Optuna TPE + chromosome LOCO-CV."
    )
    parser.add_argument("--train", default="data/processed/train_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_chrom1.jsonl.gz")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=6)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default 50; 50–100 recommended).")
    parser.add_argument("--objective", default="reg:squarederror",
                        choices=["reg:squarederror", "reg:pseudohubererror"],
                        help="XGBoost training objective (default: reg:squarederror).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--final-val-chrom", default="22",
        help="Chromosome held out for early stopping when retraining the final model "
             "(never used during CV evaluation). Default: 22.",
    )
    parser.add_argument("--output-dir", default=".",
                        help="Root directory for all outputs (default: current directory).")
    parser.add_argument("--nthread", type=int, default=-1,
                        help="XGBoost CPU threads per model fit (-1 = all cores, the default). "
                             "Cap this on shared servers to avoid starving other users.")
    args = parser.parse_args()

    _obj_tag = {"reg:squarederror": "mse", "reg:pseudohubererror": "huber"}[args.objective]

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "data" / "model"
    cv_dir = out_dir / "results" / "cv"
    model_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    # ── Load records ──────────────────────────────────────────────────────────
    print(f"Loading training records from {args.train} ...")
    train_records = load_jsonl(args.train)
    print(f"  {len(train_records):,} records")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test)
    print(f"  {len(test_records):,} records")

    chrom_arr = np.array([r.chrom for r in train_records])

    # ── Determine CV chromosome folds ─────────────────────────────────────────
    chrom_counts = Counter(chrom_arr)
    cv_chroms = sorted(
        [str(c) for c, n in chrom_counts.items() if c and n >= _MIN_FOLD_RECORDS],
        key=lambda x: (len(x), x),
    )
    print(f"\nCV chromosomes ({len(cv_chroms)} folds): {cv_chroms}")
    print(f"  Records with no chromosome annotation: "
          f"{chrom_counts.get('', 0):,} (always in training, never held out)")

    if args.final_val_chrom not in cv_chroms:
        fallback = cv_chroms[-1]
        print(f"  Warning: --final-val-chrom '{args.final_val_chrom}' not found; "
              f"using '{fallback}' instead.")
        final_val_chrom = fallback
    else:
        final_val_chrom = args.final_val_chrom

    # ── Pre-compute per-fold feature matrices ─────────────────────────────────
    # Vocab is fitted on each fold's training rows only (excludes val + early-stop
    # chromosomes) so k-mers unique to held-out sequences don't inflate the feature space.
    # Matrices are built once here — the Optuna objective reuses them across all trials.
    print(f"\nFitting per-fold vocabularies and building feature matrices ...")
    fold_data: dict[str, tuple] = {}
    feature_cols: list[str] = []
    for i, val_chrom in enumerate(cv_chroms):
        es_chrom = cv_chroms[(i + 1) % len(cv_chroms)]
        val_mask = chrom_arr == val_chrom
        es_mask = chrom_arr == es_chrom
        train_mask = ~val_mask & ~es_mask

        train_recs_fold = [r for r, m in zip(train_records, train_mask) if m]
        val_recs_fold   = [r for r, m in zip(train_records, val_mask)   if m]
        es_recs_fold    = [r for r, m in zip(train_records, es_mask)    if m]

        fold_vocab = fit_vocab([r.target_sequence for r in train_recs_fold], args.k)
        X_tr, y_tr, cols = build_features(
            train_recs_fold, k=args.k, include_distance=args.include_distance,
            sparse=True, vocab=fold_vocab,
        )
        X_val, y_val, _ = build_features(
            val_recs_fold, k=args.k, include_distance=args.include_distance,
            sparse=True, vocab=fold_vocab,
        )
        X_es, y_es, _ = build_features(
            es_recs_fold, k=args.k, include_distance=args.include_distance,
            sparse=True, vocab=fold_vocab,
        )
        fold_data[val_chrom] = (X_tr, y_tr, X_val, y_val, X_es, y_es)
        if not feature_cols:
            feature_cols = cols
        print(f"  fold chr{val_chrom}: {len(fold_vocab)}/{4**args.k} k-mers  "
              f"train={X_tr.shape[0]:,}  val={X_val.shape[0]:,}  es={X_es.shape[0]:,}")
        gc.collect()

    # ── Optuna objective ───────────────────────────────────────────────────────
    cv_rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        # Scale early-stopping patience with learning rate: lower LR needs more rounds
        # to detect real improvement vs. noise.
        es_rounds = max(50, int(0.5 / lr))
        base_params = dict(
            n_estimators=5000,
            learning_rate=lr,
            max_depth=trial.suggest_int("max_depth", 3, 9),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            objective=args.objective,
            tree_method="hist",
            nthread=args.nthread,
            random_state=args.seed,
        )

        fold_rhos: list[float] = []
        for i, val_chrom in enumerate(cv_chroms):
            X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[val_chrom]

            # Fresh callback per fold — EarlyStopping holds per-training state
            model = xgb.XGBRegressor(
                **base_params,
                callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
            )
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
            del X_tr, y_tr, X_es, y_es
            gc.collect()

            y_pred = model.predict(X_val)
            rho, _ = spearmanr(y_val, y_pred)
            n_trees = model.best_iteration + 1
            del X_val, y_val

            fold_rhos.append(float(rho))
            cv_rows.append({
                "trial": trial.number,
                "chromosome": val_chrom,
                "n_val": int(val_mask.sum()),
                "spearman_rho": float(rho),
                "best_n_estimators": int(n_trees),
            })

            # Intermediate report so MedianPruner can kill poor trials early
            trial.report(float(np.mean(fold_rhos)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rhos))

    # ── Run study ─────────────────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    # Don't prune until 10 trials complete; within a trial, wait 5 folds before pruning
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print(f"\nRunning Optuna TPE study: {args.n_trials} trials × {len(cv_chroms)} CV folds")
    print(f"  (up to {args.n_trials * len(cv_chroms):,} model fits)\n")

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_value
        pruned = trial.state == optuna.trial.TrialState.PRUNED
        status = "PRUNED" if pruned else f"rho={trial.value:.4f}"
        print(f"  Trial {trial.number:>3d}  {status:<18}  best={best:.4f}", flush=True)
        if cv_rows:
            pruned_set = [t.number for t in study.trials
                          if t.state == optuna.trial.TrialState.PRUNED]
            snap = pl.DataFrame(cv_rows).with_columns(
                pl.col("trial").is_in(pruned_set).alias("pruned")
            )
            snap.write_csv(cv_dir / "cv_scores.csv")

    try:
        study.optimize(objective, n_trials=args.n_trials, callbacks=[_trial_callback])
    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results ...")

    if not cv_rows:
        print("No completed CV folds — exiting.")
        return

    # ── Save CV scores ─────────────────────────────────────────────────────────
    pruned_trials = [t.number for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED]
    cv_df = pl.DataFrame(cv_rows).with_columns(
        pl.col("trial").is_in(pruned_trials).alias("pruned")
    )
    cv_path = cv_dir / "cv_scores.csv"
    cv_df.write_csv(cv_path)
    print(f"\nCV scores saved  -> {cv_path}")

    # ── Report best trial ──────────────────────────────────────────────────────
    best_trial = study.best_trial
    best_cv = cv_df.filter(pl.col("trial") == best_trial.number)
    mean_rho = best_cv["spearman_rho"].mean()
    std_rho = best_cv["spearman_rho"].std()

    print(f"\nBest trial: #{best_trial.number}  CV Spearman rho = {mean_rho:.4f} ± {std_rho:.4f}")
    print(f"\nPer-chromosome CV scores (best trial):")
    print(f"  {'chrom':<8}  {'n_val':>8}  {'rho':>8}  {'n_trees':>8}")
    sorted_cv = best_cv.sort(
        by=[pl.col("chromosome").str.len_chars(), pl.col("chromosome")]
    )
    for row in sorted_cv.iter_rows(named=True):
        print(f"  chr{row['chromosome']:<5}  {int(row['n_val']):>8,}  "
              f"{row['spearman_rho']:>8.4f}  {int(row['best_n_estimators']):>8}")

    # ── Save best hyperparameter config ───────────────────────────────────────
    median_n_est = int(np.median(best_cv["best_n_estimators"].to_numpy()))
    best_params_doc = {
        "k": args.k,
        "objective": args.objective,
        "include_distance": args.include_distance,
        "seed": args.seed,
        "cv_mean_spearman_rho": float(mean_rho),
        "cv_std_spearman_rho": float(std_rho),
        "n_estimators_cv_median": median_n_est,
        **{
            k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
            for k, v in best_trial.params.items()
        },
    }
    params_path = model_dir / f"xgboost_best_params_k{args.k}_{_obj_tag}.json"
    with open(params_path, "w") as fh:
        json.dump(best_params_doc, fh, indent=2)
    print(f"\nBest params saved -> {params_path}")
    print(json.dumps(best_params_doc, indent=2))

    # ── Retrain final model on all training data ───────────────────────────────
    print(f"\nRetraining final model on all training data ...")
    print(f"  early-stop val chromosome: {final_val_chrom}")

    final_val_mask = chrom_arr == final_val_chrom
    final_train_mask = ~final_val_mask
    final_train_recs = [r for r, m in zip(train_records, final_train_mask) if m]
    final_val_recs   = [r for r, m in zip(train_records, final_val_mask)   if m]
    print(f"  final train: {len(final_train_recs):,}  early-stop val: {len(final_val_recs):,}")

    # Fit vocab on final training rows only (excludes early-stop chromosome).
    final_vocab = fit_vocab([r.target_sequence for r in final_train_recs], args.k)
    print(f"  Final vocab: {len(final_vocab)}/{4**args.k} k-mers observed")
    X_final_tr, y_final_tr, _ = build_features(
        final_train_recs, k=args.k, include_distance=args.include_distance,
        sparse=True, vocab=final_vocab,
    )
    X_final_val, y_final_val, _ = build_features(
        final_val_recs, k=args.k, include_distance=args.include_distance,
        sparse=True, vocab=final_vocab,
    )
    gc.collect()

    bp = best_trial.params
    final_es_rounds = max(50, int(0.5 / bp["learning_rate"]))
    final_model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=bp["learning_rate"],
        max_depth=bp["max_depth"],
        subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"],
        min_child_weight=bp["min_child_weight"],
        reg_alpha=bp["reg_alpha"],
        reg_lambda=bp["reg_lambda"],
        objective=args.objective,
        tree_method="hist",
        nthread=args.nthread,
        random_state=args.seed,
        callbacks=[xgb.callback.EarlyStopping(rounds=final_es_rounds, save_best=True)],
    )
    final_model.fit(
        X_final_tr, y_final_tr,
        eval_set=[(X_final_val, y_final_val)],
        verbose=100,
    )
    del X_final_tr, y_final_tr, X_final_val, y_final_val
    gc.collect()
    final_n_trees = final_model.best_iteration + 1
    print(f"  Final model: {final_n_trees} trees")

    model_path = model_dir / f"xgboost_k{args.k}_tuned_{_obj_tag}.ubj"
    final_model.save_model(str(model_path))
    print(f"  Model saved  -> {model_path}")

    vocab_path = model_dir / f"xgboost_vocab_k{args.k}_{_obj_tag}.json"
    with open(vocab_path, "w") as fh:
        json.dump(final_vocab, fh)
    print(f"  Vocab saved  -> {vocab_path}")

    best_params_doc["n_estimators_final_model"] = final_n_trees
    with open(params_path, "w") as fh:
        json.dump(best_params_doc, fh, indent=2)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\nEvaluating on held-out test set ...")
    X_test, y_test, _ = build_features(
        test_records, k=args.k, include_distance=args.include_distance, sparse=True,
        vocab=final_vocab,
    )
    y_test_pred = final_model.predict(X_test)
    del X_test
    gc.collect()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = out_dir / "results" / f"final_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Filter by cell line / day using the original records — no copy of X_test needed.
    print("\nTest set metrics:")
    metrics_rows = [compute_metrics("Overall", y_test, y_test_pred)]
    for cl in _CELL_LINES:
        mask = np.array([r.cell_line == cl for r in test_records])
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(cl, y_test[mask], y_test_pred[mask]))
    for day in _DAYS:
        mask = np.array([r.day == day for r in test_records])
        if mask.sum() == 0:
            continue
        metrics_rows.append(compute_metrics(f"day_{day}", y_test[mask], y_test_pred[mask]))
    for cl in _CELL_LINES:
        for day in _DAYS:
            mask = np.array([r.cell_line == cl and r.day == day for r in test_records])
            if mask.sum() == 0:
                continue
            metrics_rows.append(compute_metrics(f"{cl}_day{day}", y_test[mask], y_test_pred[mask]))

    metrics_path = eval_dir / "metrics.csv"
    pl.DataFrame(metrics_rows).write_csv(metrics_path)
    print(f"\nMetrics CSV      -> {metrics_path}")

    preds_rows = [
        {
            "target": rec.target,
            "cell_line": rec.cell_line,
            "day": rec.day,
            "replicate": rec.replicate,
            "y_true": float(y_t),
            "y_pred": float(y_p),
        }
        for rec, y_t, y_p in zip(test_records, y_test, y_test_pred)
    ]
    preds_df = pl.DataFrame(preds_rows)
    preds_path = eval_dir / "predictions.csv"
    preds_df.write_csv(preds_path)
    print(f"Predictions CSV  -> {preds_path}")

    present_cls = [cl for cl in _CELL_LINES if (preds_df["cell_line"] == cl).any()]
    n_panels = 1 + len(present_cls)
    fig = plt.figure(figsize=(4 * n_panels, 4))
    gs = gridspec.GridSpec(1, n_panels, figure=fig, wspace=0.4)
    _scatter_panel(
        fig.add_subplot(gs[0, 0]),
        preds_df["y_true"].to_numpy(), preds_df["y_pred"].to_numpy(), "Overall",
    )
    for i, cl in enumerate(present_cls):
        sub = preds_df.filter(pl.col("cell_line") == cl)
        _scatter_panel(
            fig.add_subplot(gs[0, i + 1]),
            sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), cl,
        )
    fig.suptitle(
        f"Predicted vs. Observed log2FC  (k={args.k}, tuned)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    scatter_path = eval_dir / "scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot     -> {scatter_path}")

    run_info = {
        "train_file": str(args.train),
        "test_file": str(args.test),
        "k": args.k,
        "objective": args.objective,
        "include_distance": args.include_distance,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_cv_folds": len(cv_chroms),
        "cv_chroms": cv_chroms,
        "best_trial": best_trial.number,
        "cv_mean_spearman_rho": float(mean_rho),
        "cv_std_spearman_rho": float(std_rho),
        "final_val_chrom": final_val_chrom,
        "final_n_estimators": final_n_trees,
        "n_test_records": len(test_records),
        "timestamp": timestamp,
        "git_commit": _git_commit(),
    }
    run_info_path = eval_dir / "run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2)
    print(f"Run info JSON    -> {run_info_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
