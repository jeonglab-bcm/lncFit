"""Hyperparameter tuning for the lncRNA RRA-hit XGBoost classifier (issue #62).

Same manner as scripts/tune_xgboost.py — Optuna TPE + chromosome LOCO-CV — adapted
for the Day-14 lncRNA-level binary classification task from issue #60/#61:

  - Search space: the same 7 hyperparameters as tune_xgboost.py, PLUS a
    scale_pos_weight_mult that scales each fold's natural neg/pos ratio (~21x on
    the full train set, but varies per fold). scale_pos_weight was fixed in the
    initial classifier (#61); here it is tuned, not just computed once.
  - CV objective: mean AUPRC across folds (more informative than AUROC/accuracy
    at this ~4.5% positive rate).
  - After tuning, retrains a final model on all training data and evaluates on the
    held-out chr1 test split.

Outputs:
  data/model/xgboost_lncrna_best_params_k<K>.json   best hyperparameter configuration
  results/lncrna_rra_day14/tune_k<K>/cv_scores.csv   per-chromosome AUPRC for every trial
  results/lncrna_rra_day14/tune_k<K>/final_eval/     final held-out test evaluation bundle
"""

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.cv import build_lncrna_folds
from lncfit.features import build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group


def _classifier_kwargs(params: dict, scale_pos_weight: float, nthread: int, seed: int, n_estimators: int) -> dict:
    return dict(
        n_estimators=n_estimators,
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        nthread=nthread,
        random_state=seed,
    )


def _natural_ratio(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / n_pos if n_pos > 0 else 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Tune the lncRNA RRA-hit XGBoost classifier via Optuna TPE + chromosome LOCO-CV."
    )
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default 50; 50-100 recommended).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--final-val-chrom", default="22",
        help="Chromosome held out for early stopping when retraining the final model "
             "(never used during CV evaluation). Default: 22.",
    )
    parser.add_argument("--output-dir", default=".",
                        help="Root directory for all outputs (default: current directory).")
    parser.add_argument("--nthread", type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "data" / "model"
    tune_dir = out_dir / "results" / "lncrna_rra_day14" / f"tune_k{args.k}"
    model_dir.mkdir(parents=True, exist_ok=True)
    tune_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training records from {args.train} ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    print(f"  {len(train_records):,} records")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  {len(test_records):,} records")

    print()
    cv_chroms, fold_data, feature_cols = build_lncrna_folds(
        train_records, k=args.k, include_distance=args.include_distance,
    )
    print(f"\nCV chromosomes ({len(cv_chroms)} folds): {cv_chroms}")

    if args.final_val_chrom not in cv_chroms:
        fallback = cv_chroms[-1]
        print(f"  Warning: --final-val-chrom '{args.final_val_chrom}' not found; using '{fallback}' instead.")
        final_val_chrom = fallback
    else:
        final_val_chrom = args.final_val_chrom

    cv_rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
        es_rounds = max(50, int(0.5 / lr))
        trial_params = {
            "learning_rate": lr,
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        scale_pos_weight_mult = trial.suggest_float("scale_pos_weight_mult", 0.25, 4.0, log=True)

        fold_auprcs: list[float] = []
        for i, val_chrom in enumerate(cv_chroms):
            X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[val_chrom]
            scale_pos_weight = _natural_ratio(y_tr) * scale_pos_weight_mult

            model = xgb.XGBClassifier(
                **_classifier_kwargs(trial_params, scale_pos_weight, args.nthread, args.seed, n_estimators=2000),
                callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
            )
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

            y_pred = model.predict_proba(X_val)[:, 1]
            n_pos_val = int(y_val.sum())
            auprc = float(average_precision_score(y_val, y_pred)) if 0 < n_pos_val < len(y_val) else float("nan")
            n_trees = model.best_iteration + 1
            n_val = X_val.shape[0]

            fold_auprcs.append(auprc)
            cv_rows.append({
                "trial": trial.number,
                "chromosome": val_chrom,
                "n_val": int(n_val),
                "n_pos_val": n_pos_val,
                "auprc": auprc,
                "scale_pos_weight": scale_pos_weight,
                "best_n_estimators": int(n_trees),
            })

            mean_so_far = float(np.nanmean(fold_auprcs))
            trial.report(mean_so_far, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.nanmean(fold_auprcs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print(f"\nRunning Optuna TPE study: {args.n_trials} trials x {len(cv_chroms)} CV folds")
    print(f"  (up to {args.n_trials * len(cv_chroms):,} model fits)\n")

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        best = study.best_value
        pruned = trial.state == optuna.trial.TrialState.PRUNED
        status = "PRUNED" if pruned else f"auprc={trial.value:.4f}"
        print(f"  Trial {trial.number:>3d}  {status:<18}  best={best:.4f}", flush=True)
        if cv_rows:
            pd.DataFrame(cv_rows).to_csv(tune_dir / "cv_scores.csv", index=False)

    try:
        study.optimize(objective, n_trials=args.n_trials, callbacks=[_trial_callback])
    except KeyboardInterrupt:
        print("\nInterrupted -- saving partial results ...")

    if not cv_rows:
        print("No completed CV folds -- exiting.")
        return

    cv_df = pd.DataFrame(cv_rows)
    cv_path = tune_dir / "cv_scores.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\nCV scores saved  -> {cv_path}")

    best_trial = study.best_trial
    best_cv = cv_df[cv_df["trial"] == best_trial.number]
    mean_auprc = best_cv["auprc"].mean()
    std_auprc = best_cv["auprc"].std()

    print(f"\nBest trial: #{best_trial.number}  CV AUPRC = {mean_auprc:.4f} +/- {std_auprc:.4f}")
    print(f"\nPer-chromosome CV scores (best trial):")
    print(f"  {'chrom':<8}  {'n_val':>8}  {'n_pos':>6}  {'auprc':>8}  {'n_trees':>8}")
    for _, row in best_cv.sort_values(by="chromosome", key=lambda s: s.str.zfill(2)).iterrows():
        print(f"  chr{row['chromosome']:<5}  {int(row['n_val']):>8,}  {int(row['n_pos_val']):>6}  "
              f"{row['auprc']:>8.4f}  {int(row['best_n_estimators']):>8}")

    median_n_est = int(np.median(best_cv["best_n_estimators"].to_numpy()))
    best_params_doc = {
        "k": args.k,
        "include_distance": args.include_distance,
        "seed": args.seed,
        "cv_mean_auprc": float(mean_auprc),
        "cv_std_auprc": float(std_auprc),
        "n_estimators_cv_median": median_n_est,
        **{
            key: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
            for key, v in best_trial.params.items()
        },
    }
    params_path = model_dir / f"xgboost_lncrna_best_params_k{args.k}.json"
    with open(params_path, "w") as fh:
        json.dump(best_params_doc, fh, indent=2)
    print(f"\nBest params saved -> {params_path}")
    print(json.dumps(best_params_doc, indent=2))

    # ── Retrain final model on all training data ───────────────────────────────
    print(f"\nRetraining final model on all training data ...")
    print(f"  early-stop val chromosome: {final_val_chrom}")

    chrom_arr = np.array([r.chrom for r in train_records])
    final_val_mask = chrom_arr == final_val_chrom
    final_train_mask = ~final_val_mask
    final_train_recs = [r for r, m in zip(train_records, final_train_mask) if m]
    final_val_recs   = [r for r, m in zip(train_records, final_val_mask)   if m]
    print(f"  final train: {len(final_train_recs):,}  early-stop val: {len(final_val_recs):,}")

    final_guide_seqs = [seq for r in final_train_recs for seq in r.guide_sequences]
    final_vocab = fit_vocab(final_guide_seqs, args.k)
    print(f"  Final vocab: {len(final_vocab)}/{4**args.k} k-mers observed")

    X_final_tr, y_final_tr, _ = build_lncrna_features(
        final_train_recs, k=args.k, include_distance=args.include_distance, vocab=final_vocab, sparse=True,
    )
    X_final_val, y_final_val, _ = build_lncrna_features(
        final_val_recs, k=args.k, include_distance=args.include_distance, vocab=final_vocab, sparse=True,
    )
    gc.collect()

    bp = best_trial.params
    final_scale_pos_weight = _natural_ratio(y_final_tr) * bp["scale_pos_weight_mult"]
    final_es_rounds = max(50, int(0.5 / bp["learning_rate"]))
    final_model = xgb.XGBClassifier(
        **_classifier_kwargs(bp, final_scale_pos_weight, args.nthread, args.seed, n_estimators=2000),
        callbacks=[xgb.callback.EarlyStopping(rounds=final_es_rounds, save_best=True)],
    )
    final_model.fit(X_final_tr, y_final_tr, eval_set=[(X_final_val, y_final_val)], verbose=100)
    del X_final_tr, y_final_tr, X_final_val, y_final_val
    gc.collect()
    final_n_trees = final_model.best_iteration + 1
    print(f"  Final model: {final_n_trees} trees  scale_pos_weight={final_scale_pos_weight:.2f}")

    model_path = model_dir / f"xgboost_lncrna_day14_k{args.k}_tuned.ubj"
    final_model.save_model(str(model_path))
    print(f"  Model saved  -> {model_path}")

    vocab_path = model_dir / f"xgboost_lncrna_day14_k{args.k}_tuned_vocab.json"
    with open(vocab_path, "w") as fh:
        json.dump(final_vocab, fh)
    print(f"  Vocab saved  -> {vocab_path}")

    best_params_doc["n_estimators_final_model"] = final_n_trees
    best_params_doc["final_scale_pos_weight"] = final_scale_pos_weight
    with open(params_path, "w") as fh:
        json.dump(best_params_doc, fh, indent=2)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\nEvaluating on held-out test set ...")
    X_test, y_test, _ = build_lncrna_features(
        test_records, k=args.k, include_distance=args.include_distance, vocab=final_vocab, sparse=True,
    )
    y_test_pred = final_model.predict_proba(X_test)[:, 1]
    del X_test
    gc.collect()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = tune_dir / f"final_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\nTest set metrics:")
    metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_test_pred)

    metrics_path = eval_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"\nMetrics CSV      -> {metrics_path}")

    preds_rows = [
        {"target": rec.target, "cell_line": rec.cell_line, "y_true": float(y_t), "y_pred_proba": float(y_p)}
        for rec, y_t, y_p in zip(test_records, y_test, y_test_pred)
    ]
    preds_path = eval_dir / "predictions.csv"
    pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
    print(f"Predictions CSV  -> {preds_path}")

    run_info = {
        "train_file": str(args.train),
        "test_file": str(args.test),
        "k": args.k,
        "include_distance": args.include_distance,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_cv_folds": len(cv_chroms),
        "cv_chroms": cv_chroms,
        "best_trial": best_trial.number,
        "cv_mean_auprc": float(mean_auprc),
        "cv_std_auprc": float(std_auprc),
        "final_val_chrom": final_val_chrom,
        "final_n_estimators": final_n_trees,
        "final_scale_pos_weight": final_scale_pos_weight,
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
