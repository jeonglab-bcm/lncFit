"""Hyperparameter tuning for the lncRNA RRA-hit classifier via row-level stratified CV.

Companion to scripts/tune_lncrna_xgboost.py, which uses chromosome LOCO-CV. This
script instead uses plain StratifiedKFold(n_splits=5) over the binary label,
ignoring chromosome entirely (lncfit.cv.build_lncrna_stratified_folds). Requested
explicitly to compare against the chromosome-grouped numbers -- note this is NOT
leak-free: every cell-line row for a given lncRNA shares one k-mer vector (only the
cell-line one-hot differs), so the same lncRNA's sequence can appear in both a
fold's train and validation split via its other cell-line rows. Expect this CV's
scores to run optimistic relative to the chromosome-grouped CV and the true chr1
held-out numbers.

Supports two models (--model xgboost|logreg) and a class-weight toggle
(--class-weight on|off) to isolate how much of any performance sits on
imbalance-reweighting vs the model/features themselves:
  - xgboost: class-weight on tunes scale_pos_weight_mult (as in
    tune_lncrna_xgboost.py); off fixes scale_pos_weight=1 (no reweighting).
  - logreg: class-weight on sets class_weight="balanced"; off sets None.

CV objective is mean AUPRC across folds. After tuning, retrains a final model on
all training data (XGBoost early-stops on a random stratified 10% carve-out;
logreg needs no carve-out) and evaluates once on the existing chr1 held-out test
split, so results are directly comparable to metrics_k*.csv / tune_k*/.

Outputs (under --output-dir):
  data/model/<model>_lncrna_stratified_k<K>_cw<on|off>_best_params.json
  results/lncrna_rra_day14/tune_stratified/<model>_k<K>_cw<on|off>/cv_scores.csv
  results/lncrna_rra_day14/tune_stratified/<model>_k<K>_cw<on|off>/final_eval_<ts>/
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
from scipy.sparse import vstack
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.cv import build_lncrna_stratified_folds
from lncfit.features import build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group


def _natural_ratio(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / n_pos if n_pos > 0 else 1.0


def _xgb_kwargs(params: dict, scale_pos_weight: float, nthread: int, seed: int, n_estimators: int) -> dict:
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


def load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Tune the lncRNA RRA-hit classifier via row-level stratified 5-fold CV "
                    "(chromosome-agnostic; see module docstring for the leakage caveat)."
    )
    parser.add_argument("--model", required=True, choices=["xgboost", "logreg"])
    parser.add_argument("--class-weight", required=True, choices=["on", "off"],
                        help="on: tune/apply imbalance reweighting; off: disable it entirely.")
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--variance-threshold", type=float, default=0.0,
                        help="Drop k-mer feature columns with train-fold variance at or below this "
                             "value (fit per fold / on final train only, never on val/test). 0 = off. "
                             "Mainly useful at k=6 (4096 columns, most near-constant).")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--nthread", type=int, default=-1, help="XGBoost only.")
    args = parser.parse_args()

    class_weight_on = args.class_weight == "on"
    run_tag = f"{args.model}_k{args.k}_cw{args.class_weight}"

    out_dir = Path(args.output_dir)
    model_dir = out_dir / "data" / "model"
    tune_dir = out_dir / "results" / "lncrna_rra_day14" / "tune_stratified" / run_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    tune_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training records from {args.train} ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    print(f"  {len(train_records):,} records")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  {len(test_records):,} records")

    print(f"Loading transcript sequences from {args.transcript_sequences} ...")
    transcript_sequences = load_transcript_sequences(args.transcript_sequences)
    print(f"  {len(transcript_sequences):,} lncRNAs")

    print()
    fold_ids, fold_data, feature_cols = build_lncrna_stratified_folds(
        train_records, transcript_sequences, k=args.k, n_splits=args.n_splits,
        include_distance=args.include_distance, seed=args.seed,
        variance_threshold=args.variance_threshold,
    )
    print(f"\nCV folds ({len(fold_ids)}, stratified, chromosome-agnostic)")

    cv_rows: list[dict] = []

    # ── Optuna objective, dispatched by model ───────────────────────────────────
    if args.model == "xgboost":
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
            scale_pos_weight_mult = (
                trial.suggest_float("scale_pos_weight_mult", 0.25, 4.0, log=True) if class_weight_on else None
            )

            fold_auprcs: list[float] = []
            for i in fold_ids:
                X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[i]
                spw = _natural_ratio(y_tr) * scale_pos_weight_mult if class_weight_on else 1.0

                model = xgb.XGBClassifier(
                    **_xgb_kwargs(trial_params, spw, args.nthread, args.seed, n_estimators=2000),
                    callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
                )
                model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)

                y_pred = model.predict_proba(X_val)[:, 1]
                n_pos_val = int(y_val.sum())
                auprc = float(average_precision_score(y_val, y_pred)) if 0 < n_pos_val < len(y_val) else float("nan")
                n_trees = model.best_iteration + 1

                fold_auprcs.append(auprc)
                cv_rows.append({
                    "trial": trial.number, "fold": i,
                    "n_val": int(X_val.shape[0]), "n_pos_val": n_pos_val,
                    "auprc": auprc, "scale_pos_weight": spw, "best_n_estimators": int(n_trees),
                })

                mean_so_far = float(np.nanmean(fold_auprcs))
                trial.report(mean_so_far, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.nanmean(fold_auprcs))

    else:  # logreg
        def objective(trial: optuna.Trial) -> float:
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            class_weight = "balanced" if class_weight_on else None

            fold_auprcs: list[float] = []
            for i in fold_ids:
                X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[i]
                # logreg has no early-stopping need; fold the ES carve-out back into train.
                X_tr_full = vstack([X_tr, X_es])
                y_tr_full = np.concatenate([y_tr, y_es])

                model = LogisticRegression(
                    C=C, max_iter=1000, class_weight=class_weight, random_state=args.seed,
                )
                model.fit(X_tr_full, y_tr_full)

                y_pred = model.predict_proba(X_val)[:, 1]
                n_pos_val = int(y_val.sum())
                auprc = float(average_precision_score(y_val, y_pred)) if 0 < n_pos_val < len(y_val) else float("nan")

                fold_auprcs.append(auprc)
                cv_rows.append({
                    "trial": trial.number, "fold": i,
                    "n_val": int(X_val.shape[0]), "n_pos_val": n_pos_val,
                    "auprc": auprc, "C": C,
                })

                mean_so_far = float(np.nanmean(fold_auprcs))
                trial.report(mean_so_far, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.nanmean(fold_auprcs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print(f"\nRunning Optuna TPE study: {args.n_trials} trials x {len(fold_ids)} CV folds "
          f"[{args.model}, class_weight={args.class_weight}, k={args.k}]")

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

    best_params_doc = {
        "model": args.model,
        "class_weight": args.class_weight,
        "k": args.k,
        "include_distance": args.include_distance,
        "variance_threshold": args.variance_threshold,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "cv_mean_auprc": float(mean_auprc),
        "cv_std_auprc": float(std_auprc),
        **{
            key: (float(v) if isinstance(v, float) else int(v) if isinstance(v, int) else v)
            for key, v in best_trial.params.items()
        },
    }
    bp = best_trial.params

    # ── Retrain final model on all training data ───────────────────────────────
    print(f"\nRetraining final {args.model} model on all training data ...")

    if args.model == "xgboost":
        y_all_train = np.array([r.label for r in train_records])
        idx = np.arange(len(train_records))
        final_train_idx, final_es_idx = train_test_split(
            idx, test_size=0.1, stratify=y_all_train, random_state=args.seed,
        )
        final_train_recs = [train_records[i] for i in final_train_idx]
        final_es_recs = [train_records[i] for i in final_es_idx]
        print(f"  final train: {len(final_train_recs):,}  early-stop val: {len(final_es_recs):,} "
              f"(random stratified 90/10 carve-out; no chromosome grouping)")

        final_targets = {r.target for r in final_train_recs}
        final_seqs = [transcript_sequences[t] for t in final_targets if t in transcript_sequences]
        final_vocab = fit_vocab(final_seqs, args.k)
        print(f"  Final vocab: {len(final_vocab)}/{4**args.k} k-mers observed")

        X_final_tr, y_final_tr, _ = build_lncrna_features(
            final_train_recs, transcript_sequences, k=args.k, include_distance=args.include_distance,
            vocab=final_vocab, sparse=True,
        )
        X_final_es, y_final_es, _ = build_lncrna_features(
            final_es_recs, transcript_sequences, k=args.k, include_distance=args.include_distance,
            vocab=final_vocab, sparse=True,
        )
        gc.collect()

        final_selector = None
        if args.variance_threshold > 0:
            n_before = X_final_tr.shape[1]
            final_selector = VarianceThreshold(threshold=args.variance_threshold).fit(X_final_tr)
            X_final_tr = final_selector.transform(X_final_tr)
            X_final_es = final_selector.transform(X_final_es)
            print(f"  Variance filter: {X_final_tr.shape[1]}/{n_before} columns kept "
                  f"(threshold={args.variance_threshold})")

        final_scale_pos_weight = _natural_ratio(y_final_tr) * bp["scale_pos_weight_mult"] if class_weight_on else 1.0
        final_es_rounds = max(50, int(0.5 / bp["learning_rate"]))
        final_model = xgb.XGBClassifier(
            **_xgb_kwargs(bp, final_scale_pos_weight, args.nthread, args.seed, n_estimators=2000),
            callbacks=[xgb.callback.EarlyStopping(rounds=final_es_rounds, save_best=True)],
        )
        final_model.fit(X_final_tr, y_final_tr, eval_set=[(X_final_es, y_final_es)], verbose=100)
        n_features_final = X_final_tr.shape[1]
        del X_final_tr, y_final_tr, X_final_es, y_final_es
        gc.collect()
        final_n_trees = final_model.best_iteration + 1
        print(f"  Final model: {final_n_trees} trees  scale_pos_weight={final_scale_pos_weight:.2f}")

        model_path = model_dir / f"xgboost_lncrna_stratified_k{args.k}_cw{args.class_weight}.ubj"
        final_model.save_model(str(model_path))
        best_params_doc["n_estimators_final_model"] = final_n_trees
        best_params_doc["final_scale_pos_weight"] = final_scale_pos_weight
        best_params_doc["n_features_final"] = int(n_features_final)
        predict_fn = lambda X: final_model.predict_proba(X)[:, 1]  # noqa: E731

    else:  # logreg -- no ES carve-out needed
        final_vocab = fit_vocab(
            [transcript_sequences[t] for t in {r.target for r in train_records} if t in transcript_sequences],
            args.k,
        )
        print(f"  Final vocab: {len(final_vocab)}/{4**args.k} k-mers observed")
        X_final_tr, y_final_tr, _ = build_lncrna_features(
            train_records, transcript_sequences, k=args.k, include_distance=args.include_distance,
            vocab=final_vocab, sparse=True,
        )
        final_selector = None
        if args.variance_threshold > 0:
            n_before = X_final_tr.shape[1]
            final_selector = VarianceThreshold(threshold=args.variance_threshold).fit(X_final_tr)
            X_final_tr = final_selector.transform(X_final_tr)
            print(f"  Variance filter: {X_final_tr.shape[1]}/{n_before} columns kept "
                  f"(threshold={args.variance_threshold})")

        class_weight = "balanced" if class_weight_on else None
        final_model = LogisticRegression(C=bp["C"], max_iter=1000, class_weight=class_weight, random_state=args.seed)
        final_model.fit(X_final_tr, y_final_tr)
        best_params_doc["n_features_final"] = int(X_final_tr.shape[1])
        del X_final_tr, y_final_tr
        gc.collect()
        predict_fn = lambda X: final_model.predict_proba(X)[:, 1]  # noqa: E731

    params_path = model_dir / f"{args.model}_lncrna_stratified_k{args.k}_cw{args.class_weight}_best_params.json"
    with open(params_path, "w") as fh:
        json.dump(best_params_doc, fh, indent=2)
    print(f"\nBest params saved -> {params_path}")
    print(json.dumps(best_params_doc, indent=2))

    vocab_path = model_dir / f"{args.model}_lncrna_stratified_k{args.k}_cw{args.class_weight}_vocab.json"
    with open(vocab_path, "w") as fh:
        json.dump(final_vocab, fh)
    print(f"Vocab saved      -> {vocab_path}")

    if final_selector is not None:
        mask_path = model_dir / f"{args.model}_lncrna_stratified_k{args.k}_cw{args.class_weight}_variance_mask.json"
        with open(mask_path, "w") as fh:
            json.dump(final_selector.get_support().tolist(), fh)
        print(f"Variance mask    -> {mask_path} "
              f"(boolean, over [vocab + cell-line one-hot] columns in vocab order)")

    # ── Evaluate on the existing chr1 held-out test set ─────────────────────────
    print("\nEvaluating on held-out chr1 test set ...")
    X_test, y_test, _ = build_lncrna_features(
        test_records, transcript_sequences, k=args.k, include_distance=args.include_distance,
        vocab=final_vocab, sparse=True,
    )
    if final_selector is not None:
        X_test = final_selector.transform(X_test)
    y_test_pred = predict_fn(X_test)
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
        "model": args.model,
        "class_weight": args.class_weight,
        "cv_scheme": "stratified_row_level",
        "train_file": str(args.train),
        "test_file": str(args.test),
        "k": args.k,
        "include_distance": args.include_distance,
        "variance_threshold": args.variance_threshold,
        "n_features_final": best_params_doc.get("n_features_final"),
        "n_splits": args.n_splits,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial": best_trial.number,
        "cv_mean_auprc": float(mean_auprc),
        "cv_std_auprc": float(std_auprc),
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
