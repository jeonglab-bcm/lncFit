#!/usr/bin/env python3
"""Leave-one-cell-line-out (LOCO) evaluation for the lncRNA-hit classifier.

Unlike scripts/run_pipeline.py (which holds out chromosome 1 and asks "does
this generalize to an unseen lncRNA?"), this asks a different question: "does
this generalize to a cell line the model never saw a single row of during
training?" There are 5 cell lines in the day14 dataset (HAP1, HEK293FT, K562,
MDA-MB-231, THP1); for each one in turn, a model is trained on the other 4
and used to predict that cell line's rows. Every record ends up predicted by
a model that never trained on its cell line, and the 5 folds' predictions are
stitched into one predictions.csv covering the whole dataset -- this is what
the lncrna_rra_day14_cellline_loco leaderboard challenge scores.

Config schema is the model/features subset of scripts/run_pipeline.py's YAML
(see configs/README.md), plus an optional `tuning` section:
  tuning:
    method: fixed (default) or optuna
    search_space: configs/search_spaces/<model>.yaml
    n_trials: 50
    metric: auprc or auroc
When tuning.method is "optuna", hyperparameters are selected ONCE via a
stratified 5-fold CV over the full (non-held-out-cell-line) dataset, then
that single fixed set of best_params is used for all 4-5 outer
leave-one-cell-line-out folds. This is a deliberate simplification -- a
fully nested search (nest an inner nested tuning CV inside each outer fold)
would cost ~n_folds times as many model fits for a fairly marginal rigor
gain, since hyperparameters like max_depth/learning_rate are structural
choices, not label-specific to whichever cell line ends up held out.

Usage:
  python scripts/run_cellline_loco.py --config configs/cellline_loco/xgboost_kmer.yaml
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.classifiers import build_classifier
from lncfit.cv import make_cv_splits
from lncfit.embeddings import load_embeddings
from lncfit.features import build_lncrna_embedding_features, build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

_DEFAULT_DATA_PATH = "data/processed/lncrna_rra_day14.jsonl.gz"
_METRIC_FNS = {"auprc": average_precision_score, "auroc": roc_auc_score}


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def _cv_score(model_name, params, seed, X, y, splits, metric) -> float:
    scores = []
    for train_mask, val_mask, _ in splits:
        model = build_classifier(model_name, **{"seed": seed, **params})
        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict_proba(X[val_mask])
        y_val = y[val_mask]
        if len(np.unique(y_val)) < 2:
            continue
        scores.append(_METRIC_FNS[metric](y_val, y_pred))
    return float(np.nanmean(scores)) if scores else float("nan")


def _tune_optuna(model_name, seed, X, y, splits, search_space_path, n_trials, metric) -> dict:
    with open(search_space_path) as fh:
        search_space = yaml.safe_load(fh)

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, spec in search_space.items():
            param_type = spec.get("type", "float")
            if param_type == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif param_type == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type {param_type!r} for param {name!r}")
        return _cv_score(model_name, params, seed, X, y, splits, metric)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  trial {trial.number:>3d}  {metric}={trial.value:.4f}  best={study.best_value:.4f}", flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[_trial_callback])
    return study.best_params


def run(config: dict) -> dict:
    data_cfg = config["data"]
    data_path = data_cfg.get("path", _DEFAULT_DATA_PATH)
    features_cfg = config["features"]
    feature_type = features_cfg.get("type", "kmer")
    if feature_type not in ("kmer", "dnabert2"):
        raise ValueError(f"features.type must be 'kmer' or 'dnabert2', got {feature_type!r}")
    k = features_cfg.get("k", 5)
    include_distance = features_cfg.get("include_distance", False)
    celligner_dim = features_cfg.get("cell_embedding_dim", 0)

    model_cfg = config["model"]
    model_name = model_cfg["name"]
    params = dict(model_cfg.get("params") or {})
    seed = config.get("seed", 42)
    output_dir = Path(config.get("output_dir", "results/lncrna_rra_day14_cellline_loco/runs"))

    tuning_cfg = config.get("tuning") or {"method": "fixed"}
    tuning_method = tuning_cfg.get("method", "fixed")
    if tuning_method not in ("fixed", "optuna"):
        raise ValueError(f"tuning.method must be 'fixed' or 'optuna', got {tuning_method!r}")
    metric = tuning_cfg.get("metric", "auprc")

    exclude_cell_lines = set(data_cfg.get("exclude_cell_lines") or [])

    print(f"Loading records from {data_path} ...")
    records = load_jsonl(data_path, record_cls=LncRnaRecord)
    if exclude_cell_lines:
        n_before = len(records)
        records = [r for r in records if r.cell_line not in exclude_cell_lines]
        print(f"  excluded {sorted(exclude_cell_lines)}: {n_before:,} -> {len(records):,} records")
    else:
        print(f"  {len(records):,} records")

    transcript_sequences = None
    embeddings = None
    if feature_type == "kmer":
        transcript_sequences = _load_transcript_sequences(data_cfg["transcript_sequences"])
    else:
        embeddings = load_embeddings(features_cfg["embeddings"])

    def build_features(recs, vocab=None):
        if feature_type == "kmer":
            return build_lncrna_features(
                recs, transcript_sequences, k=k, vocab=vocab,
                include_distance=include_distance, celligner_embedding_dim=celligner_dim,
                sparse=False,
            )
        return build_lncrna_embedding_features(
            recs, embeddings, include_distance=include_distance, celligner_embedding_dim=celligner_dim,
        )

    if tuning_method == "optuna":
        search_space_path = tuning_cfg["search_space"]
        n_trials = tuning_cfg.get("n_trials", 50)
        print(f"Tuning via optuna ({n_trials} trials, metric={metric}) on a stratified "
              "5-fold CV over the full dataset ...")
        vocab_for_tuning = None
        if feature_type == "kmer":
            all_targets = {r.target for r in records}
            all_seqs = [transcript_sequences[t] for t in all_targets if t in transcript_sequences]
            vocab_for_tuning = fit_vocab(all_seqs, k)
        X_all, y_all, _ = build_features(records, vocab=vocab_for_tuning)
        tuning_splits = make_cv_splits(records, strategy="stratified", n_splits=5, seed=seed)
        params = _tune_optuna(model_name, seed, X_all, y_all, tuning_splits, search_space_path, n_trials, metric)
        print(f"Best params: {params}")

    splits = make_cv_splits(records, strategy="cellline")
    print(f"Leave-one-cell-line-out: {len(splits)} folds ({[label for _, _, label in splits]})")

    n = len(records)
    y_pred_proba = np.full(n, np.nan, dtype=np.float64)
    y_true = np.array([r.label for r in records], dtype=int)

    for train_mask, val_mask, fold_label in splits:
        train_recs = [r for r, m in zip(records, train_mask) if m]
        val_recs = [r for r, m in zip(records, val_mask) if m]

        vocab = None
        if feature_type == "kmer":
            train_targets = {r.target for r in train_recs}
            train_seqs = [transcript_sequences[t] for t in train_targets if t in transcript_sequences]
            vocab = fit_vocab(train_seqs, k)

        X_train, y_train, _ = build_features(train_recs, vocab=vocab)
        X_val, _, _ = build_features(val_recs, vocab=vocab)

        model = build_classifier(model_name, **{"seed": seed, **params})
        model.fit(X_train, y_train)
        fold_pred = model.predict_proba(X_val)
        y_pred_proba[val_mask] = fold_pred

        print(f"  fold {fold_label:<12} train={len(train_recs):,}  val={len(val_recs):,}")

    assert not np.isnan(y_pred_proba).any(), "every record should be predicted by exactly one fold"

    metrics_rows = evaluate_lncrna_by_group(records, y_true, y_pred_proba)
    overall = next(r for r in metrics_rows if r["split"] == "Overall")
    print(f"\nOverall (leave-one-cell-line-out) AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config["model"]["params"] = params  # reflect the tuned params actually used, not the config's originals
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)

    preds_rows = [
        {"target": r.target, "cell_line": r.cell_line, "y_true": int(y_t), "y_pred_proba": float(y_p)}
        for r, y_t, y_p in zip(records, y_true, y_pred_proba)
    ]
    pd.DataFrame(preds_rows).to_csv(run_dir / "predictions.csv", index=False)

    run_info = {
        "model": model_name,
        "params": params,
        "tuning_method": tuning_method,
        "features": feature_type,
        "cell_embedding_dim": celligner_dim,
        "eval_strategy": "cellline_loco",
        "n_folds": len(splits),
        "auroc": overall["auroc"],
        "auprc": overall["auprc"],
        "timestamp": timestamp,
        "git_commit": git_commit(),
    }
    with open(run_dir / "run_info.json", "w") as fh:
        json.dump(run_info, fh, indent=2, default=str)
        fh.write("\n")

    print(f"\nRun saved -> {run_dir}")
    return {"run_dir": str(run_dir), "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a cellline_loco YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override the config's output_dir.")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    if args.output_dir:
        config["output_dir"] = args.output_dir

    run(config)


if __name__ == "__main__":
    main()
