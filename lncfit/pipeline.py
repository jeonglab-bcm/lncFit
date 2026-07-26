"""Pluggable, YAML-configured lncRNA classifier pipeline (issue #78 follow-up).

Single entry point (`scripts/run_pipeline.py --config <path>.yaml`) that lets a
user choose every axis that previously required picking a different script:

  - model            lncfit.classifiers registry: xgboost, logreg, randomforest, knn, mlp, null
  - features          k-mer frequencies vs. precomputed DNABERT-2 embeddings
  - cell embedding    one-hot only, or + Celligner UMAP(2)/PCA(10/70) (issue #78)
  - tuning            fixed params, grid search, or Optuna TPE
  - cross-validation  none, chromosome LOCO, or stratified K-fold

See configs/README.md for the full config schema and configs/pipeline/*.yaml for
ready-to-run examples.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from lncfit.classifiers import build_classifier
from lncfit.cv import make_cv_splits
from lncfit.embeddings import load_embeddings, reduce_embeddings_pca
from lncfit.features import build_lncrna_embedding_features, build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.resample import METHODS as RESAMPLE_METHODS, resample
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

_METRIC_FNS = {
    "auprc": average_precision_score,
    "auroc": roc_auc_score,
}


def _load_transcript_sequences(path: str) -> dict[str, str]:
    """{target: [spliced_seq, ""]} JSON from lncfit.sequence -> {target: spliced_seq}."""
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


class LncRnaPipeline:
    """Config-driven train + tune + evaluate pipeline for the lncRNA RRA-hit task."""

    def __init__(self, config: dict):
        self.config = config
        self.seed = config.get("seed", 42)

        features_cfg = config["features"]
        self.feature_type = features_cfg.get("type", "kmer")
        if self.feature_type not in ("kmer", "dnabert2"):
            raise ValueError(f"features.type must be 'kmer' or 'dnabert2', got {self.feature_type!r}")
        self.k = features_cfg.get("k", 5)
        self.include_distance = features_cfg.get("include_distance", False)
        self.celligner_embedding_dim = features_cfg.get("cell_embedding_dim", 0)
        self.embeddings_path = features_cfg.get("embeddings")
        if self.feature_type == "dnabert2" and not self.embeddings_path:
            raise ValueError("features.embeddings is required when features.type is 'dnabert2'.")
        # 0 / absent = use the raw embedding dims; >0 = PCA-reduce to that many
        # components, fit on training targets only (see reduce_embeddings_pca).
        self.embedding_pca = int(features_cfg.get("embedding_pca", 0) or 0)

        model_cfg = config.get("model", {})
        if "name" not in model_cfg:
            raise ValueError("model.name is required (see lncfit.classifiers.available_classifiers()).")
        self.model_name = model_cfg["name"]
        self.fixed_params = dict(model_cfg.get("params") or {})
        # Training-set resampling (see lncfit.resample) -- applied to train splits
        # only, never to validation/test.
        resample_cfg = model_cfg.get("resample") or {}
        self.resample_method = resample_cfg.get("method", "none")
        if self.resample_method not in RESAMPLE_METHODS:
            raise ValueError(
                f"model.resample.method must be one of {RESAMPLE_METHODS}, "
                f"got {self.resample_method!r}"
            )
        self.resample_ratio = resample_cfg.get("ratio", "auto")

        tuning_cfg = config.get("tuning") or {"method": "fixed"}
        self.tuning_method = tuning_cfg.get("method", "fixed")
        if self.tuning_method not in ("fixed", "grid", "optuna"):
            raise ValueError(f"tuning.method must be 'fixed', 'grid', or 'optuna', got {self.tuning_method!r}")
        self.search_space_path = tuning_cfg.get("search_space")
        self.n_trials = tuning_cfg.get("n_trials", 50)
        self.metric = tuning_cfg.get("metric", "auprc")
        if self.metric not in _METRIC_FNS:
            raise ValueError(f"tuning.metric must be one of {list(_METRIC_FNS)}, got {self.metric!r}")

        cv_cfg = config.get("cv") or {"strategy": "none"}
        self.cv_strategy = cv_cfg.get("strategy", "none")
        if self.cv_strategy not in ("none", "chrom", "stratified"):
            raise ValueError(f"cv.strategy must be 'none', 'chrom', or 'stratified', got {self.cv_strategy!r}")
        self.cv_n_splits = int(cv_cfg.get("n_splits", 5))

        if self.tuning_method in ("grid", "optuna") and self.cv_strategy == "none":
            raise ValueError(
                f"tuning.method={self.tuning_method!r} needs a validation split -- set cv.strategy to "
                "'chrom' or 'stratified'."
            )

        data_cfg = config["data"]
        self.train_path = data_cfg["train"]
        self.test_path = data_cfg["test"]
        self.transcript_sequences_path = data_cfg.get("transcript_sequences")
        if self.feature_type == "kmer" and not self.transcript_sequences_path:
            raise ValueError("data.transcript_sequences is required when features.type is 'kmer'.")

        self.output_dir = Path(config.get("output_dir", "results/pipeline_runs"))

        self.vocab: list[str] | None = None
        self.transcript_sequences: dict[str, str] | None = None
        self.embeddings: tuple[np.ndarray, dict[str, int]] | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "LncRnaPipeline":
        with open(path) as fh:
            config = yaml.safe_load(fh)
        return cls(config)

    def _load_search_space(self) -> dict:
        if not self.search_space_path:
            raise ValueError(f"tuning.method={self.tuning_method!r} requires tuning.search_space to be set.")
        with open(self.search_space_path) as fh:
            return yaml.safe_load(fh)

    def _build_features(self, records: list[LncRnaRecord]):
        if self.feature_type == "kmer":
            assert self.transcript_sequences is not None, "run() must load transcript_sequences first"
            return build_lncrna_features(
                records, self.transcript_sequences, k=self.k, vocab=self.vocab,
                include_distance=self.include_distance,
                celligner_embedding_dim=self.celligner_embedding_dim,
                sparse=False,
            )
        assert self.embeddings is not None, "run() must load embeddings first"
        return build_lncrna_embedding_features(
            records, self.embeddings,
            include_distance=self.include_distance,
            celligner_embedding_dim=self.celligner_embedding_dim,
        )

    def _fit_resampled(self, model, X_train, y_train):
        """Fit `model`, resampling the TRAINING data first if configured.

        Kept as one helper so every fit site (CV folds and the final fit) goes
        through the same path -- resampling one but not the other would make the
        CV score describe a different procedure than the model actually shipped.
        """
        X_res, y_res = resample(
            X_train, y_train, method=self.resample_method,
            ratio=self.resample_ratio, seed=self.seed,
        )
        model.fit(X_res, y_res)
        return model

    def _build_model(self, params: dict):
        return build_classifier(self.model_name, **{"seed": self.seed, **params})

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(_METRIC_FNS[self.metric](y_true, y_pred))

    def _cv_score(self, params: dict, X, y: np.ndarray, splits) -> tuple[float, list[dict]]:
        fold_rows = []
        scores = []
        for train_mask, val_mask, fold_label in splits:
            model = self._build_model(params)
            self._fit_resampled(model, X[train_mask], y[train_mask])
            y_pred = model.predict_proba(X[val_mask])
            score = self._score(y[val_mask], y_pred)
            scores.append(score)
            fold_rows.append({"fold": fold_label, "n_val": int(val_mask.sum()), self.metric: score})
        return float(np.nanmean(scores)), fold_rows

    def _tune_grid(self, X, y: np.ndarray, splits) -> tuple[dict, pd.DataFrame]:
        search_space = self._load_search_space()
        grid_params = {name: spec["grid"] for name, spec in search_space.items() if "grid" in spec}
        if not grid_params:
            raise ValueError(f"No 'grid' entries found in {self.search_space_path} for grid search.")
        keys = list(grid_params)

        rows = []
        best_score, best_params = -np.inf, dict(zip(keys, [v[0] for v in grid_params.values()]))
        for combo in itertools.product(*[grid_params[k] for k in keys]):
            params = dict(zip(keys, combo))
            mean_score, _ = self._cv_score(params, X, y, splits)
            rows.append({**params, f"mean_{self.metric}": mean_score})
            if mean_score > best_score:
                best_score, best_params = mean_score, params
            print(f"  grid {params} -> {self.metric}={mean_score:.4f}", flush=True)
        return best_params, pd.DataFrame(rows)

    def _tune_optuna(self, X, y: np.ndarray, splits) -> tuple[dict, pd.DataFrame]:
        search_space = self._load_search_space()

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
            mean_score, _ = self._cv_score(params, X, y, splits)
            return mean_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            print(f"  trial {trial.number:>3d}  {self.metric}={trial.value:.4f}  best={study.best_value:.4f}",
                  flush=True)

        study.optimize(objective, n_trials=self.n_trials, callbacks=[_trial_callback])
        return study.best_params, study.trials_dataframe()

    def run(self) -> dict:
        print(f"Loading train records from {self.train_path} ...")
        train_records = load_jsonl(self.train_path, record_cls=LncRnaRecord)
        print(f"  {len(train_records):,} records")
        print(f"Loading test records from {self.test_path} ...")
        test_records = load_jsonl(self.test_path, record_cls=LncRnaRecord)
        print(f"  {len(test_records):,} records")

        if self.feature_type == "kmer":
            print(f"Loading transcript sequences from {self.transcript_sequences_path} ...")
            self.transcript_sequences = _load_transcript_sequences(self.transcript_sequences_path)
            train_targets = {r.target for r in train_records}
            train_seqs = [self.transcript_sequences[t] for t in train_targets if t in self.transcript_sequences]
            self.vocab = fit_vocab(train_seqs, self.k)
            print(f"  k={self.k} vocab: {len(self.vocab)}/{4 ** self.k} k-mers observed")
        else:
            print(f"Loading DNABERT-2 embeddings from {self.embeddings_path} ...")
            self.embeddings = load_embeddings(self.embeddings_path)
            print(f"  {self.embeddings[0].shape[0]:,} lncRNAs x {self.embeddings[0].shape[1]} dims")
            if self.embedding_pca > 0:
                # Fit on training targets only -- the test set's own embeddings must
                # not influence the projection.
                train_targets = {r.target for r in train_records}
                self.embeddings = reduce_embeddings_pca(
                    self.embeddings, train_targets, self.embedding_pca, seed=self.seed
                )
                print(f"  PCA -> {self.embeddings[0].shape[1]} components "
                      f"(fit on {len(train_targets):,} training targets)")

        print(f"Building features (type={self.feature_type}, cell_embedding_dim={self.celligner_embedding_dim}) ...")
        X_train, y_train, feature_cols = self._build_features(train_records)
        X_test, y_test, _ = self._build_features(test_records)
        print(f"  n_features={X_train.shape[1]}")

        cv_scores_df = None
        if self.tuning_method == "fixed":
            best_params = dict(self.fixed_params)
            if self.cv_strategy != "none":
                splits = make_cv_splits(train_records, self.cv_strategy, self.cv_n_splits, self.seed)
                mean_score, fold_rows = self._cv_score(best_params, X_train, y_train, splits)
                cv_scores_df = pd.DataFrame(fold_rows)
                print(f"CV ({self.cv_strategy}, {len(splits)} folds) mean {self.metric} = {mean_score:.4f}")
        else:
            splits = make_cv_splits(train_records, self.cv_strategy, self.cv_n_splits, self.seed)
            print(f"Tuning via {self.tuning_method} (cv={self.cv_strategy}, {len(splits)} folds) ...")
            if self.tuning_method == "grid":
                best_params, cv_scores_df = self._tune_grid(X_train, y_train, splits)
            else:
                best_params, cv_scores_df = self._tune_optuna(X_train, y_train, splits)
            print(f"Best params: {best_params}")

        print(f"\nFitting final {self.model_name} model on all training data ...")
        model = self._build_model(best_params)
        self._fit_resampled(model, X_train, y_train)

        print("Evaluating on held-out test set ...")
        y_pred_proba = model.predict_proba(X_test)
        metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred_proba)
        overall = next(r for r in metrics_rows if r["split"] == "Overall")
        print(f"  Overall AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")

        run_dir = self._save(best_params, metrics_rows, test_records, y_test, y_pred_proba,
                              feature_cols, cv_scores_df)
        return {"run_dir": str(run_dir), "best_params": best_params, "overall": overall}

    def _save(self, best_params, metrics_rows, test_records, y_test, y_pred_proba,
              feature_cols, cv_scores_df) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{self.model_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.yaml", "w") as fh:
            yaml.safe_dump(self.config, fh, sort_keys=False)

        with open(run_dir / "best_params.json", "w") as fh:
            json.dump(best_params, fh, indent=2, default=str)

        if cv_scores_df is not None:
            cv_scores_df.to_csv(run_dir / "cv_scores.csv", index=False)

        pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)

        preds_rows = [
            {"target": rec.target, "cell_line": rec.cell_line, "y_true": float(y_t), "y_pred_proba": float(y_p)}
            for rec, y_t, y_p in zip(test_records, y_test, y_pred_proba)
        ]
        pd.DataFrame(preds_rows).to_csv(run_dir / "predictions.csv", index=False)

        overall = next(r for r in metrics_rows if r["split"] == "Overall")
        run_info = {
            "model": self.model_name,
            "best_params": best_params,
            "features": self.feature_type,
            "cell_embedding_dim": self.celligner_embedding_dim,
            "tuning_method": self.tuning_method,
            "cv_strategy": self.cv_strategy,
            "n_features": len(feature_cols),
            "auroc": overall["auroc"],
            "auprc": overall["auprc"],
            "timestamp": timestamp,
            "git_commit": git_commit(),
        }
        with open(run_dir / "run_info.json", "w") as fh:
            json.dump(run_info, fh, indent=2, default=str)
            fh.write("\n")

        print(f"\nRun saved -> {run_dir}")
        return run_dir
