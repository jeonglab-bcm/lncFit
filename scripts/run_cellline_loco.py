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
(see configs/README.md) -- no data.train/data.test (one dataset, no held-out
split) and no tuning/cv sections (only fixed hyperparameters; nested
per-fold tuning is out of scope here).

Usage:
  python scripts/run_cellline_loco.py --config configs/cellline_loco/xgboost_kmer.yaml
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.classifiers import build_classifier
from lncfit.cv import make_cv_splits
from lncfit.embeddings import load_embeddings
from lncfit.features import build_lncrna_embedding_features, build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

_DEFAULT_DATA_PATH = "data/processed/lncrna_rra_day14.jsonl.gz"


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


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

    print(f"Loading records from {data_path} ...")
    records = load_jsonl(data_path, record_cls=LncRnaRecord)
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
