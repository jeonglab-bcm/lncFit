#!/usr/bin/env python3
"""Predict Day-14 RRA hits from strictly earlier Day-7 measurements.

This is a challenge-specific longitudinal model.  It uses Day-7 RRA values and
streamed guide-level Day-7 replicate summaries, never Day-14 p-values or fold
changes.  Chromosome 1 remains fully held out for final evaluation.

Usage:
  uv run python scripts/run_day7_longitudinal.py \
      --config configs/pipeline/xgboost_day7_longitudinal.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.io import git_commit
from lncfit.longitudinal import (
    build_day7_longitudinal_features,
    load_day7_rra_table,
    stream_day7_guide_stats,
)
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map arbitrary model scores to [0, 1] while preserving their ordering."""
    return rankdata(values, method="average") / len(values)


def run(config: dict) -> dict:
    data = config["data"]
    excluded = set(data.get("exclude_cell_lines") or [])
    train_records = load_jsonl(data["train"], record_cls=LncRnaRecord)
    test_records = load_jsonl(data["test"], record_cls=LncRnaRecord)
    train_records = [record for record in train_records if record.cell_line not in excluded]
    test_records = [record for record in test_records if record.cell_line not in excluded]

    print("Loading Day-7 RRA features ...", flush=True)
    rra_day7 = load_day7_rra_table(data["target_workbook"], data["screen_workbook"])
    print("Streaming Day-7 guide summaries ...", flush=True)
    guide_day7 = stream_day7_guide_stats(data["guide_records"])

    X_train, y_train, columns = build_day7_longitudinal_features(train_records, rra_day7, guide_day7)
    X_test, y_test, test_columns = build_day7_longitudinal_features(test_records, rra_day7, guide_day7)
    if columns != test_columns:
        raise RuntimeError("train/test feature columns do not match")
    print(f"Training {len(train_records):,} rows with {len(columns):,} features ...", flush=True)

    common_params = {
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": config.get("seed", 42),
    }
    classifier_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        **common_params,
        **config["model"]["classifier_params"],
    }
    classifier = XGBClassifier(**classifier_params)
    classifier.fit(X_train, y_train)
    classifier_predictions = classifier.predict_proba(X_test)[:, 1]

    # The continuous training p-values provide more information than the
    # thresholded hit label alone.  Regress their capped -log10 transform on
    # training chromosomes, then rank-blend with the classifier.  The held-out
    # chromosome's Day-14 p-values are used only by the evaluator, never here.
    significance_clip = float(config["model"].get("significance_clip", 4.0))
    significance_target = np.clip(
        -np.log10(np.clip([record.rra_pvalue for record in train_records], 1e-300, 1.0)),
        0.0,
        significance_clip,
    )
    regressor_params = {
        "objective": "reg:squarederror",
        **common_params,
        **config["model"]["significance_regressor_params"],
    }
    significance_regressor = XGBRegressor(**regressor_params)
    significance_regressor.fit(X_train, significance_target)
    significance_predictions = significance_regressor.predict(X_test)

    classifier_weight = float(config["model"].get("classifier_weight", 0.5))
    predictions = (
        classifier_weight * _percentile_rank(classifier_predictions)
        + (1.0 - classifier_weight) * _percentile_rank(significance_predictions)
    )

    metrics_rows = evaluate_lncrna_by_group(test_records, y_test, predictions)
    overall = next(row for row in metrics_rows if row["split"] == "Overall")
    print(f"AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["output_dir"]) / f"run_xgboost_day7_ensemble_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "target": [record.target for record in test_records],
            "cell_line": [record.cell_line for record in test_records],
            "y_true": y_test,
            "y_pred_proba": predictions,
        }
    ).to_csv(run_dir / "predictions.csv", index=False)
    run_info = {
        "model": "xgboost_day7_longitudinal_rank_ensemble",
        "classifier_params": classifier_params,
        "significance_regressor_params": regressor_params,
        "classifier_weight": classifier_weight,
        "significance_clip": significance_clip,
        "feature_count": len(columns),
        "train_rows": len(train_records),
        "test_rows": len(test_records),
        "auroc": overall["auroc"],
        "auprc": overall["auprc"],
        "git_commit": git_commit(),
        "timestamp": timestamp,
    }
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Run saved -> {run_dir}", flush=True)
    return {"run_dir": str(run_dir), "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    run(config)


if __name__ == "__main__":
    main()
