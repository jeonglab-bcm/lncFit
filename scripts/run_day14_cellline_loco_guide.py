#!/usr/bin/env python3
"""Leakage-safe Day-14 leave-one-cell-line-out guide-feature evaluation.

Each outer fold holds out one of HAP1, K562, MDA-MB-231, or THP1. Models may
use pre-screen annotations, expression, transcript sequence, and guide-design
features, plus Day-14 outcomes from the *other* three cell lines. Day-0,
Day-7, and held-out-cell-line outcomes are never model inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import LncRnaRecord, load_jsonl
from scripts.run_day14_compliant_multimodal import (
    _is_effect_model,
    _is_regression_model,
    _is_strength_model,
    _make_model,
    _predict_values,
    build_features,
)


EVALUATED_CELL_LINES = ("HAP1", "K562", "MDA-MB-231", "THP1")
DEFAULT_MODELS = (
    "xgboost",
    "xgboost_strength",
    "xgboost_d3",
    "xgboost_d3_strength",
    "xgboost_d7",
    "xgboost_d7_strength",
)


def _metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, predictions)),
        "auprc": float(average_precision_score(y, predictions)),
    }


def _rank_within_cell(
    predictions: np.ndarray, cell_lines: np.ndarray
) -> np.ndarray:
    ranked = np.empty(len(predictions), dtype=float)
    for cell_line in EVALUATED_CELL_LINES:
        mask = cell_lines == cell_line
        ranked[mask] = rankdata(predictions[mask]) / int(mask.sum())
    return ranked


def _training_cell_priors(
    records: list[LncRnaRecord],
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute target propensities using outer-training cell lines only."""
    by_target: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record, is_train in zip(records, train_mask, strict=True):
        if not is_train:
            continue
        by_target[record.target]["label"].append(float(record.label))
        by_target[record.target]["strength"].append(
            min(-np.log10(max(record.rra_pvalue, 1e-12)), 8.0)
        )
        by_target[record.target]["depletion"].append(-record.fold_change)
        by_target[record.target]["effect"].append(max(-record.fold_change, 0.0))

    valid_records = [
        record
        for record, is_valid in zip(records, valid_mask, strict=True)
        if is_valid
    ]
    priors: dict[str, np.ndarray] = {}
    for outcome in ("label", "strength", "depletion", "effect"):
        values = [by_target[record.target][outcome] for record in valid_records]
        priors[f"prior_{outcome}_mean"] = np.asarray(
            [np.mean(row) for row in values], dtype=float
        )
        priors[f"prior_{outcome}_max"] = np.asarray(
            [np.max(row) for row in values], dtype=float
        )
    return priors


def _cross_cell_prior_matrix(
    records: list[LncRnaRecord], outer_train_mask: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Build row features from other outer-training cell lines only."""
    allowed_cells = {
        record.cell_line
        for record, is_train in zip(records, outer_train_mask, strict=True)
        if is_train
    }
    lookup = {
        (record.target, record.cell_line): record
        for record, is_train in zip(records, outer_train_mask, strict=True)
        if is_train
    }
    names = [
        "cross_cell_label_mean",
        "cross_cell_label_max",
        "cross_cell_strength_mean",
        "cross_cell_strength_max",
        "cross_cell_strength_std",
        "cross_cell_depletion_mean",
        "cross_cell_depletion_max",
        "cross_cell_depletion_min",
        "cross_cell_depletion_std",
        "cross_cell_effect_mean",
        "cross_cell_effect_max",
        "cross_cell_effect_min",
        "cross_cell_effect_std",
    ]
    matrix: list[list[float]] = []
    for record in records:
        source_records = [
            lookup[(record.target, cell_line)]
            for cell_line in allowed_cells
            if cell_line != record.cell_line
        ]
        labels = np.asarray([source.label for source in source_records], dtype=float)
        strengths = np.asarray(
            [
                min(-np.log10(max(source.rra_pvalue, 1e-12)), 8.0)
                for source in source_records
            ],
            dtype=float,
        )
        depletions = np.asarray(
            [-source.fold_change for source in source_records], dtype=float
        )
        effects = np.asarray(
            [max(-source.fold_change, 0.0) for source in source_records],
            dtype=float,
        )
        matrix.append(
            [
                float(labels.mean()),
                float(labels.max()),
                float(strengths.mean()),
                float(strengths.max()),
                float(strengths.std()),
                float(depletions.mean()),
                float(depletions.max()),
                float(depletions.min()),
                float(depletions.std()),
                float(effects.mean()),
                float(effects.max()),
                float(effects.min()),
                float(effects.std()),
            ]
        )
    return np.asarray(matrix, dtype=np.float32), names


def run(args: argparse.Namespace) -> dict:
    records: list[LncRnaRecord] = [
        record
        for record in load_jsonl(args.data, record_cls=LncRnaRecord)
        if record.cell_line in EVALUATED_CELL_LINES
    ]
    if any(record.day != 14 for record in records):
        raise ValueError("Input contains a non-Day-14 record")

    X, feature_names = build_features(
        records,
        args.mmc2,
        args.sequences,
        args.embeddings,
        args.gtf,
        tuple(args.feature_blocks),
        None,
        "multimodal",
    )
    y = np.asarray([record.label for record in records], dtype=np.int8)
    strength = np.minimum(
        -np.log10(np.maximum([record.rra_pvalue for record in records], 1e-12)),
        8.0,
    )
    depletion = -np.asarray(
        [record.fold_change for record in records], dtype=float
    )
    cell_lines = np.asarray([record.cell_line for record in records])
    positive_weight = float((y == 0).sum() / (y == 1).sum())

    predictions = {
        model_name: np.full(len(records), np.nan, dtype=float)
        for model_name in args.models
    }
    prior_feature_predictions = {
        f"prior_features_{model_name}": np.full(
            len(records), np.nan, dtype=float
        )
        for model_name in args.prior_feature_models
    }
    prior_predictions = {
        name: np.full(len(records), np.nan, dtype=float)
        for name in (
            "prior_label_mean",
            "prior_label_max",
            "prior_strength_mean",
            "prior_strength_max",
            "prior_depletion_mean",
            "prior_depletion_max",
            "prior_effect_mean",
            "prior_effect_max",
        )
    }

    for fold, held_out in enumerate(EVALUATED_CELL_LINES, start=1):
        train_mask = cell_lines != held_out
        valid_mask = cell_lines == held_out
        for model_name in args.models:
            model = _make_model(model_name, args.seed + fold, positive_weight)
            target = (
                strength
                if _is_strength_model(model_name)
                else depletion
                if _is_effect_model(model_name)
                else y
            )
            model.fit(X[train_mask], target[train_mask])
            predictions[model_name][valid_mask] = _predict_values(
                model, X[valid_mask], _is_regression_model(model_name)
            )
        prior_matrix, _ = _cross_cell_prior_matrix(records, train_mask)
        X_with_priors = np.hstack([X, prior_matrix])
        for model_name in args.prior_feature_models:
            model = _make_model(model_name, args.seed + fold, positive_weight)
            target = (
                strength
                if _is_strength_model(model_name)
                else depletion
                if _is_effect_model(model_name)
                else y
            )
            model.fit(X_with_priors[train_mask], target[train_mask])
            prior_feature_predictions[f"prior_features_{model_name}"][
                valid_mask
            ] = _predict_values(
                model,
                X_with_priors[valid_mask],
                _is_regression_model(model_name),
            )
        priors = _training_cell_priors(records, train_mask, valid_mask)
        for name, values in priors.items():
            prior_predictions[name][valid_mask] = values
        print(
            json.dumps(
                {
                    "held_out": held_out,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                }
            ),
            flush=True,
        )

    all_predictions = {
        **predictions,
        **prior_feature_predictions,
        **prior_predictions,
    }
    results: list[dict[str, float | str]] = []
    for name, values in all_predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite predictions for {name}")
        row = {"model": name, **_metrics(y, values)}
        results.append(row)
        print(json.dumps(row), flush=True)

    guide_rank = np.mean(
        [_rank_within_cell(values, cell_lines) for values in predictions.values()],
        axis=0,
    )
    prior_model_rank = np.mean(
        [
            _rank_within_cell(values, cell_lines)
            for values in prior_feature_predictions.values()
        ],
        axis=0,
    )
    effect_rank = _rank_within_cell(
        prior_predictions["prior_effect_mean"], cell_lines
    )
    depletion_rank = _rank_within_cell(
        prior_predictions["prior_depletion_mean"], cell_lines
    )
    strength_rank = _rank_within_cell(
        prior_predictions["prior_strength_mean"], cell_lines
    )
    ensembles = {
        "guide_rank_mean": guide_rank,
        "prior_model_rank_mean": prior_model_rank,
        "effect_strength_rank_mean": (effect_rank + strength_rank) / 2.0,
        "depletion_strength_rank_mean": (
            depletion_rank + strength_rank
        ) / 2.0,
        "guide_depletion_25_75": 0.25 * guide_rank + 0.75 * depletion_rank,
        "guide_depletion_50_50": 0.50 * guide_rank + 0.50 * depletion_rank,
        "guide_depletion_75_25": 0.75 * guide_rank + 0.25 * depletion_rank,
        "guide_effect_25_75": 0.25 * guide_rank + 0.75 * effect_rank,
        "guide_effect_50_50": 0.50 * guide_rank + 0.50 * effect_rank,
        "guide_effect_75_25": 0.75 * guide_rank + 0.25 * effect_rank,
        "prior_model_effect_25_75": 0.25 * prior_model_rank + 0.75 * effect_rank,
        "prior_model_effect_50_50": 0.50 * prior_model_rank + 0.50 * effect_rank,
        "prior_model_effect_75_25": 0.75 * prior_model_rank + 0.25 * effect_rank,
        "prior_model_depletion_25_75": (
            0.25 * prior_model_rank + 0.75 * depletion_rank
        ),
        "prior_model_depletion_50_50": (
            0.50 * prior_model_rank + 0.50 * depletion_rank
        ),
        "prior_model_depletion_75_25": (
            0.75 * prior_model_rank + 0.25 * depletion_rank
        ),
    }
    if "prior_features_xgboost_d3_effect" in prior_feature_predictions:
        ensembles["depletion_prior_d3_effect_50_50"] = (
            0.5
            * rankdata(prior_predictions["prior_depletion_mean"])
            / len(records)
            + 0.5
            * rankdata(
                prior_feature_predictions[
                    "prior_features_xgboost_d3_effect"
                ]
            )
            / len(records)
        )
    for name, values in ensembles.items():
        row = {"model": name, **_metrics(y, values)}
        results.append(row)
        print(json.dumps(row), flush=True)

    per_cell: list[dict[str, float | str]] = []
    for name, values in {**all_predictions, **ensembles}.items():
        for cell_line in EVALUATED_CELL_LINES:
            mask = cell_lines == cell_line
            per_cell.append(
                {
                    "model": name,
                    "cell_line": cell_line,
                    **_metrics(y[mask], values[mask]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": str(args.data),
        "feature_blocks": args.feature_blocks,
        "feature_count": len(feature_names),
        "models": results,
        "per_cell": per_cell,
        "day0_features_used": False,
        "day7_features_used": False,
        "heldout_cell_outcomes_used": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        target=np.asarray([record.target for record in records]),
        cell_line=cell_lines,
        y=y,
        **all_predictions,
        **ensembles,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/lncrna_rra_day14.jsonl.gz"),
    )
    parser.add_argument("--mmc2", type=Path, default=Path("data/raw/mmc2.xlsx"))
    parser.add_argument(
        "--gtf", type=Path, default=Path("data/raw/human.lncRNA.hg19.gtf")
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("data/processed/body_sequences_transcript.json"),
    )
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument(
        "--feature-blocks",
        nargs="*",
        default=["guide_design"],
        choices=[
            "expression_specificity",
            "transcript_architecture",
            "guide_qc",
            "guide_sequence",
            "guide_context",
            "guide_design",
            "guide_flanks",
        ],
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--prior-feature-models",
        nargs="+",
        default=["xgboost", "xgboost_strength"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/lncfit_loco_guide.json"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
