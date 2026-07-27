#!/usr/bin/env python3
"""Blend leakage-safe Day-14 LOCO prediction arrays into a submission CSV.

Each component is min-max normalized before applying its fixed weight. Input
archives are produced by ``scripts/run_day14_cellline_loco_guide.py`` and must
cover the same target/cell-line rows in the same order.

Example:
  python scripts/build_day14_loco_ensemble.py \
    --component run_expression.npz::matched_guide_outcomes_xgboost_d3_effect::.36 \
    --component run_full.npz::source_features_xgboost_d1_effect::.11 \
    --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _parse_component(value: str) -> tuple[Path, str, float]:
    parts = value.rsplit("::", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "component must be PATH::ARRAY_NAME::WEIGHT"
        )
    path, array_name, weight = parts
    try:
        parsed_weight = float(weight)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid component weight: {weight!r}"
        ) from error
    if parsed_weight < 0:
        raise argparse.ArgumentTypeError("component weight must be non-negative")
    return Path(path), array_name, parsed_weight


def blend_components(
    components: list[tuple[Path, str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if not components:
        raise ValueError("at least one component is required")
    total_weight = sum(weight for _, _, weight in components)
    if not np.isclose(total_weight, 1.0):
        raise ValueError(f"component weights must sum to 1, got {total_weight}")

    targets: np.ndarray | None = None
    cell_lines: np.ndarray | None = None
    labels: np.ndarray | None = None
    prediction: np.ndarray | None = None
    for path, array_name, weight in components:
        with np.load(path, allow_pickle=True) as archive:
            if array_name not in archive:
                raise KeyError(f"{array_name!r} is not present in {path}")
            current_targets = archive["target"].astype(str)
            current_cells = archive["cell_line"].astype(str)
            current_labels = archive["y"] if "y" in archive else None
            values = np.asarray(archive[array_name], dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{path}::{array_name} contains non-finite values")
        if targets is None:
            targets = current_targets
            cell_lines = current_cells
            labels = current_labels
            prediction = np.zeros(len(values), dtype=float)
        else:
            if not np.array_equal(targets, current_targets):
                raise ValueError(f"target order differs in {path}")
            if not np.array_equal(cell_lines, current_cells):
                raise ValueError(f"cell-line order differs in {path}")
            if labels is not None and current_labels is not None:
                if not np.array_equal(labels, current_labels):
                    raise ValueError(f"labels differ in {path}")

        value_min = float(values.min())
        value_range = float(values.max() - value_min)
        if value_range <= 0:
            raise ValueError(f"{path}::{array_name} is constant")
        prediction += weight * (values - value_min) / value_range

    assert targets is not None
    assert cell_lines is not None
    assert prediction is not None
    return targets, cell_lines, prediction, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        type=_parse_component,
        action="append",
        required=True,
        help="PATH::ARRAY_NAME::WEIGHT; repeat for each component",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets, cell_lines, prediction, labels = blend_components(args.component)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target": targets,
            "cell_line": cell_lines,
            "y_pred_proba": prediction,
        }
    ).to_csv(args.output, index=False)

    if labels is not None:
        print(
            f"AUROC={roc_auc_score(labels, prediction):.4f} "
            f"AUPRC={average_precision_score(labels, prediction):.4f}"
        )
    print(f"Wrote {len(prediction):,} rows to {args.output}")


if __name__ == "__main__":
    main()
