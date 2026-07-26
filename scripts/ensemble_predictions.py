#!/usr/bin/env python3
"""Combine several runs' predictions.csv into one ensemble predictions.csv.

Motivation: the two best models on the chromosome-held-out task come from
different families (Nystroem-RBF SVM and XGBoost) and make visibly different
mistakes -- on the cell-line-LOCO task the SVM has the best AUROC of anything
tried while XGBoost has the best AUPRC. Averaging complementary rankings is the
standard way to bank that.

Combines by **rank averaging** rather than probability averaging: the inputs are
on incomparable scales (XGBoost emits calibrated-ish probabilities in ~[0, 0.5];
the SVM wrapper emits sigmoid(margin) which is not calibrated at all), so a plain
mean would silently let whichever model has the wider spread dominate. Converting
each to within-run percentile ranks first makes the blend scale-free. AUROC/AUPRC
are rank metrics, so nothing is lost by outputting ranks instead of probabilities.

Weights are EQUAL by default and that is deliberate: fitting blend weights to
maximize held-out AUPRC would be tuning on the very test set the leaderboard
scores, i.e. leaderboard overfitting dressed up as a result. If you do pass
--weights, pick them from CV on training data, not from the test score.

Usage:
  python scripts/ensemble_predictions.py \\
      --inputs runA/predictions.csv runB/predictions.csv \\
      --output ensemble/predictions.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_KEYS = ["target", "cell_line"]
_SCORE = "y_pred_proba"


def rank_average(frames: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """Rank-average several predictions frames on (target, cell_line)."""
    if weights is None:
        weights = [1.0] * len(frames)
    if len(weights) != len(frames):
        raise ValueError(f"got {len(frames)} inputs but {len(weights)} weights")

    merged = None
    for i, df in enumerate(frames):
        missing = set(_KEYS + [_SCORE]) - set(df.columns)
        if missing:
            raise ValueError(f"input {i} missing column(s): {sorted(missing)}")
        if df.duplicated(subset=_KEYS).any():
            raise ValueError(f"input {i} has duplicate (target, cell_line) rows")
        # pct=True -> within-run percentile, so every input contributes on the
        # same [0, 1] scale regardless of how its raw scores were distributed.
        part = df[_KEYS].copy()
        part[f"r{i}"] = df[_SCORE].rank(pct=True)
        merged = part if merged is None else merged.merge(part, on=_KEYS, how="inner", validate="1:1")

    assert merged is not None
    for i, df in enumerate(frames):
        if len(merged) != len(df):
            raise ValueError(
                f"inputs disagree on which rows they cover: input {i} has {len(df)} rows, "
                f"the intersection across all inputs is {len(merged)}. Ensembling runs that "
                "scored different row sets would silently drop rows the leaderboard requires."
            )

    total = sum(weights)
    merged[_SCORE] = sum(w * merged[f"r{i}"] for i, w in enumerate(weights)) / total
    return merged[_KEYS + [_SCORE]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="predictions.csv paths to combine.")
    parser.add_argument("--output", required=True, help="Where to write the ensemble predictions.csv.")
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="Optional per-input weights (default: equal). Choose these from CV on "
             "training data -- never by maximizing the held-out test score.",
    )
    args = parser.parse_args()

    frames = [pd.read_csv(p) for p in args.inputs]
    out = rank_average(frames, args.weights)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Combined {len(frames)} run(s) over {len(out):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
