"""Leakage-safe longitudinal features for the Day-14 lncRNA challenge.

The source experiment contains measurements at Day 7 and Day 14.  This module
uses only the earlier Day-7 RRA and guide-level fold changes as predictors for
the Day-14 hit label.  Guide records are aggregated while streaming so the
roughly one-million-row JSONL file never needs to be materialised in memory.
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lncfit.screen_data import (
    LncRnaRecord,
    load_annotations,
    load_rra,
    load_target_groups,
)


CELL_LINES = ("HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1")
_GUIDE_STATS = ("mean", "std", "min", "max", "negative_fraction", "strong_fraction", "very_strong_fraction")


def load_day7_rra_table(
    target_workbook: str | Path,
    screen_workbook: str | Path,
) -> pd.DataFrame:
    """Load one Day-7 RRA row per lncRNA and cell line, with stable transforms."""
    target_groups = load_target_groups(target_workbook)
    annotations = load_annotations(target_workbook)
    records = load_rra(screen_workbook, 7, target_groups, annotations)
    table = pd.DataFrame(
        {
            "target": r.target,
            "source_cell_line": r.cell_line,
            "day7_pvalue": r.rra_pvalue,
            "day7_fold_change": r.fold_change,
            "day7_hit": r.label,
        }
        for r in records
    )
    table["day7_neg_log10_pvalue"] = -np.log10(table["day7_pvalue"].clip(lower=1e-300))
    table["day7_depletion_score"] = -table["day7_fold_change"] * table["day7_neg_log10_pvalue"]
    table["day7_negative"] = (table["day7_fold_change"] < 0).astype(np.float32)
    return table


def stream_day7_guide_stats(path: str | Path) -> pd.DataFrame:
    """Aggregate Day-7 guide fold changes by target/cell/replicate in bounded memory."""
    # State: n, sum, sumsq, min, max, n<0, n<-0.5, n<-1.
    states: dict[tuple[str, str, int], list[float]] = defaultdict(
        lambda: [0, 0.0, 0.0, float("inf"), float("-inf"), 0, 0, 0]
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if raw.get("day") != 7 or not raw.get("target"):
                continue
            value = float(raw["fold_change"])
            key = (raw["target"], raw["cell_line"], int(raw["replicate"]))
            state = states[key]
            state[0] += 1
            state[1] += value
            state[2] += value * value
            state[3] = min(state[3], value)
            state[4] = max(state[4], value)
            state[5] += value < 0
            state[6] += value < -0.5
            state[7] += value < -1.0

    rows: list[dict] = []
    for (target, cell_line, replicate), state in states.items():
        n, total, total_sq, minimum, maximum, n_negative, n_strong, n_very_strong = state
        mean = total / n
        rows.append(
            {
                "target": target,
                "source_cell_line": cell_line,
                "replicate": replicate,
                "mean": mean,
                "std": math.sqrt(max(0.0, total_sq / n - mean * mean)),
                "min": minimum,
                "max": maximum,
                "negative_fraction": n_negative / n,
                "strong_fraction": n_strong / n,
                "very_strong_fraction": n_very_strong / n,
            }
        )
    return pd.DataFrame(rows)


def _flatten_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Flatten a pivot table's MultiIndex columns into deterministic feature names."""
    flattened = frame.copy()
    flattened.columns = [
        "__".join([prefix, *(str(part) for part in column if str(part))])
        for column in flattened.columns
    ]
    return flattened.reset_index()


def build_day7_longitudinal_features(
    records: list[LncRnaRecord],
    rra_day7: pd.DataFrame,
    guide_day7: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build Day-7 longitudinal features for Day-14 records.

    Features include the prediction cell's own earlier RRA/guide measurements,
    all-cell-line Day-7 vectors, cross-cell summary statistics, replicate
    disagreement, genomic distance/strand, and prediction-cell indicators.
    No Day-14 p-value or fold-change is included.
    """
    metadata = pd.DataFrame(
        {
            "target": r.target,
            "cell_line": r.cell_line,
            "distance": r.distance_to_closest_pc_gene or 0,
            "strand": r.strand,
            "label": r.label,
        }
        for r in records
    )

    own_rra = rra_day7.rename(columns={"source_cell_line": "cell_line"})
    table = metadata.merge(own_rra, on=["target", "cell_line"], how="left")

    rra_value_cols = [
        "day7_fold_change",
        "day7_neg_log10_pvalue",
        "day7_depletion_score",
        "day7_hit",
        "day7_negative",
    ]
    rra_wide = rra_day7.pivot(
        index="target",
        columns="source_cell_line",
        values=rra_value_cols,
    )
    table = table.merge(_flatten_columns(rra_wide, "rra"), on="target", how="left")

    rra_summary = rra_day7.groupby("target").agg(
        rra_fold_change_mean=("day7_fold_change", "mean"),
        rra_fold_change_std=("day7_fold_change", "std"),
        rra_fold_change_min=("day7_fold_change", "min"),
        rra_fold_change_max=("day7_fold_change", "max"),
        rra_significance_mean=("day7_neg_log10_pvalue", "mean"),
        rra_significance_max=("day7_neg_log10_pvalue", "max"),
        rra_hit_count=("day7_hit", "sum"),
        rra_depletion_mean=("day7_depletion_score", "mean"),
        rra_depletion_max=("day7_depletion_score", "max"),
    )
    table = table.merge(rra_summary.reset_index(), on="target", how="left")

    guide_own = guide_day7.pivot(
        index=["target", "source_cell_line"],
        columns="replicate",
        values=list(_GUIDE_STATS),
    )
    guide_own.columns = [f"guide_{stat}_rep{replicate}" for stat, replicate in guide_own.columns]
    guide_own = guide_own.reset_index().rename(columns={"source_cell_line": "cell_line"})
    for stat in _GUIDE_STATS:
        rep1, rep2 = f"guide_{stat}_rep1", f"guide_{stat}_rep2"
        if rep1 in guide_own and rep2 in guide_own:
            guide_own[f"guide_{stat}_replicate_difference"] = guide_own[rep1] - guide_own[rep2]
    table = table.merge(guide_own, on=["target", "cell_line"], how="left")

    guide_wide = guide_day7.pivot(
        index="target",
        columns=["source_cell_line", "replicate"],
        values=list(_GUIDE_STATS),
    )
    table = table.merge(_flatten_columns(guide_wide, "guide_all"), on="target", how="left")

    distance = pd.to_numeric(table["distance"], errors="coerce").fillna(0)
    table["distance_signed_log1p"] = np.sign(distance) * np.log1p(np.abs(distance))
    table["strand_plus"] = (table["strand"] == "+").astype(np.float32)
    for cell_line in CELL_LINES:
        table[f"prediction_cell__{cell_line}"] = (table["cell_line"] == cell_line).astype(np.float32)

    excluded = {
        "target",
        "cell_line",
        "distance",
        "strand",
        "label",
        "source_cell_line",
    }
    columns = [column for column in table.columns if column not in excluded]
    features = (
        table[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    labels = table["label"].to_numpy(dtype=np.int8)
    return features, labels, columns
