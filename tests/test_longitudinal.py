import gzip
import json

import pandas as pd
import pytest

from lncfit.longitudinal import build_day7_longitudinal_features, stream_day7_guide_stats
from lncfit.screen_data import LncRnaRecord
from scripts.run_day7_longitudinal import _percentile_rank


def test_stream_day7_guide_stats_ignores_other_days(tmp_path):
    path = tmp_path / "guides.jsonl.gz"
    rows = [
        {"target": "L1", "cell_line": "HAP1", "replicate": 1, "day": 7, "fold_change": -1.0},
        {"target": "L1", "cell_line": "HAP1", "replicate": 1, "day": 7, "fold_change": 1.0},
        {"target": "L1", "cell_line": "HAP1", "replicate": 1, "day": 14, "fold_change": -99.0},
    ]
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = stream_day7_guide_stats(path).iloc[0]
    assert result["mean"] == 0.0
    assert result["min"] == -1.0
    assert result["max"] == 1.0
    assert result["negative_fraction"] == 0.5


def test_longitudinal_features_do_not_include_day14_outcomes():
    records = [
        LncRnaRecord(
            target="L1",
            cell_line="HAP1",
            day=14,
            rra_pvalue=1e-20,
            fold_change=-99.0,
            label=1,
            strand="+",
            distance_to_closest_pc_gene=100,
        )
    ]
    rra = pd.DataFrame(
        [
            {
                "target": "L1",
                "source_cell_line": "HAP1",
                "day7_pvalue": 0.01,
                "day7_fold_change": -0.5,
                "day7_hit": 1,
                "day7_neg_log10_pvalue": 2.0,
                "day7_depletion_score": 1.0,
                "day7_negative": 1.0,
            }
        ]
    )
    guide = pd.DataFrame(
        [
            {
                "target": "L1",
                "source_cell_line": "HAP1",
                "replicate": replicate,
                "mean": -0.25,
                "std": 0.1,
                "min": -0.5,
                "max": 0.0,
                "negative_fraction": 0.5,
                "strong_fraction": 0.0,
                "very_strong_fraction": 0.0,
            }
            for replicate in (1, 2)
        ]
    )

    X, y, columns = build_day7_longitudinal_features(records, rra, guide)
    assert X.shape == (1, len(columns))
    assert y.tolist() == [1]
    assert all("day14" not in column.lower() for column in columns)
    assert "rra_pvalue" not in columns
    assert "fold_change" not in columns


def test_percentile_rank_preserves_order_and_averages_ties():
    ranked = _percentile_rank(pd.Series([10.0, 2.0, 10.0, -1.0]).to_numpy())
    assert ranked.tolist() == pytest.approx([0.875, 0.5, 0.875, 0.25])
