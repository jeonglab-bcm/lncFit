from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lncfit.screen_data import LncRnaRecord
from scripts.run_day14_compliant_multimodal import _supplementary_features


def _write_mmc2(path: Path, leaked_counts: list[int]) -> None:
    s1a = pd.DataFrame(
        {
            "lncRNA": ["T1", "T2"],
            "Transcript length": [1000, 2000],
            "Exons": [2, 3],
            "Tissue tau": [0.5, 0.7],
            "Time tau": [0.2, 0.4],
            "Dynamic": [True, False],
            "Count dynamic tissues": [1, 0],
            "Dynamic tissues": ["Brain", np.nan],
            "Distance to closest protein-coding gene": [100, 200],
            "Closest protein-coding gene Cas9 - DepMap score (23Q2, median)": [
                -0.1,
                np.nan,
            ],
            "CRISPRi hit": [True, np.nan],
            "Strand": ["+", "-"],
            "Genomic class": ["sense", "antisense"],
            "Age": ["old", "young"],
            # This same-screen outcome is present in S1A but must never enter X.
            "Number of cell lines showing essentiality": leaked_counts,
        }
    )
    s1c = pd.DataFrame(
        {
            "lncRNA": ["T1", "T2"],
            "HAP1": [1.0, 2.0],
            "HAP1 RfxCas13d": [1.5, 2.5],
            "HEK293FT": [2.0, 3.0],
            "HEK293FT RfxCas13d": [2.5, 3.5],
            "K562": [3.0, 4.0],
            "K562 RfxCas13d": [3.5, 4.5],
            "MDA-MB-231": [4.0, 5.0],
            "THP1": [5.0, 6.0],
            "THP1 RfxCas13d": [5.5, 6.5],
        }
    )
    s1e = pd.DataFrame(
        {
            "lncRNA": ["T1", "T2"],
            "HAP1 RfxCas13d": [2.0, 3.0],
            "HEK293FT RfxCas13d": [3.0, 4.0],
            "K562 RfxCas13d": [4.0, 5.0],
            "MDA-MB-231 RfxCas13d": [5.0, 6.0],
            "THP1 RfxCas13d": [6.0, 7.0],
        }
    )
    with pd.ExcelWriter(path) as writer:
        s1a.to_excel(writer, sheet_name="S1A", startrow=2, index=False)
        s1c.to_excel(writer, sheet_name="S1C", startrow=2, index=False)
        s1e.to_excel(writer, sheet_name="S1E", startrow=2, index=False)


def test_same_screen_hit_count_is_excluded_from_features(tmp_path: Path) -> None:
    records = [
        LncRnaRecord(
            target="T1",
            cell_line="HAP1",
            day=14,
            rra_pvalue=0.01,
            fold_change=-1.0,
            label=1,
        ),
        LncRnaRecord(
            target="T2",
            cell_line="THP1",
            day=14,
            rra_pvalue=1.0,
            fold_change=0.0,
            label=0,
        ),
    ]
    first = tmp_path / "first.xlsx"
    second = tmp_path / "second.xlsx"
    _write_mmc2(first, [0, 4])
    _write_mmc2(second, [4, 0])

    X_first, columns_first = _supplementary_features(records, first)
    X_second, columns_second = _supplementary_features(records, second)
    X_mutated_targets, _ = _supplementary_features(
        [
            replace(record, label=1 - record.label, rra_pvalue=1.0 - record.rra_pvalue)
            for record in records
        ],
        first,
    )

    assert columns_first == columns_second
    assert not any("essentiality" in column.lower() for column in columns_first)
    assert not any(
        "day7" in column.lower() or "day_7" in column.lower()
        for column in columns_first
    )
    assert X_first.shape[1] == len(columns_first)
    np.testing.assert_array_equal(X_first, X_second)
    np.testing.assert_array_equal(X_first, X_mutated_targets)
    assert np.isfinite(X_first).all()
