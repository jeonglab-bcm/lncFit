from __future__ import annotations

from dataclasses import replace

import numpy as np

from lncfit.screen_data import LncRnaRecord
from scripts.run_day14_cellline_loco_guide import (
    _cross_cell_guide_outcome_matrix,
    _cross_cell_prior_matrix,
    _cross_cell_source_matrix,
    _training_cell_priors,
)


def _records() -> list[LncRnaRecord]:
    records = []
    for target_index, target in enumerate(("T1", "T2")):
        for cell_index, cell_line in enumerate(
            ("HAP1", "K562", "MDA-MB-231", "THP1")
        ):
            records.append(
                LncRnaRecord(
                    target=target,
                    cell_line=cell_line,
                    chrom="1",
                    day=14,
                    rra_pvalue=0.01 * (cell_index + 1),
                    fold_change=-float(target_index + cell_index + 1),
                    label=int((target_index + cell_index) % 2 == 0),
                )
            )
    return records


def test_outer_heldout_outcomes_cannot_change_cross_cell_features() -> None:
    records = _records()
    train_mask = np.asarray(
        [record.cell_line != "THP1" for record in records], dtype=bool
    )
    valid_mask = ~train_mask
    mutated = [
        replace(
            record,
            label=1 - record.label,
            rra_pvalue=1e-12,
            fold_change=record.fold_change + 1000.0,
        )
        if record.cell_line == "THP1"
        else record
        for record in records
    ]

    matrix, names = _cross_cell_prior_matrix(records, train_mask)
    mutated_matrix, mutated_names = _cross_cell_prior_matrix(
        mutated, train_mask
    )
    assert names == mutated_names
    np.testing.assert_array_equal(matrix, mutated_matrix)

    source_matrix, source_names = _cross_cell_source_matrix(
        records, train_mask
    )
    mutated_source_matrix, mutated_source_names = _cross_cell_source_matrix(
        mutated, train_mask
    )
    assert source_names == mutated_source_names
    np.testing.assert_array_equal(source_matrix, mutated_source_matrix)

    priors = _training_cell_priors(records, train_mask, valid_mask)
    mutated_priors = _training_cell_priors(
        mutated, train_mask, valid_mask
    )
    assert priors.keys() == mutated_priors.keys()
    for name in priors:
        np.testing.assert_array_equal(priors[name], mutated_priors[name])


def test_outer_heldout_guide_outcomes_cannot_change_features() -> None:
    records = _records()
    train_mask = np.asarray(
        [record.cell_line != "THP1" for record in records], dtype=bool
    )
    summaries = {
        cell_line: {
            target: np.asarray(
                [cell_index + target_index, cell_index - target_index],
                dtype=np.float32,
            )
            for target_index, target in enumerate(("T1", "T2"))
        }
        for cell_index, cell_line in enumerate(
            ("HAP1", "K562", "MDA-MB-231", "THP1")
        )
    }
    mutated = {
        cell_line: {
            target: values.copy() for target, values in by_target.items()
        }
        for cell_line, by_target in summaries.items()
    }
    for values in mutated["THP1"].values():
        values += 1000.0

    matrix, names = _cross_cell_guide_outcome_matrix(
        records, train_mask, summaries, ["metric_a", "metric_b"]
    )
    mutated_matrix, mutated_names = _cross_cell_guide_outcome_matrix(
        records, train_mask, mutated, ["metric_a", "metric_b"]
    )
    assert names == mutated_names
    np.testing.assert_array_equal(matrix, mutated_matrix)
