import numpy as np
import pytest

import lncfit.cv as cv_module
from lncfit.constants import MIN_FOLD_RECORDS
from lncfit.cv import build_lncrna_folds, make_cv_splits
from lncfit.screen_data import LncRnaRecord


def _make_records(n_per_chrom, chroms):
    records = []
    seqs = {}
    for c in chroms:
        for i in range(n_per_chrom):
            target = f"T_{c}_{i}"
            records.append(LncRnaRecord(
                target=target,
                cell_line="HAP1",
                day=14,
                rra_pvalue=0.5,
                fold_change=0.0,
                label=1 if i % 5 == 0 else 0,
                chrom=c,
            ))
            seqs[target] = "AAACCCGGGTTT"
    return records, seqs


def test_excludes_chroms_below_min_fold_records():
    records1, seqs1 = _make_records(MIN_FOLD_RECORDS - 1, ["1"])
    records2, seqs2 = _make_records(MIN_FOLD_RECORDS + 1, ["2"])
    records = records1 + records2
    seqs = {**seqs1, **seqs2}
    cv_chroms, fold_data, _ = build_lncrna_folds(records, seqs, k=3, verbose=False)
    assert cv_chroms == ["2"]


def test_val_and_es_and_train_partition_without_overlap():
    records, seqs = _make_records(MIN_FOLD_RECORDS + 10, ["1", "2", "3"])
    cv_chroms, fold_data, _ = build_lncrna_folds(records, seqs, k=3, verbose=False)
    assert set(cv_chroms) == {"1", "2", "3"}
    for val_chrom in cv_chroms:
        X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[val_chrom]
        total = X_tr.shape[0] + X_val.shape[0] + X_es.shape[0]
        assert total == len(records)
        assert X_val.shape[0] == MIN_FOLD_RECORDS + 10


def test_make_cv_splits_stratified_partitions_without_overlap():
    # lncfit.pipeline's generic, feature-type-agnostic CV splitter (issue #78
    # pipeline follow-up) -- returns boolean masks over records, not feature matrices.
    records, _ = _make_records(40, ["1"])
    splits = make_cv_splits(records, strategy="stratified", n_splits=4, seed=0)
    assert len(splits) == 4
    n = len(records)
    seen_val = np.zeros(n, dtype=bool)
    for train_mask, val_mask, fold_label in splits:
        assert train_mask.shape == (n,)
        assert val_mask.shape == (n,)
        assert not np.any(train_mask & val_mask)  # no row is both train and val
        assert np.all(train_mask | val_mask)  # every row is one or the other
        assert not np.any(seen_val & val_mask)  # each row is held out exactly once
        seen_val |= val_mask
        assert isinstance(fold_label, str)
    assert np.all(seen_val)


def test_make_cv_splits_chrom_groups_by_chromosome(monkeypatch):
    monkeypatch.setattr(cv_module, "MIN_FOLD_RECORDS", 5)
    records, _ = _make_records(5, ["1"])
    records += _make_records(8, ["2"])[0]
    records += _make_records(2, ["3"])[0]  # below the (patched) MIN_FOLD_RECORDS -> excluded

    splits = make_cv_splits(records, strategy="chrom")
    fold_labels = {label for _, _, label in splits}
    assert fold_labels == {"chr1", "chr2"}
    for train_mask, val_mask, label in splits:
        chrom = label.removeprefix("chr")
        expected_val = np.array([r.chrom == chrom for r in records])
        assert np.array_equal(val_mask, expected_val)
        assert np.array_equal(train_mask, ~expected_val)


def test_make_cv_splits_rejects_unknown_strategy():
    records, _ = _make_records(5, ["1"])
    with pytest.raises(ValueError):
        make_cv_splits(records, strategy="bogus")


def _make_multi_cellline_records(n_per_cell, cell_lines):
    records = []
    for cl in cell_lines:
        for i in range(n_per_cell):
            records.append(LncRnaRecord(
                target=f"T_{cl}_{i}",
                cell_line=cl,
                day=14,
                rra_pvalue=0.5,
                fold_change=0.0,
                label=1 if i % 5 == 0 else 0,
                chrom="1",
            ))
    return records


def test_make_cv_splits_cellline_one_fold_per_cell_line():
    cell_lines = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
    records = _make_multi_cellline_records(10, cell_lines)

    splits = make_cv_splits(records, strategy="cellline")
    fold_labels = {label for _, _, label in splits}
    assert fold_labels == set(cell_lines)

    n = len(records)
    seen_val = np.zeros(n, dtype=bool)
    for train_mask, val_mask, label in splits:
        assert isinstance(label, str)
        expected_val = np.array([r.cell_line == label for r in records])
        assert np.array_equal(val_mask, expected_val)
        assert np.array_equal(train_mask, ~expected_val)  # every other cell line's rows
        assert not np.any(seen_val & val_mask)  # each row held out exactly once
        seen_val |= val_mask
    assert np.all(seen_val)
