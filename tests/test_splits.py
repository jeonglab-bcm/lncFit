import pytest
from lncfit.screen_data import ScreenRecord
from lncfit.splits import split_by_chrom, split_by_cell_line


def _rec(guide_id, target, chrom, cell_line):
    return ScreenRecord(
        guide_id=guide_id,
        target=target,
        target_sequence="ACGT",
        cell_line=cell_line,
        day=7,
        replicate=1,
        fold_change=1.0,
        chrom=chrom,
    )


# 6 records: 3 lncRNAs × 2 cell lines, spread across 3 chromosomes
@pytest.fixture
def records():
    return [
        _rec("g1", "lncA", "1", "HAP1"),
        _rec("g2", "lncA", "1", "THP1"),
        _rec("g3", "lncB", "2", "HAP1"),
        _rec("g4", "lncB", "2", "THP1"),
        _rec("g5", "lncC", "X", "HAP1"),
        _rec("g6", "lncC", "X", "THP1"),
    ]


class TestSplitByChrom:
    def test_test_set_contains_only_target_chrom(self, records):
        _, test = split_by_chrom(records, "1")
        assert all(r.chrom == "1" for r in test)

    def test_train_set_contains_no_target_chrom(self, records):
        train, _ = split_by_chrom(records, "1")
        assert all(r.chrom != "1" for r in train)

    def test_partition_is_complete(self, records):
        train, test = split_by_chrom(records, "1")
        assert sorted(train + test, key=lambda r: r.guide_id) == sorted(records, key=lambda r: r.guide_id)

    def test_unknown_chrom_yields_empty_test(self, records):
        train, test = split_by_chrom(records, "99")
        assert test == []
        assert len(train) == len(records)


class TestSplitByCellLine:
    def test_test_set_contains_only_target_cell_line(self, records):
        _, test = split_by_cell_line(records, "THP1")
        assert all(r.cell_line == "THP1" for r in test)

    def test_train_set_contains_no_target_cell_line(self, records):
        train, _ = split_by_cell_line(records, "THP1")
        assert all(r.cell_line != "THP1" for r in train)

    def test_partition_is_complete(self, records):
        train, test = split_by_cell_line(records, "THP1")
        assert sorted(train + test, key=lambda r: r.guide_id) == sorted(records, key=lambda r: r.guide_id)

    def test_unknown_cell_line_yields_empty_test(self, records):
        train, test = split_by_cell_line(records, "UNKNOWN")
        assert test == []
        assert len(train) == len(records)
