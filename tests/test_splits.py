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


# split_by_chrom and split_by_cell_line are independently implemented but must
# satisfy the same invariants -- parametrized across both rather than
# duplicating the same test bodies under two class names.
SPLIT_CASES = [
    pytest.param(split_by_chrom, "chrom", "1", "99", id="by_chrom"),
    pytest.param(split_by_cell_line, "cell_line", "THP1", "UNKNOWN", id="by_cell_line"),
]


@pytest.mark.parametrize("split_fn, attr, target_value, unknown_value", SPLIT_CASES)
class TestSplitInvariants:
    def test_split_is_correct_and_complete(self, records, split_fn, attr, target_value, unknown_value):
        train, test = split_fn(records, target_value)
        assert all(getattr(r, attr) == target_value for r in test)
        assert all(getattr(r, attr) != target_value for r in train)
        assert sorted(train + test, key=lambda r: r.guide_id) == sorted(records, key=lambda r: r.guide_id)

    def test_unknown_value_yields_empty_test(self, records, split_fn, attr, target_value, unknown_value):
        train, test = split_fn(records, unknown_value)
        assert test == []
        assert len(train) == len(records)
