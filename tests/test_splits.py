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


def test_split_by_chrom_and_cell_line():
    # 6 records: 3 lncRNAs x 2 cell lines, spread across 3 chromosomes
    records = [
        _rec("g1", "lncA", "1", "HAP1"),
        _rec("g2", "lncA", "1", "THP1"),
        _rec("g3", "lncB", "2", "HAP1"),
        _rec("g4", "lncB", "2", "THP1"),
        _rec("g5", "lncC", "X", "HAP1"),
        _rec("g6", "lncC", "X", "THP1"),
    ]
    key = lambda r: r.guide_id  # noqa: E731

    for split_fn, attr, target_value, unknown_value in [
        (split_by_chrom, "chrom", "1", "99"),
        (split_by_cell_line, "cell_line", "THP1", "UNKNOWN"),
    ]:
        train, test = split_fn(records, target_value)
        assert all(getattr(r, attr) == target_value for r in test)
        assert all(getattr(r, attr) != target_value for r in train)
        assert sorted(train + test, key=key) == sorted(records, key=key)

        train2, test2 = split_fn(records, unknown_value)
        assert test2 == [] and len(train2) == len(records)
