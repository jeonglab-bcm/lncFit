from lncfit.screen_data import ScreenRecord


def split_by_chrom(
    records: list[ScreenRecord],
    test_chrom: str,
) -> tuple[list[ScreenRecord], list[ScreenRecord]]:
    """Partition records by chromosome, holding out all records whose lncRNA maps to test_chrom.

    Loci on the same chromosome share chromatin state and cis-regulatory elements, so nearby
    lncRNAs have correlated essentiality. Holding out an entire chromosome prevents the model
    from exploiting co-regulation rather than sequence-level features.

    Returns (train, test). Every record appears in exactly one partition; none are dropped.
    Unknown test_chrom yields empty test and full train.
    """
    train = [r for r in records if r.chrom != test_chrom]
    test = [r for r in records if r.chrom == test_chrom]
    return train, test


def split_by_cell_line(
    records: list[ScreenRecord],
    test_cell_line: str,
) -> tuple[list[ScreenRecord], list[ScreenRecord]]:
    """Partition records by cell line, holding out all records from test_cell_line.

    The practical goal of lncFit is to predict essentiality in cell types not yet screened.
    If all cell lines appear in training the model memorises cell-type-specific biases instead
    of learning transferable features. Holding out one complete cell line measures genuine
    cross-context generalisation.

    Returns (train, test). Every record appears in exactly one partition; none are dropped.
    Unknown test_cell_line yields empty test and full train.
    """
    train = [r for r in records if r.cell_line != test_cell_line]
    test = [r for r in records if r.cell_line == test_cell_line]
    return train, test
