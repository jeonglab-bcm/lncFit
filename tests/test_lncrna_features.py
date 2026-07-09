import numpy as np
import pytest

from lncfit.screen_data import LncRnaRecord
from lncfit.features import all_kmers, build_lncrna_features


def _rec(target, cell_line="HAP1", label=0, guide_sequences=(), distance=None):
    return LncRnaRecord(
        target=target,
        cell_line=cell_line,
        day=14,
        rra_pvalue=0.5,
        fold_change=0.0,
        label=label,
        guide_sequences=guide_sequences,
        distance_to_closest_pc_gene=distance,
    )


def test_shape_matches_vocab_plus_cell_columns():
    vocab = all_kmers(3)
    records = [_rec("T1", guide_sequences=("AAACCC",))]
    X, y, cols = build_lncrna_features(records, k=3, vocab=vocab)
    assert X.shape == (1, len(vocab) + 5)
    assert len(cols) == len(vocab) + 5


def test_y_is_binary_label_not_fold_change():
    records = [_rec("T1", label=1, guide_sequences=("AAACCC",)),
               _rec("T2", label=0, guide_sequences=("GGGTTT",))]
    _, y, _ = build_lncrna_features(records, k=3)
    assert list(y) == [1.0, 0.0]


def test_pools_kmers_across_multiple_guides():
    vocab = ["AAA", "TTT"]
    records = [_rec("T1", guide_sequences=("AAA", "TTT"))]
    X, _, cols = build_lncrna_features(records, k=3, vocab=vocab)
    aaa_idx = cols.index("AAA")
    ttt_idx = cols.index("TTT")
    # 1 AAA window + 1 TTT window pooled across the two guides -> 0.5 / 0.5
    assert X[0, aaa_idx] == pytest.approx(0.5)
    assert X[0, ttt_idx] == pytest.approx(0.5)


def test_same_target_shares_pooled_vector_across_cell_lines():
    vocab = ["AAA", "TTT"]
    records = [
        _rec("T1", cell_line="HAP1", guide_sequences=("AAA",)),
        _rec("T1", cell_line="K562", guide_sequences=("AAA",)),
    ]
    X, _, cols = build_lncrna_features(records, k=3, vocab=vocab)
    aaa_idx = cols.index("AAA")
    assert X[0, aaa_idx] == X[1, aaa_idx] == pytest.approx(1.0)


def test_cell_line_one_hot():
    records = [_rec("T1", cell_line="K562", guide_sequences=("AAACCC",))]
    X, _, cols = build_lncrna_features(records, k=3, vocab=["AAA"])
    assert X[0, cols.index("cell_K562")] == 1.0
    assert X[0, cols.index("cell_HAP1")] == 0.0


def test_no_day_column_present():
    records = [_rec("T1", guide_sequences=("AAACCC",))]
    _, _, cols = build_lncrna_features(records, k=3, vocab=["AAA"])
    assert not any(c.startswith("day_") for c in cols)


def test_include_distance_uses_negative_one_sentinel_when_missing():
    records = [_rec("T1", guide_sequences=("AAACCC",), distance=None)]
    X, _, cols = build_lncrna_features(records, k=3, vocab=["AAA"], include_distance=True)
    assert cols[-1] == "distance_to_closest_pc_gene"
    assert X[0, -1] == -1.0


def test_target_with_no_guides_gets_zero_kmer_vector():
    records = [_rec("T1", guide_sequences=())]
    X, _, cols = build_lncrna_features(records, k=3, vocab=all_kmers(3))
    assert np.all(X[0, : len(all_kmers(3))] == 0.0)


def test_sparse_matches_dense():
    vocab = all_kmers(3)
    records = [
        _rec("T1", cell_line="HAP1", label=1, guide_sequences=("AAACCCGGGTTT", "TTTGGGCCCAAA")),
        _rec("T2", cell_line="THP1", label=0, guide_sequences=("ACGTACGTACGT",)),
    ]
    X_dense, y_dense, cols_dense = build_lncrna_features(records, k=3, vocab=vocab, sparse=False)
    X_sparse, y_sparse, cols_sparse = build_lncrna_features(records, k=3, vocab=vocab, sparse=True)
    assert cols_dense == cols_sparse
    assert np.allclose(X_dense, X_sparse.toarray())
    assert np.allclose(y_dense, y_sparse)
