import numpy as np
import pytest

from lncfit.screen_data import LncRnaRecord
from lncfit.features import all_kmers, build_lncrna_features


def _rec(target, cell_line="HAP1", label=0, distance=None):
    return LncRnaRecord(
        target=target,
        cell_line=cell_line,
        day=14,
        rra_pvalue=0.5,
        fold_change=0.0,
        label=label,
        distance_to_closest_pc_gene=distance,
    )


def test_shape_and_binary_label():
    vocab = all_kmers(3)
    records = [_rec("T1", label=1), _rec("T2", label=0)]
    seqs = {"T1": "AAACCC", "T2": "GGGTTT"}
    X, y, cols = build_lncrna_features(records, seqs, k=3, vocab=vocab)
    assert X.shape == (2, len(vocab) + 5)
    assert len(cols) == len(vocab) + 5
    assert list(y) == [1.0, 0.0]  # binary label, not continuous fold-change


def test_kmer_freq_computed_from_own_transcript_sequence():
    vocab = ["AAA", "TTT"]
    records = [_rec("T1")]
    # Restricted vocab: only AAA/TTT windows count toward the normalising total.
    X, _, cols = build_lncrna_features(records, {"T1": "AAATTT"}, k=3, vocab=vocab)
    aaa_idx = cols.index("AAA")
    ttt_idx = cols.index("TTT")
    assert X[0, aaa_idx] == pytest.approx(0.5)
    assert X[0, ttt_idx] == pytest.approx(0.5)


def test_same_target_shares_feature_vector_across_cell_lines():
    vocab = ["AAA", "TTT"]
    records = [
        _rec("T1", cell_line="HAP1"),
        _rec("T1", cell_line="K562"),
    ]
    X, _, cols = build_lncrna_features(records, {"T1": "AAA"}, k=3, vocab=vocab)
    aaa_idx = cols.index("AAA")
    assert X[0, aaa_idx] == X[1, aaa_idx] == pytest.approx(1.0)


def test_no_day_column_and_distance_sentinel():
    records = [_rec("T1", distance=None)]
    X, _, cols = build_lncrna_features(records, {"T1": "AAACCC"}, k=3, vocab=["AAA"], include_distance=True)
    assert not any(c.startswith("day_") for c in cols)  # Day-14-only task, no day dimension
    assert cols[-1] == "distance_to_closest_pc_gene"
    assert X[0, -1] == -1.0  # missing distance -> sentinel, not NaN/0


def test_target_missing_from_transcript_sequences_gets_zero_kmer_vector():
    records = [_rec("T1")]
    X, _, cols = build_lncrna_features(records, {}, k=3, vocab=all_kmers(3))
    assert np.all(X[0, : len(all_kmers(3))] == 0.0)


def test_sparse_matches_dense():
    vocab = all_kmers(3)
    records = [
        _rec("T1", cell_line="HAP1", label=1),
        _rec("T2", cell_line="THP1", label=0),
    ]
    seqs = {"T1": "AAACCCGGGTTTTTTGGGCCCAAA", "T2": "ACGTACGTACGT"}
    X_dense, y_dense, cols_dense = build_lncrna_features(records, seqs, k=3, vocab=vocab, sparse=False)
    X_sparse, y_sparse, cols_sparse = build_lncrna_features(records, seqs, k=3, vocab=vocab, sparse=True)
    assert cols_dense == cols_sparse
    assert np.allclose(X_dense, X_sparse.toarray())
    assert np.allclose(y_dense, y_sparse)
