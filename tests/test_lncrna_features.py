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


def test_build_lncrna_features_core_contract():
    vocab = all_kmers(3)
    records = [_rec("T1", label=1), _rec("T2", label=0)]
    seqs = {"T1": "AAACCC", "T2": "GGGTTT"}
    X, y, cols = build_lncrna_features(records, seqs, k=3, vocab=vocab)
    assert X.shape == (2, len(vocab) + 5)
    assert list(y) == [1.0, 0.0]  # binary label, not continuous fold-change
    assert not any(c.startswith("day_") for c in cols)  # Day-14-only task, no day dimension

    # frequency computed and normalized from the lncRNA's own transcript sequence
    small_vocab = ["AAA", "TTT"]
    X2, _, cols2 = build_lncrna_features([_rec("T1")], {"T1": "AAATTT"}, k=3, vocab=small_vocab)
    assert X2[0, cols2.index("AAA")] == pytest.approx(0.5)
    assert X2[0, cols2.index("TTT")] == pytest.approx(0.5)

    # THE property issue #65 fixed: same lncRNA shares one feature vector
    # across every cell-line row, regardless of cell line.
    shared = [_rec("T1", cell_line="HAP1"), _rec("T1", cell_line="K562")]
    X3, _, cols3 = build_lncrna_features(shared, {"T1": "AAA"}, k=3, vocab=["AAA"])
    assert X3[0, cols3.index("AAA")] == X3[1, cols3.index("AAA")] == pytest.approx(1.0)

    # missing distance -> sentinel; missing transcript sequence -> zero vector, not a crash
    X4, _, _ = build_lncrna_features(
        [_rec("T1", distance=None)], {"T1": "AAACCC"}, k=3, vocab=["AAA"], include_distance=True,
    )
    assert X4[0, -1] == -1.0
    X5, _, _ = build_lncrna_features([_rec("T1")], {}, k=3, vocab=vocab)
    assert np.all(X5[0, : len(vocab)] == 0.0)


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
