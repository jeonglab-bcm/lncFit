import numpy as np
import pytest

from lncfit.screen_data import LncRnaRecord
from lncfit.features import all_kmers, build_lncrna_embedding_features, build_lncrna_features


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

    # issue #78: Celligner cell-line embedding at each dimensionality (dim=2 UMAP,
    # dim=10/70 pre-UMAP PCA), dense/sparse equivalence + the zero-fill fallback
    # for cell lines Celligner has no coordinates for (HEK293FT)
    records_hek = records + [_rec("T1", cell_line="HEK293FT", label=0)]
    for dim in [2, 10, 70]:
        Xd, _, cols_d = build_lncrna_features(records_hek, seqs, k=3, vocab=vocab, celligner_embedding_dim=dim)
        Xs, _, cols_s = build_lncrna_features(
            records_hek, seqs, k=3, vocab=vocab, celligner_embedding_dim=dim, sparse=True,
        )
        assert cols_d == cols_s
        embed_cols = [c for c in cols_d if c.startswith("cell_embed_")]
        assert len(embed_cols) == dim
        assert np.allclose(Xd, Xs.toarray())
        embed_slice = slice(cols_d.index(embed_cols[0]), cols_d.index(embed_cols[-1]) + 1)
        assert not np.allclose(Xd[0, embed_slice], 0.0)  # HAP1: has real Celligner coordinates
        assert np.allclose(Xd[2, embed_slice], 0.0)  # HEK293FT: not in CCLE/DepMap, zero-filled


def test_embedding_features_celligner_dim():
    # issue #78 pipeline follow-up: build_lncrna_embedding_features (the DNABERT-2
    # feature path) got the same celligner_embedding_dim option as build_lncrna_features,
    # so k-mer and DNABERT-2 runs can be compared at the same cell-embedding setting.
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    index = {"T1": 0, "T2": 1}
    records = [
        _rec("T1", cell_line="HAP1", label=1),
        _rec("T1", cell_line="HEK293FT", label=0),
    ]

    X0, _, cols0 = build_lncrna_embedding_features(records, (matrix, index))
    assert not any(c.startswith("cell_embed_") for c in cols0)
    assert X0.shape == (2, 2 + 5)  # 2 dnabert dims + 5-way cell one-hot

    for dim in [2, 10]:
        X, _, cols = build_lncrna_embedding_features(records, (matrix, index), celligner_embedding_dim=dim)
        embed_cols = [c for c in cols if c.startswith("cell_embed_")]
        assert len(embed_cols) == dim
        assert X.shape == (2, 2 + 5 + dim)
        embed_slice = slice(cols.index(embed_cols[0]), cols.index(embed_cols[-1]) + 1)
        assert not np.allclose(X[0, embed_slice], 0.0)  # HAP1: real Celligner coordinates
        assert np.allclose(X[1, embed_slice], 0.0)  # HEK293FT: zero-filled
        # the dnabert embedding block itself is untouched by the celligner addition
        assert np.allclose(X[0, :2], matrix[0])
        assert np.allclose(X[1, :2], matrix[0])  # same target T1 -> same dnabert row
