import numpy as np
import pytest
from lncfit.screen_data import ScreenRecord
from lncfit.features import all_kmers, build_features, fit_vocab


def _rec(seq, cell_line="HAP1", day=7, fold_change=1.0, distance=None, target="T1"):
    return ScreenRecord(
        guide_id="g1",
        target=target,
        target_sequence=seq,
        cell_line=cell_line,
        day=day,
        replicate=1,
        fold_change=fold_change,
        distance_to_closest_pc_gene=distance,
    )


def test_build_features_core_contract():
    assert all_kmers(3) == sorted(all_kmers(3))

    # shape formula, raw fold-change passthrough, exact k-mer frequency value,
    # and day/cell-line one-hot columns, all from one build_features call.
    records = [_rec("AAAAAA", fold_change=v, day=7, cell_line="K562") for v in [1.0, -2.0, 0.5]]
    X, y, cols = build_features(records, k=3)
    assert X.shape == (3, 64 + 2 + 5)
    assert list(y) == pytest.approx([1.0, -2.0, 0.5])
    assert X[0, cols.index("AAA")] == pytest.approx(1.0)  # "AAAAAA" is 4 windows, all "AAA"
    assert "day_7" in cols and "day_14" in cols
    for cl in ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]:
        assert f"cell_{cl}" in cols

    # non-ACGT windows excluded (partial and total), through the real
    # production entry point (build_features -> _fill_kmer_row -> _count_kmers)
    partial = build_features([_rec("ACGNACG")], k=3)[0][0, :64]
    assert abs(partial.sum() - 1.0) < 1e-9
    all_invalid = build_features([_rec("NNNNN")], k=3)[0][0, :64]
    assert np.all(all_invalid == 0.0)

    # include_distance: present+value, missing->sentinel, disabled->absent
    X_d, _, cols_d = build_features([_rec("ACGT", distance=500)], k=3, include_distance=True)
    assert X_d[0, cols_d.index("distance_to_closest_pc_gene")] == pytest.approx(500.0)
    X_none, _, cols_none = build_features([_rec("ACGT", distance=None)], k=3, include_distance=True)
    assert X_none[0, cols_none.index("distance_to_closest_pc_gene")] == pytest.approx(-1.0)
    _, _, cols_off = build_features([_rec("ACGT", distance=500)], k=3, include_distance=False)
    assert "distance_to_closest_pc_gene" not in cols_off


def test_sparse_dense_and_vocab_handling():
    records = [_rec("ACGTACGTACGTACGTACGTACG", fold_change=v) for v in [1.0, -2.0, 0.5]]

    # Load-bearing equivalence: XGBoost treats sparse implicit zeros as
    # *missing*, not the real "k-mer absent" value a dense zero represents.
    X_dense, y_dense, cols_dense = build_features(records, k=3)
    X_sparse, y_sparse, cols_sparse = build_features(records, k=3, sparse=True)
    assert cols_dense == cols_sparse
    assert np.allclose(X_dense, X_sparse.toarray())
    assert np.allclose(y_dense, y_sparse)

    # restricted vocab shrinks the matrix and preserves column order
    small_vocab = all_kmers(3)[:10]
    X_small, _, cols_small = build_features(records, k=3, vocab=small_vocab)
    assert X_small.shape == (3, 10 + 2 + 5)
    assert cols_small[:10] == small_vocab

    # a k-mer seen only in holdout data is silently zero, not an error or a
    # misaligned extra column -- every chr1-holdout evaluation in this project
    # relies on exactly this.
    train_vocab = ["AAA"]
    X_train, _, _ = build_features([_rec("AAAAAA")], k=3, vocab=train_vocab)
    X_hold, _, _ = build_features([_rec("TTTTTT")], k=3, vocab=train_vocab)
    assert X_train.shape[1] == X_hold.shape[1]
    assert X_hold[0, 0] == pytest.approx(0.0)


def test_signed_overlap_negates_shared_kmers_only_when_present():
    # Reverse-complement overlap between a guide and its target body flips the
    # sign of shared k-mers -- both directions checked since getting either
    # branch wrong (negating when it shouldn't, or not negating when it
    # should) is an independent bug, not two views of one property.
    overlap_body = {"G1": ("ACGTGGGTTTACGT", "AAAAAAAAAA")}  # contains revcomp(AAACCC) = GGGTTT
    rec = _rec("AAACCC", target="G1")
    X, _, cols = build_features([rec], k=3, body_sequences=overlap_body, signed_overlap=True)
    for kmer in ("GGG", "GGT", "GTT", "TTT"):
        assert X[0, cols.index(f"body_signed_{kmer}")] < 0
    assert X[0, cols.index("body_signed_AAA")] > 0

    no_overlap_body = {"G1": ("ACGTACGTACGT", "CCCCCCCCCC")}  # does not contain GGGTTT
    X2, _, cols2 = build_features([rec], k=3, body_sequences=no_overlap_body, signed_overlap=True)
    kmer_end = cols2.index("day_7")
    assert (X2[0, :kmer_end] >= 0).all()


def test_fit_vocab_observed_sorted_and_edge_cases():
    assert fit_vocab(["AAAAAA"], k=3) == ["AAA"]  # only observed k-mers, sorted
    vocab = fit_vocab(["ACGNACG"], k=3)
    assert all("N" not in km for km in vocab)  # non-ACGT k-mers skipped
    assert fit_vocab([], k=3) == []  # empty input -> empty output, not an error
