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


def test_all_kmers_sorted_order():
    # Length (4**k) has no standalone test: build_features's shape check below
    # uses all_kmers(k) as its default vocab, so a wrong-length bug there would
    # already fail here. Sortedness isn't exercised by anything downstream, so
    # it has no downstream substitute and stays as its own check.
    kmers = all_kmers(3)
    assert kmers == sorted(kmers)


class TestBuildFeatures:
    def test_shape_label_and_exact_kmer_value(self):
        # One combined "core contract" check: shape formula, raw fold-change
        # passthrough, and a specific k-mer frequency landing in the right
        # column, all from one build_features call on a known sequence.
        records = [_rec("AAAAAA", fold_change=v) for v in [1.0, -2.0, 0.5]]
        X, y, cols = build_features(records, k=3)
        assert X.shape == (3, 64 + 2 + 5)
        assert list(y) == pytest.approx([1.0, -2.0, 0.5])
        # "AAAAAA" has 4 windows, all "AAA" -> that column must be exactly 1.0.
        assert X[0, cols.index("AAA")] == pytest.approx(1.0)

    @pytest.mark.parametrize("seq, all_invalid", [("ACGNACG", False), ("NNNNN", True)])
    def test_non_acgt_windows_excluded(self, seq, all_invalid):
        # Exercises the shared _count_kmers logic through the real production
        # entry point (build_features -> _fill_kmer_row -> _count_kmers).
        X, _, _ = build_features([_rec(seq)], k=3)
        kmer_row = X[0, :64]
        if all_invalid:
            assert np.all(kmer_row == 0.0)
        else:
            assert abs(kmer_row.sum() - 1.0) < 1e-9

    def test_day_and_cell_line_onehot_columns_present(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", day=7, cell_line="K562")]
        _, _, cols = build_features(records, k=3)
        assert "day_7" in cols and "day_14" in cols
        for cl in ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]:
            assert f"cell_{cl}" in cols

    @pytest.mark.parametrize(
        "include_distance, distance, expect_present, expect_value",
        [
            (True, 500, True, 500.0),
            (True, None, True, -1.0),  # missing distance -> sentinel, not NaN/0
            (False, 500, False, None),
        ],
    )
    def test_include_distance(self, include_distance, distance, expect_present, expect_value):
        records = [_rec("ACGTACGTACGTACGTACGTACG", distance=distance)]
        X, _, cols = build_features(records, k=3, include_distance=include_distance)
        assert ("distance_to_closest_pc_gene" in cols) == expect_present
        if expect_present:
            assert X[0, cols.index("distance_to_closest_pc_gene")] == pytest.approx(expect_value)

    def test_sparse_matches_dense(self):
        # This equivalence matters for real: XGBoost treats a CSR matrix's implicit
        # zeros as *missing*, not as the real "k-mer absent" value a dense zero
        # represents -- lncfit/classifiers/xgboost_clf.py builds dense specifically
        # because of this.
        records = [_rec("ACGTACGTACGTACGTACGTACG", fold_change=v) for v in [1.0, -2.0, 0.5]]
        X_dense, y_dense, cols_dense = build_features(records, k=3)
        X_sparse, y_sparse, cols_sparse = build_features(records, k=3, sparse=True)
        assert cols_dense == cols_sparse
        assert np.allclose(X_dense, X_sparse.toarray())
        assert np.allclose(y_dense, y_sparse)

    def test_custom_vocab_reduces_columns_and_preserves_order(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG")] * 3
        small_vocab = all_kmers(3)[:10]
        X, _, cols = build_features(records, k=3, vocab=small_vocab)
        assert X.shape == (3, 10 + 2 + 5)
        assert cols[:10] == small_vocab

    def test_holdout_unseen_kmer_silently_dropped(self):
        # Real scenario, not a hypothetical: every chr1-holdout evaluation in this
        # project reuses a vocab fit on train-only sequences. A k-mer that only
        # appears in holdout data must be silently ignored (zero contribution),
        # not raise or silently add an extra column that would break alignment.
        train_vocab = ["AAA"]
        X_train, _, _ = build_features([_rec("AAAAAA")], k=3, vocab=train_vocab)
        X_hold, _, _ = build_features([_rec("TTTTTT")], k=3, vocab=train_vocab)
        assert X_train.shape[1] == X_hold.shape[1]
        assert X_hold[0, 0] == pytest.approx(0.0)


class TestSignedOverlap:
    # Reverse-complement overlap between a guide and its target body flips the sign
    # of shared k-mers -- both directions of this conditional need their own test,
    # since getting either branch wrong (negating when it shouldn't, or not negating
    # when it should) is an independent bug, not two views of the same property.
    def test_overlap_negates_shared_kmers(self):
        # guide AAACCC -> revcomp GGGTTT; body contains GGGTTT -> those k-mers negate
        body_seqs = {"G1": ("ACGTGGGTTTACGT", "AAAAAAAAAA")}
        rec = _rec("AAACCC", target="G1")
        X_dense, _, cols = build_features([rec], k=3, body_sequences=body_seqs, signed_overlap=True)
        for kmer in ("GGG", "GGT", "GTT", "TTT"):
            assert X_dense[0, cols.index(f"body_signed_{kmer}")] < 0, f"{kmer} should be negative"
        assert X_dense[0, cols.index("body_signed_AAA")] > 0

    def test_no_overlap_all_nonnegative(self):
        # guide AAACCC -> revcomp GGGTTT; body does not contain it -> no negation
        body_seqs = {"G1": ("ACGTACGTACGT", "CCCCCCCCCC")}
        rec = _rec("AAACCC", target="G1")
        X_dense, _, cols = build_features([rec], k=3, body_sequences=body_seqs, signed_overlap=True)
        kmer_end = cols.index("day_7")
        assert (X_dense[0, :kmer_end] >= 0).all()


def test_fit_vocab_observed_sorted_and_edge_cases():
    assert fit_vocab(["AAAAAA"], k=3) == ["AAA"]  # only observed k-mers, sorted
    vocab = fit_vocab(["ACGNACG"], k=3)
    assert all("N" not in km for km in vocab)  # non-ACGT k-mers skipped
    assert fit_vocab([], k=3) == []  # empty input -> empty output, not an error
