import pytest
from lncfit.screen_data import ScreenRecord
from lncfit.features import all_kmers, kmer_freq_vector, build_features, fit_vocab


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


class TestAllKmers:
    def test_k3_length(self):
        assert len(all_kmers(3)) == 64

    def test_k6_length(self):
        assert len(all_kmers(6)) == 4096

    def test_sorted_order(self):
        kmers = all_kmers(3)
        assert kmers == sorted(kmers)


class TestKmerFreqVector:
    def test_sums_to_one(self):
        vocab = all_kmers(3)
        vec = kmer_freq_vector("ACGTACGT", 3, vocab)
        assert abs(sum(vec) - 1.0) < 1e-9

    def test_correct_counts(self):
        vocab = all_kmers(3)
        # "AAAAAA" has 4 windows, all "AAA"
        vec = kmer_freq_vector("AAAAAA", 3, vocab)
        aaa_idx = vocab.index("AAA")
        assert vec[aaa_idx] == pytest.approx(1.0)

    def test_skips_non_acgt(self):
        vocab = all_kmers(3)
        # N in the middle — windows containing N are skipped
        vec = kmer_freq_vector("ACGNACG", 3, vocab)
        assert abs(sum(vec) - 1.0) < 1e-9

    def test_all_non_acgt_returns_zeros(self):
        vocab = all_kmers(3)
        vec = kmer_freq_vector("NNNNN", 3, vocab)
        assert all(v == 0.0 for v in vec)


class TestBuildFeatures:
    def test_x_shape_k3(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG")] * 5
        X, y, cols = build_features(records, k=3)
        assert X.shape == (5, 64 + 2 + 5)

    def test_x_shape_k6(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG")] * 3
        X, y, cols = build_features(records, k=6)
        assert X.shape == (3, 4096 + 2 + 5)

    def test_y_values(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", fold_change=v) for v in [1.0, -2.0, 0.5]]
        _, y, _ = build_features(records, k=3)
        assert list(y) == pytest.approx([1.0, -2.0, 0.5])

    def test_day_onehot_columns_present(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", day=7)]
        _, _, cols = build_features(records, k=3)
        assert "day_7" in cols
        assert "day_14" in cols

    def test_cell_line_onehot_columns_present(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", cell_line="K562")]
        _, _, cols = build_features(records, k=3)
        for cl in ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]:
            assert f"cell_{cl}" in cols

    def test_include_distance_true(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", distance=500)]
        X, _, cols = build_features(records, k=3, include_distance=True)
        assert "distance_to_closest_pc_gene" in cols
        assert X[0, cols.index("distance_to_closest_pc_gene")] == pytest.approx(500.0)

    def test_include_distance_none_becomes_sentinel(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", distance=None)]
        X, _, cols = build_features(records, k=3, include_distance=True)
        assert X[0, cols.index("distance_to_closest_pc_gene")] == pytest.approx(-1.0)

    def test_include_distance_false(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG", distance=500)]
        _, _, cols = build_features(records, k=3, include_distance=False)
        assert "distance_to_closest_pc_gene" not in cols

    def test_sparse_matches_dense(self):
        import numpy as np
        records = [_rec("ACGTACGTACGTACGTACGTACG", fold_change=v) for v in [1.0, -2.0, 0.5]]
        X_dense, y_dense, cols_dense = build_features(records, k=3)
        X_sparse, y_sparse, cols_sparse = build_features(records, k=3, sparse=True)
        assert cols_dense == cols_sparse
        assert np.allclose(X_dense, X_sparse.toarray())
        assert np.allclose(y_dense, y_sparse)

    def test_custom_vocab_reduces_columns(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG")] * 3
        full_vocab = all_kmers(3)
        small_vocab = full_vocab[:10]
        X, _, cols = build_features(records, k=3, vocab=small_vocab)
        assert X.shape == (3, 10 + 2 + 5)
        assert cols[:10] == small_vocab

    def test_holdout_unseen_kmer_silently_dropped(self):
        import numpy as np
        # train on sequences that only produce "AAA"; holdout has "TTT" too
        train_vocab = ["AAA"]
        train = [_rec("AAAAAA")]
        holdout = [_rec("TTTTTT")]
        X_train, _, _ = build_features(train, k=3, vocab=train_vocab)
        X_hold, _, _ = build_features(holdout, k=3, vocab=train_vocab)
        # holdout TTT is not in vocab — its row should be all zeros (k-mer part)
        assert X_train.shape[1] == X_hold.shape[1]
        assert X_hold[0, 0] == pytest.approx(0.0)

    def test_custom_vocab_sparse_matches_dense(self):
        import numpy as np
        records = [_rec("ACGTACGTACGTACGTACGTACG", fold_change=v) for v in [1.0, -2.0]]
        vocab = all_kmers(3)[:20]
        X_dense, _, _ = build_features(records, k=3, vocab=vocab)
        X_sparse, _, _ = build_features(records, k=3, vocab=vocab, sparse=True)
        assert np.allclose(X_dense, X_sparse.toarray())


class TestSignedOverlap:
    def test_overlap_negates_shared_kmers_and_sparse_matches_dense(self):
        import numpy as np
        # guide AAACCC → revcomp GGGTTT; body first window contains GGGTTT
        # rc_guide k-mers GGG, GGT, GTT, TTT must be negative; others positive
        body_seqs = {"G1": ("ACGTGGGTTTACGT", "AAAAAAAAAA")}
        rec = _rec("AAACCC", target="G1")
        X_dense, _, cols = build_features([rec], k=3, body_sequences=body_seqs, signed_overlap=True)
        X_sparse, _, _ = build_features([rec], k=3, body_sequences=body_seqs, signed_overlap=True, sparse=True)
        for kmer in ("GGG", "GGT", "GTT", "TTT"):
            assert X_dense[0, cols.index(f"body_signed_{kmer}")] < 0, f"{kmer} should be negative"
        assert X_dense[0, cols.index("body_signed_AAA")] > 0
        assert np.allclose(X_dense, X_sparse.toarray())

    def test_no_overlap_all_nonnegative_and_sparse_matches_dense(self):
        import numpy as np
        # guide AAACCC → revcomp GGGTTT; body does not contain GGGTTT → no negation
        body_seqs = {"G1": ("ACGTACGTACGT", "CCCCCCCCCC")}
        rec = _rec("AAACCC", target="G1")
        X_dense, _, cols = build_features([rec], k=3, body_sequences=body_seqs, signed_overlap=True)
        X_sparse, _, _ = build_features([rec], k=3, body_sequences=body_seqs, signed_overlap=True, sparse=True)
        kmer_end = cols.index("day_7")
        assert (X_dense[0, :kmer_end] >= 0).all()
        assert np.allclose(X_dense, X_sparse.toarray())


class TestFitVocab:
    def test_returns_only_observed_kmers(self):
        vocab = fit_vocab(["AAAAAA"], k=3)
        assert vocab == ["AAA"]

    def test_sorted(self):
        vocab = fit_vocab(["ACGT"], k=2)
        assert vocab == sorted(vocab)

    def test_skips_non_acgt(self):
        vocab = fit_vocab(["ACGNACG"], k=3)
        assert all("N" not in km for km in vocab)

    def test_subset_of_all_kmers(self):
        seqs = ["ACGTACGTACGTACGTACGTACG"]
        vocab = fit_vocab(seqs, k=6)
        assert set(vocab) <= set(all_kmers(6))

    def test_empty_seqs_returns_empty(self):
        assert fit_vocab([], k=3) == []
