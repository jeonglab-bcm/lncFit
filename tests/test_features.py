import pytest
from lncfit.screen_data import ScreenRecord
from lncfit.features import all_kmers, kmer_freq_vector, build_features


def _rec(seq, cell_line="HAP1", day=7, fold_change=1.0, distance=None):
    return ScreenRecord(
        guide_id="g1",
        target="T1",
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

    def test_k3_sorted_starts_aaa(self):
        assert all_kmers(3)[0] == "AAA"

    def test_k3_sorted_ends_ttt(self):
        assert all_kmers(3)[-1] == "TTT"

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

    def test_sparse_shape(self):
        records = [_rec("ACGTACGTACGTACGTACGTACG")] * 5
        X, y, cols = build_features(records, k=3, sparse=True)
        assert X.shape == (5, 64 + 2 + 5)

    def test_sparse_matches_dense(self):
        import numpy as np
        records = [_rec("ACGTACGTACGTACGTACGTACG", fold_change=v) for v in [1.0, -2.0, 0.5]]
        X_dense, y_dense, cols_dense = build_features(records, k=3)
        X_sparse, y_sparse, cols_sparse = build_features(records, k=3, sparse=True)
        assert cols_dense == cols_sparse
        assert np.allclose(X_dense, X_sparse.toarray())
        assert np.allclose(y_dense, y_sparse)
