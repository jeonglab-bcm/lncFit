import numpy as np

from lncfit.screen_data import LncRnaRecord
from lncfit.features import build_lncrna_embedding_features


def _rec(target, cell_line="HAP1", label=0, distance=None):
    return LncRnaRecord(
        target=target, cell_line=cell_line, day=14,
        rra_pvalue=0.5, fold_change=0.0, label=label,
        distance_to_closest_pc_gene=distance,
    )


def _emb(n_dims=4):
    matrix = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)[:, :n_dims]
    index = {"T1": 0, "T2": 1}
    return matrix, index


def test_shape_is_dims_plus_cell_onehot():
    X, y, cols = build_lncrna_embedding_features([_rec("T1")], _emb())
    assert X.shape == (1, 4 + 5)
    assert cols[:4] == ["dnabert_0", "dnabert_1", "dnabert_2", "dnabert_3"]
    assert cols[4:] == [f"cell_{c}" for c in ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]]


def test_embedding_row_copied_for_target():
    X, _, _ = build_lncrna_embedding_features([_rec("T2")], _emb())
    assert list(X[0, :4]) == [5.0, 6.0, 7.0, 8.0]


def test_cell_line_one_hot_set():
    X, _, cols = build_lncrna_embedding_features([_rec("T1", cell_line="K562")], _emb())
    assert X[0, cols.index("cell_K562")] == 1.0
    assert X[0, cols.index("cell_HAP1")] == 0.0


def test_missing_target_gets_zero_embedding():
    X, _, _ = build_lncrna_embedding_features([_rec("UNKNOWN")], _emb())
    assert np.all(X[0, :4] == 0.0)


def test_y_is_binary_label():
    recs = [_rec("T1", label=1), _rec("T2", label=0)]
    _, y, _ = build_lncrna_embedding_features(recs, _emb())
    assert list(y) == [1.0, 0.0]


def test_no_cell_line_block_when_disabled():
    _, _, cols = build_lncrna_embedding_features([_rec("T1")], _emb(), include_cell_line=False)
    assert not any(c.startswith("cell_") for c in cols)
    assert len(cols) == 4


def test_include_distance_appends_column():
    X, _, cols = build_lncrna_embedding_features([_rec("T1", distance=None)], _emb(), include_distance=True)
    assert cols[-1] == "distance_to_closest_pc_gene"
    assert X[0, -1] == -1.0


def test_same_target_shares_embedding_across_cell_lines():
    recs = [_rec("T1", cell_line="HAP1"), _rec("T1", cell_line="K562")]
    X, _, _ = build_lncrna_embedding_features(recs, _emb())
    assert np.array_equal(X[0, :4], X[1, :4])
