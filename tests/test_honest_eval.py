import numpy as np
import pytest

from lncfit.honest_eval import predict_cell_and_gene_holdout, score


class _SpyModel:
    """Records which (cell, gene) rows it trained on; column 0 is the gene id, column 1 the
    cell id."""

    fitted: list[np.ndarray] = []

    def fit(self, X, y):
        _SpyModel.fitted.append(X.copy())
        return self

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), X[:, 0] / 100.0])


def _toy(n_genes=20):
    cells = ["A", "B", "C"]
    X = {c: np.column_stack([np.arange(n_genes), np.full(n_genes, i)]).astype(float)
         for i, c in enumerate(cells)}
    y = {c: (np.arange(n_genes) % 4 == 0).astype(int) for c in cells}
    return X, y


def test_no_eval_gene_and_no_holdout_line_reaches_training():
    X, y = _toy()
    _SpyModel.fitted = []
    predict_cell_and_gene_holdout(X, y, "B", _SpyModel, n_gene_folds=4, seed=1)
    assert len(_SpyModel.fitted) == 4
    all_genes = set(range(20))
    eval_sets = []
    for X_train in _SpyModel.fitted:
        assert 1.0 not in set(X_train[:, 1])           # cell line B never trained on
        trained = set(X_train[:, 0].astype(int))
        eval_sets.append(all_genes - trained)
    # the genes left out across folds partition the gene set exactly once
    assert sorted(g for s in eval_sets for g in s) == sorted(all_genes)


def test_every_gene_gets_one_score():
    X, y = _toy()
    scores = predict_cell_and_gene_holdout(X, y, "A", _SpyModel, n_gene_folds=5)
    assert not np.isnan(scores).any()
    assert np.allclose(scores, np.arange(20) / 100.0)


def test_rejects_single_cell_line():
    X, y = _toy()
    with pytest.raises(ValueError):
        predict_cell_and_gene_holdout({"A": X["A"]}, {"A": y["A"]}, "A", _SpyModel)


def test_score_reports_pos_rate():
    out = score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert out == pytest.approx({"auprc": 1.0, "auroc": 1.0, "pos_rate": 0.5})
