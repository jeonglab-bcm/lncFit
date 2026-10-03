"""Hold out a cell line AND a fold of genes at the same time.

Holding out only the cell line lets a model score well by recognising genes: every gene is
seen in the other lines with its label, and gene-level features (length, exon count, ...)
act as an identifier. On the 40-feature model from #118 that recognition was worth about
0.03 AUPRC (0.173 -> 0.145). This module scores each held-out cell line with the genes being
predicted also absent from training, in every line, so the score cannot come from
remembering a gene.

    X_by_cell  {cell_line: (n_genes, n_features)}, rows in one shared gene order
    y_by_cell  {cell_line: (n_genes,)} binary labels, same gene order
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold


def predict_cell_and_gene_holdout(
    X_by_cell: dict[str, np.ndarray],
    y_by_cell: dict[str, np.ndarray],
    holdout: str,
    make_model: Callable[[], object],
    n_gene_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Predict every gene in `holdout`, each from a model that never saw that gene or line.

    Genes are split into `n_gene_folds` folds. For each fold, the model trains on the other
    cell lines' rows for the remaining genes and predicts the held-out line's rows for the
    fold's genes. Returns one score per gene, in the shared gene order.
    """
    sources = [c for c in X_by_cell if c != holdout]
    if not sources:
        raise ValueError("need at least one training cell line besides the holdout")
    n_genes = len(y_by_cell[holdout])
    scores = np.full(n_genes, np.nan)
    folds = KFold(n_splits=n_gene_folds, shuffle=True, random_state=seed)
    for train_genes, eval_genes in folds.split(np.arange(n_genes)):
        X_train = np.vstack([X_by_cell[c][train_genes] for c in sources])
        y_train = np.concatenate([y_by_cell[c][train_genes] for c in sources])
        model = make_model()
        model.fit(X_train, y_train)
        scores[eval_genes] = model.predict_proba(X_by_cell[holdout][eval_genes])[:, 1]
    return scores


def score(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUPRC and AUROC, plus the positive rate that AUPRC should be read against."""
    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "pos_rate": float(np.mean(y_true)),
    }
