"""Abstract base for pluggable lncRNA hit classifiers.

Modeled on the teddy project's model layer (LiuzLab/teddy, teddy/model/base.py) —
a shared abstract interface + a registry so architectures are drop-in — adapted here
for sklearn-style binary classifiers (fit / predict_proba on feature matrices) rather
than teddy's PyTorch nn.Modules. Different project, different domain; only the
plug-in *pattern* is borrowed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix


class ClassifierModel(ABC):
    """Abstract base for all lncRNA-hit classifier wrappers.

    Every wrapper must:

    1. Set a class-level ``model_type`` string matching its registry key.
    2. Accept hyperparameters as keyword args in ``__init__`` and store them
       on ``self.params`` (so a run can serialize exactly what it fit).
    3. Implement :meth:`fit` returning ``self``.
    4. Implement :meth:`predict_proba` returning P(positive class) as a 1-D array.

    The feature matrix ``X`` is whatever ``lncfit.features.build_lncrna_features``
    produces (dense ndarray or CSR sparse); ``y`` is the binary hit label. Class
    imbalance handling (~5% positive rate) is each wrapper's own responsibility —
    e.g. XGBoost's ``scale_pos_weight``, logistic regression's ``class_weight`` —
    computed from the training ``y`` inside :meth:`fit`, not pushed onto callers.
    """

    model_type: str

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def fit(self, X: np.ndarray | csr_matrix, y: np.ndarray) -> "ClassifierModel":
        """Fit the model. Returns self for chaining."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray | csr_matrix) -> np.ndarray:
        """Return P(positive class) for each row, as a float 1-D array of shape (n,).

        Deliberately NOT sklearn's 2-column ``predict_proba`` — downstream evaluation
        (``evaluate_lncrna_by_group``, ROC/PR) only ever wants the positive-class
        column, and standardizing on a 1-D array here removes a recurring ``[:, 1]``
        footgun across the runner and plots.
        """
        ...

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({param_str})"
