"""Small feedforward neural classification head (no boosted-tree/linear/kNN model)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch import nn

from lncfit.classifiers.base import ClassifierModel
from lncfit.classifiers.registry import register_classifier


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register_classifier("mlp")
class MLPClassifier(ClassifierModel):
    """One-hidden-layer MLP -> sigmoid, trained by gradient descent (BCEWithLogitsLoss + Adam).

    Meant to sit directly on top of a feature vector such as a frozen DNABERT-2
    embedding concatenated with a cell-line one-hot block
    (lncfit.features.build_lncrna_embedding_features) -- a genuine trainable
    classification layer, rather than routing the embedding through a separate
    boosted-tree/linear/kNN model. ``BCEWithLogitsLoss(pos_weight=...)`` handles the
    ~5% positive rate (computed from the training y inside fit(), same convention as
    XGBoost's scale_pos_weight). A 10% stratified slice of the training data is held
    out inside fit() purely for early stopping -- keeps the fit/predict_proba contract
    at (X, y) only, same as every other wrapper.
    """

    model_type = "mlp"

    def __init__(
        self,
        hidden: int = 128,
        dropout: float = 0.2,
        lr: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 10,
        batch_size: int = 256,
        seed: int = 42,
        **params,
    ) -> None:
        super().__init__(
            hidden=hidden, dropout=dropout, lr=lr, max_epochs=max_epochs,
            patience=patience, batch_size=batch_size, seed=seed, **params,
        )
        self._model: _MLP | None = None

    def fit(self, X, y) -> "MLPClassifier":
        # Pin to a single thread: torch's own OpenMP-based thread pool deadlocks when
        # it initializes in the same process after xgboost/sklearn (n_jobs=-1) have
        # already spun up their own OpenMP threads (macOS-specific; this model is small
        # enough that single-threaded training is not a real slowdown).
        torch.set_num_threads(1)

        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        torch.manual_seed(self.params["seed"])
        idx = np.arange(len(y))
        tr_idx, es_idx = train_test_split(idx, test_size=0.1, stratify=y, random_state=self.params["seed"])
        X_tr, y_tr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        X_es, y_es = torch.from_numpy(X[es_idx]), torch.from_numpy(y[es_idx])

        n_pos = float(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0)

        self._model = _MLP(X.shape[1], hidden=self.params["hidden"], dropout=self.params["dropout"])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.params["lr"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        batch_size = self.params["batch_size"]
        n_train = len(y_tr)
        best_val_loss = float("inf")
        best_state = None
        patience_left = self.params["patience"]

        for _ in range(self.params["max_epochs"]):
            self._model.train()
            perm = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                batch = perm[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self._model(X_tr[batch]), y_tr[batch])
                loss.backward()
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(X_es), y_es).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_left = self.params["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        assert self._model is not None, "call fit() before predict_proba()"
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(self._model(torch.from_numpy(X))).numpy()
        return proba
