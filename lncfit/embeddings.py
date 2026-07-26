from __future__ import annotations

import json

import numpy as np


def load_embeddings(path: str) -> tuple[np.ndarray, dict[str, int]]:
    """Load a pre-computed embedding matrix saved by scripts/embed_sequences.py.

    Returns (matrix, index) where matrix is float32 (n_seqs, n_dims)
    and index maps sequence id -> row in matrix.
    """
    data = np.load(path, allow_pickle=False)
    matrix = data["embeddings"].astype(np.float32)
    index: dict[str, int] = json.loads(str(data["index"]))
    return matrix, index


def reduce_embeddings_pca(
    embeddings: tuple[np.ndarray, dict[str, int]],
    train_targets: set[str] | None,
    n_components: int,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """PCA-reduce an embedding matrix, returning a new (matrix, index) pair.

    The 768 dims of a DNABERT-2 mean-pooled embedding are highly correlated, which
    is awkward for tree models (``colsample_bytree`` samples redundant columns).
    Projecting onto the top ``n_components`` PCs gives a compact, decorrelated
    basis. Standardizes columns first (PCA is scale-sensitive) via a mean/std
    computed on the training rows only.

    train_targets: the ids whose rows PCA (and the standardization) may be fit on.
    Every row is then transformed, including held-out ones. This is the whole
    reason this takes train_targets rather than just fitting on the full matrix --
    fitting on all rows would leak held-out targets' embedding distribution into
    the projection. Pass None to deliberately fit on everything (only sensible when
    there is no held-out split to protect).

    The returned index is the same mapping as the input's; only the matrix's
    column space changes, so callers can swap the result in without any other
    changes.
    """
    from sklearn.decomposition import PCA

    matrix, index = embeddings
    n_components = min(n_components, matrix.shape[1])

    if train_targets is None:
        fit_rows = np.arange(matrix.shape[0])
    else:
        fit_rows = np.array(
            sorted(index[t] for t in train_targets if t in index), dtype=int
        )
        if len(fit_rows) < n_components:
            raise ValueError(
                f"PCA needs at least n_components={n_components} fit rows, "
                f"got {len(fit_rows)} training targets present in the embedding index"
            )

    fit_block = matrix[fit_rows]
    mean = fit_block.mean(axis=0, keepdims=True)
    std = fit_block.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # constant columns -> leave at 0 after centering

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit((fit_block - mean) / std)
    reduced = pca.transform((matrix - mean) / std).astype(np.float32)
    return reduced, index
