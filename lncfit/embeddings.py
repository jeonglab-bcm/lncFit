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
