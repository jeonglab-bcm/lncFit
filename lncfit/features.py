from __future__ import annotations

import itertools

import numpy as np

from lncfit.screen_data import ScreenRecord

_BASES = "ACGT"
_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_DAYS = [7, 14]


def all_kmers(k: int) -> list[str]:
    """Return all 4^k k-mers over ACGT in sorted alphabetical order."""
    return ["".join(p) for p in itertools.product(_BASES, repeat=k)]


def kmer_freq_vector(seq: str, k: int, vocab: list[str]) -> list[float]:
    """Normalized k-mer frequency vector for seq. Non-ACGT characters are skipped."""
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    counts = [0] * len(vocab)
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in _BASES for c in kmer):
            continue
        idx = vocab_index.get(kmer)
        if idx is not None:
            counts[idx] += 1
            total += 1
    if total == 0:
        return [0.0] * len(vocab)
    return [c / total for c in counts]


def _fill_kmer_row(row: np.ndarray, seq: str, k: int, vocab_index: dict[str, int]) -> None:
    """Fill a pre-allocated float32 row with normalised k-mer frequencies in-place."""
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in _BASES for c in kmer):
            continue
        idx = vocab_index.get(kmer)
        if idx is not None:
            row[idx] += 1
            total += 1
    if total > 0:
        row /= total


def build_features(
    records: list[ScreenRecord],
    k: int = 6,
    include_distance: bool = False,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X, target vector y, and column names from ScreenRecords.

    Returns (X, y, columns):
      X        ndarray of shape (n_records, n_features), dtype controlled by `dtype`
      y        float32 ndarray of shape (n_records,)
      columns  list[str] of feature column names

    Memory layout: preallocates a single array and fills rows in-place.
    For k=6 at 1M records: float32 ~16 GB, float16 ~8 GB.
    Pass dtype=np.float16 in the Optuna CV loop to halve the steady-state matrix size;
    convert individual fold slices to float32 before passing to XGBoost.
    Returns bare numpy arrays — no pandas/polars wrapper.
    """
    vocab = all_kmers(k)
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    n_kmer = len(vocab)

    day_cols = [f"day_{d}" for d in _DAYS]
    cell_cols = [f"cell_{c}" for c in _CELL_LINES]
    columns = vocab + day_cols + cell_cols
    if include_distance:
        columns.append("distance_to_closest_pc_gene")

    n = len(records)
    n_cols = len(columns)
    day_offset = n_kmer
    cell_offset = n_kmer + len(_DAYS)

    X = np.zeros((n, n_cols), dtype=dtype)
    y = np.empty(n, dtype=np.float32)

    for i, r in enumerate(records):
        _fill_kmer_row(X[i, :n_kmer], r.target_sequence, k, vocab_index)
        for j, d in enumerate(_DAYS):
            if r.day == d:
                X[i, day_offset + j] = 1.0
        for j, c in enumerate(_CELL_LINES):
            if r.cell_line == c:
                X[i, cell_offset + j] = 1.0
        if include_distance:
            dist = r.distance_to_closest_pc_gene if r.distance_to_closest_pc_gene is not None else -1
            X[i, n_cols - 1] = float(dist)
        y[i] = r.fold_change

    return X, y, columns
