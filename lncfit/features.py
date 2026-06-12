from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import csr_matrix

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
    sparse: bool = False,
) -> tuple[np.ndarray | csr_matrix, np.ndarray, list[str]]:
    """Build feature matrix X, target vector y, and column names from ScreenRecords.

    Returns (X, y, columns):
      X        ndarray or csr_matrix of shape (n_records, n_features)
      y        float32 ndarray of shape (n_records,)
      columns  list[str] of feature column names

    Pass sparse=True for a CSR sparse matrix (~0.15 GB for k=6 at 1M records vs ~8 GB
    dense float16). XGBoost accepts csr_matrix directly; no float16/float32 cast needed.
    The dense path still accepts a dtype parameter for backwards compatibility.
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

    y = np.empty(n, dtype=np.float32)

    if sparse:
        # Build via COO arrays — only ~20 non-zeros per row for 23-nt sequences at k=6.
        row_idx: list[int] = []
        col_idx: list[int] = []
        vals: list[float] = []

        for i, r in enumerate(records):
            seq = r.target_sequence
            kmer_counts: dict[int, int] = {}
            total = 0
            for j in range(len(seq) - k + 1):
                kmer = seq[j : j + k]
                if any(c not in _BASES for c in kmer):
                    continue
                idx = vocab_index.get(kmer)
                if idx is not None:
                    kmer_counts[idx] = kmer_counts.get(idx, 0) + 1
                    total += 1
            if total > 0:
                for col, count in kmer_counts.items():
                    row_idx.append(i)
                    col_idx.append(col)
                    vals.append(count / total)
            for j, d in enumerate(_DAYS):
                if r.day == d:
                    row_idx.append(i)
                    col_idx.append(day_offset + j)
                    vals.append(1.0)
                    break
            for j, c in enumerate(_CELL_LINES):
                if r.cell_line == c:
                    row_idx.append(i)
                    col_idx.append(cell_offset + j)
                    vals.append(1.0)
                    break
            if include_distance:
                dist = r.distance_to_closest_pc_gene if r.distance_to_closest_pc_gene is not None else -1
                row_idx.append(i)
                col_idx.append(n_cols - 1)
                vals.append(float(dist))
            y[i] = r.fold_change

        X = csr_matrix(
            (np.array(vals, dtype=np.float32), (np.array(row_idx), np.array(col_idx))),
            shape=(n, n_cols),
        )
        return X, y, columns

    X_dense = np.zeros((n, n_cols), dtype=dtype)
    for i, r in enumerate(records):
        _fill_kmer_row(X_dense[i, :n_kmer], r.target_sequence, k, vocab_index)
        for j, d in enumerate(_DAYS):
            if r.day == d:
                X_dense[i, day_offset + j] = 1.0
        for j, c in enumerate(_CELL_LINES):
            if r.cell_line == c:
                X_dense[i, cell_offset + j] = 1.0
        if include_distance:
            dist = r.distance_to_closest_pc_gene if r.distance_to_closest_pc_gene is not None else -1
            X_dense[i, n_cols - 1] = float(dist)
        y[i] = r.fold_change

    return X_dense, y, columns
