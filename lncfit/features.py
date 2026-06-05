from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from lncfit.screen_data import ScreenRecord

_BASES = "ACGT"
_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_DAYS = [7, 14]


def all_kmers(k: int) -> list[str]:
    """Return all 4^k k-mers over ACGT in sorted alphabetical order."""
    return ["".join(p) for p in itertools.product(_BASES, repeat=k)]


def kmer_freq_vector(seq: str, k: int, vocab: list[str]) -> list[float]:
    """Normalized k-mer frequency vector for seq. Non-ACGT characters are skipped.

    Sliding window of size k; each window that contains a non-ACGT character is dropped.
    Result sums to 1.0 (or is all zeros if no valid k-mers exist).
    """
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
    """Fill a pre-allocated float32 row with normalised k-mer frequencies in-place.

    Uses a pre-built vocab_index dict so it is not rebuilt on every call.
    """
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
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y from a list of ScreenRecords.

    X columns: k-mer frequency features, day one-hot, cell-line one-hot,
               and optionally distance_to_closest_pc_gene.
    y: fold_change (one row per record — both replicates kept as independent examples).

    Memory layout: preallocates a single float32 numpy array instead of building a
    Python list-of-lists. For k=6 at 1M records this reduces peak RAM from ~57 GB
    to ~16 GB by eliminating the intermediate Python object overhead.
    """
    vocab = all_kmers(k)
    # Build vocab_index once here; _fill_kmer_row reuses it across all records.
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

    X = np.zeros((n, n_cols), dtype=np.float32)
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

    return pd.DataFrame(X, columns=columns), pd.Series(y, name="fold_change")
