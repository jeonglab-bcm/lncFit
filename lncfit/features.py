from __future__ import annotations

import itertools

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


def build_features(
    records: list[ScreenRecord],
    k: int = 6,
    include_distance: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y from a list of ScreenRecords.

    X columns: k-mer frequency features, day one-hot, cell-line one-hot,
               and optionally distance_to_closest_pc_gene.
    y: fold_change (one row per record — both replicates kept as independent examples).
    """
    vocab = all_kmers(k)
    day_cols = [f"day_{d}" for d in _DAYS]
    cell_cols = [f"cell_{c}" for c in _CELL_LINES]

    rows = []
    targets = []
    for r in records:
        kmer_feats = kmer_freq_vector(r.target_sequence, k, vocab)
        day_feats = [1 if r.day == d else 0 for d in _DAYS]
        cell_feats = [1 if r.cell_line == c else 0 for c in _CELL_LINES]
        row = kmer_feats + day_feats + cell_feats
        if include_distance:
            dist = r.distance_to_closest_pc_gene if r.distance_to_closest_pc_gene is not None else -1
            row.append(float(dist))
        rows.append(row)
        targets.append(r.fold_change)

    columns = vocab + day_cols + cell_cols
    if include_distance:
        columns.append("distance_to_closest_pc_gene")

    X = pd.DataFrame(rows, columns=columns)
    y = pd.Series(targets, name="fold_change")
    return X, y
