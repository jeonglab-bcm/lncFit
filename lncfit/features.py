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


def fit_vocab(seqs: list[str], k: int) -> list[str]:
    """Return sorted list of k-mers observed in at least one sequence in seqs."""
    observed: set[str] = set()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            if not any(c not in _BASES for c in kmer):
                observed.add(kmer)
    return sorted(observed)


def _count_kmers(seq: str, k: int, vocab_index: dict[str, int]) -> tuple[dict[int, int], int]:
    """Return (col_index -> raw_count, total_valid_windows) for seq."""
    counts: dict[int, int] = {}
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(c not in _BASES for c in kmer):
            continue
        idx = vocab_index.get(kmer)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
            total += 1
    return counts, total


def kmer_freq_vector(seq: str, k: int, vocab: list[str]) -> list[float]:
    """Normalized k-mer frequency vector for seq. Non-ACGT characters are skipped."""
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    counts, total = _count_kmers(seq, k, vocab_index)
    if total == 0:
        return [0.0] * len(vocab)
    result = [0.0] * len(vocab)
    for idx, count in counts.items():
        result[idx] = count / total
    return result


def _fill_kmer_row(row: np.ndarray, seq: str, k: int, vocab_index: dict[str, int]) -> None:
    """Fill a pre-allocated float32 row with normalised k-mer frequencies in-place."""
    counts, total = _count_kmers(seq, k, vocab_index)
    if total > 0:
        for idx, count in counts.items():
            row[idx] = count / total


def _precompute_body_kmer_cache(
    records: list[ScreenRecord],
    body_sequences: dict[str, tuple[str, str]],
    k: int,
    vocab_index: dict[str, int],
) -> dict[str, tuple[list[tuple[int, float]], list[tuple[int, float]]]]:
    """Compute body k-mer frequency vectors once per unique gene, not once per record.

    Returns {gene_id: (first_window_entries, last_window_entries)} where each entry
    list is [(col_index, frequency), ...] ready to scatter into a feature matrix.
    """
    cache: dict[str, tuple[list, list]] = {}
    for r in records:
        gene_id = r.target
        if gene_id in cache or gene_id not in body_sequences:
            continue
        first_seq, last_seq = body_sequences[gene_id]
        first_counts, first_total = _count_kmers(first_seq, k, vocab_index)
        last_counts, last_total = _count_kmers(last_seq, k, vocab_index)
        first_entries = [(col, cnt / first_total) for col, cnt in first_counts.items()] if first_total > 0 else []
        last_entries = [(col, cnt / last_total) for col, cnt in last_counts.items()] if last_total > 0 else []
        cache[gene_id] = (first_entries, last_entries)
    return cache


def build_features(
    records: list[ScreenRecord],
    k: int = 6,
    include_distance: bool = False,
    dtype: np.dtype = np.float32,
    sparse: bool = False,
    vocab: list[str] | None = None,
    body_sequences: dict[str, tuple[str, str]] | None = None,
) -> tuple[np.ndarray | csr_matrix, np.ndarray, list[str]]:
    """Build feature matrix X, target vector y, and column names from ScreenRecords.

    Returns (X, y, columns):
      X        ndarray or csr_matrix of shape (n_records, n_features)
      y        float32 ndarray of shape (n_records,)
      columns  list[str] of feature column names

    Pass sparse=True for a CSR sparse matrix (~0.15 GB for k=6 at 1M records vs ~8 GB
    dense float16). XGBoost accepts csr_matrix directly; no float16/float32 cast needed.
    The dense path still accepts a dtype parameter for backwards compatibility.

    Pass body_sequences={gene_id: (first_1000bp, last_1000bp)} to supplement guide k-mers
    with k-mers from the lncRNA body windows (from lncfit.sequence.load_body_sequences).
    Records whose gene_id is absent from body_sequences get zero body k-mer columns.
    """
    if vocab is None:
        vocab = all_kmers(k)
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    n_kmer = len(vocab)

    use_body = body_sequences is not None
    body_first_cols = [f"body_first_{kmer}" for kmer in vocab] if use_body else []
    body_last_cols = [f"body_last_{kmer}" for kmer in vocab] if use_body else []
    day_cols = [f"day_{d}" for d in _DAYS]
    cell_cols = [f"cell_{c}" for c in _CELL_LINES]
    columns = vocab + body_first_cols + body_last_cols + day_cols + cell_cols
    if include_distance:
        columns.append("distance_to_closest_pc_gene")

    n = len(records)
    n_cols = len(columns)
    body_first_offset = n_kmer
    body_last_offset = n_kmer + (n_kmer if use_body else 0)
    day_offset = n_kmer + (2 * n_kmer if use_body else 0)
    cell_offset = day_offset + len(_DAYS)

    y = np.empty(n, dtype=np.float32)

    # Pre-compute body k-mer vectors once per gene — avoids recomputing identical
    # sequences for every record that maps to the same gene target.
    body_cache: dict | None = None
    if use_body:
        body_cache = _precompute_body_kmer_cache(records, body_sequences, k, vocab_index)

    if sparse:
        # Build via COO arrays — only ~20 non-zeros per row for 23-nt sequences at k=6.
        row_idx: list[int] = []
        col_idx: list[int] = []
        vals: list[float] = []

        for i, r in enumerate(records):
            seq = r.target_sequence
            kmer_counts, total = _count_kmers(seq, k, vocab_index)
            if total > 0:
                for col, count in kmer_counts.items():
                    row_idx.append(i)
                    col_idx.append(col)
                    vals.append(count / total)
            if body_cache is not None:
                cached = body_cache.get(r.target)
                if cached is not None:
                    first_entries, last_entries = cached
                    for col, val in first_entries:
                        row_idx.append(i)
                        col_idx.append(body_first_offset + col)
                        vals.append(val)
                    for col, val in last_entries:
                        row_idx.append(i)
                        col_idx.append(body_last_offset + col)
                        vals.append(val)
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
        if body_cache is not None:
            cached = body_cache.get(r.target)
            if cached is not None:
                first_entries, last_entries = cached
                for col, val in first_entries:
                    X_dense[i, body_first_offset + col] = val
                for col, val in last_entries:
                    X_dense[i, body_last_offset + col] = val
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
