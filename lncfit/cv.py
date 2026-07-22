"""Chromosome LOCO-CV fold building, shared across training scripts."""
from __future__ import annotations

import gc
from collections import Counter

import numpy as np

from lncfit.constants import MIN_FOLD_RECORDS
from lncfit.features import build_features, build_lncrna_features, fit_vocab


def build_folds(
    train_records: list,
    k: int,
    include_distance: bool = False,
    body_sequences: dict | None = None,
    signed_overlap: bool = False,
    body_embeddings: tuple | None = None,
    guide_embeddings: tuple | None = None,
    verbose: bool = True,
) -> tuple[list[str], dict[str, tuple], list[str]]:
    """Build per-fold feature matrices for chromosome LOCO-CV.

    Returns (cv_chroms, fold_data, feature_cols).
    fold_data maps val_chrom -> (X_tr, y_tr, X_val, y_val, X_es, y_es).
    """
    chrom_arr = np.array([r.chrom for r in train_records])
    chrom_counts = Counter(chrom_arr)
    cv_chroms = sorted(
        [str(c) for c, n in chrom_counts.items() if c and n >= MIN_FOLD_RECORDS],
        key=lambda x: (len(x), x),
    )

    if verbose:
        print(f"Fitting per-fold vocabularies and building feature matrices ...")

    fold_data: dict[str, tuple] = {}
    feature_cols: list[str] = []
    for i, val_chrom in enumerate(cv_chroms):
        es_chrom = cv_chroms[(i + 1) % len(cv_chroms)]
        val_mask = chrom_arr == val_chrom
        es_mask = chrom_arr == es_chrom
        train_mask = ~val_mask & ~es_mask

        train_recs_fold = [r for r, m in zip(train_records, train_mask) if m]
        val_recs_fold   = [r for r, m in zip(train_records, val_mask)   if m]
        es_recs_fold    = [r for r, m in zip(train_records, es_mask)    if m]

        guide_seqs = [r.target_sequence for r in train_recs_fold]
        if body_sequences is not None:
            seen_targets = {r.target for r in train_recs_fold}
            body_seqs_for_vocab = [
                seq for t in seen_targets for seq in body_sequences.get(t, ())
            ]
        else:
            body_seqs_for_vocab = []
        fold_vocab = fit_vocab(guide_seqs + body_seqs_for_vocab, k)

        X_tr, y_tr, cols = build_features(
            train_recs_fold, k=k, include_distance=include_distance,
            sparse=True, vocab=fold_vocab, body_sequences=body_sequences,
            signed_overlap=signed_overlap, body_embeddings=body_embeddings,
            guide_embeddings=guide_embeddings,
        )
        X_val, y_val, _ = build_features(
            val_recs_fold, k=k, include_distance=include_distance,
            sparse=True, vocab=fold_vocab, body_sequences=body_sequences,
            signed_overlap=signed_overlap, body_embeddings=body_embeddings,
            guide_embeddings=guide_embeddings,
        )
        X_es, y_es, _ = build_features(
            es_recs_fold, k=k, include_distance=include_distance,
            sparse=True, vocab=fold_vocab, body_sequences=body_sequences,
            signed_overlap=signed_overlap, body_embeddings=body_embeddings,
            guide_embeddings=guide_embeddings,
        )
        fold_data[val_chrom] = (X_tr, y_tr, X_val, y_val, X_es, y_es)
        if not feature_cols:
            feature_cols = cols
        if verbose:
            print(f"  fold chr{val_chrom}: {len(fold_vocab)}/{4**k} k-mers  "
                  f"train={X_tr.shape[0]:,}  val={X_val.shape[0]:,}  es={X_es.shape[0]:,}")
        gc.collect()

    return cv_chroms, fold_data, feature_cols


def build_lncrna_folds(
    train_records: list,
    transcript_sequences: dict[str, str],
    k: int,
    include_distance: bool = False,
    verbose: bool = True,
) -> tuple[list[str], dict[str, tuple], list[str]]:
    """Build per-fold feature matrices for chromosome LOCO-CV over LncRnaRecords.

    Same fold structure as build_folds (rotating early-stop chromosome, everything
    else in train), but vocab is fit on each fold's targets' own transcript_sequences
    (not guide sequences — see issue #65) and features come from build_lncrna_features
    (binary label y, no day dimension).

    Returns (cv_chroms, fold_data, feature_cols).
    fold_data maps val_chrom -> (X_tr, y_tr, X_val, y_val, X_es, y_es).
    """
    chrom_arr = np.array([r.chrom for r in train_records])
    chrom_counts = Counter(chrom_arr)
    cv_chroms = sorted(
        [str(c) for c, n in chrom_counts.items() if c and n >= MIN_FOLD_RECORDS],
        key=lambda x: (len(x), x),
    )

    if verbose:
        print(f"Fitting per-fold vocabularies and building lncRNA feature matrices ...")

    fold_data: dict[str, tuple] = {}
    feature_cols: list[str] = []
    for i, val_chrom in enumerate(cv_chroms):
        es_chrom = cv_chroms[(i + 1) % len(cv_chroms)]
        val_mask = chrom_arr == val_chrom
        es_mask = chrom_arr == es_chrom
        train_mask = ~val_mask & ~es_mask

        train_recs_fold = [r for r, m in zip(train_records, train_mask) if m]
        val_recs_fold   = [r for r, m in zip(train_records, val_mask)   if m]
        es_recs_fold    = [r for r, m in zip(train_records, es_mask)    if m]

        fold_targets = {r.target for r in train_recs_fold}
        fold_seqs = [transcript_sequences[t] for t in fold_targets if t in transcript_sequences]
        fold_vocab = fit_vocab(fold_seqs, k)

        X_tr, y_tr, cols = build_lncrna_features(
            train_recs_fold, transcript_sequences, k=k, include_distance=include_distance,
            vocab=fold_vocab, sparse=True,
        )
        X_val, y_val, _ = build_lncrna_features(
            val_recs_fold, transcript_sequences, k=k, include_distance=include_distance,
            vocab=fold_vocab, sparse=True,
        )
        X_es, y_es, _ = build_lncrna_features(
            es_recs_fold, transcript_sequences, k=k, include_distance=include_distance,
            vocab=fold_vocab, sparse=True,
        )
        fold_data[val_chrom] = (X_tr, y_tr, X_val, y_val, X_es, y_es)
        if not feature_cols:
            feature_cols = cols
        if verbose:
            n_pos_tr = int(y_tr.sum())
            n_pos_val = int(y_val.sum())
            print(f"  fold chr{val_chrom}: {len(fold_vocab)}/{4**k} k-mers  "
                  f"train={X_tr.shape[0]:,} (pos={n_pos_tr})  "
                  f"val={X_val.shape[0]:,} (pos={n_pos_val})  es={X_es.shape[0]:,}")
        gc.collect()

    return cv_chroms, fold_data, feature_cols


def build_lncrna_stratified_folds(
    train_records: list,
    transcript_sequences: dict[str, str],
    k: int,
    n_splits: int = 5,
    include_distance: bool = False,
    seed: int = 42,
    variance_threshold: float = 0.0,
    verbose: bool = True,
) -> tuple[list[int], dict[int, tuple], list[str]]:
    """Build per-fold feature matrices for row-level stratified K-fold CV.

    Unlike build_lncrna_folds (chromosome LOCO-CV), folds are plain
    StratifiedKFold splits over the binary label, ignoring chromosome
    entirely. Every cell-line row for a given lncRNA shares one k-mer
    frequency vector (only the cell-line one-hot differs), so the same
    lncRNA's sequence can appear in both a fold's train and validation split
    via its other cell-line rows. That's a real leakage risk relative to
    build_lncrna_folds's chromosome grouping -- kept here on request, for a
    direct comparison against the chromosome-grouped CV numbers, not because
    it's leak-free.

    variance_threshold: if > 0, drop feature columns whose variance on the
    fold's *training* split is at or below this value (fit per fold, applied
    to that fold's train/val/es -- never fit on val/es, to avoid leaking the
    selection itself). K-mer frequency columns are heavily right-skewed in
    variance (most barely vary row to row); this is mainly useful for large
    k (e.g. k=6's 4096 columns) where most columns carry near-zero signal
    and dropping them cuts compute without touching the informative ones.
    The 5 cell-line one-hot columns have too much variance to ever be
    dropped by any threshold worth using here, so this only prunes k-mers.

    Returns (fold_ids, fold_data, feature_cols).
    fold_data maps fold_id -> (X_tr, y_tr, X_val, y_val, X_es, y_es).
    """
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import StratifiedKFold

    y_all = np.array([r.label for r in train_records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_mask_by_fold = [np.zeros(len(train_records), dtype=bool) for _ in range(n_splits)]
    for i, (_, val_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        val_mask_by_fold[i][val_idx] = True

    if verbose:
        print(f"Fitting per-fold vocabularies and building lncRNA feature matrices "
              f"(stratified {n_splits}-fold, chromosome-agnostic) ...")

    fold_ids = list(range(n_splits))
    fold_data: dict[int, tuple] = {}
    feature_cols: list[str] = []
    for i in fold_ids:
        es_i = (i + 1) % n_splits
        val_mask = val_mask_by_fold[i]
        es_mask = val_mask_by_fold[es_i]
        train_mask = ~val_mask & ~es_mask

        train_recs_fold = [r for r, m in zip(train_records, train_mask) if m]
        val_recs_fold   = [r for r, m in zip(train_records, val_mask)   if m]
        es_recs_fold    = [r for r, m in zip(train_records, es_mask)    if m]

        fold_targets = {r.target for r in train_recs_fold}
        fold_seqs = [transcript_sequences[t] for t in fold_targets if t in transcript_sequences]
        fold_vocab = fit_vocab(fold_seqs, k)

        X_tr, y_tr, cols = build_lncrna_features(
            train_recs_fold, transcript_sequences, k=k, include_distance=include_distance,
            vocab=fold_vocab, sparse=True,
        )
        X_val, y_val, _ = build_lncrna_features(
            val_recs_fold, transcript_sequences, k=k, include_distance=include_distance,
            vocab=fold_vocab, sparse=True,
        )
        X_es, y_es, _ = build_lncrna_features(
            es_recs_fold, transcript_sequences, k=k, include_distance=include_distance,
            vocab=fold_vocab, sparse=True,
        )

        n_cols_before = X_tr.shape[1]
        if variance_threshold > 0:
            selector = VarianceThreshold(threshold=variance_threshold).fit(X_tr)
            X_tr = selector.transform(X_tr)
            X_val = selector.transform(X_val)
            X_es = selector.transform(X_es)
            cols = [c for c, keep in zip(cols, selector.get_support()) if keep]

        fold_data[i] = (X_tr, y_tr, X_val, y_val, X_es, y_es)
        if not feature_cols:
            feature_cols = cols
        if verbose:
            n_pos_tr = int(y_tr.sum())
            n_pos_val = int(y_val.sum())
            vt_note = f"  vt: {X_tr.shape[1]}/{n_cols_before} cols kept" if variance_threshold > 0 else ""
            print(f"  fold {i}: {len(fold_vocab)}/{4**k} k-mers  "
                  f"train={X_tr.shape[0]:,} (pos={n_pos_tr})  "
                  f"val={X_val.shape[0]:,} (pos={n_pos_val})  es={X_es.shape[0]:,}{vt_note}")
        gc.collect()

    return fold_ids, fold_data, feature_cols


def make_cv_splits(
    records: list,
    strategy: str,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Feature-type-agnostic CV split masks: (train_mask, val_mask, fold_label) triples.

    Unlike build_lncrna_folds/build_lncrna_stratified_folds above (which refit a
    k-mer vocabulary per fold and build feature matrices themselves), this only
    returns boolean masks over `records` -- it doesn't know or care whether the
    caller's feature matrix came from k-mers, DNABERT-2 embeddings, or anything
    else. Used by lncfit.pipeline.LncRnaPipeline so the same tuning/CV code path
    works regardless of --features choice.

    strategy="chrom": chromosome LOCO-CV, one fold per chromosome with
    >= MIN_FOLD_RECORDS records (same grouping as build_lncrna_folds).
    strategy="stratified": plain StratifiedKFold over the binary label, ignoring
    chromosome. As documented in build_lncrna_stratified_folds, this is NOT
    leak-free for k-mer features (every cell-line row of a given lncRNA shares one
    k-mer vector), and the vocabulary here is fit once on all records rather than
    per fold -- a simplification accepted for a single generic code path across
    feature types, not a claim of leak-free CV.
    """
    if strategy == "chrom":
        chrom_arr = np.array([r.chrom for r in records])
        counts = Counter(chrom_arr)
        chroms = sorted(
            [str(c) for c, n in counts.items() if c and n >= MIN_FOLD_RECORDS],
            key=lambda x: (len(x), x),
        )
        return [(chrom_arr != c, chrom_arr == c, f"chr{c}") for c in chroms]
    elif strategy == "stratified":
        from sklearn.model_selection import StratifiedKFold

        y = np.array([r.label for r in records])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = []
        for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            train_mask = np.zeros(len(y), dtype=bool)
            val_mask = np.zeros(len(y), dtype=bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            splits.append((train_mask, val_mask, f"fold{i}"))
        return splits
    else:
        raise ValueError(f"Unknown CV strategy {strategy!r}. Expected 'chrom' or 'stratified'.")
