"""Chromosome LOCO-CV fold building, shared across training scripts."""
from __future__ import annotations

import gc
from collections import Counter

import numpy as np

from lncfit.constants import MIN_FOLD_RECORDS
from lncfit.features import build_features, fit_vocab


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
