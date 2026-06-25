"""Preprocessing utilities for lncRNA screen data."""
from __future__ import annotations

import numpy as np


def symmetric_quantile_clip(
    y: np.ndarray,
    quantile: float = 0.95,
    minimum_limit: float = 1.0,
    clip_limit: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Clip y symmetrically at the given quantile of absolute values.

    Args:
        y: 1-D float array of log2FC values.
        quantile: Quantile of |y| to use as clip boundary. 1.0 = no clipping.
        minimum_limit: Floor on the clip limit (prevents clipping near-zero values).
        clip_limit: If provided, skip computing from quantile and use this directly.
            Use when applying the same limit derived from training data to val/test sets.

    Returns:
        (clipped_y, clip_limit, pct_clipped)
        clipped_y: copy of y with values outside [-limit, limit] set to ±limit.
        clip_limit: the actual limit used.
        pct_clipped: fraction of entries that were clipped (0.0–1.0).
    """
    if clip_limit is None:
        if quantile >= 1.0:
            clip_limit = float(np.max(np.abs(y))) if len(y) > 0 else minimum_limit
        else:
            clip_limit = float(np.nanquantile(np.abs(y), quantile))
        clip_limit = max(minimum_limit, clip_limit)

    clipped = np.clip(y, -clip_limit, clip_limit)
    pct_clipped = float(np.mean(np.abs(y) > clip_limit))
    return clipped.astype(y.dtype), float(clip_limit), pct_clipped
