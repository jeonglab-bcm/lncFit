"""Training-set resampling strategies for the ~4.5%-positive lncRNA-hit task.

Resampling is a property of the *training procedure*, not of the model, so it
lives here rather than inside any one classifier wrapper: any registered model
can be combined with any strategy below.

CRITICAL: only ever apply this to a training split. Resampling a validation or
test split changes the class balance you are measuring against and makes the
resulting AUROC/AUPRC meaningless (and incomparable to every other run). The
call sites in lncfit.pipeline / scripts.run_cellline_loco resample X_train
immediately before fit() and never touch X_val / X_test.
"""
from __future__ import annotations

import numpy as np

METHODS = ("none", "random_over", "random_under", "smote", "smote_tomek")


def resample(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "none",
    ratio: float | str = "auto",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a TRAINING set to rebalance its classes. Returns (X_res, y_res).

    method:
      none          -- passthrough, returns the inputs unchanged.
      random_over   -- duplicate randomly-drawn minority rows. Cheap; risks the
                       model memorizing exact duplicates.
      random_under  -- drop randomly-drawn majority rows. Balances without
                       synthesizing anything, but discards real data (at ~4.5%
                       positive, a 1:1 target throws away ~90% of the majority).
      smote         -- synthesize new minority rows by interpolating between a
                       minority sample and one of its minority k-NN. Adds
                       variety a plain duplicate cannot, but interpolating in a
                       768-dim frozen-embedding space is a real assumption, not
                       a free lunch.
      smote_tomek   -- SMOTE, then drop Tomek-link pairs (nearest neighbours of
                       opposite class) to clean the decision boundary.

    ratio: "auto" targets a fully balanced 1:1 result. A float is passed through
    as imbalanced-learn's sampling_strategy, i.e. the desired
    minority:majority ratio after resampling (0.5 => minority becomes half the
    majority's size), so values between the natural rate and 1.0 give partial
    rebalancing.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown resample method {method!r}. Expected one of {METHODS}.")
    if method == "none":
        return X, y

    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    strategy = "auto" if ratio == "auto" else float(ratio)
    samplers = {
        "random_over": lambda: RandomOverSampler(sampling_strategy=strategy, random_state=seed),
        "random_under": lambda: RandomUnderSampler(sampling_strategy=strategy, random_state=seed),
        "smote": lambda: SMOTE(sampling_strategy=strategy, random_state=seed),
        "smote_tomek": lambda: SMOTETomek(sampling_strategy=strategy, random_state=seed),
    }
    X_res, y_res = samplers[method]().fit_resample(X, y)
    return np.asarray(X_res), np.asarray(y_res)
