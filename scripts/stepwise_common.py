"""Shared plumbing for the scripts/stepNN_*.py analyses (see docs/findings.html).

Every step scores a handful of candidates under lncfit.honest_eval (cell line AND genes held
out), over several gene-split seeds, and prints one AUPRC row per candidate per cell line. A
candidate is either a fixed score per held-out line (nothing fitted) or a feature dict plus a
model factory.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lncfit.honest_eval import predict_cell_and_gene_holdout, score  # noqa: E402
from sweep_gene_level_priors import load_gene_table  # noqa: E402
from sweep_tpm_features import _TRAIN, load_tpm, tpm_block  # noqa: E402


def load_labels() -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    """(genes, training cell lines, {cell: labels}) from the THP1-free training file."""
    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    cells = sorted({c for obs in table.values() for c in obs})
    y = {c: np.array([table[g][c]["label"] for g in genes], dtype=int) for c in cells}
    return genes, cells, y


def own_expression(genes: list[str], cells: list[str]) -> dict[str, np.ndarray]:
    """{cell: (n_genes, 2)} own-line log1p TPM, columns [total RNA-seq, mRNA-seq]."""
    total, mrna = load_tpm()
    return {c: tpm_block(genes, c, total, mrna)[0][:, :2] for c in cells}


def lr() -> object:
    """The untuned linear model every step adds features through.

    Smooth scores keep the fine ordering at the top of the ranking that AUPRC rewards; tree
    ensembles collapse it (step 2).
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000))


Candidate = tuple[str, "dict[str, np.ndarray] | None", "Callable[[], object] | None",
                  "Callable[[str], np.ndarray] | None"]


def fitted(name: str, X: dict[str, np.ndarray], make_model: Callable[[], object]) -> Candidate:
    return (name, X, make_model, None)


def fixed(name: str, score_for: Callable[[str], np.ndarray]) -> Candidate:
    return (name, None, None, score_for)


def run(candidates: list[Candidate], y: dict[str, np.ndarray], cells: list[str],
        seeds: int, out: str) -> dict[str, dict[str, float]]:
    """Score every candidate; print a per-line AUPRC table; write per-seed rows to `out`.

    Returns {candidate: {cell: mean AUPRC}}.
    """
    rows = []
    for holdout in cells:
        for seed in range(seeds):
            for name, X, make_model, score_for in candidates:
                if score_for is not None:
                    if seed:  # nothing fitted, so every seed would repeat seed 0
                        continue
                    p = score_for(holdout)
                else:
                    p = predict_cell_and_gene_holdout(X, y, holdout, make_model, seed=seed)
                rows.append({"holdout": holdout, "candidate": name, "seed": seed,
                             **{k: round(v, 4) for k, v in score(y[holdout], p).items()}})
            print(f"  {holdout} seed {seed} done", flush=True)

    means: dict[str, dict[str, float]] = {}
    print(f"\nAUPRC, mean over gene-split seeds ({seeds} for fitted candidates)")
    print(f"{'candidate':<34}" + "".join(f"{c:>12}" for c in cells) + f"{'mean':>9}")
    for name, *_ in candidates:
        means[name] = {c: float(np.mean([r["auprc"] for r in rows
                                         if r["candidate"] == name and r["holdout"] == c]))
                       for c in cells}
        print(f"{name:<34}" + "".join(f"{means[name][c]:>12.4f}" for c in cells)
              + f"{np.mean(list(means[name].values())):>9.4f}")

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-fold/per-seed -> {path}")
    return means
