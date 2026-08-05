"""Sweep gene-level pan-essentiality priors under leave-one-cell-line-out validation.

Week-1 harness for the THP1 hold-out challenge (docs/PARTICIPATE.md). The test set is
one cell line and train/test cover the same 5,496 genes, so the whole task is gene-level
ranking: per gene you get fold_change / rra_pvalue / label in three training cell lines
and must order those same genes for an unseen fourth.

scripts/make_barebones_submission.py reaches AUPRC 0.2000 with a single aggregation of
that signal -- -mean(fold_change). This sweeps alternatives under the only honest local
validation available: hold out one TRAINING cell line, build features from the other two,
predict the held-out line's labels. That mirrors the real transfer task, unlike the
stratified CV in the starter config, which puts the same cell lines on both sides and so
flatters models that memorize cell-line-specific signal.

Every scorer here is a closed-form aggregation with no fitted parameters, so results are
seed-free and differences come only from the aggregation itself.

Two caveats before trusting any ordering this prints:
  * Each LOCO fold aggregates 2 cell lines; a real submission aggregates 3. Absolute
    numbers are therefore pessimistic relative to the board.
  * 3 folds with 157-401 positives each. docs/PARTICIPATE.md puts the single-run 95% CI
    near +/-0.02 AUPRC and reports CV vs test AUPRC coming out Spearman -1.0 on an
    earlier version of this task. Treat sub-0.02 gaps as noise, not findings.

Reads only the training file -- never THP1's labels.

Usage:
  uv run python scripts/sweep_gene_level_priors.py
  uv run python scripts/sweep_gene_level_priors.py --out results/gene_level_priors.csv
"""
import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

_TRAIN = "data/holdout_thp1/train_thp1_holdout.jsonl.gz"
_BAREBONES = "neg_mean_fc"  # the aggregation scripts/make_barebones_submission.py uses

# rra_pvalue is 0 for some genes; -log10(0) is inf, which would poison every mean.
_P_FLOOR = 1e-12


def _read_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_gene_table(path: str) -> dict[str, dict[str, dict]]:
    """Return {target: {cell_line: {fold_change, rra_pvalue, label}}}."""
    table: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in _read_jsonl_gz(path):
        table[row["target"]][row["cell_line"]] = {
            "fold_change": float(row["fold_change"]),
            "rra_pvalue": float(row["rra_pvalue"]),
            "label": int(row["label"]),
        }
    return dict(table)


def _fold_changes(obs: list[dict]) -> list[float]:
    return [o["fold_change"] for o in obs]


def _signed_logp(o: dict) -> float:
    """Significance magnitude, signed so depletion is positive and enrichment negative.

    label is defined as (rra_pvalue < 0.05 and fold_change < 0), so an unsigned p-value
    would rank strongly *enriched* genes alongside strongly depleted ones.
    """
    magnitude = -math.log10(max(o["rra_pvalue"], _P_FLOOR))
    return magnitude if o["fold_change"] < 0 else -magnitude


# Each scorer maps one gene's observations in the non-held-out cell lines to a score
# where higher = more likely essential in the unseen line.
SCORERS: dict[str, callable] = {
    _BAREBONES: lambda obs: -sum(_fold_changes(obs)) / len(obs),
    "neg_min_fc": lambda obs: -min(_fold_changes(obs)),
    "neg_max_fc": lambda obs: -max(_fold_changes(obs)),
    "neg_median_fc": lambda obs: -float(np.median(_fold_changes(obs))),
    "hit_count": lambda obs: float(sum(o["label"] for o in obs)),
    "mean_signed_logp": lambda obs: sum(_signed_logp(o) for o in obs) / len(obs),
    "max_signed_logp": lambda obs: max(_signed_logp(o) for o in obs),
    "mean_fc_x_logp": lambda obs: sum(
        -o["fold_change"] * -math.log10(max(o["rra_pvalue"], _P_FLOOR)) for o in obs
    ) / len(obs),
}


def _zscore(a: np.ndarray) -> np.ndarray:
    sd = float(a.std())
    return (a - a.mean()) / sd if sd > 0 else np.zeros_like(a)


def score_fold(table: dict, holdout: str, cell_lines: list[str]) -> tuple[dict, np.ndarray]:
    """Score every candidate for one LOCO fold. Returns ({name: array}, y_true)."""
    others = [c for c in cell_lines if c != holdout]
    genes = sorted(table)

    y = np.array([table[g][holdout]["label"] for g in genes], dtype=int)

    scores: dict[str, np.ndarray] = {}
    for name, fn in SCORERS.items():
        scores[name] = np.array(
            [fn([table[g][c] for c in others if c in table[g]]) for g in genes],
            dtype=float,
        )

    # Composites need all genes at once (z-scoring is population-relative), so they are
    # built from the base arrays rather than declared in SCORERS.
    scores["z_meanfc_plus_hits"] = _zscore(scores[_BAREBONES]) + _zscore(scores["hit_count"])
    scores["z_meanfc_plus_logp"] = _zscore(scores[_BAREBONES]) + _zscore(scores["mean_signed_logp"])

    return scores, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=_TRAIN)
    parser.add_argument("--out", default="results/gene_level_priors.csv",
                        help="Long-format per-fold results CSV.")
    args = parser.parse_args()

    table = load_gene_table(args.train)
    cell_lines = sorted({c for obs in table.values() for c in obs})
    print(f"Loaded {len(table):,} genes x {len(cell_lines)} training cell lines: "
          f"{', '.join(cell_lines)}\n")

    rows: list[dict] = []
    per_scorer: dict[str, list[float]] = defaultdict(list)
    per_scorer_auroc: dict[str, list[float]] = defaultdict(list)

    for holdout in cell_lines:
        scores, y = score_fold(table, holdout, cell_lines)
        others = ", ".join(c for c in cell_lines if c != holdout)
        print(f"--- hold out {holdout} ({y.sum()} pos / {len(y)} = {y.mean():.3%}), "
              f"features from {others} ---")

        fold_rows = []
        for name, s in scores.items():
            auroc = float(roc_auc_score(y, s))
            auprc = float(average_precision_score(y, s))
            per_scorer[name].append(auprc)
            per_scorer_auroc[name].append(auroc)
            rows.append({"holdout": holdout, "scorer": name, "n": len(y),
                         "n_pos": int(y.sum()), "auroc": round(auroc, 4),
                         "auprc": round(auprc, 4)})
            fold_rows.append((name, auroc, auprc))

        for name, auroc, auprc in sorted(fold_rows, key=lambda r: -r[2]):
            flag = "  <- barebones" if name == _BAREBONES else ""
            print(f"  {name:<22} AUROC {auroc:.4f}  AUPRC {auprc:.4f}{flag}")
        print()

    # Mean across folds. Each fold has a different base rate, so the mean is a rough
    # comparator, not an estimate of board AUPRC.
    baseline = float(np.mean(per_scorer[_BAREBONES]))
    print("=== mean across the 3 LOCO folds (sorted by AUPRC) ===")
    print(f"{'scorer':<22} {'AUROC':>7} {'AUPRC':>7} {'vs barebones':>13}")
    for name, auprcs in sorted(per_scorer.items(), key=lambda kv: -np.mean(kv[1])):
        mean_auprc = float(np.mean(auprcs))
        delta = mean_auprc - baseline
        note = "" if name == _BAREBONES else f"{delta:+.4f}"
        if name != _BAREBONES and abs(delta) < 0.02:
            note += " (noise)"
        print(f"{name:<22} {np.mean(per_scorer_auroc[name]):>7.4f} {mean_auprc:>7.4f} "
              f"{note:>13}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["holdout", "scorer", "n", "n_pos",
                                                "auroc", "auprc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-fold results -> {out_path}")
    print("Gaps under ~0.02 AUPRC are within noise -- see this script's docstring.")


if __name__ == "__main__":
    main()
