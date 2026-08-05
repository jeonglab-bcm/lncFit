"""Weight training cell lines by their similarity to the cell line being predicted.

Celligner coordinates cannot help this challenge as *features*: every scored row is THP1, so
a cell-line column takes one value across all 5,496 and cannot reorder them (measured in
results/thp1_holdout_loco_sweep.csv -- dim 0/2/10/70 all flat). This uses them differently,
as sample WEIGHTS: train the model to care more about the training cell line that most
resembles the target. That changes what the model learns rather than what it reads at
scoring time, so the constant-column argument does not apply.

Legitimate under the challenge rules: Celligner is built from RNA-seq expression, not from
the CRISPR screen, so using THP1's coordinates is not reading THP1's labels. The starter
config says as much -- Celligner coordinates are "what let a model say anything
cell-line-specific about a line it never trained on."

Validation design. Weighting toward THP1 cannot be tested by LOCO, whose folds predict
HAP1/K562/MDA-MB-231. So what is validated is the METHOD -- "weight training cell lines by
similarity to whatever line you are predicting" -- with each fold weighting toward its own
held-out line. Evidence that this helps when the target is a training line is the honest
basis for applying it with THP1 as the target.

Schemes:
  uniform    every training row weighted 1.0 (the baseline used everywhere else)
  umap       softmax of negative distance in the 2-D Celligner UMAP
  pca70      same in the 70-D pre-UMAP PCA space
  lineage    same-lineage cell lines up-weighted, different-lineage down

Read data/external/README.md before trusting the coordinate schemes. HAP1 is flagged
UNRELIABLE there on three independent signals (lineage-purity outlier inside an otherwise
88.9%-pure lineage, no raw-expression explanation, and the least stable position across
reruns at 7.04 UMAP units vs 1.25-3.10). That matters twice over: the UMAP scheme's main
effect is down-weighting HAP1 to 0.58 on a coordinate the data says not to trust, and HAP1
is in fact Myeloid like THP1, so biology and coordinate disagree. Every LOCO fold also uses
HAP1's coordinates somewhere, so the validation itself is contaminated. The lineage scheme
exists to express the same intent without depending on that coordinate.

Usage:
  uv run python scripts/sweep_cellline_weighting.py
  uv run python scripts/sweep_cellline_weighting.py --seeds 8
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sweep_gene_level_priors import load_gene_table
from sweep_prescreen_features import _GUIDE, _KMER, _ONEHOT, _S1A, _TPM, build, load_guides, load_s1a
from sweep_tpm_features import _TRAIN, kmer_matrix, load_tpm

_UMAP_PATH = "data/external/celligner_cell_line_umap.csv"
_PCA_PATH = "data/external/celligner_cell_line_pca.csv"
_BLOCKS = [_KMER, _ONEHOT, _TPM, _S1A, _GUIDE]

# From data/external/README.md's lineage table. HAP1 is Myeloid despite its UMAP position.
_LINEAGE = {"HAP1": "Myeloid", "K562": "Myeloid", "THP1": "Myeloid", "MDA-MB-231": "Breast"}

# "posrate" is a control, not a proposal. The UMAP scheme's gain appears only in the two
# folds where it happens to down-weight K562, which carries an anomalous positive rate
# (7.3% vs 4.3% and 2.9%). Down-weighting it also moves the training positive rate toward
# the target's -- a different mechanism from similarity. If weighting purely by inverse
# positive rate reproduces the gain, the similarity story is not what is doing the work.
# That distinction decides whether the method transfers to THP1, where UMAP up-weights
# K562 (1.74) rather than down-weighting it.
SCHEMES = ["uniform", "umap", "pca70", "lineage", "posrate"]


def _coords(path: str, prefix: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(path).set_index("cell_line")
    cols = [c for c in df.columns if c.startswith(prefix)]
    return {str(i): df.loc[i, cols].to_numpy(dtype=float) for i in df.index}


def weights_for(scheme: str, target: str, sources: list[str],
                umap: dict, pca: dict, lineage_ratio: float,
                pos_rate: dict[str, float] | None = None) -> dict[str, float]:
    """Per-cell-line weights, normalised to average 1.0 so totals stay comparable."""
    if scheme == "uniform":
        return {c: 1.0 for c in sources}

    if scheme == "posrate":
        # Control: uses only the training cell lines' own positive rates, no coordinates
        # and nothing about the target.
        raw = np.array([1.0 / max(pos_rate[c], 1e-6) for c in sources], dtype=float)
    elif scheme == "lineage":
        raw = np.array([lineage_ratio if _LINEAGE.get(c) == _LINEAGE.get(target) else 1.0
                        for c in sources], dtype=float)
    else:
        coords = umap if scheme == "umap" else pca
        d = np.array([np.linalg.norm(coords[c] - coords[target]) for c in sources],
                     dtype=float)
        # Softmax of negative distance, scaled by the mean distance so the sharpness does
        # not depend on the units of the space (UMAP and 70-D PCA differ by ~10x).
        raw = np.exp(-d / (d.mean() if d.mean() > 0 else 1.0))

    raw = raw / raw.sum() * len(sources)
    return dict(zip(sources, raw))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--lineage-ratio", type=float, default=3.0,
                        help="How much more a same-lineage cell line counts.")
    parser.add_argument("--out", default="results/cellline_weighting_sweep.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    pos_rate = {c: float(labels[c].mean()) for c in train_cells}

    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}")
    umap = _coords(_UMAP_PATH, "UMAP_")
    pca = _coords(_PCA_PATH, "PC")
    total, mrna = load_tpm()
    s1a, s1a_ext, _, _ = load_s1a(genes)
    guide, _ = load_guides(genes)
    kmer_X = kmer_matrix(genes, args.k)

    # Show the weights that would be used for the real submission, where THP1 is the target
    # and all three training cell lines are sources.
    print("\nweights if the target were THP1 (the real submission):")
    for scheme in SCHEMES:
        w = weights_for(scheme, "THP1", train_cells, umap, pca, args.lineage_ratio, pos_rate)
        print(f"  {scheme:<9} " + "  ".join(f"{c}={w[c]:.2f}" for c in train_cells))

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        sources = [c for c in train_cells if c != holdout]
        y_eval = labels[holdout]
        X_eval = build(genes, holdout, _BLOCKS, kmer_X, total, mrna, s1a, s1a_ext, guide)
        X_train = np.vstack([build(genes, c, _BLOCKS, kmer_X, total, mrna, s1a, s1a_ext,
                                   guide) for c in sources])
        y_train = np.concatenate([labels[c] for c in sources])

        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for scheme in SCHEMES:
            w = weights_for(scheme, holdout, sources, umap, pca, args.lineage_ratio, pos_rate)
            sample_weight = np.concatenate([np.full(len(genes), w[c]) for c in sources])

            auprcs, aurocs = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=3,
                    subsample=0.8, colsample_bytree=0.3, tree_method="hist",
                    objective="binary:logistic", eval_metric="aucpr",
                    scale_pos_weight=1.0, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train, sample_weight=sample_weight)
                p = m.predict_proba(X_eval)[:, 1]
                auprcs.append(float(average_precision_score(y_eval, p)))
                aurocs.append(float(roc_auc_score(y_eval, p)))
                results[scheme].append(auprcs[-1])
                rows.append({"holdout": holdout, "scheme": scheme, "seed": seed,
                             "weights": ";".join(f"{c}={w[c]:.3f}" for c in sources),
                             "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
            wstr = " ".join(f"{c}={w[c]:.2f}" for c in sources)
            print(f"  {scheme:<9} AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}   [{wstr}]",
                  flush=True)

    base = float(np.mean(results["uniform"]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'scheme':<9} {'AUPRC':>7} {'sd':>7} {'vs uniform':>12}")
    for scheme, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if scheme == "uniform" else f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else "")
        print(f"{scheme:<9} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>12}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["holdout", "scheme", "seed", "weights",
                                           "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")
    print("Every fold's weights involve HAP1's coordinates, which data/external/README.md "
          "flags as UNRELIABLE -- read coordinate-scheme results with that in mind.")


if __name__ == "__main__":
    main()
