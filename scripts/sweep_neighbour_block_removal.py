"""Does dropping nb_depmap from the current best 42-column model cost anything?

The neighbour blocks were only ever measured by ADDING each to a base. That is a
different question from REMOVING one from the full model, which is what we would
actually be doing. This runs the paired leave-one-cell-line-out comparison:

  count only (42 cols)          the current best, all four neighbour blocks
  count only, no nb_depmap      same minus the neighbour's DepMap Cas9 essentiality

Same folds, same seeds, same hyperparameters, so the two are directly paired.

Usage:
  python scripts/sweep_neighbour_block_removal.py --seeds 20
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sweep_gene_level_priors import load_gene_table
from sweep_neighbour_features import (_NB_CLASS, _NB_DEPMAP, _NB_DIST, _NB_EXPR,
                                      _ONEHOT, _S1A_GENE, _TPM, load_blocks)
from sweep_prescreen_features import load_guides
from sweep_tpm_features import _ALL_CELLS, _TRAIN, load_tpm, tpm_block

_COUNT = "guide_count"

CONFIGS = [
    ("count only (current best)",
     [_ONEHOT, _TPM, _S1A_GENE, _NB_DIST, _NB_CLASS, _NB_DEPMAP, _NB_EXPR, _COUNT]),
    ("minus nb_depmap",
     [_ONEHOT, _TPM, _S1A_GENE, _NB_DIST, _NB_CLASS, _NB_EXPR, _COUNT]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", default="results/neighbour_block_removal.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}", flush=True)

    total, mrna = load_tpm()
    guide, guide_names = load_guides(genes)
    s1a_gene, nb_dist, nb_class, nb_depmap, nb_expr, _tissue = load_blocks(genes)
    guide_count = guide[:, [guide_names.index("guide_count")]]
    print(f"  nb_depmap is {nb_depmap.shape[1]} columns", flush=True)

    def make(cell_line: str, blocks: list[str]) -> np.ndarray:
        cols = []
        if _ONEHOT in blocks:
            oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
            oh[:, _ALL_CELLS.index(cell_line)] = 1.0
            cols.append(oh)
        if _TPM in blocks:
            cols.append(tpm_block(genes, cell_line, total, mrna)[0])
        if _S1A_GENE in blocks:
            cols.append(s1a_gene)
        if _NB_DIST in blocks:
            cols.append(nb_dist)
        if _NB_CLASS in blocks:
            cols.append(nb_class)
        if _NB_DEPMAP in blocks:
            cols.append(nb_depmap)
        if _NB_EXPR in blocks:
            cols.append(nb_expr[cell_line])
        if _COUNT in blocks:
            cols.append(guide_count)
        return np.hstack(cols).astype(np.float32)

    rows, results = [], defaultdict(list)
    # paired[(fold, seed)] -> {config: auprc}, for a per-pair win/loss count
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)

    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---",
              flush=True)
        for name, blocks in CONFIGS:
            X_eval = make(holdout, blocks)
            X_train = np.vstack([make(c, blocks) for c in train_cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])

            auprcs, aurocs = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                    colsample_bytree=0.3, tree_method="hist", objective="binary:logistic",
                    eval_metric="aucpr", scale_pos_weight=1.0, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_eval)[:, 1]
                auprcs.append(float(average_precision_score(y_eval, p)))
                aurocs.append(float(roc_auc_score(y_eval, p)))
                results[name].append(auprcs[-1])
                paired[(holdout, seed)][name] = auprcs[-1]
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4),
                             "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<26} ({X_eval.shape[1]:>3} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    a, b = CONFIGS[0][0], CONFIGS[1][0]
    ma, mb = float(np.mean(results[a])), float(np.mean(results[b]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===", flush=True)
    print(f"{a:<26} {ma:.4f}")
    print(f"{b:<26} {mb:.4f}   delta {mb - ma:+.4f}")

    wins = sum(1 for v in paired.values() if v[b] > v[a])
    ties = sum(1 for v in paired.values() if v[b] == v[a])
    print(f"\ndropping nb_depmap wins {wins}/{len(paired)} paired fold x seed runs "
          f"({ties} ties)")
    print("per-fold means:")
    for holdout in train_cells:
        fa = np.mean([v[a] for (h, _), v in paired.items() if h == holdout])
        fb = np.mean([v[b] for (h, _), v in paired.items() if h == holdout])
        print(f"  {holdout:<14} {fa:.4f} -> {fb:.4f}  ({fb - fa:+.4f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["holdout", "config", "seed", "n_features",
                                           "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")


if __name__ == "__main__":
    main()
