"""How much of the guide-feature gain is just guide COUNT, an artifact of the label?

The paper's methods define essentiality from "up to five most depleted gRNAs per target gene
and robust rank aggregation". That makes the number of guides a direct lever on the label: a
gene with 9 guides gets 9 draws to supply 5 extreme ones, a gene with 4 cannot fill the set
at all. Measured on the training cell lines, the effect is enormous and perfectly monotonic:

  4 guides  0.000%   (0 essential in 1,110 gene x cell-line observations)
  5 guides  1.010%
  6 guides  1.781%
  7 guides  4.245%
  8 guides  6.725%
  9 guides 10.467%

Fisher exact p = 4e-25 comparing >=5 against <5 guides.

Guide design was this project's second-largest gain (+0.0124, 9/9 folds), and it was
interpreted as capturing knockdown efficacy -- whether poorly designed guides failed to knock
the target down. This splits the block to find out how much of it was ever about guide
quality at all:

  guide_count   the count alone (1 column)
  guide_seq     the eight sequence-derived summaries (GC mean/std/min/max, homopolymer
                mean/max, 3-mer complexity, self-complementarity) with the count removed
  guide9        both, as used until now

Note the count is legitimate to use: library composition is fixed before the screen runs and
is known for THP1 as well, so it is not outcome data and not leakage. But if it carries the
block, then the honest description of the model changes -- it is exploiting a property of how
the label was computed rather than measuring guide efficacy, and any write-up should say so.

Model settings are the tuned ones; features are the final configuration minus the guide block.

Usage:
  uv run python scripts/sweep_guide_count_confound.py --seeds 20
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
from sweep_neighbour_features import (_ALL_NB, _BASE, _GUIDE, _NB_CLASS, _NB_DEPMAP,
                                      _NB_DIST, _NB_EXPR, _ONEHOT, _S1A_GENE, _TPM,
                                      load_blocks)
from sweep_prescreen_features import load_guides
from sweep_tpm_features import _ALL_CELLS, _TRAIN, load_tpm, tpm_block

_COUNT, _SEQ = "guide_count", "guide_seq"

# Everything except the guide block; the guide pieces are added per config.
_NO_GUIDE = [_ONEHOT, _TPM, _S1A_GENE] + _ALL_NB

CONFIGS: list[tuple[str, list[str]]] = [
    ("final (guide9)", _NO_GUIDE + [_GUIDE]),
    ("count only", _NO_GUIDE + [_COUNT]),
    ("sequence only (no count)", _NO_GUIDE + [_SEQ]),
    ("no guide block", _NO_GUIDE),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", default="results/guide_count_confound.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}")

    total, mrna = load_tpm()
    guide, guide_names = load_guides(genes)
    s1a_gene, nb_dist, nb_class, nb_depmap, nb_expr, _tissue = load_blocks(genes)

    count_idx = guide_names.index("guide_count")
    guide_count = guide[:, [count_idx]]
    guide_seq = np.delete(guide, count_idx, axis=1)
    print(f"  guide block split: count={guide_count.shape[1]} col, "
          f"sequence={guide_seq.shape[1]} cols")

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
        if _GUIDE in blocks:
            cols.append(guide)
        if _COUNT in blocks:
            cols.append(guide_count)
        if _SEQ in blocks:
            cols.append(guide_seq)
        return np.hstack(cols).astype(np.float32)

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
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
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<26} ({X_eval.shape[1]:>3} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    ref = float(np.mean(results["no guide block"]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<26} {'AUPRC':>7} {'sd':>7} {'vs no guide':>14}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - ref
        note = "" if name == "no guide block" else f"{d:+.4f}"
        print(f"{name:<26} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>14}")

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
