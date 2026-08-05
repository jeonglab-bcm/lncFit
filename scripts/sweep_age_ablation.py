"""Does `Age` contribute? It is the 2026 paper's only conservation measure, and a gate test.

The corrected paper (Liang et al., Cell Genomics 2026, doi 10.1016/j.xgen.2026.101253) makes
exactly one conservation claim, in Figure 2A: essential lncRNAs show "fewer human-specific
transcripts than expected and a larger fraction of older (180 Mya) transcripts, indicating the
essential lncRNAs are more likely to be evolutionarily conserved". The measure behind it is
S1A's `Age` category, tested by Fisher's exact. There is no phyloP, no phastCons, and no
alignment-based conservation score anywhere in the paper.

`Age` is already in our model, one-hot encoded inside the s1a_gene block (7 of its 14 columns),
but it has never been isolated -- the block was only ever adopted or ablated whole. So this
ablation does double duty. It says what `Age` is worth, and it gates whether finer conservation
scores are worth fetching: continuous per-base constraint (phyloP/phastCons over exons) measures
the same underlying axis at higher resolution, so if seven coarse age bins contribute nothing,
the refinement has little room to help.

s1a_gene column layout, for the ablation indices: 0-4 the numeric fields (transcript length,
exons, tissue tau, time tau, count dynamic tissues), 5 log1p(transcript length), 6 `Dynamic`,
7-13 the `Age` one-hot.

Usage:
  python scripts/sweep_age_ablation.py --seeds 20
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
from sweep_neighbour_features import load_blocks
from sweep_prescreen_features import load_guides
from sweep_tpm_features import _ALL_CELLS, _TRAIN, load_tpm, tpm_block

_AGE_COLS = list(range(7, 14))
_WITH, _WITHOUT = "base (with Age)", "minus Age"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/age_ablation.csv")
    args = ap.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int) for c in cells}
    print(f"{len(genes):,} genes, folds: {', '.join(cells)}")

    total, mrna = load_tpm()
    guide, guide_names = load_guides(genes)
    s1a, nb_dist, nb_class, _dep, nb_expr, _t = load_blocks(genes)
    guide_count = guide[:, [guide_names.index("guide_count")]]
    print(f"  s1a_gene has {s1a.shape[1]} cols; Age occupies {len(_AGE_COLS)}")

    def make(cell_line: str, drop_age: bool) -> np.ndarray:
        oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
        oh[:, _ALL_CELLS.index(cell_line)] = 1.0
        s = np.delete(s1a, _AGE_COLS, axis=1) if drop_age else s1a
        return np.hstack([oh, tpm_block(genes, cell_line, total, mrna)[0], s, nb_dist,
                          nb_class, nb_expr[cell_line], guide_count]).astype(np.float32)

    rows, results = [], defaultdict(list)
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)

    for holdout in cells:
        print(f"\n--- hold out {holdout} ({labels[holdout].sum()} pos) ---", flush=True)
        for name, drop in [(_WITH, False), (_WITHOUT, True)]:
            X_eval = make(holdout, drop)
            X_train = np.vstack([make(c, drop) for c in cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in cells if c != holdout])

            aps, aus = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                    colsample_bytree=0.3, tree_method="hist", objective="binary:logistic",
                    eval_metric="aucpr", scale_pos_weight=1.0, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_eval)[:, 1]
                aps.append(float(average_precision_score(labels[holdout], p)))
                aus.append(float(roc_auc_score(labels[holdout], p)))
                results[name].append(aps[-1])
                paired[(holdout, seed)][name] = aps[-1]
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1], "auroc": round(aus[-1], 4),
                             "auprc": round(aps[-1], 4)})
            print(f"  {name:<18} ({X_eval.shape[1]:>2} cols)  AUROC {np.mean(aus):.4f}  "
                  f"AUPRC {np.mean(aps):.4f} +/- {np.std(aps):.4f}", flush=True)

    mw, mo = float(np.mean(results[_WITH])), float(np.mean(results[_WITHOUT]))
    print(f"\n=== mean over {len(cells)} folds x {args.seeds} seeds ===")
    print(f"{_WITH:<18} {mw:.4f}")
    print(f"{_WITHOUT:<18} {mo:.4f}   delta {mo - mw:+.4f}")
    wins = sum(1 for v in paired.values() if v[_WITH] > v[_WITHOUT])
    print(f"\nkeeping Age wins {wins}/{len(paired)} paired runs")
    print("per-fold:")
    signs = []
    for holdout in cells:
        fw = np.mean([v[_WITH] for (h, _), v in paired.items() if h == holdout])
        fo = np.mean([v[_WITHOUT] for (h, _), v in paired.items() if h == holdout])
        signs.append(fw - fo)
        print(f"  {holdout:<14} with {fw:.4f}  without {fo:.4f}  ({fw - fo:+.4f})")
    print(f"\nsign consistent across folds: "
          f"{'yes' if min(signs) > 0 or max(signs) < 0 else 'NO'}")

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
