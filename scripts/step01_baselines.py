"""Step 1: the simplest baselines under a cell-line AND gene hold-out.

Every later step is compared against these numbers, so they use no tuning at all: an
untuned logistic regression, and two scorers that fit nothing.

  random              uniform noise; its AUPRC is the positive rate
  tpm rank            the gene's own total-RNA TPM in the held-out line, used directly
  mrna rank           the same with mRNA-seq TPM
  tpm (LR)            logistic regression on own-line log TPM (total + mRNA), 2 columns
  tpm + annotation    adds S1A gene annotations: length, exons, tissue/time tau, dynamic
                      tissue count, dynamic flag, evolutionary age -- 16 columns in all

Evaluation is lncfit.honest_eval: each training line is held out in turn, and the genes being
predicted are also left out of training (5 gene folds), so a score cannot come from
recognising a gene. Repeated over 5 gene-split seeds.

Not used, deliberately: guide_count and the other S1B guide columns (ruled out -- they predict
the label through how it is computed), any fold_change / rra_pvalue, and THP1 labels.

Usage:
  uv run python scripts/step01_baselines.py
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lncfit.honest_eval import predict_cell_and_gene_holdout, score
from sweep_gene_level_priors import load_gene_table
from sweep_neighbour_features import load_blocks
from sweep_tpm_features import _TRAIN, load_tpm, tpm_block


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/step01_baselines.csv")
    args = ap.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    cells = sorted({c for obs in table.values() for c in obs})
    y = {c: np.array([table[g][c]["label"] for g in genes], dtype=int) for c in cells}

    total, mrna = load_tpm()
    s1a_gene = load_blocks(genes)[0]
    # tpm_block columns 0-1 are own-line log total TPM and log mRNA TPM. The panel columns
    # (2-5) are left out: they average over all five lines and belong to a later step.
    tpm = {c: tpm_block(genes, c, total, mrna)[0][:, :2] for c in cells}
    X = {
        "tpm (LR)": tpm,
        "tpm + annotation (LR)": {c: np.hstack([tpm[c], s1a_gene]) for c in cells},
    }

    rows = []
    for holdout in cells:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            preds = {
                "random": rng.random(len(genes)),
                "tpm rank": tpm[holdout][:, 0],
                "mrna rank": tpm[holdout][:, 1],
            }
            for name, Xc in X.items():
                preds[name] = predict_cell_and_gene_holdout(Xc, y, holdout, _lr, seed=seed)
            for name, p in preds.items():
                rows.append({"holdout": holdout, "baseline": name, "seed": seed,
                             **{k: round(v, 4) for k, v in score(y[holdout], p).items()}})

    names = list(dict.fromkeys(r["baseline"] for r in rows))
    print(f"\nAUPRC, mean over {args.seeds} gene-split seeds (sd in brackets)")
    print(f"{'baseline':<24}" + "".join(f"{c:>18}" for c in cells) + f"{'mean':>9}")
    for name in names:
        per = [[r["auprc"] for r in rows if r["baseline"] == name and r["holdout"] == c]
               for c in cells]
        cells_txt = "".join(f"{np.mean(v):>10.4f} ({np.std(v):.4f})" for v in per)
        print(f"{name:<24}{cells_txt}{np.mean([np.mean(v) for v in per]):>9.4f}")
    print("positive rate: " + ", ".join(f"{c} {y[c].mean():.4f}" for c in cells))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-fold/per-seed -> {out}")


if __name__ == "__main__":
    main()
