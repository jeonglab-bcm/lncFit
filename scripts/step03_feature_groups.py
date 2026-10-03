"""Step 3: every feature group from #118, each added alone to the same base.

Base: logistic regression on own-line mRNA expression (step 2's vehicle; it matches the
mRNA sort). Each group is added to that base on its own -- never stacked -- so every
comparison is against the same reference and nothing depends on the order groups were
tried in. A group passes only if it beats the base on every held-out cell line.

  expr_panel   expression context: mean TPM over the 5-line panel, breadth (#lines with
               TPM >= 1), own/panel ratio, expressed flag
  expr_total   own-line total-RNA TPM (step 2 found it hurts in a linear model)
  annot        S1A numeric annotations: length (raw + log), exons, tissue tau, time tau,
               dynamic-tissue count, dynamic flag
  age          S1A evolutionary age, one-hot
  tissues      which of 7 developmental tissues the gene is dynamic in
  nb_dist      distance to the closest protein-coding gene, raw + log
  nb_class     S1A genomic class (orientation relative to that gene), one-hot
  nb_expr      that gene's expression in this line, plus panel mean / breadth
  kmer4        256 normalised 4-mer frequencies of the transcript
  dev_*        Sarropoulos developmental atlas: level, breadth, timing, per-organ
  conservation phastCons conserved-element overlap of exons
  prolif_*     co-expression with PCNA / MKI67 across development (all, per-organ)

Then a diagnostic for why most groups hurt: univariate AUROC of single features in the
bottom 90% vs the top 10% of mRNA expression.

Usage:
  uv run python scripts/step03_feature_groups.py
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from stepwise_common import fitted, load_groups, load_labels, lr, own_expression, run

_BASE = "base: LR on mRNA"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/step03_feature_groups.csv")
    args = ap.parse_args()

    genes, cells, y = load_labels()
    E = own_expression(genes, cells)
    base = {c: E[c][:, 1:2] for c in cells}
    groups = load_groups(genes, cells, E)

    candidates = [fitted(_BASE, base, lr)]
    for name, block in groups.items():
        X = {c: np.hstack([base[c], block[c]]).astype(np.float32) for c in cells}
        candidates.append(fitted(f"+ {name} ({block[cells[0]].shape[1]} cols)", X, lr))
    means = run(candidates, y, cells, args.seeds, args.out)

    ref = means[_BASE]
    print("\nChange vs base, per line (PASS = better on every line):")
    for name, m in means.items():
        if name != _BASE:
            d = [m[c] - ref[c] for c in cells]
            print(f"  {name:<34}" + "".join(f"{x:>+10.4f}" for x in d)
                  + f"{np.mean(d):>+10.4f}  {'PASS' if min(d) > 0 else ''}")

    g = {k: v[cells[0]] for k, v in groups.items()}  # gene-level blocks
    singles = {"log transcript length": g["annot"][:, 5], "exons": g["annot"][:, 1],
               "tissue tau": g["annot"][:, 2], "log nb distance": g["nb_dist"][:, 1],
               "conserved fraction": g["conservation"][:, 0],
               "dev_breadth[0]": g["dev_breadth"][:, 0], "dev_level[0]": g["dev_level"][:, 0]}
    top = {c: E[c][:, 1] >= np.quantile(E[c][:, 1], 0.9) for c in cells}
    print("\nUnivariate AUROC, pooled over lines: bottom 90% vs top 10% of mRNA expression")
    for name, f in singles.items():
        auc = [roc_auc_score(np.concatenate([y[c][sel(c)] for c in cells]),
                             np.concatenate([f[sel(c)] for c in cells]))
               for sel in (lambda c: ~top[c], lambda c: top[c])]
        print(f"  {name:<24}{auc[0]:>8.3f}{auc[1]:>8.3f}")


if __name__ == "__main__":
    main()
