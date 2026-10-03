"""Step 4: order only the most-expressed genes, with a model fit only on them.

Step 3 found that features relate to the label differently in the top 10% of mRNA
expression -- where AUPRC is decided -- than in the bulk, so a model fit on all genes
learns the bulk pattern and scrambles the top. stepwise_common.TopDecileModel ranks the
top-decile genes first and orders them with logistic regression trained on top-decile
training rows only; everything else keeps the mRNA sort. The 10% cut comes from step 2,
where the hit rate is flat for nine deciles and jumps in the top one.

Same groups as step 3, each added alone to the same base (top-decile LR on log mRNA). A
group passes only if it beats the base on every held-out line. The passing groups are then
combined in one last row.

Usage:
  uv run python scripts/step04_top_decile.py
"""
import argparse

import numpy as np

from stepwise_common import (TopDecileModel, fitted, load_groups, load_labels,
                             own_expression, percentile, run)

_BASE = "base: top-decile LR on mRNA"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/step04_top_decile.csv")
    args = ap.parse_args()

    genes, cells, y = load_labels()
    E = own_expression(genes, cells)
    head = {c: np.column_stack([percentile(E[c][:, 1]), E[c][:, 1]]) for c in cells}
    groups = load_groups(genes, cells, E)

    def with_(names):
        return {c: np.hstack([head[c]] + [groups[n][c] for n in names]).astype(np.float32)
                for c in cells}

    candidates = [fitted(_BASE, head, TopDecileModel)]
    candidates += [fitted(f"+ {n} ({groups[n][cells[0]].shape[1]} cols)", with_([n]),
                          TopDecileModel) for n in groups]
    means = run(candidates, y, cells, args.seeds, args.out)

    ref = means[_BASE]
    passed = []
    print("\nChange vs base, per line (PASS = better on every line):")
    for (name, *_), group in zip(candidates[1:], groups):
        d = [means[name][c] - ref[c] for c in cells]
        ok = min(d) > 0
        passed += [group] if ok else []
        print(f"  {name:<34}" + "".join(f"{x:>+10.4f}" for x in d)
              + f"{np.mean(d):>+10.4f}  {'PASS' if ok else ''}")

    if len(passed) > 1:
        label = "+ " + " + ".join(passed)
        combo = run([fitted(_BASE, head, TopDecileModel),
                     fitted(label, with_(passed), TopDecileModel)],
                    y, cells, args.seeds, args.out.replace(".csv", "_combined.csv"))
        d = [combo[label][c] - combo[_BASE][c] for c in cells]
        print(f"\ncombined vs base: " + " ".join(f"{x:+.4f}" for x in d)
              + f"  mean {np.mean(d):+.4f}  {'PASS' if min(d) > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
