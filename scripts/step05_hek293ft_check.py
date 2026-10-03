"""Step 5: one untouched check of steps 2-4 on HEK293FT.

HEK293FT was excluded from the THP1 challenge and used by no earlier step, so it is data
the step-4 design was not chosen on. Plan fixed before this was run (committed first):

  1. mRNA sort                                  (step-1/2 bar)
  2. top-decile LR on mRNA                      (step-4 base)
  3. top-decile LR on mRNA + dev_breadth        (the one atlas group carried forward,
                                                 picked for fewest columns, 2, not score)

Train on HAP1, K562 and MDA-MB-231; predict HEK293FT with the genes being predicted held
out of training (5 gene folds), as in every step. Pass = 3 beats both 1 and 2.

Only HEK293FT rows are read from the five-line file; THP1 rows are skipped while reading.

Usage:
  uv run python scripts/step05_hek293ft_check.py
"""
import argparse
import gzip
import json

import numpy as np

from stepwise_common import (TopDecileModel, fitted, fixed, load_groups, load_labels,
                             own_expression, percentile, run)

_ALL_LINES = "data/processed/lncrna_rra_day14.jsonl.gz"
_TARGET = "HEK293FT"


def load_target_labels(genes: list[str]) -> np.ndarray:
    labels = {}
    with gzip.open(_ALL_LINES, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["cell_line"] == _TARGET:
                labels[row["target"]] = int(row["label"])
    return np.array([labels[g] for g in genes], dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/step05_hek293ft_check.csv")
    args = ap.parse_args()

    genes, train_cells, y = load_labels()
    cells = train_cells + [_TARGET]
    y[_TARGET] = load_target_labels(genes)
    E = own_expression(genes, cells)
    head = {c: np.column_stack([percentile(E[c][:, 1]), E[c][:, 1]]) for c in cells}
    breadth = load_groups(genes, cells, E)["dev_breadth"]

    # Holding out HEK293FT trains on every other line in X: the three training lines.
    means = run([
        fixed("1. mRNA sort", lambda h: E[h][:, 1]),
        fitted("2. top-decile LR on mRNA", head, TopDecileModel),
        fitted("3. + dev_breadth",
               {c: np.hstack([head[c], breadth[c]]) for c in cells}, TopDecileModel),
    ], y, [_TARGET], args.seeds, args.out)

    m = {k: v[_TARGET] for k, v in means.items()}
    ok = m["3. + dev_breadth"] > max(m["1. mRNA sort"], m["2. top-decile LR on mRNA"])
    print(f"\nHEK293FT positive rate {y[_TARGET].mean():.4f}; check {'PASSES' if ok else 'FAILS'}")


if __name__ == "__main__":
    main()
