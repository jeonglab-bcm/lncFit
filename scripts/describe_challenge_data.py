#!/usr/bin/env python3
"""Print the descriptive tables quoted in docs/PARTICIPATE.md's biology section.

Exists so the numbers in that document are reproducible rather than asserted. It
reports, per cell line: lineage, hit count and rate; pairwise hit-set overlap;
single-source transfer AUPRC (how well one line's measured depletion ranks
another line's hits); and pairwise depletion rank-correlation.

This is descriptive analysis of the screen, not a model and not a submission
path. It necessarily reads `fold_change` and `label` for every cell line
including THP1, which is fine for understanding the dataset but is exactly what
you may NOT feed a model -- see the feature ban in docs/PARTICIPATE.md. Nothing
here is importable as a feature builder; it only prints.

  python scripts/describe_challenge_data.py
"""
import argparse
import gzip
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# Lineage per this repo's Celligner analysis; see data/external/README.md, which also
# records that K562/THP1 cluster cleanly (15/15 same-lineage neighbours), HAP1 is an
# unexplained outlier within Myeloid, and MDA-MB-231's Breast cluster is noisier.
LINEAGE = {
    "HAP1": "Myeloid (near-haploid, ex-CML)",
    "K562": "Myeloid (CML, BCR-ABL+)",
    "THP1": "Myeloid (AML, monocytic)",
    "MDA-MB-231": "Breast (triple-negative)",
    "HEK293FT": "Embryonic kidney (non-cancer)",
}
_ORDER = ["HAP1", "K562", "THP1", "MDA-MB-231", "HEK293FT"]
_DEFAULT = "data/processed/lncrna_rra_day14.jsonl.gz"


def _load(path):
    fc, lab = defaultdict(dict), defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            fc[r["target"]][r["cell_line"]] = r["fold_change"]
            lab[r["target"]][r["cell_line"]] = r["label"]
    return fc, lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=_DEFAULT, help=f"day-14 jsonl.gz (default {_DEFAULT})")
    args = ap.parse_args()

    fc, lab = _load(args.path)
    targets = sorted(fc)
    lines = [c for c in _ORDER if any(c in fc[t] for t in targets)]
    print(f"{len(targets):,} lncRNAs x {len(lines)} cell lines\n")

    print("=== Cell lines ===")
    print(f"{'line':<12} {'lineage':<34} {'n':>6} {'hits':>6} {'rate':>7}")
    for c in lines:
        ys = [lab[t][c] for t in targets if c in lab[t]]
        print(f"{c:<12} {LINEAGE.get(c, '?'):<34} {len(ys):>6} {sum(ys):>6} {np.mean(ys):>7.4f}")

    print("\n=== Hit-set overlap (Jaccard) ===")
    hits = {c: {t for t in targets if lab[t].get(c) == 1} for c in lines}
    print(f"{'':<12}" + "".join(f"{c:>13}" for c in lines))
    for a in lines:
        row = f"{a:<12}"
        for b in lines:
            row += f"{len(hits[a] & hits[b]) / max(1, len(hits[a] | hits[b])):>13.3f}"
        print(row)

    print("\n=== Single-source transfer (rows = depletion source, cols = line predicted) ===")
    print("AUPRC / lift over the predicted line's base rate.")
    print(f"{'source':<12}" + "".join(f"{c:>16}" for c in lines))
    for a in lines:
        row = f"{a:<12}"
        for b in lines:
            if a == b:
                row += f"{'--':>16}"
                continue
            ts = [t for t in targets if a in fc[t] and b in lab[t]]
            y = [lab[t][b] for t in ts]
            ap_ = average_precision_score(y, [-fc[t][a] for t in ts])
            row += f"{ap_:>10.4f}/{ap_ / np.mean(y):>4.1f}x"
        print(row)
    print("\nNote: compare within a COLUMN, not across. Lift shares a denominator only")
    print("down a column, so cross-column lift comparisons are confounded by base rate.")

    print("\n=== Depletion rank-correlation (Spearman) ===")
    print("The ceiling on cross-line transfer -- every pair is weak.")
    print(f"{'':<12}" + "".join(f"{c:>13}" for c in lines))
    for a in lines:
        row = f"{a:<12}"
        for b in lines:
            ts = [t for t in targets if a in fc[t] and b in fc[t]]
            rho = spearmanr([fc[t][a] for t in ts], [fc[t][b] for t in ts])[0]
            row += f"{rho:>13.3f}"
        print(row)


if __name__ == "__main__":
    main()
