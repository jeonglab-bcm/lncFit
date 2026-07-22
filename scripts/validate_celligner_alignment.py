"""Validate the from-scratch Celligner realignment (issue #78) by nearest-neighbor
lineage purity, not by eyeballing distances among just the 4 target cell lines.

For each CCLE cell line, finds its k=15 nearest neighbors (by UMAP distance) among
all other CCLE cell lines and computes what fraction share its true Oncotree
lineage (Model.csv). Reports this for all lineage-annotated cell lines (a baseline
for what "good" purity looks like in this specific alignment) and for the 4 target
cell lines specifically (K562, MDA-MB-231, THP1, HAP1).

Reads a full_umap_coords.csv (all tumor + CL samples) produced by the one-off R
realignment script (not checked into this repo -- see data/external/README.md) and
the DepMap Model.csv used to build it.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

TARGET_CELL_LINES = {
    "ACH-000551": ("K562", "Myeloid"),
    "ACH-000768": ("MDA-MB-231", "Breast"),
    "ACH-000146": ("THP1", "Myeloid"),
    "ACH-002475": ("HAP1", "Myeloid"),
}
K = 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-coords", required=True, help="full_umap_coords.csv from the R realignment script")
    args = parser.parse_args()

    full = pd.read_csv(args.full_coords)
    cls = full[full["type"] == "CL"].copy()
    cls = cls[cls["lineage"].notna()].reset_index(drop=True)

    coords = cls[["UMAP_1", "UMAP_2"]].to_numpy()
    lineages = cls["lineage"].to_numpy()
    ids = cls["sampleID"].to_numpy()
    n = len(cls)

    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)

    purities = np.empty(n)
    for i in range(n):
        nn_idx = np.argsort(D[i])[:K]
        purities[i] = (lineages[nn_idx] == lineages[i]).mean()

    print(f"Baseline over all {n} lineage-annotated CCLE lines:")
    print(f"  mean purity={purities.mean():.3f}  median={np.median(purities):.3f}")
    print()

    summary = pd.DataFrame({"lineage": lineages, "purity": purities}).groupby("lineage")["purity"].agg(["mean", "count"])
    print(summary.sort_values("mean").to_string())
    print()

    print(f"Target cell lines (k={K} nearest CCLE neighbors):")
    for depmap_id, (name, true_lineage) in TARGET_CELL_LINES.items():
        i = np.where(ids == depmap_id)[0][0]
        nn_idx = np.argsort(D[i])[:K]
        same = (lineages[nn_idx] == true_lineage).sum()
        lineage_avg = purities[lineages == true_lineage].mean()
        print(f"  {name:<12} true_lineage={true_lineage:<10} same_lineage_neighbors={same}/{K}  "
              f"lineage_own_average_purity={lineage_avg:.3f}")


if __name__ == "__main__":
    main()
