"""Validation plot for the from-scratch Celligner realignment (issue #78).

Highlights only the tissue lineages relevant to the validation story -- Myeloid
(K562/THP1/HAP1's true lineage), Breast (MDA-MB-231's true lineage), and Lymphoid
(the lineage most easily confused with Myeloid by eye) -- with every other CCLE
cell line and all tumor samples shown as neutral gray background. Highlights +
labels our 5 target cell lines (K562, MDA-MB-231, THP1, HAP1; HEK293FT excluded --
not in CCLE/DepMap) so their position relative to the highlighted lineages can be
checked by eye, not just by the summary distance/purity numbers in the README.

Reads a full_umap_coords.csv (all tumor + CL samples: sampleID, type, lineage,
UMAP_1, UMAP_2) produced by the one-off R realignment script (not checked into
this repo -- see data/external/README.md).
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

TARGET_CELL_LINES = {
    "ACH-000551": "K562", "ACH-000768": "MDA-MB-231", "ACH-000146": "THP1", "ACH-002475": "HAP1",
}
HIGHLIGHT_LINEAGES = {
    "Myeloid": "tab:purple", "Breast": "tab:blue", "Lymphoid": "tab:cyan", "Skin": "tab:orange",
}
N_NEIGHBORS_CHECKED = 3  # how many nearest CCLE neighbors to check per target


def _nearest_neighbor_lineage_pairs(cls):
    """For each target, return its nearest neighbors' (lineage, target) pairs,
    deduplicated by lineage and kept in nearest-first order -- e.g. MDA-MB-231's
    3 nearest neighbors are Pancreas/Cervix/Liver, all different, so it gets 3
    pairs; K562's 3 nearest are all Myeloid, so it gets just 1."""
    coords = cls[["UMAP_1", "UMAP_2"]].to_numpy()
    ids = cls["sampleID"].to_numpy()
    lineages = cls["lineage"].to_numpy()
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)

    pairs = []
    for depmap_id, target_name in TARGET_CELL_LINES.items():
        matches = np.where(ids == depmap_id)[0]
        if len(matches) == 0:
            continue
        i = matches[0]
        order = np.argsort(D[i])[:N_NEIGHBORS_CHECKED]
        seen = set()
        for j in order:
            lin = lineages[j]
            if lin not in seen:
                seen.add(lin)
                pairs.append((lin, target_name))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-coords", default="data/external/celligner_validation_data/full_umap_coords.csv",
                        help="full_umap_coords.csv from the R realignment script")
    parser.add_argument("--output", default="results/lncrna_rra_day14/celligner_embedding_comparison/alignment_validation.png")
    args = parser.parse_args()

    df = pd.read_csv(args.full_coords)
    tumors = df[df["type"] == "tumor"]
    cls = df[df["type"] == "CL"].copy()
    cls["lineage"] = cls["lineage"].fillna("Unknown")

    neighbor_pairs = _nearest_neighbor_lineage_pairs(cls)

    other = cls[~cls["lineage"].isin(HIGHLIGHT_LINEAGES)]

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(tumors["UMAP_1"], tumors["UMAP_2"], s=4, c="lightgray", alpha=0.35, linewidths=0, label="tumor (background)")
    ax.scatter(other["UMAP_1"], other["UMAP_2"], s=8, c="darkgray", alpha=0.5, linewidths=0,
               label=f"other CCLE lineages (n={len(other)})")

    for lin, color in HIGHLIGHT_LINEAGES.items():
        sub = cls[cls["lineage"] == lin]
        label = f"{lin} (n={len(sub)})"
        ax.scatter(sub["UMAP_1"], sub["UMAP_2"], s=16, c=color, alpha=0.85, linewidths=0, label=label)

    target_rows = df[df["sampleID"].isin(TARGET_CELL_LINES)]
    for _, row in target_rows.iterrows():
        name = TARGET_CELL_LINES[row["sampleID"]]
        ax.scatter(row["UMAP_1"], row["UMAP_2"], s=260, facecolors="none", edgecolors="black",
                   linewidths=2.2, marker="o", zorder=5)
        ax.annotate(name, (row["UMAP_1"], row["UMAP_2"]), xytext=(8, 8), textcoords="offset points",
                    fontsize=12, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))

    # Proxy (invisible) legend entries: one per (nearest-neighbor lineage, target) pair,
    # e.g. "(Liver - MDA-MB-231)" -- these aren't separate plotted groups, just labels.
    for lin, name in neighbor_pairs:
        ax.scatter([], [], s=0, label=f"({lin} - {name})")

    ax.set_xlabel("UMAP_1")
    ax.set_ylabel("UMAP_2")
    ax.set_title("Celligner realignment (from-scratch, current DepMap data)\n"
                 "Myeloid / Breast / Lymphoid / Skin highlighted, rest gray; "
                 "target cell lines circled + labeled")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False,
              title=f"Lineage / nearest-neighbor tissue (top {N_NEIGHBORS_CHECKED})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
