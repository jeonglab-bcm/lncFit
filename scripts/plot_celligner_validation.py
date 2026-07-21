"""Validation plot for the from-scratch Celligner realignment (issue #78).

Colors every CCLE cell line by its Oncotree tissue lineage (Model.csv), shows all
tumor samples as a neutral gray background cloud for context, and highlights +
labels our 5 target cell lines (K562, MDA-MB-231, THP1, HAP1; HEK293FT excluded --
not in CCLE/DepMap) so their position relative to the rest of the landscape can be
checked by eye, not just by the summary distance numbers in the README.

Reads a full_umap_coords.csv (all tumor + CL samples: sampleID, type, lineage,
UMAP_1, UMAP_2) produced by the one-off R realignment script (not checked into
this repo -- see data/external/README.md).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TARGET_CELL_LINES = {
    "ACH-000551": "K562", "ACH-000768": "MDA-MB-231", "ACH-000146": "THP1", "ACH-002475": "HAP1",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-coords", required=True, help="full_umap_coords.csv from the R realignment script")
    parser.add_argument("--output", default="results/lncrna_rra_day14/celligner_embedding_comparison/alignment_validation.png")
    args = parser.parse_args()

    df = pd.read_csv(args.full_coords)
    tumors = df[df["type"] == "tumor"]
    cls = df[df["type"] == "CL"].copy()
    cls["lineage"] = cls["lineage"].fillna("Unknown")

    lineage_counts = cls["lineage"].value_counts()
    lineages = list(lineage_counts.index)
    cmap = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
    color_map = {lin: cmap[i % len(cmap)] for i, lin in enumerate(lineages)}

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.scatter(tumors["UMAP_1"], tumors["UMAP_2"], s=4, c="lightgray", alpha=0.4, linewidths=0, label="tumor (background)")

    for lin in lineages:
        sub = cls[cls["lineage"] == lin]
        ax.scatter(sub["UMAP_1"], sub["UMAP_2"], s=10, c=[color_map[lin]], alpha=0.8,
                   linewidths=0, label=f"{lin} (n={len(sub)})")

    target_rows = df[df["sampleID"].isin(TARGET_CELL_LINES)]
    for _, row in target_rows.iterrows():
        name = TARGET_CELL_LINES[row["sampleID"]]
        ax.scatter(row["UMAP_1"], row["UMAP_2"], s=260, facecolors="none", edgecolors="black",
                   linewidths=2.2, marker="o", zorder=5)
        ax.annotate(name, (row["UMAP_1"], row["UMAP_2"]), xytext=(8, 8), textcoords="offset points",
                    fontsize=12, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))

    ax.set_xlabel("UMAP_1")
    ax.set_ylabel("UMAP_2")
    ax.set_title("Celligner realignment (from-scratch, current DepMap data): "
                 "all CCLE cell lines colored by tissue lineage, tumors as gray background,\n"
                 "our 5 target cell lines circled + labeled (HEK293FT absent from CCLE/DepMap)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6, ncol=2, frameon=False,
              title=f"CCLE lineage ({len(lineages)} categories)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
