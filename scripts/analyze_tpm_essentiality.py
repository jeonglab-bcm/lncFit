"""TPM vs. lncRNA essentiality analysis.

Compares TPM distributions between essential and non-essential lncRNAs across
five cell lines using:
  - Essentiality: RRA p < 0.05 at Day 14 (mmc3, sheets S2F–S2J)
  - Expression filter: TPM > 0 in S1C (total RNA-seq) OR S1E (mRNA-seq)
  - TPM values for boxplot y-axis: S1C base cell line column

Outputs:
  results/tpm_essentiality/boxplot.png       per-cell-line boxplots
  results/tpm_essentiality/summary.csv       n and median TPM per group per cell line
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_RRA_SHEETS = {
    "HAP1":       "S2F",
    "HEK293FT":   "S2G",
    "K562":       "S2H",
    "MDA-MB-231": "S2I",
    "THP1":       "S2J",
}
_CL_EXPR_COLS = {
    "HAP1":       ["HAP1", "HAP1_RfxCas13d", "HAP1_s1e"],
    "HEK293FT":   ["HEK293FT", "HEK293FT_RfxCas13d", "HEK293FT_s1e"],
    "K562":       ["K562", "K562_RfxCas13d", "K562_s1e"],
    "MDA-MB-231": ["MDA-MB-231", "MDA-MB-231_s1e"],
    "THP1":       ["THP1", "THP1_RfxCas13d", "THP1_s1e"],
}


def load_s1c(path):
    raw = pd.read_excel(path, sheet_name="S1C", header=None, skiprows=2)
    raw.columns = ["target", "HAP1", "HAP1_RfxCas13d",
                   "HEK293FT", "HEK293FT_RfxCas13d",
                   "K562", "K562_RfxCas13d",
                   "MDA-MB-231", "THP1", "THP1_RfxCas13d"]
    for col in raw.columns[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
    return raw


def load_s1e(path):
    raw = pd.read_excel(path, sheet_name="S1E", header=None, skiprows=2)
    raw.columns = raw.iloc[0]
    raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.rename(columns={
        "lncRNA":               "target",
        "HAP1 RfxCas13d":       "HAP1_s1e",
        "HEK293FT RfxCas13d":   "HEK293FT_s1e",
        "K562 RfxCas13d":       "K562_s1e",
        "MDA-MB-231 RfxCas13d": "MDA-MB-231_s1e",
        "THP1 RfxCas13d":       "THP1_s1e",
    })
    for col in raw.columns[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)
    return raw


def load_rra(path, cell_line):
    rra = pd.read_excel(path, sheet_name=_RRA_SHEETS[cell_line],
                        header=None, skiprows=2)
    rra.columns = rra.iloc[0]
    rra = rra.iloc[1:].reset_index(drop=True)
    rra = rra[rra["Gene"].str.startswith("Hum_", na=False)].copy()
    rra["Day 14 - P value"] = pd.to_numeric(rra["Day 14 - P value"], errors="coerce")
    rra["essential"] = rra["Day 14 - P value"] < 0.05
    return rra[["Gene", "essential"]]


def main():
    parser = argparse.ArgumentParser(description="TPM vs. lncRNA essentiality analysis.")
    parser.add_argument("--tpm", default="data/raw/mmc2.xlsx",
                        help="Path to TPM table Excel file (default: data/raw/mmc2.xlsx)")
    parser.add_argument("--rra", default="data/raw/mmc3.xlsx",
                        help="Path to RRA p-value Excel file (default: data/raw/mmc3.xlsx)")
    parser.add_argument("--output-dir", default="results/tpm_essentiality",
                        help="Output directory (default: results/tpm_essentiality)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TPM tables ...")
    s1c = load_s1c(args.tpm)
    s1e = load_s1e(args.tpm)
    tpm = s1c.merge(s1e, on="target", how="inner")
    print(f"  {len(tpm):,} lncRNAs after inner join of S1C and S1E")

    rng = np.random.default_rng(42)
    colors = {"Essential": "#d73027", "Non-essential": "#4575b4"}
    order = ["Essential", "Non-essential"]

    fig, axes = plt.subplots(1, 5, figsize=(14, 5), sharey=True)
    summary_rows = []

    for ax, cl in zip(axes, _CELL_LINES):
        rra = load_rra(args.rra, cl)

        # Expressed = TPM > 0 in any column for this cell line
        expressed_mask = (tpm[_CL_EXPR_COLS[cl]] > 0).any(axis=1)
        expressed = (tpm[expressed_mask][["target"]]
                     .merge(rra, left_on="target", right_on="Gene", how="inner")
                     .merge(tpm[["target", cl]], on="target", how="left"))
        expressed["log2tpm"] = np.log2(expressed[cl] + 1)
        expressed["label"] = expressed["essential"].map(
            {True: "Essential", False: "Non-essential"}
        )

        n_ess = int(expressed["essential"].sum())
        n_non = int((~expressed["essential"]).sum())
        e  = expressed[expressed["essential"] == True]["log2tpm"].values
        ne = expressed[expressed["essential"] == False]["log2tpm"].values
        _, p = mannwhitneyu(e, ne, alternative="two-sided")

        print(f"  {cl}: {n_ess} essential, {n_non} non-essential  p={p:.2e}")

        summary_rows.append({
            "cell_line":                    cl,
            "n_essential":                  n_ess,
            "n_non_essential":              n_non,
            "n_total_expressed":            n_ess + n_non,
            "median_log2tpm_essential":     float(np.median(e)),
            "median_log2tpm_non_essential": float(np.median(ne)),
            "mannwhitney_p":                float(p),
        })

        for i, grp in enumerate(order):
            vals = expressed[expressed["label"] == grp]["log2tpm"].values
            ax.boxplot(vals, positions=[i], widths=0.4, patch_artist=True,
                       showfliers=False,
                       boxprops=dict(facecolor=colors[grp], alpha=0.5),
                       medianprops=dict(color="black", linewidth=1.5),
                       whiskerprops=dict(color="black"),
                       capprops=dict(color="black"))
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(i + jitter, vals, s=4, alpha=0.3, color=colors[grp], linewidths=0)

        ax.set_title(f"{cl}\np={p:.1e}", fontsize=8, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"Essential\n(n={n_ess})", f"Non-essential\n(n={n_non})"],
            fontsize=7, ha="center",
        )
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("log₂(TPM + 1)", fontsize=9)
    fig.suptitle(
        "TPM distribution: essential vs. non-essential lncRNAs\n"
        "(RRA p < 0.05, Day 14, expressed in S1C or S1E)",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    plot_path = out_dir / "boxplot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBoxplot saved  -> {plot_path}")

    summary_path = out_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Summary saved  -> {summary_path}")


if __name__ == "__main__":
    main()
