"""EDA 6: what are the lncRNAs essential in three or more cell lines, and why do they separate?

EDA 5 found that the 40 screened lncRNAs essential in >= 3 of the 5 lines (the "core" set) are
the most separable group in several embeddings, including sequence ones that GC + length does
not explain. This describes them against the 4,484 lncRNAs never essential, using only
pre-screen annotation (mmc2 S1A, the GTF, expression), and then tests three explanations for the
sequence signal:

  family      eight of the 40 are annotated 'multimember_family'; copies of a family share
              sequence, so they would be each other's neighbours
  host gene   many core lncRNAs sit next to housekeeping protein-coding genes (TAF12, TFRC,
              TOMM7, HNRNPR); a guide that also hits the neighbour's mRNA would make the lncRNA
              look essential. Tested with S1A's DepMap Cas9 score for the closest protein-coding
              gene (more negative = more essential) and the distance to it.
  similarity  how close core lncRNAs are to each other in k-mer space, and how the EDA-5 z falls
              when the explanation's members are removed.

Comparisons are core vs never essential: Fisher exact test for categories, Mann-Whitney U for
numbers. The S1A column 'Number of cell lines showing essentiality' is not used (it is the
answer); the groups come from the labels.

Usage:
  uv run python eda/core_shared.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from embedding_separation import _LINES, load_embeddings_all, load_labels, neighbours  # noqa: E402
from within_tissue_separation import z_score  # noqa: E402

_OUT = "eda/core_shared_annotations.csv"
_DEPMAP = "Closest protein-coding gene Cas9 - DepMap score (23Q2, median)"


def main() -> None:
    genes, y = load_labels()
    n_lines = np.column_stack([y[c] for c in _LINES]).sum(1)
    core, never = n_lines >= 3, n_lines == 0
    s1a = pd.read_excel("data/raw/mmc2.xlsx", sheet_name="S1A", header=2).set_index("lncRNA")
    s1a.index = s1a.index.astype(str)
    a = s1a.reindex(genes)
    emb, mrna = load_embeddings_all(genes)
    a["tau"] = a["Tissue tau"]
    a["mean_log_mrna"] = np.mean(list(mrna.values()), axis=0)
    a["n_lines_expressed"] = np.sum([np.expm1(m) >= 1 for m in mrna.values()], axis=0)
    a["gc"] = emb["gc_length"][:, 0]
    a["group"] = np.where(core, "core (3+ lines)", np.where(never, "never", "other"))
    a.to_csv(_OUT)
    print(f"core {core.sum()} vs never {never.sum()} screened lncRNAs\n")

    print("Numeric (median core / never, Mann-Whitney p):")
    for col in ["Transcript length", "Exons", "tau", "mean_log_mrna", "n_lines_expressed", "gc",
                "Distance to closest protein-coding gene", _DEPMAP]:
        x = pd.to_numeric(a.loc[core, col], errors="coerce").dropna()
        z = pd.to_numeric(a.loc[never, col], errors="coerce").dropna()
        print(f"  {col[:42]:<44}{x.median():>10.2f} / {z.median():<10.2f} p = "
              f"{mannwhitneyu(x, z).pvalue:.1e}  (n {len(x)} / {len(z)})")

    print("\nCategories (core n / share vs never share, Fisher p; only p < 0.05):")
    for col in ["Genomic class", "Age", "CRISPRi hit"]:
        v = a[col].astype(str).fillna("missing")
        for cat in sorted(v.unique()):
            k, m = (core & (v == cat)).sum(), (never & (v == cat)).sum()
            if k < 3:
                continue
            p = fisher_exact([[k, core.sum() - k], [m, never.sum() - m]])[1]
            tag = "  <--" if p < 0.05 else ""
            print(f"  {col[:14]:<15}{cat:<24}{k:>3} / {k / core.sum():>5.0%}  vs {m / never.sum():>5.0%}"
                  f"   p = {p:.1e}{tag}")

    fam = (a["Age"].astype(str) == "multimember_family").to_numpy()
    dep = pd.to_numeric(a[_DEPMAP], errors="coerce").to_numpy()
    print("\nWhy sequence separates them:")
    print(f"  multimember family: {fam[core].sum()}/40 of core vs {fam[never].mean():.1%} of never")
    for name, X in [("kmer4", emb["kmer4"]), ("dnabert2_mean", emb["dnabert2_mean"]),
                    ("dev_atlas", emb["dev_atlas"])]:
        rng = np.random.default_rng(0)
        out = []
        for label, drop in [("all", np.zeros(len(genes), bool)), ("no family members", fam),
                            ("no neighbour DepMap < -0.5", np.nan_to_num(dep, nan=0) < -0.5)]:
            keep = ~drop
            idx = np.flatnonzero((core | never) & keep)
            nbr = neighbours(X[idx], 20)
            out.append(f"{label}: z {z_score(core[idx].astype(int), nbr, rng):.1f} "
                       f"(core {core[idx].sum()})")
        print(f"  {name:<14} " + " | ".join(out))

    # Are core lncRNAs each other's nearest neighbours in k-mer space?
    idx = np.flatnonzero(core | never)
    nbr = neighbours(emb["kmer4"][idx], 20)
    is_core = core[idx]
    share = is_core[nbr[is_core]].mean()
    print(f"\n  k-mer neighbours of a core lncRNA that are also core: {share:.1%} "
          f"(chance {is_core.mean():.1%})")

    # How concentrated is that signal? Count each core lncRNA's core neighbours among its 20.
    cnt = is_core[nbr].sum(1)
    top = pd.DataFrame({"lncRNA": np.array(genes)[idx][is_core], "core_neighbours_of_20": cnt[is_core],
                        "age": a["Age"].astype(str).fillna("missing").to_numpy()[idx][is_core],
                        "closest_pc_gene": a["Closest protein-coding gene symbol"].to_numpy()[idx][is_core],
                        "depmap": pd.to_numeric(a[_DEPMAP], errors="coerce").round(2).to_numpy()[idx][is_core]})
    top = top.sort_values("core_neighbours_of_20", ascending=False)
    print(f"  core lncRNAs with >= 1 core neighbour: {(cnt[is_core] >= 1).sum()}/{is_core.sum()}; "
          f"the 10 most connected hold {top.core_neighbours_of_20.head(10).sum() / top.core_neighbours_of_20.sum():.0%} "
          f"of all core-to-core links")
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
