"""EDA 2: cluster every lncRNA by tissue specificity, then ask which clusters hold the
essential ones.

Developmental atlas (Sarropoulos et al. 2019), all 31,678 lncRNAs, 7 organs. Per lncRNA:

  organ profile  mean RPKM over each organ's samples (all stages pooled), 7 values
  tau            tissue-specificity index on that profile: 0 = equal in every organ,
                 1 = expressed in one organ only
  shape          log2(organ RPKM + 0.05) minus the gene's own mean -- the pattern across
                 organs with overall level removed, which is what the clustering sees. The
                 small pseudocount keeps ratios visible for weakly expressed lncRNAs, which
                 log2(x + 1) would flatten.

lncRNAs whose highest organ mean is below 0.1 RPKM (10,092 of 31,678; 282 of the 5,496
screened) are set aside as "not expressed" rather than clustered: their pattern is noise. The rest are clustered with k-means on the shape; k is
chosen by silhouette over 4-12 on a 5,000-gene sample. Each cluster is named by its top
organ(s) and median tau.

Then, for the 5,496 screened lncRNAs, the essential rate in each cluster per cell line,
relative to that line's overall rate, with a Fisher exact p-value.

Usage:
  uv run python eda/tissue_clusters.py
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_ATLAS = "data/external/sarropoulos_all_lncrna_rpkm.tsv.gz"
_LABELS = "data/processed/lncrna_rra_day14.jsonl.gz"
_OUT = "eda/tissue_clusters.csv"
_OUT_ENRICH = "eda/tissue_cluster_enrichment.csv"
_ORGANS = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]
_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_MIN_RPKM = 0.1
_PSEUDO = 0.05


def tau(profile: np.ndarray) -> np.ndarray:
    m = profile.max(1, keepdims=True)
    return ((1 - profile / np.where(m > 0, m, 1)).sum(1) / (profile.shape[1] - 1))


def main() -> None:
    atlas = pd.read_csv(_ATLAS, sep="\t").set_index("lncRNA")
    organ_of = np.array([c.split(".")[0] for c in atlas.columns])
    prof = np.column_stack([atlas.loc[:, organ_of == o].mean(axis=1) for o in _ORGANS])
    genes = atlas.index.astype(str).to_numpy()

    expressed = prof.max(1) >= _MIN_RPKM
    t = tau(prof)
    shape = np.log2(prof + _PSEUDO)
    shape = shape - shape.mean(1, keepdims=True)
    print(f"{len(genes):,} lncRNAs; expressed (max organ mean >= {_MIN_RPKM} RPKM): "
          f"{expressed.sum():,}; median tau {np.median(t[expressed]):.2f}")

    Xe = shape[expressed]
    rng = np.random.default_rng(0)
    sample = rng.choice(len(Xe), size=min(5000, len(Xe)), replace=False)
    sil = {k: silhouette_score(Xe[sample], KMeans(k, n_init=10, random_state=0)
                               .fit_predict(Xe[sample])) for k in range(4, 13)}
    k = max(sil, key=sil.get)
    print("silhouette by k: " + "  ".join(f"{kk}:{v:.3f}" for kk, v in sil.items())
          + f"  -> k = {k}")

    km = KMeans(k, n_init=20, random_state=0).fit(Xe)
    # Name clusters by organ(s) above the gene mean in the centroid, ordered by median tau.
    names = {}
    for c in range(k):
        cen, members = km.cluster_centers_[c], np.flatnonzero(km.labels_ == c)
        top = [o for o, v in sorted(zip(_ORGANS, cen), key=lambda x: -x[1]) if v > 0.5][:3]
        mt = np.median(t[expressed][members])
        names[c] = ("broad" if not top or mt < 0.35 else "+".join(top)) + f" (tau {mt:.2f})"
    cluster = np.full(len(genes), "not expressed", dtype=object)
    cluster[expressed] = [names[c] for c in km.labels_]

    out = pd.DataFrame({"lncRNA": genes, "cluster": cluster, "tau": t.round(3),
                        "expressed": expressed, **{o: prof[:, i].round(3)
                                                   for i, o in enumerate(_ORGANS)}})
    out.to_csv(_OUT, index=False)

    # Essentiality per cluster, screened lncRNAs only.
    lab: dict[str, dict[str, int]] = {}
    with gzip.open(_LABELS, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            lab.setdefault(r["target"], {})[r["cell_line"]] = int(r["label"])
    scr = out[out.lncRNA.isin(lab)].copy()
    for c in _LINES:
        scr[c] = scr.lncRNA.map(lambda g: lab[g][c])

    rows = []
    order = scr.groupby("cluster").tau.median().sort_values().index
    for name in order:
        inside = scr.cluster == name
        row = {"cluster": name, "n_all": int((out.cluster == name).sum()),
               "n_screened": int(inside.sum())}
        for c in _LINES:
            a, b = scr.loc[inside, c].sum(), scr.loc[~inside, c].sum()
            table = [[a, inside.sum() - a], [b, (~inside).sum() - b]]
            row[f"{c}_ratio"] = round(scr.loc[inside, c].mean() / scr[c].mean(), 2)
            row[f"{c}_p"] = float(f"{fisher_exact(table)[1]:.1e}")
        rows.append(row)
    enr = pd.DataFrame(rows)
    enr.to_csv(_OUT_ENRICH, index=False)

    print(f"\nEssential rate in cluster / line average (Fisher p < 0.001 marked *), "
          f"{len(scr):,} screened lncRNAs")
    print(f"{'cluster':<34}{'all':>7}{'screened':>9}" + "".join(f"{c:>12}" for c in _LINES))
    for r in rows:
        cells = "".join(f"{r[c + '_ratio']:>11.2f}{'*' if r[c + '_p'] < 1e-3 else ' '}"
                        for c in _LINES)
        print(f"{r['cluster']:<34}{r['n_all']:>7,}{r['n_screened']:>9,}{cells}")
    print(f"\n-> {_OUT}, {_OUT_ENRICH}")


if __name__ == "__main__":
    main()
