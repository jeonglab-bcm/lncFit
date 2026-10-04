"""EDA 4: inside each tissue-specificity cluster, do essential lncRNAs sit close together?

EDA 1 asked this over all 5,496 screened lncRNAs at once. If essential lncRNAs only group
together within a tissue pattern (for example among brain-expressed lncRNAs), pooling the tissue
groups together can hide it. Here the neighbours are found INSIDE each tissue cluster from
EDA 2 (testis-only, testis + ovary, cerebellum + brain, liver, not expressed), after
standardising and PCA within that cluster, so a lncRNA is only compared with lncRNAs that share
its tissue pattern.

Labels tested in each cluster: essential in each of the five cell lines, and essential in AT
LEAST ONE cell line ("any line"), which pools the positives of the five lines into a single,
larger group. Score: z of the AUPRC of the 20-nearest-neighbour essential fraction against 100
label shuffles inside the same cluster. Tests with fewer than 15 positives or fewer than 150
lncRNAs are skipped, and the number of positives is reported alongside every z.

Embeddings are the eight of EDA 1 (k-mer, DNABERT-2 x2, Nucleotide Transformer, developmental
atlas, cell-line expression, GC + length control, random control). The cluster itself was
defined from the atlas, so the atlas row measures what remains after tissue pattern is held
roughly constant, not an independent signal.

Usage:
  uv run python eda/within_tissue_separation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from embedding_separation import _LINES, load_embeddings_all, load_labels, neighbours  # noqa: E402

_CLUSTERS = "eda/tissue_clusters.csv"
_OUT = "eda/within_tissue_separation.csv"
_MIN_POS, _MIN_N, _K, _SHUFFLES = 15, 150, 20, 100


def z_score(y: np.ndarray, nbr: np.ndarray, rng) -> float:
    auprc = average_precision_score(y, y[nbr].mean(1))
    null = []
    for _ in range(_SHUFFLES):
        p = rng.permutation(y)
        null.append(average_precision_score(p, p[nbr].mean(1)))
    return float((auprc - np.mean(null)) / np.std(null))


def main() -> None:
    genes, y = load_labels()
    emb, _ = load_embeddings_all(genes)
    y["any line"] = np.max([y[c] for c in _LINES], axis=0)
    labels = _LINES + ["any line"]
    cl = pd.read_csv(_CLUSTERS).set_index("lncRNA").cluster.reindex(genes).to_numpy()
    groups = [("all screened", np.ones(len(genes), bool))] + [
        (c, cl == c) for c in pd.Series(cl).value_counts().index]
    rng = np.random.default_rng(0)

    rows = []
    for gname, mask in groups:
        idx = np.flatnonzero(mask)
        if len(idx) < _MIN_N:
            continue
        pos = {lab: int(y[lab][idx].sum()) for lab in labels}
        todo = [lab for lab in labels if pos[lab] >= _MIN_POS]
        for name, X in emb.items():
            nbr = neighbours(X[idx], _K)
            for lab in todo:
                rows.append({"cluster": gname, "n": len(idx), "label": lab,
                             "positives": pos[lab], "embedding": name,
                             "z": round(z_score(y[lab][idx], nbr, rng), 1)})
        print(f"  {gname}: {len(idx):,} lncRNAs, positives " +
              ", ".join(f"{lab} {pos[lab]}" for lab in labels), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(_OUT, index=False)
    order = list(emb)
    for lab in labels:
        sub = df[df.label == lab]
        if sub.empty:
            continue
        print(f"\nz, essential in: {lab}  (rows: embedding; columns: cluster [positives])")
        t = sub.pivot(index="embedding", columns="cluster", values="z").reindex(order)
        pos = sub.drop_duplicates("cluster").set_index("cluster").positives
        t.columns = [f"{c} [{pos[c]}]" for c in t.columns]
        print(t.to_string())
    print(f"\n-> {_OUT}")


if __name__ == "__main__":
    main()
