"""2-D t-SNE and UMAP maps of every EDA-1 embedding, for viewing.

Same preprocessing as eda/embedding_separation.py (standardise, PCA to <= 50 dims), then
t-SNE (perplexity 30) and UMAP (15 neighbours, cosine). Writes one JSON with coordinates
scaled to 0-1000, per-line essential labels, GC fraction and lncRNA ids.

Projections are for looking only: distances between clusters in either map are not
meaningful, which is why separation is measured in the original space by
embedding_separation.py.

Usage (umap-learn is not a project dependency):
  uv run --with umap-learn python eda/projections.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from embedding_separation import _LINES, _SEQUENCES, load_embeddings_all, load_labels  # noqa: E402

_OUT = "eda/projections.json"
_SEPARATION = "eda/embedding_separation.csv"


def _prepare(X: np.ndarray) -> np.ndarray:
    Z = np.nan_to_num(StandardScaler().fit_transform(X))
    return PCA(n_components=50, random_state=0).fit_transform(Z) if Z.shape[1] > 50 else Z


def _scale(P: np.ndarray) -> list[list[int]]:
    P = (P - P.min(0)) / (P.max(0) - P.min(0))
    return np.round(P * 1000).astype(int).tolist()


def main() -> None:
    genes, y = load_labels()
    emb, mrna = load_embeddings_all(genes)
    with open(_SEQUENCES) as fh:
        seqs = {g: s.upper() for g, (s, _) in json.load(fh).items()}
    gc = [round((seqs[g].count("G") + seqs[g].count("C")) / max(len(seqs[g]), 1), 3)
          for g in genes]

    maps = {}
    for name, X in emb.items():
        Z = _prepare(X)
        maps[name] = {
            "tsne": _scale(TSNE(perplexity=30, init="pca", random_state=0).fit_transform(Z)),
            "umap": _scale(umap.UMAP(n_neighbors=15, metric="cosine",
                                     random_state=0).fit_transform(Z)),
        }
        print(f"  {name} done", flush=True)

    sep = pd.read_csv(_SEPARATION)
    out = {
        "genes": genes,
        "lines": _LINES,
        "labels": {c: "".join(map(str, y[c])) for c in _LINES},
        "gc": gc,
        "maps": maps,
        "z": {r.embedding: {} for r in sep.itertuples()},
    }
    for r in sep.itertuples():
        out["z"][r.embedding][r.cell_line] = r.z
    Path(_OUT).write_text(json.dumps(out, separators=(",", ":")))
    print(f"-> {_OUT} ({Path(_OUT).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
