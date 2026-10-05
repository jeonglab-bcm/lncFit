"""EDA 8: one embedding map per cell line, using only that line's lncRNAs.

For each cell line L, the population is
  - the lncRNAs expressed specifically in L (EDA 7: mRNA TPM >= 1 in some line, tau >= 0.8 over the
    five lines, highest in L), split into
        N  never essential in any line
        T  essential in L and in no other line
  - plus P, the lncRNAs essential in two or more lines (the control; the same set in every map).
lncRNAs specific to L that are essential only in another line are left out of L's map.

Each map is computed from scratch on that population only: its features are standardised and reduced
(PCA to 50 dimensions at most) inside the population, then laid out with UMAP and with t-SNE. The
embeddings are the existing representations, not trained here: sequence (k-mer, DNABERT-2 mean and last
chunk, Nucleotide Transformer) and, for contrast, the developmental atlas and cell-line expression.

Alongside each map the 20-neighbour test of EDA 1: z of T against N and of P against N, inside the
population. T has only 10-20 members per line, so its z is noisy.

Writes eda/line_maps.json for the page. umap-learn is not a project dependency:
  uv run --with umap-learn python eda/line_maps.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from embedding_separation import _LINES, load_embeddings_all, load_labels, neighbours  # noqa: E402
from within_tissue_separation import z_score  # noqa: E402

_OUT = "eda/line_maps.json"
_TAU = 0.8
_EMBEDDINGS = ["kmer4", "dnabert2_mean", "dnabert2_last", "nt_mean", "dev_atlas", "cellline_expr"]


def layout(X: np.ndarray) -> dict[str, list[list[int]]]:
    Z = np.nan_to_num(StandardScaler().fit_transform(X))
    if Z.shape[1] > 50:
        Z = PCA(n_components=50, random_state=0).fit_transform(Z)

    def scale(P):
        P = (P - P.min(0)) / (P.max(0) - P.min(0))
        return np.round(P * 1000).astype(int).tolist()

    return {"umap": scale(umap.UMAP(n_neighbors=15, metric="cosine", random_state=0).fit_transform(Z)),
            "tsne": scale(TSNE(perplexity=30, init="pca", random_state=0).fit_transform(Z))}


def main() -> None:
    genes, y = load_labels()
    emb, mrna = load_embeddings_all(genes)
    Y = np.column_stack([y[c] for c in _LINES])
    n_ess = Y.sum(1)
    M = np.column_stack([np.expm1(mrna[c]) for c in _LINES])
    top = M.argmax(1)
    mx = M.max(1, keepdims=True)
    tau = (1 - M / np.where(mx > 0, mx, 1)).sum(1) / (len(_LINES) - 1)
    specific = (M.max(1) >= 1) & (tau >= _TAU)
    P = n_ess >= 2
    rng = np.random.default_rng(0)

    out = {"lines": _LINES, "embeddings": _EMBEDDINGS, "per_line": {}}
    for li, line in enumerate(_LINES):
        own = specific & (top == li)
        N = own & (n_ess == 0)
        T = own & (n_ess == 1) & (Y[:, li] == 1)
        pop = np.flatnonzero(N | T | P)
        code = np.where(P[pop], 2, np.where(T[pop], 1, 0))            # 0 N, 1 T, 2 P
        entry = {
            "genes": [genes[i] for i in pop],
            "group": code.tolist(),
            "tau": [round(float(tau[i]), 2) for i in pop],
            "tpm": [round(float(M[i, li]), 1) for i in pop],
            "essential_in": ["".join(str(int(v)) for v in Y[i]) for i in pop],
            "counts": {"N": int((code == 0).sum()), "T": int((code == 1).sum()), "P": int((code == 2).sum())},
            "maps": {}, "z": {},
        }
        for en in _EMBEDDINGS:
            X = emb[en][pop]
            entry["maps"][en] = layout(X)
            zs = {}
            for name, g in (("T", 1), ("P", 2)):
                keep = np.flatnonzero((code == 0) | (code == g))
                nbr = neighbours(X[keep], 20)
                zs[name] = round(z_score((code[keep] == g).astype(int), nbr, rng), 1)
            entry["z"][en] = zs
        out["per_line"][line] = entry
        print(f"  {line}: {entry['counts']}  " + "  ".join(f"{e} T {entry['z'][e]['T']} P {entry['z'][e]['P']}"
                                                          for e in ("dnabert2_mean", "kmer4")), flush=True)
    Path(_OUT).write_text(json.dumps(out, separators=(",", ":")))
    print(f"-> {_OUT} ({Path(_OUT).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
