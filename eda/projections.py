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
# Tissue-aware contrastive embeddings (contrastive/train_tissue_aware.py, lambda=1, seed 0),
# trained only on unscreened lncRNAs, so every screened lncRNA mapped here is held out. The
# earlier in-sample models (train_clip.py, train_tissue.py) are not shown: their sequence
# embeddings had memorised each gene's own expression.
_AWARE = "data/processed/ssl_tissue_aware_seed0.npz"
_AWARE_EVAL = "contrastive/tissue_aware_eval.json"
# Fine-tuned head (contrastive/finetune.py): its fold-0 held-out genes only, one map per cell
# line since the head is line-conditioned; the frozen embedding on the same genes alongside.
_FT = "data/processed/ssl_finetuned_fold0.npz"
_FT_EVAL = "contrastive/finetune_eval.json"
_AWARE_NAMES = {"aware_seq": "lambda=1: sequence", "aware_expr": "lambda=1: expression",
                "aware_both": "lambda=1: both"}


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

    learned_z = {}
    sys.path.insert(0, str(Path(__file__).parent.parent / "contrastive"))
    from train_tissue import tissue_view
    if Path(_AWARE).exists():
        L = np.load(_AWARE)
        row = {g: i for i, g in enumerate(L["genes"])}
        idx = [row[g] for g in genes]
        zs, ze = L["seq_emb"][idx], L["expr_emb"][idx]
        emb = {"aware_seq": zs, "aware_expr": ze, "aware_both": np.hstack([zs, ze]),
               "tissue_raw": tissue_view(np.array(genes)), **emb}
        ev = json.loads(Path(_AWARE_EVAL).read_text())["results"]
        learned_z = {k: {c: round(float(np.mean(ev[n][c])), 1) for c in _LINES}
                     for k, n in _AWARE_NAMES.items()}
        learned_z["tissue_raw"] = None  # filled below from its own score

    maps = {}
    if Path(_FT).exists() and Path(_AWARE).exists():
        ft = np.load(_FT)
        where = {g: i for i, g in enumerate(genes)}
        sub = [where[g] for g in ft["genes"]]

        def place(P):  # held-out genes get coordinates, everyone else null
            full: list = [None] * len(genes)
            for j, xy in zip(sub, _scale(P)):
                full[j] = xy
            return full

        def both(X):
            Z = _prepare(X)
            return (TSNE(perplexity=30, init="pca", random_state=0).fit_transform(Z),
                    umap.UMAP(n_neighbors=15, metric="cosine", random_state=0).fit_transform(Z))

        t, u = both(emb["aware_expr"][sub])
        maps["fold0_frozen"] = {"tsne": place(t), "umap": place(u)}
        maps["fold0_finetuned"] = {"tsne": {}, "umap": {}}
        for c in _LINES:
            t, u = both(ft[f"emb_{c}"])
            maps["fold0_finetuned"]["tsne"][c], maps["fold0_finetuned"]["umap"][c] = place(t), place(u)
        r = json.loads(Path(_FT_EVAL).read_text())["results"]
        learned_z["fold0_frozen"] = {c: round(v, 1) for c, v in r["z: frozen embedding"].items()}
        learned_z["fold0_finetuned"] = {c: round(v, 1) for c, v in r["z: fine-tuned embedding"].items()}
        print("  fine-tuned maps done", flush=True)

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
        "tau": [round(float(v), 3) for v in tissue_view(np.array(genes))[:, -1]],
        "maps": maps,
        "z": {r.embedding: {} for r in sep.itertuples()},
    }
    for r in sep.itertuples():
        out["z"][r.embedding][r.cell_line] = r.z
    tz = json.loads(Path("contrastive/tissue_eval.json").read_text())["z"]["raw: tissue profile"]
    learned_z["tissue_raw"] = {c: round(float(np.mean(v)), 1) for c, v in tz.items()}
    out["z"].update(learned_z)
    Path(_OUT).write_text(json.dumps(out, separators=(",", ":")))
    print(f"-> {_OUT} ({Path(_OUT).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
