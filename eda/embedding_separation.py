"""EDA step 1: does any lncRNA embedding put essential lncRNAs near each other?

For each embedding and each cell line separately: find every lncRNA's 20 nearest
neighbours (cosine, after standardising and PCA to <= 50 dims), and score it by the
fraction of those neighbours that are essential in that line. The lncRNA itself is never
its own neighbour. Then:

  AUPRC        of that neighbour score against the line's labels
  enrichment   mean neighbour score of essential lncRNAs / the line's positive rate
               (1.0 = essential lncRNAs have no more essential neighbours than chance)
  z            AUPRC against 200 label shuffles on the same neighbour graph
  AUPRC top10  the same AUPRC among the line's 10% most-expressed lncRNAs (mRNA-seq),
               where earlier work found any difference shows up

Embeddings (one vector per lncRNA, the same in every line):
  kmer4             256 normalised 4-mer frequencies of the transcript
  dnabert2_mean     DNABERT-2, mean-pooled over the full transcript
  dnabert2_last     DNABERT-2, last chunk
  nt_mean           Nucleotide Transformer, mean-pooled
  dev_atlas         log1p RPKM over 297 developmental samples (7 organs, 4 wpc - adult)
  cellline_expr     log1p TPM in the 5 screened lines, total RNA + mRNA (10 dims)
  gc_length         GC fraction + log10 length only -- a control: in K562 this alone
                    carries most of the sequence embeddings' signal
  random            64-dim Gaussian noise -- the null every row should beat

Labels are read for all five lines; fold_change and rra_pvalue are never used.

Usage:
  uv run python eda/embedding_separation.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from lncfit.embeddings import load_embeddings  # noqa: E402
from lncfit.features import _count_kmers, all_kmers  # noqa: E402

_LABELS = "data/processed/lncrna_rra_day14.jsonl.gz"
_SEQUENCES = "data/processed/body_sequences_transcript.json"
_MMC2 = "data/raw/mmc2.xlsx"
_ATLAS = "data/external/sarropoulos_human_lncrna_rpkm.tsv.gz"
_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]


def load_labels() -> tuple[list[str], dict[str, np.ndarray]]:
    rows: dict[str, dict[str, int]] = {}
    with gzip.open(_LABELS, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.setdefault(r["target"], {})[r["cell_line"]] = int(r["label"])
    genes = sorted(rows)
    return genes, {c: np.array([rows[g][c] for g in genes]) for c in _LINES}


def _expression_sheet(sheet: str, genes: list[str]) -> pd.DataFrame:
    df = pd.read_excel(_MMC2, sheet_name=sheet, header=2).set_index("lncRNA")
    df.index = df.index.astype(str)
    # The screened lines are the RfxCas13d derivatives; MDA-MB-231 has only its parental
    # column in S1C.
    cols = {c: (f"{c} RfxCas13d" if f"{c} RfxCas13d" in df.columns else c) for c in _LINES}
    out = df.reindex(genes)[list(cols.values())].apply(pd.to_numeric, errors="coerce")
    out.columns = list(cols)
    return np.log1p(out.fillna(0.0).clip(lower=0.0))


def load_embeddings_all(genes: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """({name: (n_genes, d)}, {line: own-line log mRNA TPM})."""
    emb: dict[str, np.ndarray] = {}

    with open(_SEQUENCES) as fh:
        seqs = {g: s for g, (s, _) in json.load(fh).items()}
    vocab_index = {k: i for i, k in enumerate(all_kmers(4))}
    km = np.zeros((len(genes), 256), dtype=np.float32)
    for i, g in enumerate(genes):
        counts, total = _count_kmers(seqs.get(g, "").upper(), 4, vocab_index)
        for col, n in counts.items():
            km[i, col] = n / total
    emb["kmer4"] = km
    gc = [(s.count("G") + s.count("C")) / max(len(s), 1) for s in (seqs.get(g, "").upper() for g in genes)]
    ln = [np.log10(max(len(seqs.get(g, "")), 1)) for g in genes]
    gc_length = np.column_stack([gc, ln]).astype(np.float32)

    for name, f in [("dnabert2_mean", "dnabert2_transcript_full"),
                    ("dnabert2_last", "dnabert2_transcript_last"),
                    ("nt_mean", "nucleotide_transformer_transcript_full")]:
        m, index = load_embeddings(f"data/processed/{f}.npz")
        emb[name] = m[[index[g] for g in genes]]

    atlas = pd.read_csv(_ATLAS, sep="\t").set_index("lncRNA")
    atlas.index = atlas.index.astype(str)
    emb["dev_atlas"] = np.log1p(atlas.reindex(genes).fillna(0.0).to_numpy(dtype=np.float32))

    total, mrna = _expression_sheet("S1C", genes), _expression_sheet("S1E", genes)
    emb["cellline_expr"] = np.hstack([total.to_numpy(), mrna.to_numpy()]).astype(np.float32)
    emb["gc_length"] = gc_length
    emb["random"] = np.random.default_rng(0).standard_normal((len(genes), 64)).astype(np.float32)
    return emb, {c: mrna[c].to_numpy() for c in _LINES}


def neighbours(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest other rows (cosine), after scaling and PCA."""
    Z = StandardScaler().fit_transform(X)
    Z = np.nan_to_num(Z)
    if Z.shape[1] > 50:
        Z = PCA(n_components=50, random_state=0).fit_transform(Z)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)
    # Drop self; duplicates (identical vectors) can push self off position 0.
    return np.array([[j for j in row if j != i][:k] for i, row in enumerate(idx)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--out", default="eda/embedding_separation.csv")
    args = ap.parse_args()

    genes, y = load_labels()
    print(f"{len(genes):,} lncRNAs; essential per line: "
          + ", ".join(f"{c} {y[c].sum()} ({y[c].mean():.1%})" for c in _LINES))
    emb, mrna = load_embeddings_all(genes)
    rng = np.random.default_rng(0)

    rows = []
    for name, X in emb.items():
        nbr = neighbours(X, args.k)
        for c in _LINES:
            score = y[c][nbr].mean(axis=1)
            auprc = average_precision_score(y[c], score)
            null = []
            for _ in range(args.shuffles):
                p = rng.permutation(y[c])
                null.append(average_precision_score(p, p[nbr].mean(axis=1)))
            top = mrna[c] >= np.quantile(mrna[c], 0.9)
            rows.append({
                "embedding": name, "cell_line": c, "dims": X.shape[1],
                "pos_rate": round(float(y[c].mean()), 4),
                "auprc": round(float(auprc), 4),
                "enrichment": round(float(score[y[c] == 1].mean() / y[c].mean()), 3),
                "z": round(float((auprc - np.mean(null)) / np.std(null)), 1),
                "auprc_top10": round(float(average_precision_score(y[c][top], score[top])), 4),
                "pos_rate_top10": round(float(y[c][top].mean()), 4),
            })
        print(f"  {name} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    for metric in ["enrichment", "z", "auprc", "auprc_top10"]:
        print(f"\n{metric} (rows: embedding, columns: cell line)")
        print(df.pivot(index="embedding", columns="cell_line", values=metric)
              .reindex(list(emb)).to_string())
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
