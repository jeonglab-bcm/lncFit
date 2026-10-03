"""SSL step 1: assemble every lncRNA's views for contrastive pretraining.

One row per lncRNA in data/raw/human.lncRNA.hg19.gtf (31,678 -- the screen covers only
5,496 of them). Views, each kept separate so later steps can pair any two:

  seq       normalised 3-, 4- and 5-mer frequencies of the longest spliced transcript
            (64 + 256 + 1024 = 1,344 dims)
  cellline  log1p mRNA-seq TPM in the 5 screened lines, mmc2 S1E (5 dims)
  atlas     log1p RPKM in 297 developmental samples, Sarropoulos et al. 2019 (297 dims)

plus, per lncRNA, `has_<view>` coverage masks, GC fraction and log10 length (controls, never
model inputs), and `screened` with per-line `label_<line>` for the 5,496 screened lncRNAs.
Labels are for evaluation only; pretraining must not read them.

The atlas view needs scripts/download_sarropoulos_atlas.py --all-lncrnas; if that file is
missing, the atlas view is left out and the other two are built.
Output (git-ignored, regenerable): data/processed/ssl_views.npz

Usage:
  uv run python contrastive/build_views.py
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from lncfit.features import _count_kmers, all_kmers  # noqa: E402

_SEQUENCES = "data/processed/body_sequences_transcript.json"
_MMC2 = "data/raw/mmc2.xlsx"
_ATLAS = "data/external/sarropoulos_all_lncrna_rpkm.tsv.gz"
_LABELS = "data/processed/lncrna_rra_day14.jsonl.gz"
_OUT = "data/processed/ssl_views.npz"
_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_KS = (3, 4, 5)


def seq_view(genes: list[str], seqs: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    for k in _KS:
        vocab = {km: i for i, km in enumerate(all_kmers(k))}
        X = np.zeros((len(genes), len(vocab)), dtype=np.float32)
        for r, g in enumerate(genes):
            counts, total = _count_kmers(seqs.get(g, ""), k, vocab)
            for col, n in counts.items():
                X[r, col] = n / total
        blocks.append(X)
    has = np.array([len(seqs.get(g, "")) >= max(_KS) for g in genes])
    return np.hstack(blocks), has


def cellline_view(genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(_MMC2, sheet_name="S1E", header=2).set_index("lncRNA")
    df.index = df.index.astype(str)
    df = df[~df.index.duplicated()]
    cols = [f"{c} RfxCas13d" for c in _LINES]
    has = np.array([g in df.index for g in genes])
    X = df.reindex(genes)[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0)
    return np.log1p(X.to_numpy(dtype=np.float32)), has


def atlas_view(genes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(_ATLAS, sep="\t").set_index("lncRNA")
    df.index = df.index.astype(str)
    has = np.array([g in df.index for g in genes])
    X = df.reindex(genes).fillna(0.0).clip(lower=0)
    return np.log1p(X.to_numpy(dtype=np.float32)), has, list(df.columns)


def labels(genes: list[str]) -> dict[str, np.ndarray]:
    pos = {g: i for i, g in enumerate(genes)}
    out = {c: np.full(len(genes), -1, dtype=np.int8) for c in _LINES}
    with gzip.open(_LABELS, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["cell_line"]][pos[r["target"]]] = int(r["label"])
    return out


def main() -> None:
    with open(_SEQUENCES) as fh:
        seqs = {g: s.upper() for g, (s, _) in json.load(fh).items()}
    genes = sorted(seqs)

    seq, has_seq = seq_view(genes, seqs)
    cl, has_cl = cellline_view(genes)
    if Path(_ATLAS).exists():
        atlas, has_atlas, atlas_cols = atlas_view(genes)
    else:
        print(f"note: {_ATLAS} missing -- building without the atlas view")
        atlas, has_atlas, atlas_cols = np.zeros((len(genes), 0), np.float32), np.zeros(len(genes), bool), []
    lab = labels(genes)
    screened = lab[_LINES[0]] >= 0
    gc = np.array([(seqs[g].count("G") + seqs[g].count("C")) / max(len(seqs[g]), 1)
                   for g in genes], dtype=np.float32)
    length = np.log10(np.maximum([len(seqs[g]) for g in genes], 1)).astype(np.float32)

    np.savez_compressed(
        _OUT, genes=np.array(genes), seq=seq.astype(np.float16), cellline=cl, atlas=atlas,
        atlas_samples=np.array(atlas_cols), has_seq=has_seq, has_cellline=has_cl,
        has_atlas=has_atlas, gc=gc, log10_length=length, screened=screened, lines=np.array(_LINES),
        **{f"label_{c}": lab[c] for c in _LINES})

    both = has_seq & has_cl
    print(f"{len(genes):,} lncRNAs")
    for name, h in [("seq", has_seq), ("cellline", has_cl), ("atlas", has_atlas),
                    ("seq + cellline", both)]:
        print(f"  {name:<16} {h.sum():>6,}   of screened {h[screened].sum():>5,}/{screened.sum():,}")
    print(f"  seq + cellline, not screened (extra pretraining data): {(both & ~screened).sum():,}")
    print(f"-> {_OUT} ({Path(_OUT).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
