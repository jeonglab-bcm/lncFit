"""EDA 3: how many more lncRNAs do the screen's guides reach if mismatches are allowed?

The Cas13 library (mmc2 S1B, 56,322 guides of 23 nt) was designed against 5,496 lncRNAs, and
only those have an essentiality label. A guide can still bind a transcript it was not designed
for if the two differ by a few bases. This counts, for every lncRNA in the annotation (31,678),
which library guides match its transcript with at most 0, 1 or 2 mismatches, so the number of
lncRNAs that could be pulled into the labelled set can be read off for each mismatch budget.

Matching: each guide is reverse-complemented to its target site (Cas13 binds the RNA the guide
is complementary to) and searched against every transcript (data/processed/
body_sequences_transcript.json, RNA sense, spliced). The search is exact by construction: a
23-mer with at most 2 mismatches must contain one of three non-overlapping pieces (8, 8 and 7
nt) with no mismatch, so each piece is looked up in a sorted index of all 8-mers / 7-mers of the
transcripts and every candidate is then checked over all 23 positions. Windows containing a base
other than A, C, G or T, or crossing the border between two transcripts, are ignored.

Only guides designed for ESSENTIAL lncRNAs matter for pulling in new positives, so every count
is also given for that subset: guides whose parent lncRNA is essential in at least one of the five
cell lines (label from data/processed/lncrna_rra_day14.jsonl.gz), and per cell line. Each
(guide, hit lncRNA) pair is written to eda/guide_mismatch_pairs.csv with the parent and the
cell lines in which the parent is essential, so a label could be inherited downstream.

Per lncRNA: the lowest mismatch count reached by any guide that was NOT designed for it
(`min_mm_offtarget`), how many such guides sit at <= 1 and <= 2 mismatches, and for the
screened lncRNAs the same for their own designed guides as a check (expected: 0 mismatches).

Usage:
  uv run python eda/guide_mismatch.py
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_MMC2 = "data/raw/mmc2.xlsx"
_SEQUENCES = "data/processed/body_sequences_transcript.json"
_OUT = "eda/guide_mismatch_hits.csv"
_PAIRS = "eda/guide_mismatch_pairs.csv"
_LABELS = "data/processed/lncrna_rra_day14.jsonl.gz"
_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
_CODE = np.full(256, 4, dtype=np.uint8)
for i, b in enumerate(b"ACGT"):
    _CODE[b] = i
_PIECES = [(0, 8), (8, 8), (16, 7)]   # (offset, length) inside the 23-mer
_W = 23
_MAX_MM = 2


def keys(T: np.ndarray, L: int) -> np.ndarray:
    """4^L-valued key of every length-L window of T; -1 where the window has a non-ACGT base."""
    n = len(T) - L + 1
    k = np.zeros(n, dtype=np.int32)
    bad = np.zeros(n, dtype=bool)
    for i in range(L):
        w = T[i:i + n]
        bad |= w >= 4
        k = k * 4 + np.minimum(w, 3)
    k[bad] = -1
    return k


def main() -> None:
    t0 = time.time()
    with open(_SEQUENCES) as fh:
        raw = json.load(fh)
    genes = sorted(raw)
    lens = np.array([len(raw[g][0]) for g in genes])
    offsets = np.concatenate([[0], np.cumsum(lens + 1)])          # +1: separator between genes
    T = np.full(offsets[-1], 5, dtype=np.uint8)                   # 5 = separator
    for g, o in zip(genes, offsets):
        s = raw[g][0].upper().encode()
        T[o:o + len(s)] = _CODE[np.frombuffer(s, dtype=np.uint8)]
    print(f"{len(genes):,} transcripts, {len(T) / 1e6:.1f} Mb; "
          f"{(T == 4).sum():,} non-ACGT bases", flush=True)

    s1b = pd.read_excel(_MMC2, sheet_name="S1B", header=2)
    design = s1b[s1b.columns[1]].astype(str).to_numpy()
    seq = s1b[s1b.columns[2]].astype(str).str.strip().str.upper().to_numpy()
    ok = np.array([len(s) == _W and set(s) <= set("ACGT") for s in seq])
    print(f"{len(seq):,} guides, {ok.sum():,} usable (23 nt, ACGT only)")
    design, seq = design[ok], seq[ok]
    comp = str.maketrans("ACGT", "TGCA")
    site = np.stack([_CODE[np.frombuffer(s.translate(comp)[::-1].encode(), dtype=np.uint8)]
                     for s in seq])                                # (n_guides, 23) target site
    gene_idx = {g: i for i, g in enumerate(genes)}
    parent = np.array([gene_idx.get(d, -1) for d in design])

    # Sorted index of every 8-mer and 7-mer; a guide piece is found by binary search.
    index = {}
    for L in {l for _, l in _PIECES}:
        k = keys(T, L)
        order = np.argsort(k, kind="stable").astype(np.int64)
        index[L] = (k[order], order)
    print(f"index built ({time.time() - t0:.0f} s)", flush=True)

    best = {}       # (guide, gene) -> lowest mismatches, only <= _MAX_MM
    n_cand = 0
    batch = 500
    arange = np.arange(_W)
    for b0 in range(0, len(seq), batch):
        gi_all, st_all = [], []
        for off, L in _PIECES:
            sk, order = index[L]
            piece = np.zeros(min(batch, len(seq) - b0), dtype=np.int64)
            for i in range(L):
                piece = piece * 4 + site[b0:b0 + len(piece), off + i]
            lo, hi = np.searchsorted(sk, piece, "left"), np.searchsorted(sk, piece, "right")
            cnt = hi - lo
            gi = np.repeat(np.arange(len(piece)) + b0, cnt)
            pos = order[np.concatenate([np.arange(a, c) for a, c in zip(lo, hi)])] \
                if cnt.sum() else np.empty(0, dtype=np.int64)
            gi_all.append(gi); st_all.append(pos - off)
        gi = np.concatenate(gi_all); st = np.concatenate(st_all)
        keep = (st >= 0) & (st + _W <= len(T))
        gi, st = gi[keep], st[keep]
        n_cand += len(gi)
        # Verify in chunks so the (candidates x 23) window matrix stays small.
        for c0 in range(0, len(gi), 2_000_000):
            g_, s_ = gi[c0:c0 + 2_000_000], st[c0:c0 + 2_000_000]
            win = T[s_[:, None] + arange]
            mm = (win != site[g_]).sum(1)
            good = (mm <= _MAX_MM) & (win < 4).all(1)
            g_, s_, mm = g_[good], s_[good], mm[good]
            tg = np.searchsorted(offsets, s_, side="right") - 1
            for a, t, m in zip(g_, tg, mm):
                if (a, t) not in best or m < best[(a, t)]:
                    best[(a, t)] = int(m)
        if (b0 // batch) % 20 == 0:
            print(f"  guides {b0 + batch:,}/{len(seq):,}  candidates {n_cand:,}  "
                  f"hits {len(best):,}  ({time.time() - t0:.0f} s)", flush=True)

    hits = pd.DataFrame([(a, t, m) for (a, t), m in best.items()],
                        columns=["guide", "gene", "mm"])
    hits["own"] = parent[hits.guide.to_numpy()] == hits.gene.to_numpy()

    own = hits[hits.own].groupby("gene").mm.min()
    off = hits[~hits.own]
    per = pd.DataFrame(index=pd.Index(genes, name="lncRNA"))
    per["screened"] = np.isin(np.arange(len(genes)), parent[parent >= 0])
    per["own_guide_min_mm"] = own.reindex(range(len(genes))).to_numpy()
    g = off.groupby("gene")
    per["min_mm_offtarget"] = g.mm.min().reindex(range(len(genes))).to_numpy()
    for m in range(_MAX_MM + 1):
        per[f"n_guides_le{m}"] = (off[off.mm <= m].groupby("gene").size()
                                  .reindex(range(len(genes))).fillna(0).astype(int).to_numpy())
    per.to_csv(_OUT)

    # Essential parents: label per (gene, line); a guide inherits its parent's lines.
    ess = np.zeros((len(genes), len(_LINES)), dtype=bool)
    with gzip.open(_LABELS, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if int(r["label"]) == 1 and r["target"] in gene_idx:
                ess[gene_idx[r["target"]], _LINES.index(r["cell_line"])] = True
    guide_ess = np.where((parent >= 0)[:, None], ess[np.maximum(parent, 0)], False)
    pairs = off.assign(parent=[genes[i] for i in parent[off.guide.to_numpy()]],
                       hit=[genes[i] for i in off.gene.to_numpy()],
                       parent_essential_in=[",".join(l for l, e in zip(_LINES, guide_ess[a]) if e)
                                            for a in off.guide.to_numpy()])
    pairs[["parent", "hit", "mm", "parent_essential_in"]].to_csv(_PAIRS, index=False)

    sc = per.screened.to_numpy()
    print(f"\nown designed guides, screened lncRNAs: exact match for "
          f"{(hits[hits.own].groupby('guide').mm.min() == 0).sum():,}/{(parent >= 0).sum():,} guides")
    print(f"\nlncRNAs reached by at least one guide designed for ANOTHER lncRNA "
          f"(unscreened = no label today; screened = already labelled):")
    print(f"{'mismatches allowed':<22}{'unscreened (26,182)':>22}{'screened (5,496)':>20}"
          f"{'unscreened, >=2 guides':>26}")
    for m in range(_MAX_MM + 1):
        c = per[f"n_guides_le{m}"].to_numpy()
        print(f"{'<= ' + str(m):<22}{(~sc & (c >= 1)).sum():>22,}{(sc & (c >= 1)).sum():>20,}"
              f"{(~sc & (c >= 2)).sum():>26,}")
    ess_any = guide_ess.any(1)
    print(f"\nONLY guides designed for essential lncRNAs "
          f"({ess_any.sum():,} guides, {np.unique(parent[ess_any]).size:,} lncRNAs essential in "
          f">= 1 line); lncRNAs reached that are unscreened / already screened:")
    print(f"{'subset':<24}" + "".join(f"{'<=' + str(m):>14}" for m in range(_MAX_MM + 1)))
    subsets = [("essential in >= 1 line", ess_any)] + [(l, guide_ess[:, i]) for i, l in enumerate(_LINES)]
    for name, mask in subsets:
        row = off[mask[off.guide.to_numpy()]]
        cells = []
        for m in range(_MAX_MM + 1):
            r = row[row.mm <= m]
            u = np.unique(r.gene[~sc[r.gene.to_numpy()]]).size
            k = np.unique(r.gene[sc[r.gene.to_numpy()]]).size
            cells.append(f"{u} / {k}")
        print(f"{name:<24}" + "".join(f"{c:>14}" for c in cells))
    print(f"\nunscreened lncRNAs in the annotation: {(~sc).sum():,}; "
          f"checked {n_cand:,} candidate windows in {time.time() - t0:.0f} s")
    print(f"-> {_OUT}, {_PAIRS}")


if __name__ == "__main__":
    main()
