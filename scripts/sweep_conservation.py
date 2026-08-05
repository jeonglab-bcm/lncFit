"""Does sequence conservation add anything over `Age`?

Not one of the paper's analyses -- this is ours. The paper's only evolutionary measure is S1A's
`Age`, a 7-category estimate of when a lncRNA first appeared, tested by Fisher's exact in
Figure 2A ("fewer human-specific transcripts than expected and a larger fraction of older
(180 Mya) transcripts").

The reason to try it: `Age` is the only block in this project whose contribution keeps the same
sign on every fold (sweep_age_ablation.py, +0.0058 / +0.0034 / +0.0001, 42/60 paired wins).
Everything else that looked promising -- the developmental atlas, proliferation co-expression --
flipped sign across folds. So the axis Age sits on is the one worth measuring better.

Age and conservation are related but distinct. Age asks whether a recognisable copy of the gene
exists in another species: presence or absence, one coarse label per gene, and unassignable for
384 of our targets ("multimember_family", "ambiguous"). Conservation asks how strongly selection
has held the actual sequence in place. A gene can be old but drifting freely, or young but
tightly constrained, and Age cannot separate those.

Features, from UCSC phastCons 100-way conserved elements intersected with each lncRNA's exons:

  cons_frac      fraction of exonic bp inside a conserved element -- the main measure, and
                 length-normalised, which matters because transcript length is already a feature
  cons_n         number of distinct elements overlapping the exons
  cons_max       highest element score (0-1000) overlapping
  cons_mean      element score averaged over overlapping elements, weighted by overlapped bp
  cons_maxlod    highest log-odds score overlapping

A caveat on the measurement, not the biology: elements are a thresholded call, so this is
coarser than per-base phyloP. pyBigWig will not build on this box and UCSC ships no aarch64
binaries, so the per-base tracks are unreadable here. A null result should therefore be read as
"conserved-element overlap does not help", not as closing out conservation generally.

Requires data/external/phastcons_elements_hg19_100way.tsv.gz -- see
scripts/download_phastcons_elements.py.

Legal under the no-measured-depletion rule: this is computed from genomic sequence alignment
across species. No knockdown outcome is involved.

Usage:
  python scripts/sweep_conservation.py --seeds 20
"""
import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sweep_gene_level_priors import load_gene_table
from sweep_neighbour_features import (_MODEL_NB, _NB_CLASS, _NB_DIST, _NB_EXPR, _ONEHOT,
                                      _S1A_GENE, _TPM, load_blocks)
from sweep_prescreen_features import load_guides
from sweep_tpm_features import _ALL_CELLS, _TRAIN, load_tpm, tpm_block

_ELEMENTS = REPO / "data/external/phastcons_elements_hg19_100way.tsv.gz"
_GTF = REPO / "data/raw/human.lncRNA.hg19.gtf"
_COUNT = "guide_count"
_CONS = "conservation"
_COLS = ["cons_frac", "cons_n", "cons_max", "cons_mean", "cons_maxlod"]


def _gene_exons(genes: set[str]) -> dict[str, tuple[str, list[tuple[int, int]]]]:
    """gene_id -> (UCSC chrom, merged 0-based half-open exon intervals)."""
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    chrom_of: dict[str, str] = {}
    with open(_GTF) as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            gid = f[8].split('gene_id', 1)[1].split(";")[0].strip().strip('"')
            if gid not in genes:
                continue
            raw[gid].append((int(f[3]) - 1, int(f[4])))
            chrom_of[gid] = f"chr{f[0]}"
    out = {}
    for gid, iv in raw.items():
        iv.sort()
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[gid] = (chrom_of[gid], [(s, e) for s, e in merged])
    return out


def load_conservation(genes: list[str]) -> np.ndarray:
    if not _ELEMENTS.exists():
        raise SystemExit(f"missing {_ELEMENTS}\n"
                         f"run: python scripts/download_phastcons_elements.py")

    starts: dict[str, list[int]] = defaultdict(list)
    ends: dict[str, list[int]] = defaultdict(list)
    scores: dict[str, list[float]] = defaultdict(list)
    lods: dict[str, list[float]] = defaultdict(list)
    with gzip.open(_ELEMENTS, "rt") as fh:
        fh.readline()
        for line in fh:
            c, s, e, lod, sc = line.rstrip("\n").split("\t")
            starts[c].append(int(s)); ends[c].append(int(e))
            scores[c].append(float(sc)); lods[c].append(float(lod) if lod else 0.0)
    arr = {c: (np.asarray(starts[c]), np.asarray(ends[c]),
               np.asarray(scores[c]), np.asarray(lods[c])) for c in starts}
    print(f"  {sum(len(v[0]) for v in arr.values()):,} elements on {len(arr)} chromosomes")

    exons = _gene_exons(set(genes))
    print(f"  exons found for {len(exons):,}/{len(genes):,} targets")

    out = np.zeros((len(genes), len(_COLS)), dtype=np.float32)
    for i, g in enumerate(genes):
        ex = exons.get(g)
        if ex is None:
            continue
        chrom, iv = ex
        a = arr.get(chrom)
        exonic = sum(e - s for s, e in iv)
        if a is None or exonic == 0:
            continue
        es, ee, esc, elod = a
        covered, hit, wsum, wtot, mx, mxlod = 0, set(), 0.0, 0, 0.0, 0.0
        for s, e in iv:
            lo = int(np.searchsorted(ee, s, side="right"))
            hi = int(np.searchsorted(es, e, side="left"))
            for j in range(lo, hi):
                ov = min(e, ee[j]) - max(s, es[j])
                if ov <= 0:
                    continue
                covered += ov
                hit.add(j)
                wsum += esc[j] * ov
                wtot += ov
                mx = max(mx, esc[j])
                mxlod = max(mxlod, elod[j])
        out[i] = (covered / exonic, len(hit), mx, wsum / wtot if wtot else 0.0, mxlod)
    return out


_BEST = [_ONEHOT, _TPM, _S1A_GENE] + _MODEL_NB + [_COUNT]

CONFIGS: list[tuple[str, list[str]]] = [
    ("base (current best, 40)", _BEST),
    ("+ conservation", _BEST + [_CONS]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/conservation.csv")
    args = ap.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int) for c in cells}
    print(f"{len(genes):,} genes, folds: {', '.join(cells)}", flush=True)

    total, mrna = load_tpm()
    guide, guide_names = load_guides(genes)
    s1a_gene, nb_dist, nb_class, _dep, nb_expr, _t = load_blocks(genes)
    guide_count = guide[:, [guide_names.index(_COUNT)]]
    cons = load_conservation(genes)

    ever = np.asarray([any(table[g][c]["label"] for c in table[g]) for g in genes], dtype=bool)
    print(f"\n  univariate vs ever-essential ({ever.sum()} genes):")
    for j, name in enumerate(_COLS):
        v = cons[:, j]
        a, b = v[ever], v[~ever]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"    {name:<12} essential {a.mean():8.3f}  non {b.mean():8.3f}  "
              f"AUROC {u/(len(a)*len(b)):.3f}  p {p:.2e}")

    # Is it just re-measuring Age? Age one-hot sits at s1a_gene columns 7-13.
    age_idx = np.argmax(s1a_gene[:, 7:14], axis=1)
    print(f"\n  cons_frac by Age category (rho with age-order): ", end="")
    print(f"{stats.spearmanr(cons[:, 0], age_idx).statistic:+.3f}")

    def make(cell_line: str, blocks: list[str]) -> np.ndarray:
        cols = []
        if _ONEHOT in blocks:
            oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
            oh[:, _ALL_CELLS.index(cell_line)] = 1.0
            cols.append(oh)
        if _TPM in blocks:
            cols.append(tpm_block(genes, cell_line, total, mrna)[0])
        if _S1A_GENE in blocks:
            cols.append(s1a_gene)
        if _NB_DIST in blocks:
            cols.append(nb_dist)
        if _NB_CLASS in blocks:
            cols.append(nb_class)
        if _NB_EXPR in blocks:
            cols.append(nb_expr[cell_line])
        if _COUNT in blocks:
            cols.append(guide_count)
        if _CONS in blocks:
            cols.append(cons)
        return np.hstack(cols).astype(np.float32)

    rows, results = [], defaultdict(list)
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)

    for holdout in cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---", flush=True)
        for name, blocks in CONFIGS:
            X_eval = make(holdout, blocks)
            X_train = np.vstack([make(c, blocks) for c in cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in cells if c != holdout])

            aps, aus = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                    colsample_bytree=0.3, tree_method="hist", objective="binary:logistic",
                    eval_metric="aucpr", scale_pos_weight=1.0, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_eval)[:, 1]
                aps.append(float(average_precision_score(y_eval, p)))
                aus.append(float(roc_auc_score(y_eval, p)))
                results[name].append(aps[-1])
                paired[(holdout, seed)][name] = aps[-1]
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1], "auroc": round(aus[-1], 4),
                             "auprc": round(aps[-1], 4)})
            print(f"  {name:<26} ({X_eval.shape[1]:>3} cols)  AUROC {np.mean(aus):.4f}  "
                  f"AUPRC {np.mean(aps):.4f} +/- {np.std(aps):.4f}", flush=True)

    ref_name = CONFIGS[0][0]
    ref = float(np.mean(results[ref_name]))
    print(f"\n=== mean across {len(cells)} folds x {args.seeds} seeds ===")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        if name == ref_name:
            print(f"{name:<26} {mean:.4f}")
            continue
        w = sum(1 for v in paired.values() if v[name] > v[ref_name])
        ds = [np.mean([v[name] for (h, _), v in paired.items() if h == c])
              - np.mean([v[ref_name] for (h, _), v in paired.items() if h == c]) for c in cells]
        flip = "  SIGN FLIP" if min(ds) < 0 < max(ds) else "  sign consistent"
        print(f"{name:<26} {mean:.4f}  {mean - ref:+.4f}  {w}/{len(paired)}  "
              + "  ".join(f"{c}: {d:+.4f}" for c, d in zip(cells, ds)) + flip)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["holdout", "config", "seed", "n_features",
                                           "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")


if __name__ == "__main__":
    main()
