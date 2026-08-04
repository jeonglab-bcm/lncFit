"""Do lncRNAs that move in step with the proliferation machinery score as essential?

This is the paper's Figure S11A, which we had not reproduced. It correlates each lncRNA's
expression against the proliferation markers PCNA and MKI67 across brain, heart, kidney and
liver development, and reports essential lncRNAs correlating more strongly than non-essential
ones.

Why it is worth a run when the developmental atlas itself came out flat (6664a1b): everything
tested so far that failed, failed the same way -- it re-measured ABUNDANCE, and cell-line TPM
already measures abundance more sharply. dev_level, dev_breadth, dev_organ and tau are all "how
much is this gene expressed". This asks something else: not how much, but whether the gene
RISES AND FALLS WITH the cell's growth program. A lncRNA tracking PCNA and MKI67 through
development is behaving like part of that program, and "essential" here means cells stop
growing without it. That is a tighter link to the label than abundance, on an axis nothing else
has occupied.

Blocks:

  prolif_all    Spearman rho of the lncRNA against PCNA and MKI67 over all 297 samples, plus
                the mean of the two (3 cols)
  prolif_organ  the same mean rho computed within each of the paper's four organs -- brain,
                heart, kidney, liver -- so a gene that tracks proliferation in one lineage but
                not others is distinguishable (4 cols)

Genes with no expression variance in a subset have undefined correlation; those are left as
NaN, which XGBoost routes natively rather than being filled with a sentinel that would sit
inside the real value range of a correlation.

Requires data/external/sarropoulos_human_lncrna_rpkm.tsv.gz and
data/external/sarropoulos_proliferation_markers.tsv.gz -- see download_sarropoulos_atlas.py.

Legal under the no-measured-depletion rule: this is baseline RNA abundance in developmental
tissues, correlated against two protein-coding genes' abundance in the same tissues. No
knockdown outcome from any cell line or day is involved.

Usage:
  python scripts/sweep_proliferation_coexpression.py --seeds 20
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

_ATLAS = REPO / "data/external/sarropoulos_human_lncrna_rpkm.tsv.gz"
_MARKERS = REPO / "data/external/sarropoulos_proliferation_markers.tsv.gz"
_COUNT = "guide_count"
_ALL, _ORGAN = "prolif_all", "prolif_organ"
_ALL_PROLIF = [_ALL, _ORGAN]

# The four organs Figure S11A uses.
_ORGANS = ["Brain", "Heart", "Kidney", "Liver"]


def _spearman_vs(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Spearman rho of every row of `mat` against `vec`. NaN where a row has no variance."""
    if mat.shape[1] < 3:
        return np.full(mat.shape[0], np.nan, dtype=np.float32)
    rv = stats.rankdata(vec)
    rv = (rv - rv.mean()) / (rv.std() or 1.0)
    rm = stats.rankdata(mat, axis=1).astype(np.float64)
    rm -= rm.mean(axis=1, keepdims=True)
    sd = rm.std(axis=1)
    out = np.full(mat.shape[0], np.nan, dtype=np.float64)
    ok = sd > 0
    out[ok] = (rm[ok] / sd[ok, None]) @ rv / len(vec)
    return out.astype(np.float32)


def load_prolif(genes: list[str]) -> dict[str, np.ndarray]:
    for p in (_ATLAS, _MARKERS):
        if not p.exists():
            raise SystemExit(f"missing {p}\nrun: python scripts/download_sarropoulos_atlas.py")

    with gzip.open(_ATLAS, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")[1:]
        rows = {}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows[parts[0]] = np.asarray(parts[1:], dtype=np.float64)

    marker = {}
    with gzip.open(_MARKERS, "rt") as fh:
        mh = fh.readline().rstrip("\n").split("\t")[2:]
        if mh != header:
            raise SystemExit("marker file sample columns do not match the atlas")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            marker[parts[1]] = np.asarray(parts[2:], dtype=np.float64)

    missing = [g for g in genes if g not in rows]
    if missing:
        print(f"  WARNING {len(missing)} genes absent from the atlas", file=sys.stderr)
    X = np.vstack([rows.get(g, np.zeros(len(header))) for g in genes])
    organ_of = np.asarray([c.split(".")[0] for c in header])

    pcna = _spearman_vs(X, marker["PCNA"])
    mki67 = _spearman_vs(X, marker["MKI67"])
    prolif_all = np.column_stack([pcna, mki67, np.nanmean([pcna, mki67], axis=0)])

    per_organ = []
    for o in _ORGANS:
        idx = np.flatnonzero(organ_of == o)
        a = _spearman_vs(X[:, idx], marker["PCNA"][idx])
        b = _spearman_vs(X[:, idx], marker["MKI67"][idx])
        per_organ.append(np.nanmean([a, b], axis=0))
    prolif_organ = np.column_stack(per_organ)

    print(f"  prolif_all   {prolif_all.shape[1]} cols, "
          f"{int(np.isnan(prolif_all).any(axis=1).sum())} genes with any NaN")
    print(f"  prolif_organ {prolif_organ.shape[1]} cols, "
          f"{int(np.isnan(prolif_organ).any(axis=1).sum())} genes with any NaN")
    return {_ALL: prolif_all.astype(np.float32), _ORGAN: prolif_organ.astype(np.float32)}


_BEST = [_ONEHOT, _TPM, _S1A_GENE] + _MODEL_NB + [_COUNT]

CONFIGS: list[tuple[str, list[str]]] = [
    ("base (current best, 40)", _BEST),
    ("+ prolif_all", _BEST + [_ALL]),
    ("+ prolif_organ", _BEST + [_ORGAN]),
    ("+ all proliferation", _BEST + _ALL_PROLIF),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/proliferation_coexpression.csv")
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
    prolif = load_prolif(genes)

    # Does the paper's own claim hold in our training labels?
    ever = np.asarray([any(table[g][c]["label"] for c in table[g]) for g in genes], dtype=bool)
    print(f"\n  Figure S11A check ({ever.sum()} genes essential in >=1 training line):")
    for name, j in [("rho vs PCNA", 0), ("rho vs MKI67", 1), ("mean of the two", 2)]:
        v = prolif[_ALL][:, j]
        ok = np.isfinite(v)
        a, b = v[ok & ever], v[ok & ~ever]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"    {name:<18} essential {a.mean():+.3f}  non {b.mean():+.3f}  "
              f"AUROC {u/(len(a)*len(b)):.3f}  p {p:.2e}")

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
        for key in _ALL_PROLIF:
            if key in blocks:
                cols.append(prolif[key])
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
    print(f"{'config':<26} {'AUPRC':>7} {'vs base':>9} {'wins':>8}  per-fold delta")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        if name == ref_name:
            print(f"{name:<26} {mean:>7.4f}")
            continue
        w = sum(1 for v in paired.values() if v[name] > v[ref_name])
        ds = [np.mean([v[name] for (h, _), v in paired.items() if h == c])
              - np.mean([v[ref_name] for (h, _), v in paired.items() if h == c]) for c in cells]
        flip = "  SIGN FLIP" if min(ds) < 0 < max(ds) else ""
        print(f"{name:<26} {mean:>7.4f} {mean - ref:>+9.4f} {f'{w}/{len(paired)}':>8}  "
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
