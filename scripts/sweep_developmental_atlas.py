"""Does raw developmental expression add anything over the atlas summaries we already have?

The paper's signature for essential lncRNAs is that they are "highly expressed early and
broadly in development". The model already carries this atlas indirectly: S1A's `Tissue tau`,
`Time tau`, `Dynamic` and `Count dynamic tissues` are all computed from it. So the question is
not "does development matter" -- it is what the summaries LEAVE OUT. Two things:

  magnitude   tau measures how UNEVEN a profile is, not how high it sits. A lncRNA expressed
              at RPKM 0.1 in one organ and one expressed at RPKM 100 in one organ have the
              same tau. "Highly expressed" is not represented anywhere in the model.
  direction   `Time tau` says a gene's expression is concentrated in time; it does not say
              WHEN. "Early" is a direction, and tau is unsigned.

Blocks, each testable on its own:

  dev_level    magnitude across all 297 samples: mean, max, p90, sd of log1p(RPKM)
  dev_breadth  "broadly": how many of the 7 organs express it, and what fraction of samples
  dev_timing   "early": prenatal mean, postnatal mean, their difference, and where the peak
               sits on a normalised developmental axis
  dev_organ    per-organ median log1p(RPKM), 7 columns

Requires data/external/sarropoulos_human_lncrna_rpkm.tsv.gz -- see
scripts/download_sarropoulos_atlas.py. All 5,496 targets are covered exactly.

Legal under the no-measured-depletion rule: baseline RNA abundance in developmental tissues,
not a knockdown outcome from any cell line or day.

Usage:
  python scripts/sweep_developmental_atlas.py --seeds 20
"""
import argparse
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
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
_COUNT = "guide_count"
_LEVEL, _BREADTH, _TIMING, _ORGAN = "dev_level", "dev_breadth", "dev_timing", "dev_organ"
_ALL_DEV = [_LEVEL, _BREADTH, _TIMING, _ORGAN]

_ORGANS = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]

# Postnatal stages in developmental order. Prenatal samples are named "<n>wpc" (weeks post
# conception) and sort numerically ahead of all of these.
_POSTNATAL = ["newborn", "infant", "toddler", "school", "youngteenager", "teenager",
              "oldteenager", "youngadult", "youngmidage", "oldermidage", "senior"]

# RPKM 1 as the "expressed" threshold, matching the convention the source paper uses for
# calling a lncRNA expressed in an organ.
_EXPRESSED = 1.0


def _stage_rank(stage: str) -> float:
    """Position on a single developmental axis: prenatal weeks first, then postnatal order."""
    s = stage.lower()
    if s.endswith("wpc"):
        return float(s[:-3])                      # 4..20, all below the postnatal block
    return 100.0 + _POSTNATAL.index(s)            # keeps prenatal < postnatal


def load_atlas(genes: list[str]) -> dict[str, np.ndarray]:
    """Build the four developmental blocks, row-aligned to `genes`."""
    if not _ATLAS.exists():
        raise SystemExit(f"missing {_ATLAS}\nrun: python scripts/download_sarropoulos_atlas.py")

    with gzip.open(_ATLAS, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")[1:]
        rows = {}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows[parts[0]] = np.asarray(parts[1:], dtype=np.float64)

    organ_of = np.asarray([c.split(".")[0] for c in header])
    rank_of = np.asarray([_stage_rank(c.split(".")[1]) for c in header])
    # Normalise the developmental axis to 0..1 so "where is the peak" is comparable.
    span = rank_of.max() - rank_of.min()
    norm_rank = (rank_of - rank_of.min()) / span
    prenatal = np.asarray([c.split(".")[1].lower().endswith("wpc") for c in header])
    organ_cols = {o: np.flatnonzero(organ_of == o) for o in _ORGANS}

    missing = [g for g in genes if g not in rows]
    if missing:
        print(f"  WARNING {len(missing)} genes absent from the atlas, filled with 0",
              file=sys.stderr)

    n = len(genes)
    level = np.zeros((n, 4), dtype=np.float32)
    breadth = np.zeros((n, 2), dtype=np.float32)
    timing = np.zeros((n, 4), dtype=np.float32)
    organ = np.zeros((n, len(_ORGANS)), dtype=np.float32)

    for i, g in enumerate(genes):
        rpkm = rows.get(g)
        if rpkm is None:
            continue
        lg = np.log1p(rpkm)

        level[i] = (lg.mean(), lg.max(), np.percentile(lg, 90), lg.std())

        organ_med = np.asarray([np.median(lg[idx]) for idx in organ_cols.values()])
        organ[i] = organ_med
        organ_expressed = np.asarray([np.median(rpkm[idx]) > _EXPRESSED
                                      for idx in organ_cols.values()])
        breadth[i] = (organ_expressed.sum(), float((rpkm > _EXPRESSED).mean()))

        pre, post = lg[prenatal].mean(), lg[~prenatal].mean()
        # Peak position on the normalised axis; ties take the earliest, which is the
        # conservative reading of "expressed early".
        peak = norm_rank[int(np.argmax(lg))]
        timing[i] = (pre, post, pre - post, peak)

    return {_LEVEL: level, _BREADTH: breadth, _TIMING: timing, _ORGAN: organ}


_BEST = [_ONEHOT, _TPM, _S1A_GENE] + _MODEL_NB + [_COUNT]

CONFIGS: list[tuple[str, list[str]]] = [
    ("base (current best, 40)", _BEST),
    ("+ dev_level", _BEST + [_LEVEL]),
    ("+ dev_breadth", _BEST + [_BREADTH]),
    ("+ dev_timing", _BEST + [_TIMING]),
    ("+ dev_organ", _BEST + [_ORGAN]),
    ("+ all developmental", _BEST + _ALL_DEV),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/developmental_atlas.csv")
    args = ap.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}", flush=True)

    total, mrna = load_tpm()
    guide, guide_names = load_guides(genes)
    s1a_gene, nb_dist, nb_class, _nb_depmap, nb_expr, _tissue = load_blocks(genes)
    guide_count = guide[:, [guide_names.index(_COUNT)]]
    dev = load_atlas(genes)
    for k, v in dev.items():
        print(f"  {k:<12} {v.shape[1]} cols", flush=True)

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
        for key in _ALL_DEV:
            if key in blocks:
                cols.append(dev[key])
        return np.hstack(cols).astype(np.float32)

    rows, results = [], defaultdict(list)
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)

    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---", flush=True)
        for name, blocks in CONFIGS:
            X_eval = make(holdout, blocks)
            X_train = np.vstack([make(c, blocks) for c in train_cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])

            auprcs, aurocs = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
                    colsample_bytree=0.3, tree_method="hist", objective="binary:logistic",
                    eval_metric="aucpr", scale_pos_weight=1.0, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_eval)[:, 1]
                auprcs.append(float(average_precision_score(y_eval, p)))
                aurocs.append(float(roc_auc_score(y_eval, p)))
                results[name].append(auprcs[-1])
                paired[(holdout, seed)][name] = auprcs[-1]
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<26} ({X_eval.shape[1]:>3} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    ref_name = CONFIGS[0][0]
    ref = float(np.mean(results[ref_name]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<26} {'AUPRC':>7} {'sd':>7} {'vs base':>9} {'wins':>8}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        if name == ref_name:
            note, wins = "", ""
        else:
            note = f"{mean - ref:+.4f}"
            w = sum(1 for v in paired.values() if v[name] > v[ref_name])
            wins = f"{w}/{len(paired)}"
        print(f"{name:<26} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>9} {wins:>8}")

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
