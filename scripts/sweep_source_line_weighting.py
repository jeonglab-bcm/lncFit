"""Is one training cell line a better source than the others, and does upweighting it help?

Upstream #115 reports that MDA-MB-231 is the strongest single predictor of THP1 (AUPRC 0.2133
against HAP1 0.1581 and K562 0.1019), attributes it to screen quality rather than lineage, and
says plainly that "weighting the three training lines equally is a mistake worth knowing
about". The developmental-atlas sweep then found its entire gain on the MDA-MB-231 fold. So the
question is whether to weight MDA-MB-231 heavier.

Two traps, and this script exists to avoid both.

1. Selecting on THP1. #115's 0.2133 is measured against THP1's labels. Choosing weights
   because of it is tuning on the test set, which docs/PARTICIPATE.md forbids ("pick your model
   with CV on the training cell lines"). But #115 also states the ordering holds for EVERY
   target line, which is checkable using training lines alone. Part 1 does exactly that: a
   source x target AUPRC matrix over the three training lines, no THP1 involved. If
   MDA-MB-231 is the best source for HAP1 and for K562, the premise stands on its own evidence.

2. Class balance masquerading as similarity. This repo already shipped-then-withdrew a
   Celligner similarity weighting when a posrate control with no similarity term reproduced the
   gain exactly -- the gain was class balance. That risk is acute here: MDA-MB-231 has the
   FEWEST positives (157, 2.9%) and K562 the most (401, 7.3%), so upweighting MDA-MB-231
   necessarily lowers the effective positive rate. Part 2 therefore runs, alongside each
   weighting, a control that reproduces the SAME effective positive rate by reweighting
   positives globally, using no line identity at all. If the control matches, the weighting is
   not about similarity.

A structural caveat that no amount of seeds fixes: under LOCO, when MDA-MB-231 is the held-out
target it is absent from training, so "upweight MDA-MB-231" is a no-op on that fold. Only the
HAP1 and K562 folds test the weighting, i.e. independent n = 2. For THP1 all three lines are
present, so the setting we would ship is not the setting we can validate. Reported explicitly
rather than averaged away.

Usage:
  python scripts/sweep_source_line_weighting.py --seeds 20
"""
import argparse
import csv
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

_COUNT = "guide_count"
_BEST = [_ONEHOT, _TPM, _S1A_GENE] + _MODEL_NB + [_COUNT]
_FAVOURED = "MDA-MB-231"
_WEIGHTS = [2.0, 3.0, 5.0]


def _model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
        colsample_bytree=0.3, tree_method="hist", objective="binary:logistic",
        eval_metric="aucpr", scale_pos_weight=1.0, random_state=seed, n_jobs=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/source_line_weighting.csv")
    args = ap.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int) for c in cells}
    print(f"{len(genes):,} genes, training lines: {', '.join(cells)}")
    for c in cells:
        print(f"  {c:<12} {labels[c].sum():4d} positives ({100*labels[c].mean():.2f}%)")

    total, mrna = load_tpm()
    guide, guide_names = load_guides(genes)
    s1a_gene, nb_dist, nb_class, _dep, nb_expr, _t = load_blocks(genes)
    guide_count = guide[:, [guide_names.index(_COUNT)]]

    def make(cell_line: str) -> np.ndarray:
        cols = []
        oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
        oh[:, _ALL_CELLS.index(cell_line)] = 1.0
        cols.append(oh)
        cols.append(tpm_block(genes, cell_line, total, mrna)[0])
        cols.append(s1a_gene)
        cols.append(nb_dist)
        cols.append(nb_class)
        cols.append(nb_expr[cell_line])
        cols.append(guide_count)
        return np.hstack(cols).astype(np.float32)

    X = {c: make(c) for c in cells}
    rows = []

    # ---- Part 1: single-source transfer, training lines only. No THP1 anywhere. ----
    print("\n=== Part 1: single-source transfer (rows = source, cols = target) ===")
    single = defaultdict(dict)
    for src in cells:
        for tgt in cells:
            if src == tgt:
                continue
            aps = []
            for seed in range(args.seeds):
                m = _model(seed).fit(X[src], labels[src])
                p = m.predict_proba(X[tgt])[:, 1]
                aps.append(float(average_precision_score(labels[tgt], p)))
                rows.append({"part": "single_source", "source": src, "target": tgt,
                             "weight": 1.0, "seed": seed, "auprc": round(aps[-1], 4),
                             "auroc": ""})
            single[src][tgt] = float(np.mean(aps))
    hdr = f"{'source':<14}" + "".join(f"{t:>14}" for t in cells) + f"{'mean':>10}"
    print(hdr)
    for src in cells:
        vals = [single[src].get(t) for t in cells]
        cellstr = "".join("       --     " if v is None else f"{v:>14.4f}" for v in vals)
        mean = np.mean([v for v in vals if v is not None])
        print(f"{src:<14}{cellstr}{mean:>10.4f}")

    # ---- Part 2: weighting under LOCO, each against a class-balance control. ----
    print(f"\n=== Part 2: upweight {_FAVOURED} under LOCO, vs a posrate-matched control ===")
    results = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)

    for holdout in cells:
        srcs = [c for c in cells if c != holdout]
        Xtr = np.vstack([X[c] for c in srcs])
        ytr = np.concatenate([labels[c] for c in srcs])
        line_of = np.concatenate([np.full(len(genes), i) for i, _ in enumerate(srcs)])
        favoured = np.isin(line_of, [i for i, c in enumerate(srcs) if c == _FAVOURED])
        present = favoured.any()

        configs: list[tuple[str, np.ndarray]] = [("equal", np.ones(len(ytr)))]
        for w in _WEIGHTS:
            sw = np.where(favoured, w, 1.0)
            configs.append((f"upweight {_FAVOURED} x{w:g}", sw))
            # Control: same effective positive rate, achieved by reweighting POSITIVES
            # globally. Uses no line identity, so any gain it reproduces is class balance.
            eff_pos = sw[ytr == 1].sum() / sw.sum()
            base_pos = ytr.mean()
            ratio = (eff_pos / (1 - eff_pos)) / (base_pos / (1 - base_pos))
            ctrl = np.where(ytr == 1, ratio, 1.0)
            configs.append((f"control posrate x{w:g}", ctrl))

        print(f"\n--- hold out {holdout} ({labels[holdout].sum()} pos)"
              f"{'' if present else f'  [{_FAVOURED} ABSENT from training: weighting is a no-op]'} ---")
        for name, sw in configs:
            aps, aus = [], []
            for seed in range(args.seeds):
                m = _model(seed).fit(Xtr, ytr, sample_weight=sw)
                p = m.predict_proba(X[holdout])[:, 1]
                aps.append(float(average_precision_score(labels[holdout], p)))
                aus.append(float(roc_auc_score(labels[holdout], p)))
                results[name].append(aps[-1])
                paired[(holdout, seed)][name] = aps[-1]
                rows.append({"part": "loco_weight", "source": "+".join(srcs),
                             "target": holdout, "weight": name, "seed": seed,
                             "auprc": round(aps[-1], 4), "auroc": round(aus[-1], 4)})
            print(f"  {name:<28} AUROC {np.mean(aus):.4f}  AUPRC {np.mean(aps):.4f} "
                  f"+/- {np.std(aps):.4f}")

    print(f"\n=== mean across {len(cells)} folds x {args.seeds} seeds ===")
    ref = float(np.mean(results["equal"]))
    print(f"{'config':<28} {'AUPRC':>7} {'vs equal':>10} {'wins':>8}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        if name == "equal":
            print(f"{name:<28} {mean:>7.4f}")
            continue
        w = sum(1 for v in paired.values() if v[name] > v["equal"])
        print(f"{name:<28} {mean:>7.4f} {mean - ref:>+10.4f} {f'{w}/{len(paired)}':>8}")

    # Folds where the weighting is actually active -- the only ones that carry information.
    active = [c for c in cells if c != _FAVOURED]
    print(f"\n=== active folds only ({', '.join(active)}; n = {len(active)}) ===")
    print(f"{'config':<28} " + "".join(f"{c:>14}" for c in active))
    for name in ["equal"] + [f"upweight {_FAVOURED} x{w:g}" for w in _WEIGHTS] \
                          + [f"control posrate x{w:g}" for w in _WEIGHTS]:
        cellstr = "".join(f"{np.mean([v[name] for (h, _), v in paired.items() if h == c]):>14.4f}"
                          for c in active)
        print(f"{name:<28} {cellstr}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["part", "source", "target", "weight", "seed",
                                           "auprc", "auroc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run -> {out}")


if __name__ == "__main__":
    main()
