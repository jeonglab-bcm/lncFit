"""Hyperparameter sweep for the THP1 hold-out model, under cell-line LOCO.

Every experiment so far used one fixed XGBoost setting (n_estimators=400, lr=0.05,
max_depth=9, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=balanced). Features are
now exhausted -- TPM and guide design helped, sequence/S1A/DepMap/CRISPRi did not -- so this
varies the model instead, on the architecture-compatible best feature set
(k-mer + cell one-hot + TPM + 9 guide summaries).

scale_pos_weight is the knob of interest. It has been pinned at the auto-balanced
negative/positive ratio (~20) all along, never tested, and it governs how hard the model
chases the rare positive class -- which is precisely what AUPRC rewards, since AUPRC is
decided at the very top of the ranking. Note configs/cellline_loco/xgboost_kmer.yaml uses
1.0 instead, so the repo does not treat balanced as obviously right.

Selection caution. With 3 folds and a +/-0.02 noise floor, taking the max over a grid
inflates the winner: some config will look best by luck alone. So this reports how many
folds each config wins, not just its mean, and a config that wins on mean while winning no
folds outright should be distrusted. docs/PARTICIPATE.md records CV rank and test AUPRC
coming out Spearman -1.0 across an SVM C sweep on an earlier version of this task -- model
selection here has actively misled before.

Reads only the training file, mmc2 pre-screen sheets and the transcripts. No outcome column
is used as a feature.

Usage:
  uv run python scripts/tune_thp1_holdout.py
  uv run python scripts/tune_thp1_holdout.py --seeds 3 --stage2
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sweep_gene_level_priors import load_gene_table
from sweep_guide_features import _G9, _KMER, _ONEHOT, _TPM, build, load_guide_blocks
from sweep_tpm_features import _TRAIN, kmer_matrix, load_tpm

_BLOCKS = [_KMER, _ONEHOT, _TPM, _G9]

# scale_pos_weight modes. "balanced" is what every prior experiment used.
_SPW_MODES = ["one", "sqrt", "balanced"]


def spw_value(mode: str, y: np.ndarray) -> float:
    balanced = (len(y) - y.sum()) / max(y.sum(), 1)
    if mode == "one":
        return 1.0
    if mode == "sqrt":
        return float(np.sqrt(balanced))
    return float(balanced)


def evaluate(params: dict, spw_mode: str, genes, train_cells, labels, kmer_X,
             total, mrna, g9, seeds: int) -> tuple[dict[str, float], list[dict]]:
    """Return ({holdout: mean auprc}, per-seed rows) for one hyperparameter setting."""
    per_fold, rows = {}, []
    for holdout in train_cells:
        y_eval = labels[holdout]
        X_eval = build(genes, holdout, _BLOCKS, kmer_X, total, mrna, g9, None, None)
        X_train = np.vstack([build(genes, c, _BLOCKS, kmer_X, total, mrna, g9, None, None)
                             for c in train_cells if c != holdout])
        y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])
        spw = spw_value(spw_mode, y_train)

        auprcs, aurocs = [], []
        for seed in range(seeds):
            m = xgb.XGBClassifier(
                tree_method="hist", objective="binary:logistic", eval_metric="aucpr",
                scale_pos_weight=spw, random_state=seed, n_jobs=8, **params)
            m.fit(X_train, y_train)
            p = m.predict_proba(X_eval)[:, 1]
            auprcs.append(float(average_precision_score(y_eval, p)))
            aurocs.append(float(roc_auc_score(y_eval, p)))
            rows.append({"holdout": holdout, "seed": seed, "spw_mode": spw_mode,
                         **{k: v for k, v in params.items()},
                         "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
        per_fold[holdout] = float(np.mean(auprcs))
    return per_fold, rows


def report(results: dict[str, dict[str, float]], train_cells: list[str], title: str,
           baseline: str | None = None) -> str:
    """Print mean AUPRC plus per-fold win counts; return the best config name."""
    wins = defaultdict(int)
    for c in train_cells:
        best = max(results, key=lambda n: results[n][c])
        wins[best] += 1

    print(f"\n=== {title} ===")
    print(f"{'config':<34} {'AUPRC':>7} {'folds won':>10}  per-fold")
    ordered = sorted(results, key=lambda n: -np.mean(list(results[n].values())))
    for name in ordered:
        mean = float(np.mean(list(results[name].values())))
        per = "  ".join(f"{results[name][c]:.4f}" for c in train_cells)
        star = " *" if baseline and name == baseline else ""
        print(f"{name:<34} {mean:>7.4f} {wins[name]:>10}  {per}{star}")
    best = ordered[0]
    if wins[best] == 0:
        print(f"  caution: {best} leads on mean but wins no fold outright -- "
              "likely selection noise rather than a real improvement.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--stage2", action="store_true",
                        help="After the spw x depth grid, sweep lr / n_estimators / "
                             "min_child_weight around the stage-1 winner.")
    parser.add_argument("--out", default="results/thp1_holdout_tuning.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}

    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}")
    print("Loading features (TPM, guides, k-mers) ...")
    total, mrna = load_tpm()
    g9, _, _ = load_guide_blocks(genes)
    kmer_X = kmer_matrix(genes, args.k)

    all_rows: list[dict] = []
    base_params = dict(n_estimators=400, learning_rate=0.05, max_depth=9,
                       subsample=0.8, colsample_bytree=0.8)
    baseline_name = "spw=balanced depth=9"

    # Stage 1: scale_pos_weight x max_depth.
    stage1: dict[str, dict[str, float]] = {}
    for spw_mode in _SPW_MODES:
        for depth in (3, 6, 9):
            params = {**base_params, "max_depth": depth}
            name = f"spw={spw_mode} depth={depth}"
            per_fold, rows = evaluate(params, spw_mode, genes, train_cells, labels,
                                      kmer_X, total, mrna, g9, args.seeds)
            stage1[name] = per_fold
            all_rows.extend(rows)
            print(f"  {name:<34} AUPRC {np.mean(list(per_fold.values())):.4f}", flush=True)

    best1 = report(stage1, train_cells, "stage 1: scale_pos_weight x max_depth",
                   baseline=baseline_name)
    print(f"\nbaseline (* above) = {baseline_name}: "
          f"{np.mean(list(stage1[baseline_name].values())):.4f}")
    print(f"stage-1 best = {best1}")

    if args.stage2:
        spw_mode = best1.split()[0].split("=")[1]
        depth = int(best1.split("depth=")[1])
        stage2: dict[str, dict[str, float]] = {best1: stage1[best1]}
        variants = [
            ("lr=0.02 n=1000", dict(learning_rate=0.02, n_estimators=1000)),
            ("lr=0.1 n=200", dict(learning_rate=0.1, n_estimators=200)),
            ("mcw=5", dict(min_child_weight=5)),
            ("mcw=20", dict(min_child_weight=20)),
            ("subsample=0.5", dict(subsample=0.5)),
            ("colsample=0.3", dict(colsample_bytree=0.3)),
        ]
        for label, override in variants:
            params = {**base_params, "max_depth": depth, **override}
            name = f"{best1} + {label}"
            per_fold, rows = evaluate(params, spw_mode, genes, train_cells, labels,
                                      kmer_X, total, mrna, g9, args.seeds)
            stage2[name] = per_fold
            all_rows.extend(rows)
            print(f"  {name:<34} AUPRC {np.mean(list(per_fold.values())):.4f}", flush=True)
        report(stage2, train_cells, "stage 2: around the stage-1 winner", baseline=best1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nPer-fold/per-seed -> {out}")
    print("Reference: feature-sweep best 0.1339; barebones prior 0.1728 LOCO / 0.2000 board.")


if __name__ == "__main__":
    main()
