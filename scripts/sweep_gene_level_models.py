"""Week-2 sweep: does a supervised model beat the -mean(fold_change) prior?

Companion to scripts/sweep_gene_level_priors.py, which established the Week-1 negative
result -- among closed-form aggregations of the three training outcomes, nothing beats the
plain mean (0.1728 mean LOCO AUPRC). So any further gain has to come from signal the prior
never sees. This script tests the two candidates the challenge files already carry for every
gene: distance to the nearest protein-coding gene, and the transcript sequence.

Setup. The test set is THP1 over the same 5,496 genes as the training file, so a training
example is one gene and its target is that gene's label in one cell line. With three
training cell lines we can build 3 x 5,496 such examples: for target line c, features come
from the OTHER lines -- mirroring the real task, where features come from all three training
lines and the target is an unseen fourth.

Leave-one-cell-line-out. Hold out line c: evaluate on (features from the other two, label in
c) and train only on examples whose target line is not c.

That is necessary but NOT sufficient, and an earlier version of this script got it wrong.
Excluding c only as a training *target* still lets c appear as a training *source*, which
leaks: hit_count sums the source lines' labels, so with c in the source set that feature
carries c's answer, and the 256 k-mer columns act as a near-unique per-gene fingerprint the
model can use to memorize which gene it is. Neither half leaks alone -- together they let the
model memorize gene X's fingerprint, bind it to a feature encoding X's held-out label, and
recall it at evaluation. Measured cost of the bug: 'all' scored 0.2823 leaky vs 0.1286
strict, an inflation of +0.15, larger than any real effect in this project's history.

So --strict (the default) also excludes c from every training example's source set. That
mirrors the real submission, where THP1's outcomes appear nowhere in training. Pass
--allow-holdout-in-features to reproduce the leak.

Strict has a real cost: with three training lines, excluding both the target and the held-out
line leaves training examples with a single source line while evaluation uses two. The real
submission trains on two sources and scores on three -- a milder mismatch. Strict numbers are
therefore pessimistic for the supervised approach, and the honest reading is "this did not
clear the prior here", not "this cannot work". More training cell lines would fix it; three is
what the challenge ships.

Why aggregates and not per-cell-line columns: every outcome feature is a line-agnostic
summary (mean/min/max over whatever source lines are available). A model given per-line
columns would learn positions that shift between training and scoring. Aggregates keep the
feature meaning stable when the source set grows from two lines here to three at submission
time -- though distributions do shift slightly (min of three <= min of two), which is one
reason to prefer mean-like features.

Every config runs across several seeds, because XGBoost's subsample/colsample make a single
fit noisy and docs/PARTICIPATE.md puts the noise floor near +/-0.02 AUPRC. Treat smaller
gaps as nothing.

Reads only the training file and the sequences -- never THP1's labels.

Usage:
  uv run python scripts/sweep_gene_level_models.py
  uv run python scripts/sweep_gene_level_models.py --k 5 --seeds 5
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lncfit.features import _count_kmers, all_kmers
# Reused rather than duplicated so the two sweeps cannot drift apart.
from sweep_gene_level_priors import _P_FLOOR, _read_jsonl_gz, _signed_logp, load_gene_table

_TRAIN = "data/holdout_thp1/train_thp1_holdout.jsonl.gz"
_SEQUENCES = "data/processed/body_sequences_transcript.json"

# Feature blocks, toggled per config below.
_PRIOR = "prior"            # -mean(fold_change): the barebones signal itself
_EXTRA = "outcome_extra"    # other closed-form summaries of the same outcomes
_DIST = "distance"          # distance to the nearest protein-coding gene
_KMER = "kmer"              # k-mer frequencies of the transcript sequence

_PRIOR_COLS = ["neg_mean_fc"]
_EXTRA_COLS = ["neg_min_fc", "neg_max_fc", "fc_range", "mean_signed_logp",
               "hit_count", "mean_fc_x_logp"]

CONFIGS: list[tuple[str, list[str]]] = [
    ("prior_only", [_PRIOR]),
    ("prior+extra", [_PRIOR, _EXTRA]),
    ("prior+dist", [_PRIOR, _DIST]),
    ("prior+kmer", [_PRIOR, _KMER]),
    ("prior+dist+kmer", [_PRIOR, _DIST, _KMER]),
    ("all", [_PRIOR, _EXTRA, _DIST, _KMER]),
    ("kmer_only", [_KMER]),
    ("dist_only", [_DIST]),
]


def load_gene_meta(path: str) -> dict[str, dict]:
    """Return {target: {distance}} -- gene-level, identical across cell lines."""
    meta: dict[str, dict] = {}
    for row in _read_jsonl_gz(path):
        d = row.get("distance_to_closest_pc_gene")
        meta[row["target"]] = {"distance": -1 if d is None else int(d)}
    return meta


def load_kmer_matrix(genes: list[str], k: int) -> np.ndarray:
    """Rows of k-mer frequencies for genes, in the given order.

    Uses the complete 4^k vocabulary rather than fitting on observed k-mers: no labels are
    involved either way, and a fixed vocabulary keeps column order reproducible without a
    sidecar file. Genes with no sequence get an all-zero row.
    """
    with open(_SEQUENCES) as fh:
        raw = json.load(fh)
    sequences = {gene_id: seq for gene_id, (seq, _) in raw.items()}

    vocab = all_kmers(k)
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    X = np.zeros((len(genes), len(vocab)), dtype=np.float32)
    missing = 0
    for i, gene in enumerate(genes):
        seq = sequences.get(gene)
        if not seq:
            missing += 1
            continue
        counts, total = _count_kmers(seq.upper(), k, vocab_index)
        if total > 0:
            for col, cnt in counts.items():
                X[i, col] = cnt / total
    if missing:
        print(f"  note: {missing:,} gene(s) had no sequence, zero-filled")
    return X


def outcome_features(obs: list[dict]) -> tuple[list[float], list[float]]:
    """Line-agnostic summaries of one gene's outcomes. Returns (prior, extra)."""
    fcs = [o["fold_change"] for o in obs]
    prior = [-sum(fcs) / len(fcs)]
    extra = [
        -min(fcs),
        -max(fcs),
        float(max(fcs) - min(fcs)),                              # disagreement across lines
        sum(_signed_logp(o) for o in obs) / len(obs),
        float(sum(o["label"] for o in obs)),
        sum(-o["fold_change"] * -math.log10(max(o["rra_pvalue"], _P_FLOOR))
            for o in obs) / len(obs),
    ]
    return prior, extra


def build_matrix(table: dict, genes: list[str], meta: dict, target_line: str,
                 source_lines: list[str], blocks: list[str],
                 kmer_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix + labels for one (target_line, source_lines) example set."""
    cols: list[np.ndarray] = []

    if _PRIOR in blocks or _EXTRA in blocks:
        prior_vals, extra_vals = [], []
        for gene in genes:
            obs = [table[gene][c] for c in source_lines if c in table[gene]]
            prior, extra = outcome_features(obs)
            prior_vals.append(prior)
            extra_vals.append(extra)
        if _PRIOR in blocks:
            cols.append(np.asarray(prior_vals, dtype=np.float32))
        if _EXTRA in blocks:
            cols.append(np.asarray(extra_vals, dtype=np.float32))

    if _DIST in blocks:
        d = np.asarray([[meta[g]["distance"]] for g in genes], dtype=np.float32)
        # Distances span 0 to 2.15e6 with a median of 23, so a raw linear column is
        # dominated by a few outliers; give the model both scales.
        cols.append(np.hstack([d, np.log1p(np.maximum(d, 0))]))

    if _KMER in blocks:
        cols.append(kmer_X)

    X = np.hstack(cols).astype(np.float32)
    y = np.asarray([table[g][target_line]["label"] for g in genes], dtype=int)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=_TRAIN)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6],
                        help="k-mer size. 4 -> 256 columns; 6 -> 4,096, which risks "
                             "overfitting 10,992 training rows.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--allow-holdout-in-features", action="store_true",
                        help="Let the held-out line appear as a SOURCE in training examples. "
                             "Leaks (see module docstring) -- present only to reproduce the "
                             "inflated numbers, never for model selection.")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--out", default="results/gene_level_models.csv")
    args = parser.parse_args()

    table = load_gene_table(args.train)
    meta = load_gene_meta(args.train)
    genes = sorted(table)
    cell_lines = sorted({c for obs in table.values() for c in obs})
    print(f"{len(genes):,} genes x {len(cell_lines)} training cell lines: "
          f"{', '.join(cell_lines)}")

    print(f"Building k={args.k} features ({4 ** args.k:,} columns) ...")
    kmer_X = load_kmer_matrix(genes, args.k)

    rows: list[dict] = []
    # {config: {holdout: [auprc per seed]}}
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    barebones: dict[str, float] = {}

    for holdout in cell_lines:
        train_targets = [c for c in cell_lines if c != holdout]
        eval_sources = train_targets  # features for the held-out line come from the others

        # Reference: the barebones aggregation scored directly on this fold, so the
        # supervised numbers are compared on identical ground.
        X_ref, y_eval = build_matrix(table, genes, meta, holdout, eval_sources,
                                     [_PRIOR], kmer_X)
        barebones[holdout] = float(average_precision_score(y_eval, X_ref[:, 0]))
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) "
              f"| barebones AUPRC {barebones[holdout]:.4f} ---")

        for name, blocks in CONFIGS:
            X_eval, _ = build_matrix(table, genes, meta, holdout, eval_sources,
                                     blocks, kmer_X)

            # Training examples: every target line except the held-out one. Under the
            # default (strict), the held-out line is excluded from the source set too --
            # see the module docstring for why leaving it in inflates the score.
            X_parts, y_parts = [], []
            for target in train_targets:
                sources = [c for c in cell_lines
                           if c != target
                           and (args.allow_holdout_in_features or c != holdout)]
                Xt, yt = build_matrix(table, genes, meta, target, sources, blocks, kmer_X)
                X_parts.append(Xt)
                y_parts.append(yt)
            X_train = np.vstack(X_parts)
            y_train = np.concatenate(y_parts)

            n_pos = int(y_train.sum())
            spw = (len(y_train) - n_pos) / n_pos if n_pos else 1.0

            aurocs, auprcs = [], []
            for seed in range(args.seeds):
                model = xgb.XGBClassifier(
                    n_estimators=args.n_estimators,
                    learning_rate=0.05,
                    max_depth=args.max_depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=spw,
                    random_state=seed,
                    n_jobs=8,
                )
                model.fit(X_train, y_train)
                p = model.predict_proba(X_eval)[:, 1]
                aurocs.append(float(roc_auc_score(y_eval, p)))
                auprcs.append(float(average_precision_score(y_eval, p)))
                results[name][holdout].append(auprcs[-1])
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4),
                             "auprc": round(auprcs[-1], 4)})

            print(f"  {name:<18} ({X_eval.shape[1]:>4} cols)  "
                  f"AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")

    ref = float(np.mean(list(barebones.values())))
    print(f"\n=== mean across {len(cell_lines)} folds x {args.seeds} seeds ===")
    print(f"barebones -mean(fold_change) reference: {ref:.4f}\n")
    print(f"{'config':<18} {'AUPRC':>7} {'sd':>7} {'vs barebones':>14}")
    summary = {name: float(np.mean([v for fold in folds.values() for v in fold]))
               for name, folds in results.items()}
    for name, mean_auprc in sorted(summary.items(), key=lambda kv: -kv[1]):
        sd = float(np.std([v for fold in results[name].values() for v in fold]))
        delta = mean_auprc - ref
        note = f"{delta:+.4f}" + (" (noise)" if abs(delta) < 0.02 else "")
        print(f"{name:<18} {mean_auprc:>7.4f} {sd:>7.4f} {note:>14}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["holdout", "config", "seed",
                                               "n_features", "auroc", "auprc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-fold/per-seed results -> {out_path}")
    print("Gaps under ~0.02 AUPRC are noise. LOCO folds aggregate 2 source lines "
          "while a submission aggregates 3, so absolutes run pessimistic.")


if __name__ == "__main__":
    main()
