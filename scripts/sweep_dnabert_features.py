"""Does DNABERT-2 as the sequence block beat k-mers, given the features found since?

The board already answers half of this: DNABERT-2 + distance sits at rank 2 with AUPRC
0.1696 against 0.1636 for k-mer (k=5), a +0.006 gap that is inside the noise floor. What has
never been tested is DNABERT-2 *combined with the features added in this branch* -- TPM,
S1A annotations and guide design -- and at the tuned model settings. Those board entries
predate all of it.

The prior is not encouraging. k-mers are inert here: dropping all 256 columns changed
nothing (p=0.82), and k=3 through k=6 were flat. If sequence composition carries no signal
that survives transfer to an unseen cell line, a different way of encoding the same sequence
should not either. But DNABERT-2 is a genuinely different representation -- a learned model
over context, not letter counts -- so it is the one sequence encoding this project's own
architecture permits that has not been measured directly here.

Configs isolate three questions: does DNABERT-2 add to the current best, does it beat
k-mers as a replacement, and how does it do alone (the closest analogue to the board's
existing DNABERT-2 entries).

Requires scripts/embed_sequences.py to have produced the .npz first:
  uv run python scripts/embed_sequences.py --source body \\
      --body-sequences data/processed/body_sequences_transcript.json \\
      --target-records data/holdout_thp1/train_thp1_holdout.jsonl.gz \\
      --output data/processed/dnabert2_thp1_holdout.npz --window first

Usage:
  uv run python scripts/sweep_dnabert_features.py
  uv run python scripts/sweep_dnabert_features.py --seeds 8
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

from lncfit.embeddings import load_embeddings
from sweep_gene_level_priors import load_gene_table
from sweep_prescreen_features import _GUIDE, _KMER, _ONEHOT, _S1A, _TPM, build, load_guides, load_s1a
from sweep_tpm_features import _TRAIN, kmer_matrix, load_tpm

_EMBEDDINGS = "data/processed/dnabert2_thp1_holdout.npz"

# (label, feature blocks, include DNABERT-2)
CONFIGS: list[tuple[str, list[str], bool]] = [
    ("base (kmer)", [_KMER, _ONEHOT, _TPM, _S1A, _GUIDE], False),
    ("base + dnabert", [_KMER, _ONEHOT, _TPM, _S1A, _GUIDE], True),
    ("dnabert replaces kmer", [_ONEHOT, _TPM, _S1A, _GUIDE], True),
    ("no sequence at all", [_ONEHOT, _TPM, _S1A, _GUIDE], False),
    ("dnabert + onehot only", [_ONEHOT], True),
]


def load_dnabert(genes: list[str], path: str) -> np.ndarray:
    """Embedding rows in gene order. Genes absent from the index get a zero row."""
    matrix, index = load_embeddings(path)
    X = np.zeros((len(genes), matrix.shape[1]), dtype=np.float32)
    missing = 0
    for i, g in enumerate(genes):
        row = index.get(g)
        if row is None:
            missing += 1
            continue
        X[i] = matrix[row]
    if missing:
        print(f"  note: {missing:,} gene(s) absent from the embedding index, zero-filled")
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--embeddings", default=_EMBEDDINGS)
    parser.add_argument("--out", default="results/dnabert_feature_sweep.csv")
    args = parser.parse_args()

    if not Path(args.embeddings).exists():
        raise SystemExit(
            f"No embeddings at {args.embeddings}. Generate them first -- see this "
            "script's docstring for the exact scripts/embed_sequences.py command.")

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}")

    total, mrna = load_tpm()
    s1a, s1a_ext, _, _ = load_s1a(genes)
    guide, _ = load_guides(genes)
    kmer_X = kmer_matrix(genes, args.k)
    dnabert = load_dnabert(genes, args.embeddings)
    print(f"  DNABERT-2: {dnabert.shape[1]} dims")

    def make(cell_line: str, blocks: list[str], with_dnabert: bool) -> np.ndarray:
        X = build(genes, cell_line, blocks, kmer_X, total, mrna, s1a, s1a_ext, guide)
        return np.hstack([X, dnabert]).astype(np.float32) if with_dnabert else X

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for name, blocks, with_dnabert in CONFIGS:
            X_eval = make(holdout, blocks, with_dnabert)
            X_train = np.vstack([make(c, blocks, with_dnabert)
                                 for c in train_cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])

            auprcs, aurocs = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=3,
                    subsample=0.8, colsample_bytree=0.3, tree_method="hist",
                    objective="binary:logistic", eval_metric="aucpr",
                    scale_pos_weight=1.0, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_eval)[:, 1]
                auprcs.append(float(average_precision_score(y_eval, p)))
                aurocs.append(float(roc_auc_score(y_eval, p)))
                results[name].append(auprcs[-1])
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<24} ({X_eval.shape[1]:>4} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    base = float(np.mean(results["base (kmer)"]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<24} {'AUPRC':>7} {'sd':>7} {'vs base':>12}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if name == "base (kmer)" else (
            f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else ""))
        print(f"{name:<24} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>12}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["holdout", "config", "seed", "n_features",
                                           "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")
    print("Board reference: DNABERT-2+distance 0.1696, k-mer(k=5) 0.1636 -- a +0.006 gap, "
          "inside noise.")


if __name__ == "__main__":
    main()
