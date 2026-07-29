"""Does pre-screen expression (TPM) help predict lncRNA essentiality in an unseen cell line?

Weeks 1-2 established two negatives: no aggregation of the training outcomes beats
-mean(fold_change) (0.1728 LOCO), and sequence + cell-line features plateau near 0.10 LOCO
regardless of k-mer size or Celligner dimensionality. This tests the one signal left that
is neither an outcome nor constant across the scored rows.

Why TPM can matter when the cell-line one-hot cannot. Every scored row is THP1, so any
cell-line-level feature takes one value across all 5,496 and cannot reorder them -- AUROC
and AUPRC read only the ordering. A gene's TPM *in THP1* differs for every gene, so it can.
The biology is direct: knocking out a lncRNA that is not expressed in a cell line cannot
deplete it, so low-TPM genes should be pushed down the ranking.

TPM is not an outcome. It comes from RNA-seq (mmc2 sheets S1C total and S1E mRNA), measured
independently of the CRISPR screen, and is neither label, rra_pvalue nor fold_change. Reading
THP1's TPM column is therefore not reading the answer key; the leaderboard's top entry uses
these same sheets as pre-screen features. Two things in this workbook are NOT safe and are
not touched here: the S1A column "Number of cell lines showing essentiality", which encodes
the answer across cell lines, and anything in mmc3 (sheet S2J is THP1's answer key).

Evaluation is leave-one-cell-line-out over the three training lines. No feature here
aggregates across cell lines' outcomes, so the source-set leak that inflated
scripts/sweep_gene_level_models.py cannot arise: TPM is attached per (gene, cell line) and
carries no label information.

Usage:
  uv run python scripts/sweep_tpm_features.py
  uv run python scripts/sweep_tpm_features.py --k 4 --seeds 3
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lncfit.features import _count_kmers, all_kmers
from sweep_gene_level_priors import _read_jsonl_gz, load_gene_table

_TRAIN = "data/holdout_thp1/train_thp1_holdout.jsonl.gz"
_SEQUENCES = "data/processed/body_sequences_transcript.json"
_MMC2 = "data/raw/mmc2.xlsx"

# The five cell lines the TPM sheets cover. THP1 is included deliberately: its TPM is a
# legitimate pre-screen feature, and breadth/mean statistics need the full panel.
_ALL_CELLS = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]

_KMER, _ONEHOT, _TPM = "kmer", "onehot", "tpm"

CONFIGS: list[tuple[str, list[str]]] = [
    ("kmer+onehot", [_KMER, _ONEHOT]),      # the Week-3 baseline, ~0.0985
    ("kmer+onehot+tpm", [_KMER, _ONEHOT, _TPM]),
    ("kmer+tpm", [_KMER, _TPM]),
    ("onehot+tpm", [_ONEHOT, _TPM]),
    ("tpm_only", [_TPM]),
]


def load_tpm() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return (total_rnaseq, mrna_seq) as {gene: {cell_line: tpm}} from mmc2 S1C / S1E.

    S1C carries both parental and RfxCas13d columns per line (MDA-MB-231 has only the
    parental one); the RfxCas13d derivatives are the lines the screen was actually run in,
    so they are preferred where present. S1E is RfxCas13d throughout.
    """
    out = []
    for sheet in ("S1C", "S1E"):
        df = pd.read_excel(_MMC2, sheet_name=sheet, header=2)
        df = df.set_index(df.columns[0])
        df.index = df.index.astype(str)
        per_cell: dict[str, dict[str, float]] = defaultdict(dict)
        for cell in _ALL_CELLS:
            col = f"{cell} RfxCas13d"
            if col not in df.columns:
                col = cell
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            for gene, v in values.items():
                per_cell[gene][cell] = float(v) if pd.notna(v) else 0.0
        out.append(dict(per_cell))
    return out[0], out[1]


def tpm_block(genes: list[str], cell_line: str, total: dict, mrna: dict
              ) -> tuple[np.ndarray, list[str]]:
    """Expression features for one cell line's rows, in gene order."""
    rows = []
    for g in genes:
        t = total.get(g, {})
        m = mrna.get(g, {})
        own_t = t.get(cell_line, 0.0)
        own_m = m.get(cell_line, 0.0)
        panel = [t.get(c, 0.0) for c in _ALL_CELLS]
        mean_panel = float(np.mean(panel)) if panel else 0.0
        breadth = float(sum(1 for v in panel if v >= 1.0))
        rows.append([
            np.log1p(max(own_t, 0.0)),
            np.log1p(max(own_m, 0.0)),
            np.log1p(max(mean_panel, 0.0)),
            breadth,
            # How much this gene's expression here stands out from its panel average.
            float(own_t / (mean_panel + 1e-6)),
            float(own_t >= 1.0),
        ])
    names = ["log_tpm_own_total", "log_tpm_own_mrna", "log_tpm_panel_mean",
             "tpm_breadth", "tpm_specificity", "tpm_expressed"]
    return np.asarray(rows, dtype=np.float32), names


def kmer_matrix(genes: list[str], k: int) -> np.ndarray:
    with open(_SEQUENCES) as fh:
        raw = json.load(fh)
    sequences = {g: s for g, (s, _) in raw.items()}
    vocab = all_kmers(k)
    vocab_index = {km: i for i, km in enumerate(vocab)}
    X = np.zeros((len(genes), len(vocab)), dtype=np.float32)
    for i, g in enumerate(genes):
        seq = sequences.get(g)
        if not seq:
            continue
        counts, total = _count_kmers(seq.upper(), k, vocab_index)
        if total:
            for col, cnt in counts.items():
                X[i, col] = cnt / total
    return X


def build(genes: list[str], cell_line: str, blocks: list[str], kmer_X: np.ndarray,
          total: dict, mrna: dict) -> np.ndarray:
    cols = []
    if _KMER in blocks:
        cols.append(kmer_X)
    if _ONEHOT in blocks:
        oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
        oh[:, _ALL_CELLS.index(cell_line)] = 1.0
        cols.append(oh)
    if _TPM in blocks:
        cols.append(tpm_block(genes, cell_line, total, mrna)[0])
    return np.hstack(cols).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/tpm_feature_sweep.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    print(f"{len(genes):,} genes, training cell lines: {', '.join(train_cells)}")

    print("Loading TPM from mmc2 (S1C total, S1E mRNA) ...")
    total, mrna = load_tpm()
    cov = sum(1 for g in genes if g in total)
    print(f"  TPM covers {cov:,}/{len(genes):,} genes")

    print(f"Building k={args.k} k-mer features ...")
    kmer_X = kmer_matrix(genes, args.k)

    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for name, blocks in CONFIGS:
            X_eval = build(genes, holdout, blocks, kmer_X, total, mrna)
            X_train = np.vstack([build(genes, c, blocks, kmer_X, total, mrna)
                                 for c in train_cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])
            spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

            aurocs, auprcs = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=400, learning_rate=0.05, max_depth=9,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    objective="binary:logistic", eval_metric="aucpr",
                    scale_pos_weight=spw, random_state=seed, n_jobs=8)
                m.fit(X_train, y_train)
                p = m.predict_proba(X_eval)[:, 1]
                aurocs.append(float(roc_auc_score(y_eval, p)))
                auprcs.append(float(average_precision_score(y_eval, p)))
                results[name].append(auprcs[-1])
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4),
                             "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<18} ({X_eval.shape[1]:>4} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    base = float(np.mean(results["kmer+onehot"]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<18} {'AUPRC':>7} {'sd':>7} {'vs kmer+onehot':>16}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if name == "kmer+onehot" else f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else "")
        print(f"{name:<18} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>16}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["holdout", "config", "seed", "n_features",
                                           "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")
    print("Reference: barebones fold_change prior 0.1728 LOCO / 0.2000 board; "
          "sequence-only plateau ~0.0985 LOCO / ~0.17 board.")


if __name__ == "__main__":
    main()
