"""Do richer Cas13 guide-design features push further than the nine crude ones?

Guide design was the largest single gain of this project: nine summary columns (count, GC
mean/std/min/max, homopolymer mean/max, 3-mer complexity, self-complementarity) added
+0.0124 AUPRC over sequence + TPM, improving 9 of 9 fold x seed pairs. This tests whether
finer-grained guide description adds more, using the feature families the leaderboard's top
entry describes: positional bases, dinucleotides, and target position within the transcript.

Why guide features work at all, stated honestly: they describe the tools used to knock the
lncRNA down, not the lncRNA's biology. A gene whose guides are poorly designed fails to be
knocked down and therefore reads as non-essential regardless of its true role. So these
features partly predict whether the experiment worked. That is legitimate to use -- guide
design is fixed before the screen runs and is not an outcome -- but it is a confounder, and
a write-up should say so rather than claim a biological discovery.

Three blocks, nested so each addition is measurable:
  g9    the nine existing summaries.
  g31   + 16 dinucleotide frequencies + 6 target-position statistics. All guides are
        exactly 23 nt and ~90% locate inside their transcript as reverse complements
        (Cas13 guides are complementary to the target RNA), so position is well defined:
        where along the transcript the guides bind, how spread out they are, and what
        fraction could be located at all.
  g123  + per-position base composition, 23 positions x 4 bases averaged over a gene's
        guides. Cas13 activity is known to depend on base identity at particular positions
        in the spacer, so this is the finest description available -- and the one most at
        risk of overfitting 3 folds with 157-401 positives each.

Reads mmc2 S1B, S1C/S1E and the extracted transcripts. No outcome columns
(fold_change, rra_pvalue, label) are used as features anywhere.

Usage:
  uv run python scripts/sweep_guide_features.py
  uv run python scripts/sweep_guide_features.py --k 4 --seeds 3
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

from sweep_gene_level_priors import load_gene_table
from sweep_prescreen_features import _max_homopolymer, _self_complementarity
from sweep_tpm_features import _ALL_CELLS, _MMC2, _TRAIN, kmer_matrix, load_tpm, tpm_block

_SEQUENCES = "data/processed/body_sequences_transcript.json"
_GUIDE_LEN = 23
_BASES = "ACGT"
_COMP = str.maketrans("ACGT", "TGCA")

_KMER, _ONEHOT, _TPM = "kmer", "onehot", "tpm"
_G9, _G31, _G123 = "g9", "g31", "g123"

_BASELINE = "base kmer+oh+tpm+g9"

CONFIGS: list[tuple[str, list[str]]] = [
    (_BASELINE, [_KMER, _ONEHOT, _TPM, _G9]),
    ("+dinuc+pos (g31)", [_KMER, _ONEHOT, _TPM, _G31]),
    ("+positional (g123)", [_KMER, _ONEHOT, _TPM, _G123]),
    ("nokmer g31", [_ONEHOT, _TPM, _G31]),
    ("nokmer g123", [_ONEHOT, _TPM, _G123]),
]


def _dinucs() -> list[str]:
    return [a + b for a in _BASES for b in _BASES]


def load_guide_blocks(genes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (g9, g31, g123) feature matrices in gene order, each nested in the next."""
    df = pd.read_excel(_MMC2, sheet_name="S1B", header=2)
    target_col, seq_col = df.columns[1], df.columns[2]
    by_target: dict[str, list[str]] = defaultdict(list)
    for target, seq in zip(df[target_col].astype(str), df[seq_col].astype(str)):
        s = seq.strip().upper()
        if s and set(s) <= set(_BASES):
            by_target[target].append(s)

    with open(_SEQUENCES) as fh:
        raw = json.load(fh)
    transcripts = {g: s.upper() for g, (s, _) in raw.items()}

    dinuc_list = _dinucs()
    dinuc_index = {d: i for i, d in enumerate(dinuc_list)}

    rows9, rows_extra, rows_pos = [], [], []
    no_guides = 0
    for g in genes:
        guides = by_target.get(g, [])
        if not guides:
            no_guides += 1
            rows9.append([0.0] * 9)
            rows_extra.append([0.0] * (16 + 6))
            rows_pos.append([0.0] * (_GUIDE_LEN * 4))
            continue

        gc = [(s.count("G") + s.count("C")) / len(s) for s in guides]
        homo = [float(_max_homopolymer(s)) for s in guides]
        cplx = [len({s[i:i + 3] for i in range(len(s) - 2)}) / max(len(s) - 2, 1)
                for s in guides]
        selfc = [_self_complementarity(s) for s in guides]
        rows9.append([
            float(len(guides)),
            float(np.mean(gc)), float(np.std(gc)), float(np.min(gc)), float(np.max(gc)),
            float(np.mean(homo)), float(np.max(homo)),
            float(np.mean(cplx)), float(np.mean(selfc)),
        ])

        # Dinucleotide frequencies, averaged over the gene's guides.
        dn = np.zeros(16, dtype=np.float64)
        for s in guides:
            counts = np.zeros(16, dtype=np.float64)
            for i in range(len(s) - 1):
                j = dinuc_index.get(s[i:i + 2])
                if j is not None:
                    counts[j] += 1
            total = counts.sum()
            if total:
                dn += counts / total
        dn /= len(guides)

        # Target position: guides are complementary to the transcript, so search for the
        # reverse complement. Relative position is 0 at the 5' end, 1 at the 3' end.
        transcript = transcripts.get(g, "")
        rel = []
        if transcript:
            for s in guides:
                idx = transcript.find(s.translate(_COMP)[::-1])
                if idx >= 0 and len(transcript) > _GUIDE_LEN:
                    rel.append(idx / (len(transcript) - _GUIDE_LEN))
        located = len(rel) / len(guides)
        if rel:
            pos_stats = [located, float(np.mean(rel)), float(np.std(rel)),
                         float(np.min(rel)), float(np.max(rel)),
                         float(np.max(rel) - np.min(rel))]
        else:
            pos_stats = [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        rows_extra.append(list(dn) + pos_stats)

        # Per-position base composition across the gene's guides.
        pos = np.zeros((_GUIDE_LEN, 4), dtype=np.float64)
        for s in guides:
            for i, ch in enumerate(s[:_GUIDE_LEN]):
                j = _BASES.find(ch)
                if j >= 0:
                    pos[i, j] += 1
        pos /= len(guides)
        rows_pos.append(pos.reshape(-1).tolist())

    if no_guides:
        print(f"  note: {no_guides:,} gene(s) had no usable guides, zero-filled")

    g9 = np.asarray(rows9, dtype=np.float32)
    extra = np.asarray(rows_extra, dtype=np.float32)
    posm = np.asarray(rows_pos, dtype=np.float32)
    return g9, np.hstack([g9, extra]), np.hstack([g9, extra, posm])


def build(genes, cell_line, blocks, kmer_X, total, mrna, g9, g31, g123):
    cols = []
    if _KMER in blocks:
        cols.append(kmer_X)
    if _ONEHOT in blocks:
        oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
        oh[:, _ALL_CELLS.index(cell_line)] = 1.0
        cols.append(oh)
    if _TPM in blocks:
        cols.append(tpm_block(genes, cell_line, total, mrna)[0])
    if _G9 in blocks:
        cols.append(g9)
    if _G31 in blocks:
        cols.append(g31)
    if _G123 in blocks:
        cols.append(g123)
    return np.hstack(cols).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/guide_feature_sweep.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    print(f"{len(genes):,} genes, training cell lines: {', '.join(train_cells)}")

    print("Loading TPM and guide blocks ...")
    total, mrna = load_tpm()
    g9, g31, g123 = load_guide_blocks(genes)
    print(f"  guide blocks: g9={g9.shape[1]}, g31={g31.shape[1]}, g123={g123.shape[1]} cols")
    kmer_X = kmer_matrix(genes, args.k)

    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for name, blocks in CONFIGS:
            X_eval = build(genes, holdout, blocks, kmer_X, total, mrna, g9, g31, g123)
            X_train = np.vstack([build(genes, c, blocks, kmer_X, total, mrna, g9, g31, g123)
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
                             "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<22} ({X_eval.shape[1]:>4} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    base = float(np.mean(results[_BASELINE]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<22} {'AUPRC':>7} {'sd':>7} {'vs baseline':>14}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if name == _BASELINE else f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else "")
        print(f"{name:<22} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>14}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["holdout", "config", "seed", "n_features",
                                           "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")
    print("Reference: barebones fold_change prior 0.1728 LOCO / 0.2000 board.")


if __name__ == "__main__":
    main()
