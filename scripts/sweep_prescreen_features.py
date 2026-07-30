"""Do S1A annotations and S1B guide-design features add anything over sequence + TPM?

Final feature sweep under the no-outcome constraint. Preceding results, all LOCO over the
three training cell lines:
  -mean(fold_change) prior ....... 0.1728   (excluded here -- it is an outcome)
  kmer + cell one-hot ............ 0.1128
  kmer + one-hot + TPM ........... 0.1215   (best so far; TPM helped all three folds)

This adds the two remaining pre-screen blocks in mmc2, which the leaderboard's top entry
also used:

  S1A static annotations -- transcript length, exon count, genomic class, evolutionary age,
  tissue/time specificity (tau), dynamic-tissue counts, distance to the closest
  protein-coding gene.

  S1B guide design -- per-lncRNA summaries of its Cas13 guides: how many, GC content,
  longest homopolymer run, 3-mer complexity, and self-complementarity (hairpin potential).
  Guide quality is a plausible confounder: a lncRNA whose guides work poorly looks
  non-essential regardless of biology.

Two S1A columns are kept in a SEPARATE block (s1a_ext) rather than mixed in, because they
are essentiality readouts rather than plain annotations and deserve a deliberate decision:
  'Closest protein-coding gene Cas9 - DepMap score (23Q2, median)' -- the neighbour gene's
  essentiality from DepMap, an independent Cas9 screen.
  'CRISPRi hit' -- an essentiality call from a different screen and technology.
Neither is this challenge's answer key nor cell-line specific, so both are defensible as
external knowledge; keeping them separable means their contribution is measurable on its own.

Never used: the S1A column 'Number of cell lines showing essentiality', which is derived from
the Cas13 screen across all five cell lines including THP1 and therefore encodes the answer;
and all of mmc3, whose sheet S2J is THP1's answer key.

Usage:
  uv run python scripts/sweep_prescreen_features.py
  uv run python scripts/sweep_prescreen_features.py --k 4 --seeds 3
"""
import argparse
import csv
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
from sweep_tpm_features import (_ALL_CELLS, _MMC2, _TRAIN, kmer_matrix, load_tpm, tpm_block)

_KMER, _ONEHOT, _TPM, _S1A, _S1A_EXT, _GUIDE = (
    "kmer", "onehot", "tpm", "s1a", "s1a_ext", "guide")

# Derived from the Cas13 screen across all five cell lines, THP1 included -- this IS the
# answer. Named here so the exclusion is explicit and greppable.
_LEAK_COLUMN = "Number of cell lines showing essentiality"

# The k-mer block is 256 of ~298 columns. kmer_only scored 0.1036 against a ~0.048 base
# rate, and s1a+guide only (31 columns, no sequence) scored 0.1271 -- beating the
# 267-column kmer+onehot+tpm at 0.1215. So the sequence block may be noise the model has to
# work around: with colsample_bytree=0.8, roughly eight of every nine candidate splits it
# examines are k-mer columns. These configs pair each with-k-mer setting against its
# without-k-mer twin on identical folds to settle it.
_BASELINE = "+s1a+guide"

CONFIGS: list[tuple[str, list[str]]] = [
    ("kmer+onehot+tpm", [_KMER, _ONEHOT, _TPM]),
    ("+s1a+guide", [_KMER, _ONEHOT, _TPM, _S1A, _GUIDE]),            # best with k-mers
    ("nokmer onehot+tpm+guide", [_ONEHOT, _TPM, _GUIDE]),
    ("nokmer +s1a", [_ONEHOT, _TPM, _S1A, _GUIDE]),                  # twin of +s1a+guide
    ("nokmer tpm+guide", [_TPM, _GUIDE]),
    ("s1a+guide only", [_S1A, _GUIDE]),
]

_NUMERIC_S1A = ["Transcript length", "Exons", "Tissue tau", "Time tau",
                "Count dynamic tissues", "Distance to closest protein-coding gene"]
_CATEGORICAL_S1A = ["Genomic class", "Age"]
_EXT_S1A = ["Closest protein-coding gene Cas9 - DepMap score (23Q2, median)", "CRISPRi hit"]


def load_s1a(genes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (basic_block, ext_block, basic_names, ext_names) in gene order."""
    df = pd.read_excel(_MMC2, sheet_name="S1A", header=2)
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    if _LEAK_COLUMN in df.columns:
        df = df.drop(columns=[_LEAK_COLUMN])

    df = df.reindex(genes)

    parts, names = [], []
    for col in _NUMERIC_S1A:
        v = pd.to_numeric(df[col], errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
        parts.append(v.reshape(-1, 1))
        names.append(col)
    # Transcript length spans orders of magnitude; give the model a log scale too.
    tl = pd.to_numeric(df["Transcript length"], errors="coerce").fillna(0).to_numpy()
    parts.append(np.log1p(np.maximum(tl, 0)).astype(np.float32).reshape(-1, 1))
    names.append("log_transcript_length")

    parts.append(df["Dynamic"].fillna(False).astype(bool).to_numpy()
                 .astype(np.float32).reshape(-1, 1))
    names.append("Dynamic")

    for col in _CATEGORICAL_S1A:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col)
        parts.append(dummies.to_numpy(dtype=np.float32))
        names.extend(dummies.columns.tolist())

    basic = np.hstack(parts).astype(np.float32)

    ext_parts, ext_names = [], []
    dep = pd.to_numeric(df[_EXT_S1A[0]], errors="coerce")
    ext_parts.append(dep.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1))
    ext_names.append("depmap_neighbour_score")
    ext_parts.append(dep.isna().to_numpy().astype(np.float32).reshape(-1, 1))
    ext_names.append("depmap_neighbour_missing")
    crispri = df["CRISPRi hit"].astype(str).str.lower().eq("true")
    ext_parts.append(crispri.to_numpy().astype(np.float32).reshape(-1, 1))
    ext_names.append("crispri_hit")
    ext = np.hstack(ext_parts).astype(np.float32)

    return basic, ext, names, ext_names


def _max_homopolymer(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def _self_complementarity(seq: str, w: int = 4) -> float:
    """Fraction of length-w windows whose reverse complement also occurs in the guide.

    A crude hairpin-potential proxy: guides that fold back on themselves are less
    available to hybridize with their target.
    """
    comp = str.maketrans("ACGT", "TGCA")
    windows = [seq[i:i + w] for i in range(len(seq) - w + 1)]
    if not windows:
        return 0.0
    present = set(windows)
    hits = sum(1 for x in windows if x.translate(comp)[::-1] in present)
    return hits / len(windows)


def load_guides(genes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-lncRNA summaries of its Cas13 guide set, in gene order."""
    df = pd.read_excel(_MMC2, sheet_name="S1B", header=2)
    target_col, seq_col = df.columns[1], df.columns[2]
    by_target: dict[str, list[str]] = defaultdict(list)
    for target, seq in zip(df[target_col].astype(str), df[seq_col].astype(str)):
        s = seq.strip().upper()
        if s and set(s) <= set("ACGT"):
            by_target[target].append(s)

    rows, missing = [], 0
    for g in genes:
        guides = by_target.get(g, [])
        if not guides:
            missing += 1
            rows.append([0.0] * 9)
            continue
        gc = [(s.count("G") + s.count("C")) / len(s) for s in guides]
        homo = [float(_max_homopolymer(s)) for s in guides]
        cplx = [len({s[i:i + 3] for i in range(len(s) - 2)}) / max(len(s) - 2, 1)
                for s in guides]
        selfc = [_self_complementarity(s) for s in guides]
        lens = [float(len(s)) for s in guides]
        rows.append([
            float(len(guides)),
            float(np.mean(gc)), float(np.std(gc)),
            float(np.min(gc)), float(np.max(gc)),
            float(np.mean(homo)), float(np.max(homo)),
            float(np.mean(cplx)), float(np.mean(selfc)),
        ])
    if missing:
        print(f"  note: {missing:,} gene(s) had no usable guides, zero-filled")
    names = ["guide_count", "guide_gc_mean", "guide_gc_std", "guide_gc_min",
             "guide_gc_max", "guide_homopolymer_mean", "guide_homopolymer_max",
             "guide_complexity_mean", "guide_selfcomp_mean"]
    return np.asarray(rows, dtype=np.float32), names


def build(genes, cell_line, blocks, kmer_X, total, mrna, s1a, s1a_ext, guide):
    cols = []
    if _KMER in blocks:
        cols.append(kmer_X)
    if _ONEHOT in blocks:
        oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
        oh[:, _ALL_CELLS.index(cell_line)] = 1.0
        cols.append(oh)
    if _TPM in blocks:
        cols.append(tpm_block(genes, cell_line, total, mrna)[0])
    if _S1A in blocks:
        cols.append(s1a)
    if _S1A_EXT in blocks:
        cols.append(s1a_ext)
    if _GUIDE in blocks:
        cols.append(guide)
    return np.hstack(cols).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="results/prescreen_feature_sweep.csv")
    # Model params default to the setting every feature sweep in this branch used, so
    # earlier results reproduce unchanged. scripts/tune_thp1_holdout.py later found
    # max_depth=3 with scale_pos_weight=1.0 worth +0.023 over these defaults, which is
    # more than any feature added -- so the feature rankings need re-checking there.
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--spw", choices=["one", "sqrt", "balanced"], default="balanced",
                        help="scale_pos_weight: 1.0, sqrt(balanced), or the "
                             "negative/positive ratio.")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    print(f"{len(genes):,} genes, training cell lines: {', '.join(train_cells)}")

    print("Loading TPM (S1C/S1E), S1A annotations, S1B guides ...")
    total, mrna = load_tpm()
    s1a, s1a_ext, s1a_names, ext_names = load_s1a(genes)
    guide, guide_names = load_guides(genes)
    print(f"  S1A basic {s1a.shape[1]} cols, S1A ext {s1a_ext.shape[1]} cols "
          f"({', '.join(ext_names)}), guide {guide.shape[1]} cols")
    print(f"  excluded from S1A: {_LEAK_COLUMN!r}")

    kmer_X = kmer_matrix(genes, args.k)
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for name, blocks in CONFIGS:
            X_eval = build(genes, holdout, blocks, kmer_X, total, mrna, s1a, s1a_ext, guide)
            X_train = np.vstack([build(genes, c, blocks, kmer_X, total, mrna,
                                       s1a, s1a_ext, guide)
                                 for c in train_cells if c != holdout])
            y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])
            balanced = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            spw = {"one": 1.0, "sqrt": float(np.sqrt(balanced)),
                   "balanced": float(balanced)}[args.spw]

            aurocs, auprcs = [], []
            for seed in range(args.seeds):
                m = xgb.XGBClassifier(
                    n_estimators=args.n_estimators, learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    subsample=0.8, colsample_bytree=args.colsample, tree_method="hist",
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

    base = float(np.mean(results[_BASELINE]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<26} {'AUPRC':>7} {'sd':>7} {'vs ' + _BASELINE:>20}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if name == _BASELINE else (
            f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else ""))
        print(f"{name:<26} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>20}")

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
