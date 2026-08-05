"""Does the GENCODE-annotated TPM (mmc2 S1D/S1F) add to the atlas TPM (S1C/S1E)?

TPM from S1C/S1E was one of only two features that helped this project (+0.0087, 3/3 folds).
mmc2 also ships S1D and S1F: the same RNA-seq quantified against GENCODE v36 instead of the
lncRNA developmental atlas. They are not alternate copies of S1C/S1E -- they cover the full
60,662-gene GENCODE set keyed by Ensembl ID, so they need mapping through S1A's ENSEMBL_ID
column, and they disagree with S1C/S1E where both exist (Pearson on log TPM 0.63-0.69,
Spearman 0.81-0.83, far from the ~0.99 that would mean redundancy). Different annotation,
genuinely different numbers.

The catch is coverage: only 2,992/5,496 lncRNAs carry an ENSEMBL_ID at all, and 2,576
(46.9%) map to a row in S1D/S1F. The rest get zeros plus an explicit missing indicator, so
the model can distinguish "not expressed" from "not measured" -- without that flag a zero
would be read as genuine silence in more than half the genes.

Configs test whether GENCODE TPM adds to the atlas version, replaces it, or neither. Model
params are the tuned ones (max_depth=3, scale_pos_weight=1.0, colsample_bytree=0.3).

Reads only mmc2 pre-screen sheets, the training labels and the transcripts. No outcome
column is used as a feature.

Usage:
  uv run python scripts/sweep_gencode_tpm.py
  uv run python scripts/sweep_gencode_tpm.py --seeds 8
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
from sweep_prescreen_features import _GUIDE, _KMER, _ONEHOT, _S1A, _TPM, build, load_guides, load_s1a
from sweep_tpm_features import _ALL_CELLS, _MMC2, _TRAIN, kmer_matrix, load_tpm

_GENCODE = "gencode"
_BASE = [_KMER, _ONEHOT, _TPM, _S1A, _GUIDE]

CONFIGS: list[tuple[str, list[str], bool]] = [
    ("base (atlas TPM)", _BASE, False),
    ("base + gencode TPM", _BASE, True),
    ("gencode instead of atlas", [_KMER, _ONEHOT, _S1A, _GUIDE], True),
]


def load_gencode_tpm(genes: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return {cell_line: matrix} of GENCODE TPM features, in gene order, plus names.

    Mapped lncRNA -> ENSEMBL_ID (S1A) -> row in S1D (total RNA-seq) and S1F (mRNA-seq).
    Ensembl versions are stripped, since S1D/S1F carry them and S1A does not.
    """
    s1a = pd.read_excel(_MMC2, sheet_name="S1A", header=2).set_index("lncRNA")
    s1a.index = s1a.index.astype(str)
    ens = s1a.reindex(genes)["ENSEMBL_ID"].astype(str).str.split(".").str[0]

    frames = {}
    for sheet, tag in (("S1D", "total"), ("S1F", "mrna")):
        df = pd.read_excel(_MMC2, sheet_name=sheet, header=2)
        key = df.columns[0]
        df["_g"] = df[key].astype(str).str.split(".").str[0]
        frames[tag] = df.drop_duplicates("_g").set_index("_g")

    def column_for(df: pd.DataFrame, cell: str) -> str | None:
        for candidate in (f"{cell} RfxCas13d", cell):
            if candidate in df.columns:
                return candidate
        return None

    # Panel mean is computed once per gene across all cell lines present in S1D.
    total_df, mrna_df = frames["total"], frames["mrna"]
    panel_cols = [c for c in (column_for(total_df, x) for x in _ALL_CELLS) if c]

    names = ["gencode_log_tpm_total", "gencode_log_tpm_mrna", "gencode_log_panel_mean",
             "gencode_breadth", "gencode_missing"]
    out: dict[str, np.ndarray] = {}
    n_mapped = 0
    for cell in _ALL_CELLS:
        tcol, mcol = column_for(total_df, cell), column_for(mrna_df, cell)
        rows = []
        for e in ens:
            if e not in total_df.index:
                rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
                continue
            t = pd.to_numeric(total_df.at[e, tcol], errors="coerce") if tcol else np.nan
            m = (pd.to_numeric(mrna_df.at[e, mcol], errors="coerce")
                 if mcol and e in mrna_df.index else np.nan)
            panel = pd.to_numeric(total_df.loc[e, panel_cols], errors="coerce").fillna(0.0)
            rows.append([
                float(np.log1p(max(t if pd.notna(t) else 0.0, 0.0))),
                float(np.log1p(max(m if pd.notna(m) else 0.0, 0.0))),
                float(np.log1p(max(panel.mean(), 0.0))),
                float((panel >= 1.0).sum()),
                0.0,
            ])
        arr = np.asarray(rows, dtype=np.float32)
        out[cell] = arr
        n_mapped = int((arr[:, -1] == 0).sum())
    print(f"  GENCODE TPM mapped for {n_mapped:,}/{len(genes):,} genes "
          f"({n_mapped / len(genes):.1%}); rest zero-filled with gencode_missing=1")
    return out, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", default="results/gencode_tpm_sweep.csv")
    args = parser.parse_args()

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
    gencode, _ = load_gencode_tpm(genes)

    def make(cell_line: str, blocks: list[str], with_gencode: bool) -> np.ndarray:
        X = build(genes, cell_line, blocks, kmer_X, total, mrna, s1a, s1a_ext, guide)
        return np.hstack([X, gencode[cell_line]]).astype(np.float32) if with_gencode else X

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for name, blocks, with_gencode in CONFIGS:
            X_eval = make(holdout, blocks, with_gencode)
            X_train = np.vstack([make(c, blocks, with_gencode)
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
            print(f"  {name:<26} ({X_eval.shape[1]:>4} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    base = float(np.mean(results["base (atlas TPM)"]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<26} {'AUPRC':>7} {'sd':>7} {'vs base':>12}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if name == "base (atlas TPM)" else (
            f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else ""))
        print(f"{name:<26} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>12}")

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
