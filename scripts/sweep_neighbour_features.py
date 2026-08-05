"""Measure every neighbouring protein-coding gene signal separately.

lncRNAs frequently act on the gene next door -- as a local regulator of transcription -- so a
lncRNA sitting beside a gene the cell cannot live without is a plausible candidate for being
essential itself. Earlier sweeps tested pieces of this bundled inside the S1A block and found
nothing, but never isolated them, and never included the one variant that varies by cell line
as well as by gene.

Four neighbour blocks, each measurable on its own:
  nb_dist    distance to the closest protein-coding gene, raw and log. Median is 23 bp, so
             most lncRNAs are effectively on top of a neighbour and this rarely separates.
  nb_class   S1A's "Genomic class" one-hot (downstream_antisense, etc). This describes the
             *relationship* to the neighbour -- orientation and position -- not the lncRNA
             itself, so it belongs here rather than in the gene block.
  nb_depmap  the neighbour's own Cas9 essentiality from DepMap 23Q2, precomputed in S1A,
             plus a missing indicator. Gene-level: identical across cell lines.
  nb_expr    NEW. How strongly the neighbour is expressed in THIS cell line, from GENCODE
             v36 quantifications (S1D total RNA-seq, S1F mRNA-seq), reached by mapping S1A's
             neighbour SYMBOL through data/external/gencode_v36_gene_map.csv to an Ensembl
             ID. 5,084/5,496 (92.5%) resolve; the rest carry a missing flag.

nb_expr is the one with the property that mattered before. The lncRNA's own TPM helped
because it varies by gene AND by cell line; distance, genomic class and the DepMap score are
all fixed across cell lines, so they can only ever describe a gene's average tendency.

The S1A block is split so intrinsic gene properties (length, exons, tau, age) stay separate
from neighbour properties, which were previously mixed together.

k-mers are omitted: five independent tests found the sequence block inert, and it was dropped
by agreement. Model settings are the tuned ones (max_depth=3, scale_pos_weight=1,
colsample_bytree=0.3).

Requires scripts/download_gencode_gene_map.py to have run first.

Usage:
  uv run python scripts/sweep_neighbour_features.py
  uv run python scripts/sweep_neighbour_features.py --seeds 8
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
from sweep_prescreen_features import _LEAK_COLUMN, load_guides
from sweep_tpm_features import _ALL_CELLS, _MMC2, _TRAIN, load_tpm, tpm_block

_GENE_MAP = "data/external/gencode_v36_gene_map.csv"

_ONEHOT, _TPM, _GUIDE, _S1A_GENE = "onehot", "tpm", "guide", "s1a_gene"
_NB_DIST, _NB_CLASS, _NB_DEPMAP, _NB_EXPR = "nb_dist", "nb_class", "nb_depmap", "nb_expr"
_TISSUE = "tissue"

_BASE = [_ONEHOT, _TPM, _GUIDE, _S1A_GENE]
_ALL_NB = [_NB_DIST, _NB_CLASS, _NB_DEPMAP, _NB_EXPR]

# The neighbour blocks the MODEL adopts, as opposed to _ALL_NB, which is the full set the
# configs below exist to measure. nb_depmap is excluded: it is the neighbouring gene's Cas9
# essentiality from DepMap, i.e. a measured knockdown outcome, and docs/PARTICIPATE.md now
# bans measured depletion as an input feature. DepMap is a different assay on a different
# gene in cell lines outside this screen, so it is arguably outside the letter of that rule
# -- but it is not worth the argument, because it costs nothing to remove. Dropping it moves
# training-line LOCO 0.1730 -> 0.1726 and wins 29/60 paired fold x seed runs, a coin flip
# against a noise floor where one gene is worth ~0.005 (scripts/sweep_neighbour_block_removal.py).
_MODEL_NB = [_NB_DIST, _NB_CLASS, _NB_EXPR]

# The seven developmental tissues named in S1A's "Dynamic tissues" list. The model already
# gets "Count dynamic tissues" (how many) inside s1a_gene; this asks whether WHICH tissues
# matters. Motivated by the paper's own abstract, which reports that essential lncRNAs
# "displayed dynamic expression patterns across tissues during development" -- so the
# identity of the tissue, not just the count, is the version the biology points at.
_TISSUE_NAMES = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary",
                 "Testis_NoAdults"]

CONFIGS: list[tuple[str, list[str]]] = [
    ("base (no neighbour)", _BASE),
    ("+ nb_dist", _BASE + [_NB_DIST]),
    ("+ nb_class", _BASE + [_NB_CLASS]),
    ("+ nb_depmap", _BASE + [_NB_DEPMAP]),
    ("+ nb_expr", _BASE + [_NB_EXPR]),
    ("+ all neighbour", _BASE + _ALL_NB),
    ("neighbour only", [_ONEHOT] + _ALL_NB),
    ("base + tissues", _BASE + [_TISSUE]),
    ("+ all neighbour + tissues", _BASE + _ALL_NB + [_TISSUE]),
]

_S1A_GENE_NUMERIC = ["Transcript length", "Exons", "Tissue tau", "Time tau",
                     "Count dynamic tissues"]


def _column_for(df: pd.DataFrame, cell: str) -> str | None:
    for candidate in (f"{cell} RfxCas13d", cell):
        if candidate in df.columns:
            return candidate
    return None


def load_blocks(genes: list[str]):
    """Return the S1A-derived blocks plus the per-cell-line neighbour expression block."""
    s1a = pd.read_excel(_MMC2, sheet_name="S1A", header=2).set_index("lncRNA")
    s1a.index = s1a.index.astype(str)
    if _LEAK_COLUMN in s1a.columns:
        s1a = s1a.drop(columns=[_LEAK_COLUMN])
    s1a = s1a.reindex(genes)

    # Intrinsic gene properties only.
    parts = []
    for col in _S1A_GENE_NUMERIC:
        parts.append(pd.to_numeric(s1a[col], errors="coerce").fillna(-1.0)
                     .to_numpy(dtype=np.float32).reshape(-1, 1))
    tl = pd.to_numeric(s1a["Transcript length"], errors="coerce").fillna(0).to_numpy()
    parts.append(np.log1p(np.maximum(tl, 0)).astype(np.float32).reshape(-1, 1))
    parts.append(s1a["Dynamic"].fillna(False).astype(bool).to_numpy()
                 .astype(np.float32).reshape(-1, 1))
    parts.append(pd.get_dummies(s1a["Age"].astype(str), prefix="Age")
                 .to_numpy(dtype=np.float32))
    s1a_gene = np.hstack(parts).astype(np.float32)

    d = pd.to_numeric(s1a["Distance to closest protein-coding gene"],
                      errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32).reshape(-1, 1)
    nb_dist = np.hstack([d, np.log1p(np.maximum(d, 0))]).astype(np.float32)

    nb_class = pd.get_dummies(s1a["Genomic class"].astype(str),
                              prefix="gclass").to_numpy(dtype=np.float32)

    # "Dynamic tissues" is a comma-separated list; expand to one indicator per tissue.
    # 2,482/5,496 genes name at least one; the rest are all-zero, which is the same
    # thing "Count dynamic tissues" already encodes as 0, so no missing flag is needed.
    dyn = s1a["Dynamic tissues"].fillna("").astype(str)
    tissue = np.zeros((len(genes), len(_TISSUE_NAMES)), dtype=np.float32)
    for i, val in enumerate(dyn):
        named = {t.strip() for t in val.split(",") if t.strip()}
        for j, t in enumerate(_TISSUE_NAMES):
            if t in named:
                tissue[i, j] = 1.0
        unknown = named - set(_TISSUE_NAMES)
        if unknown:
            raise SystemExit(f"unexpected tissue name(s) {sorted(unknown)} -- "
                             "_TISSUE_NAMES is out of date")

    dep_col = "Closest protein-coding gene Cas9 - DepMap score (23Q2, median)"
    dep = pd.to_numeric(s1a[dep_col], errors="coerce")
    nb_depmap = np.hstack([
        dep.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1),
        dep.isna().to_numpy().astype(np.float32).reshape(-1, 1),
    ]).astype(np.float32)

    # Neighbour expression, per cell line.
    gm = pd.read_csv(_GENE_MAP)
    # A symbol can map to several genes; prefer the protein-coding one, which is what
    # "closest protein-coding gene" means.
    gm["pc"] = (gm.gene_type == "protein_coding").astype(int)
    gm = gm.sort_values("pc", ascending=False).drop_duplicates("gene_name")
    sym2id = dict(zip(gm.gene_name, gm.gene_id_base))

    symbols = s1a["Closest protein-coding gene symbol"].astype(str)
    ids = [sym2id.get(s) for s in symbols]

    s1d = pd.read_excel(_MMC2, sheet_name="S1D", header=2)
    s1d["_g"] = s1d[s1d.columns[0]].astype(str).str.split(".").str[0]
    s1d = s1d.drop_duplicates("_g").set_index("_g")
    s1f = pd.read_excel(_MMC2, sheet_name="S1F", header=2)
    s1f["_g"] = s1f[s1f.columns[0]].astype(str).str.split(".").str[0]
    s1f = s1f.drop_duplicates("_g").set_index("_g")

    panel_cols = [c for c in (_column_for(s1d, x) for x in _ALL_CELLS) if c]
    resolved = sum(1 for e in ids if e is not None and e in s1d.index)
    print(f"  neighbour expression resolved for {resolved:,}/{len(genes):,} "
          f"({resolved / len(genes):.1%})")

    nb_expr: dict[str, np.ndarray] = {}
    for cell in _ALL_CELLS:
        tcol, mcol = _column_for(s1d, cell), _column_for(s1f, cell)
        rows = []
        for e in ids:
            if e is None or e not in s1d.index:
                rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
                continue
            t = pd.to_numeric(s1d.at[e, tcol], errors="coerce") if tcol else np.nan
            m = (pd.to_numeric(s1f.at[e, mcol], errors="coerce")
                 if mcol and e in s1f.index else np.nan)
            panel = pd.to_numeric(s1d.loc[e, panel_cols], errors="coerce").fillna(0.0)
            rows.append([
                float(np.log1p(max(t if pd.notna(t) else 0.0, 0.0))),
                float(np.log1p(max(m if pd.notna(m) else 0.0, 0.0))),
                float(np.log1p(max(panel.mean(), 0.0))),
                float((panel >= 1.0).sum()),
                0.0,
            ])
        nb_expr[cell] = np.asarray(rows, dtype=np.float32)

    return s1a_gene, nb_dist, nb_class, nb_depmap, nb_expr, tissue


def build(genes, cell_line, blocks, total, mrna, guide, s1a_gene, nb_dist, nb_class,
          nb_depmap, nb_expr, tissue) -> np.ndarray:
    cols = []
    if _ONEHOT in blocks:
        oh = np.zeros((len(genes), len(_ALL_CELLS)), dtype=np.float32)
        oh[:, _ALL_CELLS.index(cell_line)] = 1.0
        cols.append(oh)
    if _TPM in blocks:
        cols.append(tpm_block(genes, cell_line, total, mrna)[0])
    if _GUIDE in blocks:
        cols.append(guide)
    if _S1A_GENE in blocks:
        cols.append(s1a_gene)
    if _NB_DIST in blocks:
        cols.append(nb_dist)
    if _NB_CLASS in blocks:
        cols.append(nb_class)
    if _NB_DEPMAP in blocks:
        cols.append(nb_depmap)
    if _NB_EXPR in blocks:
        cols.append(nb_expr[cell_line])
    if _TISSUE in blocks:
        cols.append(tissue)
    return np.hstack(cols).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", default="results/neighbour_feature_sweep.csv")
    args = parser.parse_args()

    if not Path(_GENE_MAP).exists():
        raise SystemExit(f"No {_GENE_MAP}. Run scripts/download_gencode_gene_map.py first.")

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}")

    total, mrna = load_tpm()
    guide, _ = load_guides(genes)
    s1a_gene, nb_dist, nb_class, nb_depmap, nb_expr, tissue = load_blocks(genes)
    print(f"  block sizes: s1a_gene={s1a_gene.shape[1]} nb_dist={nb_dist.shape[1]} "
          f"nb_class={nb_class.shape[1]} nb_depmap={nb_depmap.shape[1]} "
          f"nb_expr={next(iter(nb_expr.values())).shape[1]} "
          f"tissue={tissue.shape[1]}")

    rows, results = [], defaultdict(list)
    for holdout in train_cells:
        y_eval = labels[holdout]
        print(f"\n--- hold out {holdout} ({y_eval.sum()} pos / {len(y_eval)}) ---")
        for name, blocks in CONFIGS:
            args_ = (total, mrna, guide, s1a_gene, nb_dist, nb_class, nb_depmap,
                     nb_expr, tissue)
            X_eval = build(genes, holdout, blocks, *args_)
            X_train = np.vstack([build(genes, c, blocks, *args_)
                                 for c in train_cells if c != holdout])
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
                rows.append({"holdout": holdout, "config": name, "seed": seed,
                             "n_features": X_eval.shape[1],
                             "auroc": round(aurocs[-1], 4), "auprc": round(auprcs[-1], 4)})
            print(f"  {name:<22} ({X_eval.shape[1]:>3} cols)  AUROC {np.mean(aurocs):.4f}  "
                  f"AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}", flush=True)

    base = float(np.mean(results["base (no neighbour)"]))
    print(f"\n=== mean across {len(train_cells)} folds x {args.seeds} seeds ===")
    print(f"{'config':<22} {'AUPRC':>7} {'sd':>7} {'vs base':>14}")
    for name, vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - base
        note = "" if name == "base (no neighbour)" else (
            f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else ""))
        print(f"{name:<22} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>14}")

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
