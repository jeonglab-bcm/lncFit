"""DNABERT-2 feature ablation sweep (issue #54).

Evaluates chromosome LOCO-CV performance across 7 feature configurations using
fixed hyperparameters from the best k=3 MSE run (no re-tuning per condition).
This isolates the contribution of DNABERT-2 embeddings from hyperparameter noise.

Conditions:
  1. kmer_only          — guide k=3 + day + cell (current baseline)
  2. kmer+body_first    — + DNABERT-2 body first-1000 bp
  3. kmer+body_last     — + DNABERT-2 body last-1000 bp
  4. kmer+body_mean     — + DNABERT-2 body mean(first, last)
  5. kmer+guide_emb     — + DNABERT-2 guide (23 bp spacer)
  6. kmer+body_mean+guide — + both body(mean) and guide embeddings
  7. body_mean_only     — DNABERT-2 body mean only (no k-mer); requires --include-kmer-only=false

Outputs:
  results/dnabert_ablation/ablation_results.{csv,md}

Usage:
    cd /home/kellyl/lncFit
    uv run python scripts/sweep_dnabert_ablation.py \\
        --params data/model/xgboost_best_params_k3_mse.json \\
        --body-first  data/processed/dnabert2_body_first.npz \\
        --body-last   data/processed/dnabert2_body_last.npz \\
        --body-mean   data/processed/dnabert2_body_mean.npz \\
        --guide       data/processed/dnabert2_guide.npz
"""
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.constants import MIN_FOLD_RECORDS
from lncfit.embeddings import load_embeddings
from lncfit.features import build_features, fit_vocab
from lncfit.screen_data import load_jsonl


def run_cv(
    train_records,
    k: int,
    params: dict,
    body_embeddings=None,
    guide_embeddings=None,
    n_estimators: int = 500,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Run LOCO-CV one fold at a time to avoid holding all 23 fold matrices in RAM.

    With 768-dim DNABERT-2 embeddings, pre-building all folds simultaneously needs
    ~150 GB (23 folds × ~6 GB/fold for the dense embedding block stored as CSR).
    Processing folds sequentially keeps peak usage to ~6 GB.
    """
    chrom_arr = np.array([r.chrom for r in train_records])
    chrom_counts = Counter(chrom_arr)
    cv_chroms = sorted(
        [str(c) for c, n in chrom_counts.items() if c and n >= MIN_FOLD_RECORDS],
        key=lambda x: (len(x), x),
    )

    es_rounds = max(50, int(0.5 / params["learning_rate"]))

    fold_rhos: list[float] = []
    for i, val_chrom in enumerate(cv_chroms):
        es_chrom = cv_chroms[(i + 1) % len(cv_chroms)]
        val_mask   = chrom_arr == val_chrom
        es_mask    = chrom_arr == es_chrom
        train_mask = ~val_mask & ~es_mask

        train_recs = [r for r, m in zip(train_records, train_mask) if m]
        val_recs   = [r for r, m in zip(train_records, val_mask)   if m]
        es_recs    = [r for r, m in zip(train_records, es_mask)    if m]

        fold_vocab = fit_vocab([r.target_sequence for r in train_recs], k)

        X_tr, y_tr, _ = build_features(
            train_recs, k=k, sparse=True, vocab=fold_vocab,
            body_embeddings=body_embeddings, guide_embeddings=guide_embeddings,
        )
        X_es, y_es, _ = build_features(
            es_recs, k=k, sparse=True, vocab=fold_vocab,
            body_embeddings=body_embeddings, guide_embeddings=guide_embeddings,
        )

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective="reg:squarederror",
            tree_method="hist",
            nthread=-1,
            seed=seed,
            callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        del X_tr, y_tr, X_es, y_es
        gc.collect()

        X_val, y_val, _ = build_features(
            val_recs, k=k, sparse=True, vocab=fold_vocab,
            body_embeddings=body_embeddings, guide_embeddings=guide_embeddings,
        )
        rho = spearmanr(y_val, model.predict(X_val)).statistic
        fold_rhos.append(float(rho))
        del X_val, y_val, model
        gc.collect()
        print(f"  fold chr{val_chrom}: rho={rho:.4f}", flush=True)

    return float(np.mean(fold_rhos)), float(np.std(fold_rhos)), fold_rhos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train_chrom1.jsonl.gz")
    parser.add_argument("--params", default="data/model/xgboost_best_params_k3_mse.json")
    parser.add_argument("--body-first",  default="data/processed/dnabert2_body_first.npz")
    parser.add_argument("--body-last",   default="data/processed/dnabert2_body_last.npz")
    parser.add_argument("--body-mean",   default="data/processed/dnabert2_body_mean.npz")
    parser.add_argument("--guide",       default="data/processed/dnabert2_guide.npz")
    parser.add_argument("--output-dir",  default="results/dnabert_ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading params from {args.params} ...")
    with open(args.params) as fh:
        params = json.load(fh)
    k = params["k"]
    n_est = params.get("n_estimators_final_model", params.get("n_estimators_cv_median", 500))
    print(f"  k={k}  n_estimators={n_est}")

    print(f"Loading training records from {args.train} ...")
    train_records = load_jsonl(args.train)
    print(f"  {len(train_records):,} records")

    def _load(path_str: str, label: str):
        p = Path(path_str)
        if not p.exists():
            print(f"  [skip] {label}: file not found ({p})")
            return None
        emb = load_embeddings(str(p))
        print(f"  {label}: {emb[0].shape[0]:,} seqs × {emb[0].shape[1]} dims")
        return emb

    print("\nLoading embeddings ...")
    body_first  = _load(args.body_first,  "body_first")
    body_last   = _load(args.body_last,   "body_last")
    body_mean   = _load(args.body_mean,   "body_mean")
    guide_emb   = _load(args.guide,       "guide")

    conditions: list[tuple[str, str, object, object]] = [
        ("kmer_only",           "k-mer only (baseline)",         None,       None),
        ("kmer+body_first",     "k-mer + DNABERT-2 body (first)",body_first, None),
        ("kmer+body_last",      "k-mer + DNABERT-2 body (last)", body_last,  None),
        ("kmer+body_mean",      "k-mer + DNABERT-2 body (mean)", body_mean,  None),
        ("kmer+guide",          "k-mer + DNABERT-2 guide",       None,       guide_emb),
        ("kmer+body_mean+guide","k-mer + body(mean) + guide",    body_mean,  guide_emb),
    ]

    rows: list[dict] = []
    for cond_key, cond_label, body_emb, guide_emb_arg in conditions:
        # Skip if a required embedding file was not found
        needs_body  = cond_key not in ("kmer_only", "kmer+guide")
        needs_guide = "guide" in cond_key and cond_key != "kmer_only"
        if needs_body and body_emb is None:
            print(f"\n[SKIP] {cond_label} — body embedding file missing")
            continue
        if needs_guide and guide_emb_arg is None:
            print(f"\n[SKIP] {cond_label} — guide embedding file missing")
            continue

        print(f"\n{'='*60}")
        print(f"Condition: {cond_label}")
        print(f"{'='*60}")
        mean_rho, std_rho, fold_rhos = run_cv(
            train_records, k=k, params=params,
            body_embeddings=body_emb,
            guide_embeddings=guide_emb_arg,
            n_estimators=n_est,
            seed=args.seed,
        )
        print(f"  CV Spearman rho: {mean_rho:.4f} ± {std_rho:.4f}")
        rows.append({
            "condition": cond_key,
            "label": cond_label,
            "cv_rho_mean": round(mean_rho, 6),
            "cv_rho_std":  round(std_rho, 6),
            "delta_vs_baseline": None,
        })

    if not rows:
        print("\nNo conditions completed.")
        return

    # Compute delta vs baseline
    baseline_rho = next((r["cv_rho_mean"] for r in rows if r["condition"] == "kmer_only"), None)
    for r in rows:
        if baseline_rho is not None:
            r["delta_vs_baseline"] = round(r["cv_rho_mean"] - baseline_rho, 6)

    # Save CSV
    csv_path = out_dir / "ablation_results.csv"
    with open(csv_path, "w") as fh:
        fh.write("condition,label,cv_rho_mean,cv_rho_std,delta_vs_baseline\n")
        for r in rows:
            fh.write(f"{r['condition']},{r['label']},{r['cv_rho_mean']},{r['cv_rho_std']},{r['delta_vs_baseline']}\n")
    print(f"\nSaved -> {csv_path}")

    # Save Markdown table
    md_path = out_dir / "ablation_results.md"
    with open(md_path, "w") as fh:
        fh.write("# DNABERT-2 Feature Ablation (Issue #54)\n\n")
        fh.write(f"Hyperparameters fixed from `{args.params}` (k={k}, no re-tuning per condition).\n\n")
        fh.write("| Condition | CV ρ (mean) | CV ρ (std) | Δ vs baseline |\n")
        fh.write("|---|---|---|---|\n")
        for r in rows:
            delta = f"+{r['delta_vs_baseline']:.4f}" if r["delta_vs_baseline"] > 0 else f"{r['delta_vs_baseline']:.4f}"
            fh.write(f"| {r['label']} | {r['cv_rho_mean']:.4f} | {r['cv_rho_std']:.4f} | {delta} |\n")
    print(f"Saved -> {md_path}")

    # Print table to stdout
    print("\n" + open(md_path).read())


if __name__ == "__main__":
    main()
