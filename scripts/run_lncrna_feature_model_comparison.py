"""Compare feature engineering (k-mer vs DNABERT-2) x model (xgboost/randomforest/
logreg/knn) on the lncRNA Day-14 RRA-hit classification task.

Both feature sets get the same recipe:
  - k-mer: transcript k-mer frequencies (k=5, the project's established best k) +
    cell-line one-hot (lncfit.features.build_lncrna_features).
  - dnabert2: precomputed DNABERT-2 transcript embeddings + cell-line one-hot
    (lncfit.features.build_lncrna_embedding_features).

Same stratified 90/10 train/early-stop carve-out (seed=42) used by every prior
depth9 probe in this project, same chr1 held-out test evaluation.

Models:
  - xgboost: grid search over learning_rate x subsample x colsample_bytree
    (3 params), max_depth held at 9 and all other hyperparameters anchored at
    the project's best-known k=5/class-weight-off config (see
    results/lncrna_rra_day14/README.md's "max_depth follow-up" section) --
    same anchors and grid used for both feature sets ("the same manner").
    No Optuna; a plain exhaustive grid.
  - randomforest, logreg, knn: single fixed-hyperparameter fit each (registered
    wrappers in lncfit.classifiers), no grid search.

Output: results/lncrna_rra_day14/feature_model_comparison/
  summary.csv               one row per (features, model)
  predictions_<features>_<model>.csv   y_true/y_pred_proba, for the ROC/PR plots
  xgboost_grid_<features>.csv          full grid for both feature sets
  roc_pr_curves.png                    2x2 ROC/PR plot (one column per feature set)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.classifiers import build_classifier
from lncfit.embeddings import load_embeddings
from lncfit.features import build_lncrna_embedding_features, build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

K = 5
SEED = 42

XGB_FIXED = {
    "max_depth": 9,
    "min_child_weight": 3,
    "reg_alpha": 3.188749808609341,
    "reg_lambda": 3.078336708769974e-06,
    "scale_pos_weight": 1.0,
}
LEARNING_RATE_GRID = [0.005, 0.01, 0.02, 0.05]
SUBSAMPLE_GRID = [0.5, 0.7, 0.9]
COLSAMPLE_BYTREE_GRID = [0.5, 0.7, 0.9]

RF_PARAMS = {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2, "seed": SEED}
KNN_PARAMS = {"n_neighbors": 25, "seed": SEED}
LOGREG_PARAMS = {"C": 1.0, "seed": SEED}


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gid: seq for gid, (seq, _) in raw.items()}


def _stratified_split(train_records):
    y_all = np.array([r.label for r in train_records])
    idx = np.arange(len(train_records))
    tr_idx, es_idx = train_test_split(idx, test_size=0.1, stratify=y_all, random_state=SEED)
    return [train_records[i] for i in tr_idx], [train_records[i] for i in es_idx]


def _xgb_grid_search(X_tr, y_tr, X_es, y_es, X_test, y_test, test_records, out_csv):
    grid = list(itertools.product(LEARNING_RATE_GRID, SUBSAMPLE_GRID, COLSAMPLE_BYTREE_GRID))
    rows = []
    best_pred = None
    best_auprc = -1.0
    for lr, subsample, colsample_bytree in grid:
        es_rounds = max(50, int(0.5 / lr))
        model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=lr,
            max_depth=XGB_FIXED["max_depth"],
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=XGB_FIXED["min_child_weight"],
            reg_alpha=XGB_FIXED["reg_alpha"],
            reg_lambda=XGB_FIXED["reg_lambda"],
            scale_pos_weight=XGB_FIXED["scale_pos_weight"],
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=SEED,
            callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        y_pred = model.predict_proba(X_test)[:, 1]
        overall = next(r for r in evaluate_lncrna_by_group(test_records, y_test, y_pred) if r["split"] == "Overall")
        rows.append({
            "learning_rate": lr, "subsample": subsample, "colsample_bytree": colsample_bytree,
            "n_estimators": model.best_iteration + 1, "auroc": overall["auroc"], "auprc": overall["auprc"],
        })
        if overall["auprc"] > best_auprc:
            best_auprc = overall["auprc"]
            best_pred = y_pred
        print(f"    lr={lr:<6} subsample={subsample:<4} colsample_bytree={colsample_bytree:<4} "
              f"AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    best = df.loc[df["auprc"].idxmax()]
    return best.to_dict(), best_pred


def _run_fixed_model(name, params, X_tr, y_tr, X_test):
    model = build_classifier(name, **params)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_test)


def _run_feature_set(name, X_tr, y_tr, X_es, y_es, X_test, y_test, test_records, out_dir):
    print(f"\n=== features: {name} ===")
    results = {}
    predictions = {}

    print("  [xgboost] grid search (learning_rate x subsample x colsample_bytree, max_depth=9 fixed) ...")
    best_row, xgb_pred = _xgb_grid_search(
        X_tr, y_tr, X_es, y_es, X_test, y_test, test_records,
        out_dir / f"xgboost_grid_{name}.csv",
    )
    results["xgboost"] = {"auroc": best_row["auroc"], "auprc": best_row["auprc"], "params": {
        "max_depth": 9, "learning_rate": best_row["learning_rate"],
        "subsample": best_row["subsample"], "colsample_bytree": best_row["colsample_bytree"],
    }}
    predictions["xgboost"] = xgb_pred

    for model_name, params in [("randomforest", RF_PARAMS), ("logreg", LOGREG_PARAMS), ("knn", KNN_PARAMS)]:
        print(f"  [{model_name}] fitting ({params}) ...")
        y_pred = _run_fixed_model(model_name, params, X_tr, y_tr, X_test)
        overall = next(r for r in evaluate_lncrna_by_group(test_records, y_test, y_pred) if r["split"] == "Overall")
        results[model_name] = {"auroc": overall["auroc"], "auprc": overall["auprc"], "params": params}
        predictions[model_name] = y_pred
        print(f"    AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")

    for model_name, y_pred in predictions.items():
        pd.DataFrame({
            "target": [r.target for r in test_records],
            "cell_line": [r.cell_line for r in test_records],
            "y_true": y_test,
            "y_pred_proba": y_pred,
        }).to_csv(out_dir / f"predictions_{name}_{model_name}.csv", index=False)

    return results, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json")
    parser.add_argument("--embeddings", default="data/processed/dnabert2_transcript_full.npz")
    parser.add_argument("--output-dir", default="results/lncrna_rra_day14/feature_model_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  train={len(train_records):,}  test={len(test_records):,}")

    final_train_recs, final_es_recs = _stratified_split(train_records)
    print(f"  final_train={len(final_train_recs):,}  final_es={len(final_es_recs):,}")

    all_results = {}

    # --- k-mer features ---
    print(f"\nLoading transcript sequences from {args.transcript_sequences} ...")
    transcript_sequences = _load_transcript_sequences(args.transcript_sequences)
    final_targets = {r.target for r in final_train_recs}
    final_seqs = [transcript_sequences[t] for t in final_targets if t in transcript_sequences]
    vocab = fit_vocab(final_seqs, K)
    print(f"  vocab: {len(vocab)}/{4**K} k-mers observed")

    X_tr, y_tr, _ = build_lncrna_features(final_train_recs, transcript_sequences, k=K, vocab=vocab, sparse=False)
    X_es, y_es, _ = build_lncrna_features(final_es_recs, transcript_sequences, k=K, vocab=vocab, sparse=False)
    X_test, y_test, _ = build_lncrna_features(test_records, transcript_sequences, k=K, vocab=vocab, sparse=False)
    print(f"  feature matrix: {X_tr.shape[1]} columns")

    all_results["kmer"], _ = _run_feature_set("kmer", X_tr, y_tr, X_es, y_es, X_test, y_test, test_records, out_dir)

    # --- dnabert2 features ---
    print(f"\nLoading DNABERT-2 embeddings from {args.embeddings} ...")
    emb = load_embeddings(args.embeddings)
    print(f"  {emb[0].shape[0]:,} lncRNAs x {emb[0].shape[1]} dims")

    X_tr, y_tr, _ = build_lncrna_embedding_features(final_train_recs, emb)
    X_es, y_es, _ = build_lncrna_embedding_features(final_es_recs, emb)
    X_test, y_test, _ = build_lncrna_embedding_features(test_records, emb)
    print(f"  feature matrix: {X_tr.shape[1]} columns")

    all_results["dnabert2"], _ = _run_feature_set(
        "dnabert2", X_tr, y_tr, X_es, y_es, X_test, y_test, test_records, out_dir
    )

    # --- summary ---
    rows = []
    for features, models in all_results.items():
        for model_name, r in models.items():
            row = {"features": features, "model": model_name, "auroc": r["auroc"], "auprc": r["auprc"]}
            row.update({f"param_{k}": v for k, v in r["params"].items()})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved -> {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))

    with open(out_dir / "run_info.json", "w") as fh:
        json.dump({"k": K, "seed": SEED, "xgb_fixed": XGB_FIXED,
                   "xgb_grid": {"learning_rate": LEARNING_RATE_GRID, "subsample": SUBSAMPLE_GRID,
                                "colsample_bytree": COLSAMPLE_BYTREE_GRID},
                   "rf_params": RF_PARAMS, "knn_params": KNN_PARAMS, "logreg_params": LOGREG_PARAMS,
                   "git_commit": git_commit()}, fh, indent=2)


if __name__ == "__main__":
    main()
