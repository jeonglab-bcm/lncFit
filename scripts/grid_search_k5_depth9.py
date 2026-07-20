"""Grid search around the k=5/class-weight-off/depth=9 best result (see
results/lncrna_rra_day14/README.md's "max_depth follow-up" section).

That result forced max_depth from its Optuna-tuned 4 to 9 on top of the
existing best k=5/off config and got a new project-best AUROC/AUPRC as a
one-off probe -- this script properly grid-searches the 3 hyperparameters
most likely to interact with that depth change (learning_rate, subsample,
colsample_bytree), holding max_depth fixed at 9 and everything else anchored
at the k=5/off tuned values (min_child_weight=3, reg_alpha=3.189,
reg_lambda=3.08e-6, scale_pos_weight=1.0/off).

Same final train/early-stop split as the depth9 probes (stratified 90/10
carve-out, seed=42) and the same chr1 held-out test evaluation, so results are
directly comparable to every other number in this project's history.

Output: results/lncrna_rra_day14/grid_search_k5_depth9/grid_results.csv
(one row per combo) and grid_results_best.json (best row's full run_info).
"""
import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.features import build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

K = 5
SEED = 42
FIXED = {
    "max_depth": 9,
    "min_child_weight": 3,
    "reg_alpha": 3.188749808609341,
    "reg_lambda": 3.078336708769974e-06,
    "scale_pos_weight": 1.0,
}

LEARNING_RATE_GRID = [0.005, 0.01, 0.02, 0.05]
SUBSAMPLE_GRID = [0.5, 0.7, 0.9]
COLSAMPLE_BYTREE_GRID = [0.5, 0.7, 0.9]


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gid: seq for gid, (seq, _) in raw.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json")
    parser.add_argument("--output-dir", default="results/lncrna_rra_day14/grid_search_k5_depth9")
    parser.add_argument("--nthread", type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records and transcript sequences ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    transcript_sequences = _load_transcript_sequences(args.transcript_sequences)
    print(f"  train={len(train_records):,}  test={len(test_records):,}")

    y_all_train = np.array([r.label for r in train_records])
    idx = np.arange(len(train_records))
    final_train_idx, final_es_idx = train_test_split(idx, test_size=0.1, stratify=y_all_train, random_state=SEED)
    final_train_recs = [train_records[i] for i in final_train_idx]
    final_es_recs = [train_records[i] for i in final_es_idx]

    final_targets = {r.target for r in final_train_recs}
    final_seqs = [transcript_sequences[t] for t in final_targets if t in transcript_sequences]
    final_vocab = fit_vocab(final_seqs, K)
    print(f"  vocab: {len(final_vocab)}/{4**K} k-mers observed")

    X_tr, y_tr, _ = build_lncrna_features(final_train_recs, transcript_sequences, k=K, vocab=final_vocab, sparse=True)
    X_es, y_es, _ = build_lncrna_features(final_es_recs, transcript_sequences, k=K, vocab=final_vocab, sparse=True)
    X_test, y_test, _ = build_lncrna_features(test_records, transcript_sequences, k=K, vocab=final_vocab, sparse=True)

    grid = list(itertools.product(LEARNING_RATE_GRID, SUBSAMPLE_GRID, COLSAMPLE_BYTREE_GRID))
    print(f"\nGrid: {len(LEARNING_RATE_GRID)} x {len(SUBSAMPLE_GRID)} x {len(COLSAMPLE_BYTREE_GRID)} "
          f"= {len(grid)} combos, max_depth={FIXED['max_depth']} fixed\n")

    rows = []
    for i, (lr, subsample, colsample_bytree) in enumerate(grid):
        es_rounds = max(50, int(0.5 / lr))
        model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=lr,
            max_depth=FIXED["max_depth"],
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=FIXED["min_child_weight"],
            reg_alpha=FIXED["reg_alpha"],
            reg_lambda=FIXED["reg_lambda"],
            scale_pos_weight=FIXED["scale_pos_weight"],
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            nthread=args.nthread,
            random_state=SEED,
            callbacks=[xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)],
        )
        model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        n_trees = model.best_iteration + 1

        y_pred = model.predict_proba(X_test)[:, 1]
        metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred)
        overall = next(r for r in metrics_rows if r["split"] == "Overall")

        row = {
            "learning_rate": lr, "subsample": subsample, "colsample_bytree": colsample_bytree,
            "n_estimators": n_trees, "auroc": overall["auroc"], "auprc": overall["auprc"],
        }
        rows.append(row)
        print(f"  [{i+1:>2}/{len(grid)}] lr={lr:<6} subsample={subsample:<4} colsample_bytree={colsample_bytree:<4} "
              f"n_trees={n_trees:<4} AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}", flush=True)

        pd.DataFrame(rows).to_csv(out_dir / "grid_results.csv", index=False)

    df = pd.DataFrame(rows)
    best = df.loc[df["auprc"].idxmax()]
    print(f"\nBest by AUPRC: lr={best['learning_rate']} subsample={best['subsample']} "
          f"colsample_bytree={best['colsample_bytree']}  AUROC={best['auroc']:.4f}  AUPRC={best['auprc']:.4f}")

    best_info = {
        "k": K,
        "fixed": FIXED,
        "grid": {"learning_rate": LEARNING_RATE_GRID, "subsample": SUBSAMPLE_GRID,
                 "colsample_bytree": COLSAMPLE_BYTREE_GRID},
        "n_combos": len(grid),
        "best": best.to_dict(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": git_commit(),
    }
    with open(out_dir / "grid_results_best.json", "w") as fh:
        json.dump(best_info, fh, indent=2)
    print(f"\nSaved -> {out_dir / 'grid_results.csv'}")
    print(f"Saved -> {out_dir / 'grid_results_best.json'}")


if __name__ == "__main__":
    main()
