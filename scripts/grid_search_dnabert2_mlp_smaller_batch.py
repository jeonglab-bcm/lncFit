"""Follow-up grid: smaller batch_size and lower learning_rate than the first MLP grid.

scripts/grid_search_dnabert2_mlp.py's grid bottomed out at batch_size=16 and
learning_rate=0.0005 (its best-by-AUPRC combo: batch_size=16, lr=0.002, hidden=64).
This pushes both knobs further down -- batch_size in {4, 8}, learning_rate in
{0.0001, 0.0002, 0.0005} -- holding hidden=64 fixed at that combo's value (same
"hold the other params fixed" approach used for max_depth in the xgboost grid).

Same DNABERT-2 + cell-one-hot features and chr1 held-out test evaluation as the other
MLP scripts.

Output: results/lncrna_rra_day14/feature_model_comparison/mlp_grid_dnabert2_smaller_batch.csv
"""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.classifiers import build_classifier
from lncfit.embeddings import load_embeddings
from lncfit.features import build_lncrna_embedding_features
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

BATCH_SIZE_GRID = [4, 8]
LEARNING_RATE_GRID = [0.0001, 0.0002, 0.0005]
FIXED = {"hidden": 64, "dropout": 0.2, "max_epochs": 200, "patience": 10, "seed": 42}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--embeddings", default="data/processed/dnabert2_transcript_full.npz")
    parser.add_argument("--output-dir", default="results/lncrna_rra_day14/feature_model_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records and embeddings ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    emb = load_embeddings(args.embeddings)

    X_train, y_train, _ = build_lncrna_embedding_features(train_records, emb)
    X_test, y_test, _ = build_lncrna_embedding_features(test_records, emb)
    print(f"  train={X_train.shape}  test={X_test.shape}")

    grid = list(itertools.product(BATCH_SIZE_GRID, LEARNING_RATE_GRID))
    print(f"\nGrid: {len(BATCH_SIZE_GRID)} x {len(LEARNING_RATE_GRID)} = {len(grid)} combos, "
          f"hidden={FIXED['hidden']} fixed\n")

    rows = []
    for i, (batch_size, lr) in enumerate(grid):
        model = build_classifier(
            "mlp", batch_size=batch_size, lr=lr, hidden=FIXED["hidden"],
            dropout=FIXED["dropout"], max_epochs=FIXED["max_epochs"],
            patience=FIXED["patience"], seed=FIXED["seed"],
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        overall = next(r for r in evaluate_lncrna_by_group(test_records, y_test, y_pred) if r["split"] == "Overall")

        rows.append({
            "batch_size": batch_size, "learning_rate": lr,
            "auroc": overall["auroc"], "auprc": overall["auprc"],
        })
        print(f"  [{i+1:>2}/{len(grid)}] batch_size={batch_size:<4} lr={lr:<7} "
              f"AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "mlp_grid_dnabert2_smaller_batch.csv", index=False)

    df = pd.DataFrame(rows)
    best = df.loc[df["auprc"].idxmax()]
    print(f"\nBest by AUPRC: batch_size={best['batch_size']} learning_rate={best['learning_rate']} "
          f"AUROC={best['auroc']:.4f}  AUPRC={best['auprc']:.4f}")

    best_info = {
        "fixed": FIXED,
        "grid": {"batch_size": BATCH_SIZE_GRID, "learning_rate": LEARNING_RATE_GRID},
        "n_combos": len(grid),
        "best": best.to_dict(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": git_commit(),
    }
    with open(out_dir / "mlp_grid_dnabert2_smaller_batch_best.json", "w") as fh:
        json.dump(best_info, fh, indent=2)
    print(f"\nSaved -> {out_dir / 'mlp_grid_dnabert2_smaller_batch.csv'}")
    print(f"Saved -> {out_dir / 'mlp_grid_dnabert2_smaller_batch_best.json'}")


if __name__ == "__main__":
    main()
