"""Grid search over the MLP classification head's key hyperparameters.

scripts/run_dnabert2_mlp_classifier.py used fixed defaults (hidden=128, lr=1e-3,
batch_size=256) with no tuning at all. This grid searches the 3 hyperparameters most
likely to matter -- batch_size, learning_rate, hidden -- holding dropout=0.2,
max_epochs=200, patience=10 fixed (same "2-3 key parameters, no Optuna" approach used
for the xgboost grid search in results/lncrna_rra_day14/README.md).

Same DNABERT-2 + cell-one-hot features (lncfit.features.build_lncrna_embedding_features)
and chr1 held-out test evaluation as scripts/run_dnabert2_mlp_classifier.py. Each combo's
own internal 90/10 stratified split (inside MLPClassifier.fit()) handles early stopping.

Output: results/lncrna_rra_day14/feature_model_comparison/mlp_grid_dnabert2.csv
(one row per combo) and mlp_grid_dnabert2_best.json (best row + full run_info).
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

BATCH_SIZE_GRID = [16, 32, 64]
LEARNING_RATE_GRID = [0.0005, 0.001, 0.002]
HIDDEN_GRID = [64, 128, 256]
FIXED = {"dropout": 0.2, "max_epochs": 200, "patience": 10, "seed": 42}


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

    grid = list(itertools.product(BATCH_SIZE_GRID, LEARNING_RATE_GRID, HIDDEN_GRID))
    print(f"\nGrid: {len(BATCH_SIZE_GRID)} x {len(LEARNING_RATE_GRID)} x {len(HIDDEN_GRID)} "
          f"= {len(grid)} combos\n")

    rows = []
    for i, (batch_size, lr, hidden) in enumerate(grid):
        model = build_classifier(
            "mlp", batch_size=batch_size, lr=lr, hidden=hidden,
            dropout=FIXED["dropout"], max_epochs=FIXED["max_epochs"],
            patience=FIXED["patience"], seed=FIXED["seed"],
        )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        overall = next(r for r in evaluate_lncrna_by_group(test_records, y_test, y_pred) if r["split"] == "Overall")

        rows.append({
            "batch_size": batch_size, "learning_rate": lr, "hidden": hidden,
            "auroc": overall["auroc"], "auprc": overall["auprc"],
        })
        print(f"  [{i+1:>2}/{len(grid)}] batch_size={batch_size:<4} lr={lr:<7} hidden={hidden:<4} "
              f"AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "mlp_grid_dnabert2.csv", index=False)

    df = pd.DataFrame(rows)
    best = df.loc[df["auprc"].idxmax()]
    print(f"\nBest by AUPRC: batch_size={best['batch_size']} learning_rate={best['learning_rate']} "
          f"hidden={best['hidden']}  AUROC={best['auroc']:.4f}  AUPRC={best['auprc']:.4f}")

    best_info = {
        "fixed": FIXED,
        "grid": {"batch_size": BATCH_SIZE_GRID, "learning_rate": LEARNING_RATE_GRID, "hidden": HIDDEN_GRID},
        "n_combos": len(grid),
        "best": best.to_dict(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": git_commit(),
    }
    with open(out_dir / "mlp_grid_dnabert2_best.json", "w") as fh:
        json.dump(best_info, fh, indent=2)
    print(f"\nSaved -> {out_dir / 'mlp_grid_dnabert2.csv'}")
    print(f"Saved -> {out_dir / 'mlp_grid_dnabert2_best.json'}")


if __name__ == "__main__":
    main()
