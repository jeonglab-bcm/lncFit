"""Train + evaluate the MLP classification head directly on DNABERT-2 embeddings.

No xgboost/randomforest/logreg/knn involved -- the registered "mlp" classifier
(lncfit.classifiers.mlp.MLPClassifier) is a single-hidden-layer network trained by
gradient descent (BCEWithLogitsLoss + Adam) sitting directly on top of
[DNABERT-2 transcript embedding, cell-line one-hot] (build_lncrna_embedding_features).
It holds out its own 10% stratified validation slice internally for early stopping.

Same chr1 held-out test evaluation as every other model in
results/lncrna_rra_day14/README.md, so its numbers are directly comparable.

Output: results/lncrna_rra_day14/feature_model_comparison/
  predictions_dnabert2_mlp.csv, metrics_dnabert2_mlp.csv, run_info_mlp.json
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--embeddings", default="data/processed/dnabert2_transcript_full.npz")
    parser.add_argument("--output-dir", default="results/lncrna_rra_day14/feature_model_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading records ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  train={len(train_records):,}  test={len(test_records):,}")

    print(f"Loading DNABERT-2 embeddings from {args.embeddings} ...")
    emb = load_embeddings(args.embeddings)
    print(f"  {emb[0].shape[0]:,} lncRNAs x {emb[0].shape[1]} dims")

    X_train, y_train, _ = build_lncrna_embedding_features(train_records, emb)
    X_test, y_test, _ = build_lncrna_embedding_features(test_records, emb)
    print(f"  feature matrix: {X_train.shape[1]} columns (embedding + cell one-hot)")

    model = build_classifier("mlp", seed=args.seed, hidden=args.hidden, lr=args.lr, batch_size=args.batch_size)
    print(f"\nFitting {model!r} (internal 90/10 stratified split for early stopping) ...")
    model.fit(X_train, y_train)

    print("\nEvaluating on chr1 held-out test ...")
    y_pred = model.predict_proba(X_test)
    metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred)

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_dnabert2_mlp.csv", index=False)
    pd.DataFrame({
        "target": [r.target for r in test_records],
        "cell_line": [r.cell_line for r in test_records],
        "y_true": y_test,
        "y_pred_proba": y_pred,
    }).to_csv(out_dir / "predictions_dnabert2_mlp.csv", index=False)

    overall = next(r for r in metrics_rows if r["split"] == "Overall")
    run_info = {
        "model": "mlp",
        "params": {k: v for k, v in model.params.items()},
        "features": "dnabert2",
        "n_features": int(X_train.shape[1]),
        "auroc": overall["auroc"],
        "auprc": overall["auprc"],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": git_commit(),
    }
    with open(out_dir / "run_info_mlp.json", "w") as fh:
        json.dump(run_info, fh, indent=2, default=str)
        fh.write("\n")

    print(f"\nOverall AUROC={overall['auroc']:.4f}  AUPRC={overall['auprc']:.4f}")
    print(f"Saved -> {out_dir / 'predictions_dnabert2_mlp.csv'}")
    print(f"Saved -> {out_dir / 'metrics_dnabert2_mlp.csv'}")
    print(f"Saved -> {out_dir / 'run_info_mlp.json'}")


if __name__ == "__main__":
    main()
