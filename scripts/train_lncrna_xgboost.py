"""Train an XGBoost classifier to predict lncRNA RRA-significant depletion hits (Day 14, issue #60).

Same manner as scripts/train_xgboost.py (k-mer + cell-context features, XGBoost hist
tree method, chromosome hold-out split), adapted for the lncRNA-level binary label:
significant hit = RRA P value < 0.05 and log2 fold-change < 0.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.features import build_lncrna_features, fit_vocab
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group


def load_transcript_sequences(path: str) -> dict[str, str]:
    """Load {gene_id: [seq, ""]} produced by `python -m lncfit.sequence --sequence-type transcript`
    and flatten to {gene_id: seq}."""
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost lncRNA RRA-hit classifier (Day 14).")
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json",
                        help="Path to {target: [spliced_seq, \"\"]} JSON from lncfit/sequence.py "
                             "(--sequence-type transcript). This is the lncRNA's own sequence, "
                             "not guide spacer sequences (issue #65).")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=6)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/model")
    parser.add_argument("--output-model", default=None)
    parser.add_argument("--output-metrics", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"xgboost_lncrna_day14_k{args.k}"
    model_path = Path(args.output_model) if args.output_model else out_dir / f"{tag}.ubj"
    params_path = out_dir / f"{tag}_params.json"

    print(f"Loading train records from {args.train} ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    print(f"  {len(train_records):,} records")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  {len(test_records):,} records")

    print(f"Loading transcript sequences from {args.transcript_sequences} ...")
    transcript_sequences = load_transcript_sequences(args.transcript_sequences)
    print(f"  {len(transcript_sequences):,} lncRNAs")

    print(f"\nFitting k={args.k} vocabulary on training lncRNA transcript sequences ...")
    train_targets = {r.target for r in train_records}
    train_seqs = [transcript_sequences[t] for t in train_targets if t in transcript_sequences]
    vocab = fit_vocab(train_seqs, args.k)
    print(f"  {len(vocab)} / {4**args.k} k-mers observed")

    print(f"Building features (k={args.k}, include_distance={args.include_distance}) ...")
    X_train, y_train, train_cols = build_lncrna_features(
        train_records, transcript_sequences, k=args.k, include_distance=args.include_distance, vocab=vocab,
    )
    X_test, y_test, _ = build_lncrna_features(
        test_records, transcript_sequences, k=args.k, include_distance=args.include_distance, vocab=vocab,
    )
    print(f"  Feature matrix: {X_train.shape[1]} columns")

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  Train label balance: {n_pos:,} hits / {n_neg:,} non-hits "
          f"(scale_pos_weight={scale_pos_weight:.2f})")

    print("\nFitting XGBoost classifier ...")
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating ...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred_proba)

    model.save_model(str(model_path))
    print(f"\nModel saved      -> {model_path}")

    vocab_path = out_dir / f"{tag}_vocab.json"
    with open(vocab_path, "w") as fh:
        json.dump(vocab, fh)
    print(f"Vocab saved      -> {vocab_path}")

    params_dict = {
        "k": args.k,
        "include_distance": args.include_distance,
        "n_estimators": args.n_estimators,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "seed": args.seed,
        "feature_columns": train_cols,
        "n_features": X_train.shape[1],
        "train_file": str(args.train),
        "test_file": str(args.test),
    }
    with open(params_path, "w") as fh:
        json.dump(params_dict, fh, indent=2)
    print(f"Params saved     -> {params_path}")

    if args.output_metrics:
        import pandas as pd
        pd.DataFrame(metrics_rows).to_csv(args.output_metrics, index=False)
        print(f"Metrics saved    -> {args.output_metrics}")


if __name__ == "__main__":
    main()
