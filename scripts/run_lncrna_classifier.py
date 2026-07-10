"""Train + evaluate any registered lncRNA-hit classifier via a single --model flag.

Systematic model comparison: same dataset, same chr1 hold-out split, same
transcript-sequence features (issue #65), same evaluation — swap the model with
one argument. Wraps the lncfit.classifiers registry.

Usage (from project root):
  uv run python scripts/run_lncrna_classifier.py --model xgboost --k 3
  uv run python scripts/run_lncrna_classifier.py --model logreg  --k 3
  uv run python scripts/run_lncrna_classifier.py --model null

Extra hyperparameters go through --param NAME=VALUE (repeatable), forwarded to
the wrapper's constructor (values are parsed as JSON, falling back to string):
  --model xgboost --param max_depth=4 --param learning_rate=0.03

Outputs (under --output-dir/run_<model>_<timestamp>/):
  metrics.csv       per-cell-line AUROC/AUPRC/F1/...
  predictions.csv   target, cell_line, y_true, y_pred_proba
  run_info.json     model, params, k, files, git commit
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.classifiers import available_classifiers, build_classifier
from lncfit.embeddings import load_embeddings
from lncfit.features import build_lncrna_embedding_features, build_lncrna_features, fit_vocab
from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def _parse_param(kv: str):
    """Parse NAME=VALUE; value is JSON-decoded (so 4 -> int, 0.03 -> float, true -> bool),
    falling back to the raw string for bare words like a solver name."""
    if "=" not in kv:
        raise argparse.ArgumentTypeError(f"--param must be NAME=VALUE, got {kv!r}")
    name, _, raw = kv.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return name, value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=available_classifiers(),
                        help="Registered classifier to run.")
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--features", choices=["kmer", "dnabert2"], default="kmer",
                        help="Feature type: 'kmer' = transcript k-mer frequencies (default); "
                             "'dnabert2' = precomputed DNABERT-2 transcript embeddings (--embeddings).")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json",
                        help="{target: [spliced_seq, \"\"]} JSON from lncfit/sequence.py "
                             "(--sequence-type transcript). The lncRNA's own sequence (issue #65). "
                             "Used by --features kmer.")
    parser.add_argument("--embeddings", default="data/processed/dnabert2_transcript_full.npz",
                        help="Precomputed embedding .npz from scripts/embed_sequences.py. "
                             "Used by --features dnabert2.")
    parser.add_argument("--k", type=int, choices=[3, 4, 5, 6], default=3)
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument("--param", action="append", type=_parse_param, default=[],
                        metavar="NAME=VALUE", help="Hyperparameter forwarded to the model (repeatable).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/lncrna_rra_day14/classifier_runs")
    args = parser.parse_args()

    params = dict(args.param)
    params.setdefault("seed", args.seed)

    print(f"Loading train records from {args.train} ...")
    train_records = load_jsonl(args.train, record_cls=LncRnaRecord)
    print(f"  {len(train_records):,} records")
    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  {len(test_records):,} records")

    if args.features == "dnabert2":
        print(f"Loading DNABERT-2 embeddings from {args.embeddings} ...")
        emb = load_embeddings(args.embeddings)
        print(f"  {emb[0].shape[0]:,} lncRNAs x {emb[0].shape[1]} dims")
        print(f"Building DNABERT-2 features (include_distance={args.include_distance}) ...")
        X_train, y_train, _ = build_lncrna_embedding_features(
            train_records, emb, include_distance=args.include_distance,
        )
        X_test, y_test, _ = build_lncrna_embedding_features(
            test_records, emb, include_distance=args.include_distance,
        )
        feature_desc = f"dnabert2 ({emb[0].shape[1]} dims + cell one-hot)"
    else:
        print(f"Loading transcript sequences from {args.transcript_sequences} ...")
        transcript_sequences = _load_transcript_sequences(args.transcript_sequences)
        print(f"  {len(transcript_sequences):,} lncRNAs")
        print(f"\nFitting k={args.k} vocabulary on training transcript sequences ...")
        train_targets = {r.target for r in train_records}
        train_seqs = [transcript_sequences[t] for t in train_targets if t in transcript_sequences]
        vocab = fit_vocab(train_seqs, args.k)
        print(f"  {len(vocab)} / {4**args.k} k-mers observed")
        print(f"Building k-mer features (k={args.k}, include_distance={args.include_distance}) ...")
        # Dense on purpose: XGBoost treats a CSR matrix's implicit zeros as *missing*, but a
        # dense array's zeros as present. A zero k-mer frequency means "this k-mer does not
        # occur" — a real, informative value, not missing data — so dense is the semantically
        # correct choice and also matches scripts/train_lncrna_xgboost.py's established path.
        X_train, y_train, _ = build_lncrna_features(
            train_records, transcript_sequences, k=args.k,
            include_distance=args.include_distance, vocab=vocab, sparse=False,
        )
        X_test, y_test, _ = build_lncrna_features(
            test_records, transcript_sequences, k=args.k,
            include_distance=args.include_distance, vocab=vocab, sparse=False,
        )
        feature_desc = f"kmer (k={args.k})"
    print(f"  Feature matrix: {X_train.shape[1]} columns  [{feature_desc}]")

    n_pos = int(y_train.sum())
    print(f"  Train label balance: {n_pos:,} hits / {len(y_train) - n_pos:,} non-hits")

    model = build_classifier(args.model, **params)
    print(f"\nFitting {model!r} ...")
    model.fit(X_train, y_train)

    print("\nEvaluating ...")
    y_pred_proba = model.predict_proba(X_test)
    metrics_rows = evaluate_lncrna_by_group(test_records, y_test, y_pred_proba)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{args.model}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"\nMetrics CSV      -> {metrics_path}")

    preds_rows = [
        {"target": rec.target, "cell_line": rec.cell_line, "y_true": float(y_t), "y_pred_proba": float(y_p)}
        for rec, y_t, y_p in zip(test_records, y_test, y_pred_proba)
    ]
    preds_path = out_dir / "predictions.csv"
    pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
    print(f"Predictions CSV  -> {preds_path}")

    run_info = {
        "model": args.model,
        "params": {k: v for k, v in model.params.items()},
        "features": args.features,
        "embeddings_file": args.embeddings if args.features == "dnabert2" else None,
        "k": args.k if args.features == "kmer" else None,
        "include_distance": args.include_distance,
        "n_features": int(X_train.shape[1]),
        "train_file": str(args.train),
        "test_file": str(args.test),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "timestamp": timestamp,
        "git_commit": git_commit(),
    }
    run_info_path = out_dir / "run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2, default=str)
        fh.write("\n")
    print(f"Run info JSON    -> {run_info_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
