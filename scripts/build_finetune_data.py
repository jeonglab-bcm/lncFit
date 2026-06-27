"""Build fine-tuning JSONL datasets from existing chromosome splits.

Aggregates replicates per (guide_id, cell_line, day), converts each record to a
(prompt, dna_sequences, target) triple, and verifies no guide/target leakage
between train and test.

Output files (gzip JSONL, one example per line):
  finetune_train.jsonl.gz
  finetune_val.jsonl.gz
  finetune_test.jsonl.gz

Each line:
  {
    "prompt":       "<English prompt with <DNA> placeholder>",
    "dna_sequences": ["<guide spacer sequence>"],
    "target":       "<log2FC as 4-decimal float string>",
    "guide_id":     "...",
    "target_name":  "...",
    "cell_line":    "...",
    "day":          7 | 14
  }

Usage:
    uv run python scripts/build_finetune_data.py
    uv run python scripts/build_finetune_data.py --val-chrom chr2
    uv run python scripts/build_finetune_data.py \\
        --train-glob "data/processed/train_*.jsonl.gz" \\
        --test data/processed/test_chrom1.jsonl.gz \\
        --output-dir data/processed
"""
import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.prompts import build_training_example
from lncfit.screen_data import aggregate_replicates, load_jsonl


def build_examples(records):
    examples = []
    for rec in records:
        prompt, dna_seqs, target = build_training_example(rec)
        examples.append({
            "prompt": prompt,
            "dna_sequences": dna_seqs,
            "target": target,
            "guide_id": rec.guide_id,
            "target_name": rec.target,
            "cell_line": rec.cell_line,
            "day": rec.day,
        })
    return examples


def save_jsonl(examples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def check_leakage(train_examples, test_examples):
    train_guides = {ex["guide_id"] for ex in train_examples}
    test_guides = {ex["guide_id"] for ex in test_examples}
    train_targets = {ex["target_name"] for ex in train_examples}
    test_targets = {ex["target_name"] for ex in test_examples}
    return train_guides & test_guides, train_targets & test_targets


def main():
    parser = argparse.ArgumentParser(
        description="Build ChatNT fine-tuning datasets from chromosome splits."
    )
    parser.add_argument(
        "--train-glob", default="data/processed/train_chrom1.jsonl.gz",
        help="Path or glob for training data. Must be the single split that excludes "
             "the test chromosome (train_chrom1.jsonl.gz). Do NOT pass train_*.jsonl.gz — "
             "those are LOCO-CV fold files; all of them except train_chrom1.jsonl.gz "
             "include chromosome 1 records, which leaks into test_chrom1.jsonl.gz.",
    )
    parser.add_argument("--test", default="data/processed/test_chrom1.jsonl.gz")
    parser.add_argument(
        "--val-chrom", default=None,
        help="Hold out this chromosome as validation (e.g. chr2). "
             "If omitted, holds out the last 10%% of aggregated records.",
    )
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    # Load all training splits
    train_files = sorted(glob.glob(args.train_glob))
    if not train_files:
        print(f"ERROR: no files matched: {args.train_glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(train_files)} training split(s)...")
    all_records = []
    for f in train_files:
        recs = load_jsonl(f)
        print(f"  {f}: {len(recs):,} records")
        all_records.extend(recs)
    print(f"Total before aggregation: {len(all_records):,} records")

    all_records = aggregate_replicates(all_records)
    print(f"After replicate aggregation: {len(all_records):,} records")

    # Train / val split
    if args.val_chrom:
        val_records = [r for r in all_records if r.chrom == args.val_chrom]
        train_records = [r for r in all_records if r.chrom != args.val_chrom]
        print(f"Validation chromosome {args.val_chrom}: {len(val_records):,} records")
        print(f"Training (remaining): {len(train_records):,} records")
    else:
        n_val = max(1, len(all_records) // 10)
        train_records = all_records[:-n_val]
        val_records = all_records[-n_val:]
        print(f"Validation (last 10%): {len(val_records):,} records")
        print(f"Training (remaining): {len(train_records):,} records")

    train_examples = build_examples(train_records)
    val_examples = build_examples(val_records)

    # Test set
    print(f"\nLoading test records from {args.test}...")
    test_records = aggregate_replicates(load_jsonl(args.test))
    test_examples = build_examples(test_records)
    print(f"Test examples: {len(test_examples):,}")

    # Leakage check
    print("\nLeakage check (train vs test):")
    leaked_guides, leaked_targets = check_leakage(train_examples, test_examples)
    if leaked_guides:
        print(f"  WARNING: {len(leaked_guides)} guide_id(s) appear in both train and test")
    else:
        print("  OK: no guide_id overlap")
    if leaked_targets:
        print(f"  WARNING: {len(leaked_targets)} target(s) appear in both train and test")
    else:
        print("  OK: no target overlap")

    # Save
    out_dir = Path(args.output_dir)
    for examples, name in [
        (train_examples, "finetune_train.jsonl.gz"),
        (val_examples, "finetune_val.jsonl.gz"),
        (test_examples, "finetune_test.jsonl.gz"),
    ]:
        path = out_dir / name
        n = save_jsonl(examples, path)
        print(f"Saved {n:,} examples -> {path}")


if __name__ == "__main__":
    main()
