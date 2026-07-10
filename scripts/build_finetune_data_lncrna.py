"""Build ChatNT fine-tuning datasets from the day-14 lncRNA RRA data (issue #56, v2).

Distinct from scripts/build_finetune_data.py (guide-level ScreenRecord, log2FC target).
This uses the SAME data as PR #68 — lncRNA-level day-14 RRA records + the lncRNA's own
spliced transcript — and a regression target derived from RRA:

    target = (1 - rra_pvalue)  if fold_change < 0   (a depletion — essential)
           = 0                 if fold_change >= 0  (enrichment / no change)
    formatted to 4 decimals.

Input to the model is the transcript body ONLY (one <DNA>); the 23bp guide spacer is
dropped entirely (issue #65 / PR #57 finding: the guide carries no signal the k-mer
baseline doesn't already capture).

Splits:
  - test  = the held-out chr1 file (test_lncrna_day14_chrom1.jsonl.gz), unchanged —
            the same held-out set the XGBoost ROC (PR #68) and zero-shot eval (PR #72) use.
  - val   = one held-out chromosome from the training pool (default chr2), so no lncRNA
            leaks between train and val by genomic locus.
  - train = the rest of the training pool.

Output (gzip JSONL, one example/line): finetune_lncrna_{train,val,test}.jsonl.gz
  {"prompt": ..., "dna_sequences": ["<transcript>"], "target": "0.3186",
   "target_name": "Hum_XLOC_...", "cell_line": "K562", "rra_pvalue": ..., "fold_change": ...}

Usage:
    uv run python scripts/build_finetune_data_lncrna.py
    uv run python scripts/build_finetune_data_lncrna.py --val-chrom 3
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.prompts import build_lncrna_training_example
from lncfit.screen_data import LncRnaRecord, load_jsonl


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def build_examples(records, transcript_sequences):
    examples = []
    dropped = 0
    for rec in records:
        if rec.target not in transcript_sequences:
            dropped += 1
            continue
        prompt, dna_seqs, target = build_lncrna_training_example(rec, transcript_sequences)
        examples.append({
            "prompt": prompt,
            "dna_sequences": dna_seqs,
            "target": target,
            "target_name": rec.target,
            "cell_line": rec.cell_line,
            "rra_pvalue": rec.rra_pvalue,
            "fold_change": rec.fold_change,
        })
    return examples, dropped


def save_jsonl(examples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def check_leakage(a, b):
    return {ex["target_name"] for ex in a} & {ex["target_name"] for ex in b}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json")
    parser.add_argument("--val-chrom", default="2",
                        help="Chromosome held out of the training pool for validation (default 2).")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    print(f"Loading transcript sequences from {args.transcript_sequences} ...")
    transcripts = _load_transcript_sequences(args.transcript_sequences)
    print(f"  {len(transcripts):,} lncRNAs")

    print(f"Loading train pool from {args.train} ...")
    train_pool = load_jsonl(args.train, record_cls=LncRnaRecord)
    print(f"  {len(train_pool):,} records")

    val_records = [r for r in train_pool if r.chrom == args.val_chrom]
    train_records = [r for r in train_pool if r.chrom != args.val_chrom]
    print(f"  val (chr{args.val_chrom}): {len(val_records):,}   train (rest): {len(train_records):,}")

    print(f"Loading test (held-out chr1) from {args.test} ...")
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    print(f"  {len(test_records):,} records")

    train_ex, d_tr = build_examples(train_records, transcripts)
    val_ex, d_va = build_examples(val_records, transcripts)
    test_ex, d_te = build_examples(test_records, transcripts)
    if d_tr or d_va or d_te:
        print(f"Dropped (no transcript): train={d_tr} val={d_va} test={d_te}")

    leak_tv = check_leakage(train_ex, val_ex)
    leak_tt = check_leakage(train_ex, test_ex)
    print("\nLeakage check (by lncRNA target):")
    print(f"  train∩val : {'OK (none)' if not leak_tv else f'WARNING {len(leak_tv)} shared'}")
    print(f"  train∩test: {'OK (none)' if not leak_tt else f'WARNING {len(leak_tt)} shared'}")

    def _nonzero(ex):
        return sum(1 for e in ex if float(e["target"]) > 0)
    print("\nTarget distribution (target > 0):")
    for name, ex in [("train", train_ex), ("val", val_ex), ("test", test_ex)]:
        nz = _nonzero(ex)
        print(f"  {name:5s} {len(ex):6,} examples   {nz:5,} nonzero ({100*nz/max(len(ex),1):.1f}%)")

    out_dir = Path(args.output_dir)
    for ex, name in [
        (train_ex, "finetune_lncrna_train.jsonl.gz"),
        (val_ex, "finetune_lncrna_val.jsonl.gz"),
        (test_ex, "finetune_lncrna_test.jsonl.gz"),
    ]:
        n = save_jsonl(ex, out_dir / name)
        print(f"Saved {n:,} -> {out_dir / name}")


if __name__ == "__main__":
    main()
