"""ChatNT zero-shot lncRNA essentiality CLASSIFIER (CLI).

Prints the probability of essentiality in [0, 1] (from the model's Yes/No
answer-token logits) and, unless --no-reason, the model's free-text rationale.
Exports the full result as JSON.

NOTE: lncRNA essentiality is NOT among ChatNT's documented training tasks. All
outputs are zero-shot, out-of-distribution estimates and require experimental
validation before use.

Examples:
  # dry run — prints the prompts without loading the 8B model
  uv run python scripts/run_chatnt_classifier.py --cell-line K562 --dna-sequence ACGT... --dry-run

  # real run, write JSON to a file
  uv run python scripts/run_chatnt_classifier.py \\
      --cell-line K562 --fasta examples/lncrna_regions.fa --output result.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.io import read_fasta
from lncfit.prompts import (
    build_essentiality_classification_prompt,
    build_essentiality_rationale_prompt,
)


def main():
    parser = argparse.ArgumentParser(
        description="ChatNT zero-shot lncRNA essentiality classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cell-line", required=True, help="Cell line name (e.g. K562)")
    parser.add_argument(
        "--dna-sequence", dest="dna_sequences", action="append", default=[],
        metavar="SEQ", help="DNA sequence (repeatable)",
    )
    parser.add_argument("--fasta", help="FASTA file containing DNA sequences")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability cutoff for the essential/non-essential label (default 0.5).")
    parser.add_argument("--no-reason", dest="with_reason", action="store_false",
                        help="Skip the free-text rationale generation (faster; probability only).")
    parser.add_argument("--output", help="Write the result JSON to this path (in addition to stdout).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompts and exit without loading or running the model.")
    args = parser.parse_args()

    dna_sequences = list(args.dna_sequences)
    if args.fasta:
        dna_sequences.extend(read_fasta(args.fasta))
    if not dna_sequences:
        print("Error: provide at least one sequence via --dna-sequence or --fasta.", file=sys.stderr)
        sys.exit(1)

    n = len(dna_sequences)
    print("=== Classification prompt ===")
    print(build_essentiality_classification_prompt(args.cell_line, n))
    if args.with_reason:
        print("\n=== Rationale prompt ===")
        print(build_essentiality_rationale_prompt(args.cell_line, n))
    print(f"\nSequence count : {n}")
    for i, seq in enumerate(dna_sequences, 1):
        print(f"  Sequence {i}   : length {len(seq)}")
    print()

    if args.dry_run:
        print("[dry-run] Model not loaded. Exiting.")
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Imported here so --dry-run and --help don't require torch/transformers.
    from lncfit.inference import run_chatnt_zeroshot_classifier

    result = run_chatnt_zeroshot_classifier(
        cell_line=args.cell_line,
        dna_sequences=dna_sequences,
        with_reason=args.with_reason,
        threshold=args.threshold,
    )

    prob = result["essentiality_probability"]
    print("\n=== Result ===")
    print(f"P(essential)   : {prob:.4f}" if prob == prob else "P(essential)   : NaN")
    print(f"Predicted label: {result['predicted_label']}")
    if result.get("model_rationale"):
        print(f"Model rationale: {result['model_rationale']}")

    print("\n=== JSON ===")
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
