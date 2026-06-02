"""
ChatNT baseline CLI for lncRNA essentiality log2FC estimation.

NOTE: lncRNA essentiality / CRISPR-screen log2FC prediction is NOT listed
among ChatNT's documented training tasks. All outputs are out-of-distribution
exploratory estimates and require experimental validation before use.
"""

import argparse
import sys

from lncfit.prompts import build_essentiality_prompt
from lncfit.io import read_fasta
from lncfit.inference import run_chatnt_inference


def main():
    parser = argparse.ArgumentParser(
        description="Run ChatNT inference for lncRNA essentiality estimation."
    )
    parser.add_argument("--cell-line", required=True, help="Cell line name (e.g. K562)")
    parser.add_argument(
        "--dna-sequence",
        dest="dna_sequences",
        action="append",
        default=[],
        metavar="SEQ",
        help="DNA sequence (repeatable)",
    )
    parser.add_argument("--fasta", help="FASTA file containing DNA sequences")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt without loading or running the model",
    )
    args = parser.parse_args()

    dna_sequences = list(args.dna_sequences)
    if args.fasta:
        dna_sequences.extend(read_fasta(args.fasta))

    if not dna_sequences:
        print("Error: provide at least one sequence via --dna-sequence or --fasta.", file=sys.stderr)
        sys.exit(1)

    prompt = build_essentiality_prompt(args.cell_line, len(dna_sequences))

    print("=== Prompt ===")
    print(prompt)
    print()
    print(f"Sequence count : {len(dna_sequences)}")
    for i, seq in enumerate(dna_sequences, 1):
        print(f"  Sequence {i}   : length {len(seq)}")
    print()

    if args.dry_run:
        print("[dry-run] Model not loaded. Exiting.")
        return

    run_chatnt_inference(prompt, dna_sequences)


if __name__ == "__main__":
    main()
