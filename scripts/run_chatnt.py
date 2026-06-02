"""
ChatNT baseline CLI for lncRNA essentiality log2FC estimation.

NOTE: lncRNA essentiality / CRISPR-screen log2FC prediction is NOT listed
among ChatNT's documented training tasks. All outputs are out-of-distribution
exploratory estimates and require experimental validation before use.
"""

import argparse
import re
import sys


def build_essentiality_prompt(cell_line: str, sequence_count: int) -> str:
    placeholders = " and ".join(["<DNA>"] * sequence_count)
    return (
        f"In {cell_line} cells, would targeting the lncRNA represented by "
        f"these DNA sequence regions {placeholders} alter cellular essentiality? "
        f"Return the predicted CRISPR-screen log2 fold-change (log2FC) as a single numeric value."
    )


def parse_log2fc(text: str):
    match = re.search(r"(?<![A-Za-z\d])[-+]?\d+\.?\d*", text)
    return float(match.group()) if match else None


def read_fasta(path: str) -> list:
    sequences = []
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        sequences.append("".join(current_seq))
    return sequences


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

    from transformers import pipeline  # noqa: PLC0415

    pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True)
    result = pipe(inputs={"english_sequence": prompt, "dna_sequences": dna_sequences})

    raw_response = result[0]["generated_text"] if isinstance(result, list) else str(result)

    print("=== Raw ChatNT response ===")
    print(raw_response)
    print()

    value = parse_log2fc(raw_response)
    if value is not None:
        print(f"Parsed log2FC  : {value}")
    else:
        print("Parsed log2FC  : (no numeric value found in response)")
    print()
    print(
        "WARNING: lncRNA essentiality / CRISPR-screen log2FC is outside ChatNT's "
        "documented training tasks. This is an out-of-distribution exploratory estimate "
        "and requires experimental validation against known CRISPR-screen labels before use."
    )


if __name__ == "__main__":
    main()
