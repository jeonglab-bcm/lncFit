def build_essentiality_prompt(cell_line: str, sequence_count: int) -> str:
    placeholders = " and ".join(["<DNA>"] * sequence_count)
    return (
        f"In {cell_line} cells, would targeting the lncRNA represented by "
        f"these DNA sequence regions {placeholders} alter cellular essentiality? "
        f"Return the predicted CRISPR-screen log2 fold-change (log2FC) as a single numeric value."
    )
