from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lncfit.screen_data import ScreenRecord


def build_essentiality_prompt(cell_line: str, sequence_count: int) -> str:
    placeholders = " and ".join(["<DNA>"] * sequence_count)
    return (
        f"In {cell_line} cells, would targeting the lncRNA represented by "
        f"these DNA sequence regions {placeholders} alter cellular essentiality? "
        f"Return the predicted CRISPR-screen log2 fold-change (log2FC) as a single numeric value."
    )


def build_training_example(record: ScreenRecord) -> tuple[str, list[str], str]:
    """Return (prompt, dna_sequences, target_str) for one ScreenRecord."""
    prompt = build_essentiality_prompt(record.cell_line, 1)
    return prompt, [record.target_sequence], str(record.fold_change)
