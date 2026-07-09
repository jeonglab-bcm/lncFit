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


def build_training_example(
    record: ScreenRecord, transcript_sequences: dict[str, str] | None = None
) -> tuple[str, list[str], str]:
    """Convert a ScreenRecord into a (prompt, dna_sequences, target_str) fine-tuning triple.

    The target is a bare log2FC float string — the model predicts only fold-change,
    not the input fields.

    ``transcript_sequences`` is the {target: spliced_transcript_seq} mapping from
    ``lncfit.sequence`` (``--sequence-type transcript``), keyed by ``record.target``.
    A 23bp guide spacer alone gives ChatNT nothing the guide-mer XGBoost baseline
    doesn't already capture (issue #56 diagnosis) — the correlating signal lives in
    the lncRNA's own transcript body (issue #65). When a transcript is available for
    this record's target, it's passed as a second DNA sequence alongside the guide;
    ChatNT NT-encodes each ``dna_sequences[i]`` independently and merges it at its own
    ``<DNA>`` placeholder in order, so this is not a naive concatenation. Falls back to
    guide-only when no transcript is available for the target.
    """
    body = transcript_sequences.get(record.target) if transcript_sequences else None
    if body is None:
        prompt = build_essentiality_prompt(record.cell_line, sequence_count=1)
        return prompt, [record.target_sequence], f"{record.fold_change:.4f}"

    prompt = (
        f"In {record.cell_line} cells, would targeting the lncRNA with CRISPR guide "
        f"sequence <DNA> alter cellular essentiality, given the lncRNA's own transcript "
        f"sequence <DNA>? Return the predicted CRISPR-screen log2 fold-change (log2FC) "
        f"as a single numeric value."
    )
    return prompt, [record.target_sequence, body], f"{record.fold_change:.4f}"
