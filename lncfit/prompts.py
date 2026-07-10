from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lncfit.screen_data import LncRnaRecord


def essentiality_score(rra_pvalue: float, fold_change: float) -> float:
    """RRA-derived depletion-essentiality confidence in [0, 1].

    Defined as ``1 - rra_pvalue`` for a depletion (``fold_change < 0``), and 0 for
    any non-depletion (``fold_change >= 0``, i.e. enrichment or no change). So the
    score is high only when the lncRNA is a *confident depletion* hit: a small RRA
    p-value AND a negative fold-change. A confident enrichment (small p, positive
    fold-change) is deliberately scored 0 — this target measures essentiality
    (knockout depletes the cells), not any significant effect.
    """
    if fold_change < 0:
        return 1.0 - rra_pvalue
    return 0.0


def build_lncrna_essentiality_prompt(cell_line: str) -> str:
    """Transcript-only regression prompt for the RRA essentiality score.

    A single ``<DNA>`` placeholder for the lncRNA's own spliced transcript — no
    guide spacer (issue #65: the guide is not the lncRNA's sequence, and PR #57's
    fine-tune showed the 23bp guide carries no signal the k-mer baseline lacks).
    Asks for the [0, 1] essentiality score directly, not log2FC.
    """
    return (
        f"In {cell_line} cells, how essential is the lncRNA with the following "
        f"transcript sequence <DNA>? Return the depletion-essentiality confidence "
        f"as a single value between 0 and 1."
    )


def build_lncrna_training_example(
    record: "LncRnaRecord", transcript_sequences: dict[str, str]
) -> tuple[str, list[str], str]:
    """Convert an LncRnaRecord into a (prompt, dna_sequences, target_str) triple.

    Input is the lncRNA's transcript body ONLY (``transcript_sequences[record.target]``);
    the target is the RRA :func:`essentiality_score` formatted to 4 decimals. Raises
    KeyError if the target has no transcript — callers must filter first (every record
    in the day-14 RRA pool does have one, so no drop is expected in practice).
    """
    transcript = transcript_sequences[record.target]
    prompt = build_lncrna_essentiality_prompt(record.cell_line)
    target = essentiality_score(record.rra_pvalue, record.fold_change)
    return prompt, [transcript], f"{target:.4f}"


def build_essentiality_prompt(cell_line: str, sequence_count: int) -> str:
    placeholders = " and ".join(["<DNA>"] * sequence_count)
    return (
        f"In {cell_line} cells, would targeting the lncRNA represented by "
        f"these DNA sequence regions {placeholders} alter cellular essentiality? "
        f"Return the predicted CRISPR-screen log2 fold-change (log2FC) as a single numeric value."
    )


def build_essentiality_classification_prompt(cell_line: str, sequence_count: int) -> str:
    """Yes/No question whose answer-token probabilities give P(essential).

    Phrased so a single "Yes"/"No" answer token carries the label: the zero-shot
    classifier reads the model's first-answer-position logits, aggregates the
    probability mass on yes-tokens vs no-tokens, and returns
    P(yes) / (P(yes) + P(no)) as the essentiality probability.
    """
    placeholders = " and ".join(["<DNA>"] * sequence_count)
    return (
        f"In {cell_line} cells, is the lncRNA represented by these DNA sequence "
        f"regions {placeholders} essential for cell survival, such that a CRISPR "
        f"knockout would deplete the cells? Answer Yes or No."
    )


def build_essentiality_rationale_prompt(cell_line: str, sequence_count: int) -> str:
    """Free-text prompt asking the model to explain its essentiality call.

    The generated answer is captured as a qualitative "reason" alongside the
    probability — it is the model's own explanation to a companion question, not
    a literal derivation of the logit-based probability.
    """
    placeholders = " and ".join(["<DNA>"] * sequence_count)
    return (
        f"In {cell_line} cells, is the lncRNA represented by these DNA sequence "
        f"regions {placeholders} essential for cell survival? Briefly explain your reasoning."
    )
