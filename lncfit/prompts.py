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
