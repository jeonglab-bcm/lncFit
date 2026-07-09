import logging

import numpy as np

from lncfit.parsers import parse_log2fc
from lncfit.prompts import (
    build_essentiality_classification_prompt,
    build_essentiality_rationale_prompt,
)

logger = logging.getLogger(__name__)

_MODEL_NAME = "InstaDeepAI/ChatNT"
_YES_WORDS = ("Yes", "yes", "YES")
_NO_WORDS = ("No", "no", "NO")
_OOD_CAVEAT = (
    "lncRNA essentiality is outside ChatNT's documented training tasks. This is a "
    "zero-shot, out-of-distribution estimate and requires experimental validation "
    "against known CRISPR-screen labels before use."
)


def _load_pipeline():
    """Construct the InstaDeepAI/ChatNT custom pipeline (lazy import of transformers).

    Isolated in one place so callers don't import transformers at module load and
    tests can patch it without the 8B stack installed.
    """
    from transformers import pipeline

    return pipeline(model=_MODEL_NAME, trust_remote_code=True)


def run_chatnt_inference(prompt: str, dna_sequences: list[str]) -> float | None:
    """Load ChatNT, run inference, log raw response, return parsed log2FC."""
    pipe = _load_pipeline()
    result = pipe(inputs={"english_sequence": prompt, "dna_sequences": dna_sequences})

    raw_response = result[0]["generated_text"] if isinstance(result, list) else str(result)

    logger.info("=== Raw ChatNT response ===")
    logger.info(raw_response)

    value = parse_log2fc(raw_response)
    if value is not None:
        logger.info("Parsed log2FC  : %s", value)
    else:
        logger.info("Parsed log2FC  : (no numeric value found in response)")

    logger.warning(
        "lncRNA essentiality / CRISPR-screen log2FC is outside ChatNT's "
        "documented training tasks. This is an out-of-distribution exploratory estimate "
        "and requires experimental validation against known CRISPR-screen labels before use."
    )

    return value


def essentiality_probability_from_logits(
    logits_row: np.ndarray, yes_ids: list[int], no_ids: list[int]
) -> float:
    """Convert a single answer-position logit vector into P(essential) in [0, 1].

    Softmaxes the full vocabulary logits, sums the probability mass on the
    yes-tokens and the no-tokens, and returns yes / (yes + no) — i.e. the
    model's answer renormalized to just the two relevant outcomes. Returns NaN
    if neither yes nor no tokens carry any mass (degenerate / all-elsewhere).

    Pure function (no model, no torch) so the probability math is unit-testable.
    """
    logits_row = np.asarray(logits_row, dtype=np.float64)
    probs = np.exp(logits_row - logits_row.max())
    probs /= probs.sum()
    yes_mass = float(probs[yes_ids].sum())
    no_mass = float(probs[no_ids].sum())
    total = yes_mass + no_mass
    if total <= 0.0:
        return float("nan")
    return yes_mass / total


def resolve_yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Resolve the first-token IDs the model would emit for Yes/No surface forms.

    Aggregates over case variants and the leading-space variant (sentencepiece
    emits e.g. "▁Yes" after "ASSISTANT:"). Takes each variant's FIRST token — the
    single token the model produces at the answer position.
    """

    def _ids(words: tuple[str, ...]) -> list[int]:
        ids: set[int] = set()
        for word in words:
            for variant in (word, " " + word):
                toks = tokenizer(variant, add_special_tokens=False).input_ids
                if toks:
                    ids.add(int(toks[0]))
        return sorted(ids)

    return _ids(_YES_WORDS), _ids(_NO_WORDS)


def run_chatnt_zeroshot_classifier(
    cell_line: str,
    dna_sequences: list[str],
    with_reason: bool = True,
    threshold: float = 0.5,
    max_reason_tokens: int = 80,
    pipe=None,
) -> dict:
    """Zero-shot lncRNA essentiality classifier over ChatNT.

    Returns a JSON-serialisable dict with the essentiality probability (from the
    Yes/No answer-token logits), a hard label at `threshold`, and — when
    `with_reason` — the model's free-text rationale to a companion question.

    Pass a preconstructed `pipe` (the InstaDeepAI/ChatNT custom pipeline) to reuse
    a loaded model or to inject a mock in tests; otherwise it is loaded here. The
    logit read replicates the pipeline's own generation-start logic exactly (first
    PAD position in the english tokens, logits at position first_pad - 1).
    """
    n = len(dna_sequences)
    classification_prompt = build_essentiality_classification_prompt(cell_line, n)

    if pipe is None:
        pipe = _load_pipeline()

    model_inputs = pipe.preprocess(
        {"english_sequence": classification_prompt, "dna_sequences": dna_sequences}
    )
    english_tokens = model_inputs["english_tokens"]
    bio_tokens = model_inputs["bio_tokens"]

    outs = pipe.model(
        multi_omics_tokens_ids=(english_tokens, bio_tokens),
        projection_english_tokens_ids=english_tokens,
        projected_bio_embeddings=None,
    )
    logits = np.asarray(outs["logits"].detach().cpu().numpy())

    english_ids = np.asarray(english_tokens[0].cpu().numpy())
    pad_id = pipe.english_tokenizer.pad_token_id
    pad_positions = np.where(english_ids == pad_id)[0]
    if len(pad_positions) == 0:
        raise ValueError(
            "No PAD token in the tokenized prompt — cannot locate the answer position. "
            "Try a shorter prompt or larger english_tokens_max_length."
        )
    answer_pos = int(pad_positions[0]) - 1

    yes_ids, no_ids = resolve_yes_no_token_ids(pipe.english_tokenizer)
    probability = essentiality_probability_from_logits(logits[0, answer_pos], yes_ids, no_ids)

    reason = None
    rationale_prompt = None
    if with_reason:
        rationale_prompt = build_essentiality_rationale_prompt(cell_line, n)
        reason = pipe(
            inputs={"english_sequence": rationale_prompt, "dna_sequences": dna_sequences},
            max_num_tokens_to_decode=max_reason_tokens,
        )
        if isinstance(reason, list):
            reason = reason[0].get("generated_text", str(reason)) if reason else None

    logger.warning(_OOD_CAVEAT)

    label = None
    if probability == probability:  # not NaN
        label = "essential" if probability >= threshold else "non-essential"

    return {
        "model": _MODEL_NAME,
        "cell_line": cell_line,
        "n_sequences": n,
        "essentiality_probability": probability,
        "predicted_label": label,
        "threshold": threshold,
        "model_rationale": reason,
        "classification_prompt": classification_prompt,
        "rationale_prompt": rationale_prompt,
        "caveat": _OOD_CAVEAT,
    }
