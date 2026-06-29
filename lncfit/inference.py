import logging

from transformers import pipeline

from lncfit.parsers import parse_log2fc

logger = logging.getLogger(__name__)


def run_chatnt_inference_full(prompt: str, dna_sequences: list[str]) -> tuple[str, float | None]:
    """Load ChatNT, run inference. Returns (raw_response, parsed_log2fc)."""
    pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True)
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

    return raw_response, value


def run_chatnt_inference(prompt: str, dna_sequences: list[str]) -> float | None:
    """Load ChatNT, run inference, log raw response, return parsed log2FC."""
    _, value = run_chatnt_inference_full(prompt, dna_sequences)
    return value
