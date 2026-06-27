import logging

import torch
from transformers import pipeline

from lncfit.parsers import parse_log2fc

logger = logging.getLogger(__name__)


def run_chatnt_inference(
    prompt: str,
    dna_sequences: list[str],
    lora_checkpoint: str | None = None,
) -> float | None:
    """Run ChatNT inference and return parsed log2FC.

    Args:
        prompt: English prompt string (without <DNA> expansion — the pipeline handles that).
        dna_sequences: List of DNA sequences passed through the NT encoder.
        lora_checkpoint: Path to a fine-tuned LoRA checkpoint directory produced by
            scripts/finetune_chatnt.py.  If None, runs zero-shot inference with the
            unmodified InstaDeepAI/ChatNT weights.
    """
    pipe = pipeline(
        model="InstaDeepAI/ChatNT",
        trust_remote_code=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    if lora_checkpoint is not None:
        from peft import PeftModel
        logger.info("Loading LoRA checkpoint from %s...", lora_checkpoint)
        lora_model = PeftModel.from_pretrained(pipe.model, lora_checkpoint)
        lora_model.eval()
        pipe.model = lora_model.base_model
        logger.info("Fine-tuned ChatNT loaded.")
    else:
        logger.warning(
            "Running zero-shot ChatNT inference. log2FC prediction is outside "
            "ChatNT's documented training tasks — pass lora_checkpoint= to use "
            "the fine-tuned model."
        )

    result = pipe(inputs={"english_sequence": prompt, "dna_sequences": dna_sequences})
    raw_response = result[0]["generated_text"] if isinstance(result, list) else str(result)

    logger.info("=== Raw ChatNT response ===")
    logger.info(raw_response)

    value = parse_log2fc(raw_response)
    if value is not None:
        logger.info("Parsed log2FC: %s", value)
    else:
        logger.info("Parsed log2FC: (no numeric value found in response)")

    return value
