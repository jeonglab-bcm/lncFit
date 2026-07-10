"""Evaluate fine-tuned or zero-shot ChatNT on the held-out test split.

For fine-tuned mode: loads the LoRA checkpoint, rebuilds the ChatNT pipeline
with the adapted model, and runs generation identically to zero-shot.

For zero-shot mode: runs the unmodified ChatNT pipeline.

Computes Spearman rho, Pearson r, RMSE, MAE, R² via lncfit/metrics.py.
Writes metrics.json + predictions.csv to the output directory.

Usage:
    # Fine-tuned model
    uv run python scripts/evaluate_chatnt.py \\
        --mode finetuned \\
        --model data/model/chatnt_lora/best_checkpoint \\
        --test  data/processed/finetune_test.jsonl.gz

    # Zero-shot baseline
    uv run python scripts/evaluate_chatnt.py \\
        --mode zero-shot \\
        --test  data/processed/finetune_test.jsonl.gz
"""
import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.metrics import compute_metrics
from lncfit.parsers import parse_log2fc

logger = logging.getLogger(__name__)


def load_examples(path):
    examples = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def make_pipeline_with_lora(model_dir: str, device: str):
    """Load the base ChatNT pipeline, swap its model for the LoRA-adapted one."""
    from peft import PeftModel

    # Load pipeline once (device_map="auto" → GPU, bfloat16 avoids -1e30 overflow).
    # The pipeline __init__ loads its own tokenizers, so no separate load needed.
    logger.info("Loading ChatNT pipeline (bfloat16, device_map=auto)...")
    pipe = pipeline(
        model="InstaDeepAI/ChatNT",
        trust_remote_code=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    logger.info("Attaching LoRA from %s...", model_dir)
    lora_model = PeftModel.from_pretrained(pipe.model, model_dir)
    lora_model.eval()

    # PeftModelForFeatureExtraction.forward() hardcodes input_ids=input_ids, but
    # ChatNT's pipeline calls self.model(multi_omics_tokens_ids=...).
    # BaseTuner.forward() passes *args/**kwargs through unchanged with LoRA active.
    pipe.model = lora_model.base_model
    return pipe


def run_inference(pipe, examples: list, bio_tokens_max_length: int = 2048) -> list[float | None]:
    preds = []
    for ex in examples:
        result = pipe(
            inputs={
                "english_sequence": ex["prompt"],
                "dna_sequences": ex["dna_sequences"],
            },
            bio_tokens_max_length=bio_tokens_max_length,
        )
        raw = result[0]["generated_text"] if isinstance(result, list) else str(result)
        preds.append(parse_log2fc(raw))
    return preds


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ChatNT (fine-tuned or zero-shot)."
    )
    parser.add_argument("--mode", choices=["finetuned", "zero-shot"], required=True)
    parser.add_argument(
        "--model",
        help="Path to fine-tuned LoRA checkpoint dir (required for --mode finetuned)",
    )
    parser.add_argument("--test", default="data/processed/finetune_test.jsonl.gz")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Cap the number of test examples to evaluate (default: all). "
             "34K examples × ~2 sec = ~19 hours without a cap.",
    )
    parser.add_argument(
        "--bio-max-length", type=int, default=2048,
        help="Bio-tokenizer truncation length per DNA sequence. Matches ChatNT's NT "
             "encoder architectural ceiling (nt_config.max_positions=2048) so the "
             "lncRNA transcript body (issue #56 redesign) is not silently truncated "
             "at the old pipeline default of 512 (~3kb).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "finetuned" and not args.model:
        print("ERROR: --model is required for --mode finetuned", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) / f"chatnt_{args.mode.replace('-', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.test)
    logger.info("Loaded %d test examples from %s", len(examples), args.test)
    if args.max_examples is not None and len(examples) > args.max_examples:
        import random
        random.seed(42)
        random.shuffle(examples)
        logger.info("Shuffled + capping to %d examples (--max-examples)", args.max_examples)
        examples = examples[: args.max_examples]

    if args.mode == "finetuned":
        pipe = make_pipeline_with_lora(args.model, args.device)
    else:
        logger.info("Loading zero-shot ChatNT pipeline (bfloat16, device_map=auto)...")
        pipe = pipeline(
            model="InstaDeepAI/ChatNT",
            trust_remote_code=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

    preds = run_inference(pipe, examples, bio_tokens_max_length=args.bio_max_length)

    # Filter unparseable predictions
    valid = [(ex, p) for ex, p in zip(examples, preds) if p is not None]
    n_failed = len(examples) - len(valid)
    if n_failed:
        logger.warning(
            "%d / %d examples produced no parseable float (%.1f%%)",
            n_failed, len(examples), 100 * n_failed / len(examples),
        )

    if not valid:
        print("ERROR: no parseable predictions — check model output format.", file=sys.stderr)
        sys.exit(1)

    y_true = np.array([float(ex["target"]) for ex, _ in valid])
    y_pred = np.array([float(p) for _, p in valid])

    print(f"\n=== {args.mode} ChatNT  ({len(valid):,} examples) ===")
    metrics = compute_metrics(args.mode, y_true, y_pred)
    metrics["n_failed"] = n_failed

    # Per-cell-line breakdown
    per_cl: dict = {}
    for cell_line in sorted({ex["cell_line"] for ex, _ in valid}):
        subset = [(ex, p) for ex, p in valid if ex["cell_line"] == cell_line]
        yt = np.array([float(ex["target"]) for ex, _ in subset])
        yp = np.array([float(p) for _, p in subset])
        per_cl[cell_line] = compute_metrics(cell_line, yt, yp)
    metrics["per_cell_line"] = per_cl

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics -> %s", metrics_path)

    preds_rows = [
        {
            "guide_id": ex["guide_id"],
            "cell_line": ex["cell_line"],
            "day": ex["day"],
            "y_true": float(ex["target"]),
            "y_pred": float(p),
        }
        for ex, p in valid
    ]
    preds_path = out_dir / "predictions.csv"
    pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
    logger.info("Predictions -> %s", preds_path)


if __name__ == "__main__":
    main()
