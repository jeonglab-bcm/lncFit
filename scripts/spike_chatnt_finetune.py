"""Probe ChatNT for fine-tunability via HF transformers + PEFT.

Outputs a JSON report with:
  model_class            — class name of the loaded model
  is_pretrained_model    — whether it subclasses PreTrainedModel (required for PEFT)
  forward_params         — parameter names of model.forward()
  named_modules_count    — total number of named modules
  named_modules_sample   — first 200 (name, class_name) pairs
  lora_candidates        — module names matching common attention projection patterns
  peft_attached          — whether LoraConfig + get_peft_model succeeded
  peft_target_modules    — leaf names passed to LoraConfig
  trainable_params       — trainable parameter count if PEFT attached
  total_params           — total parameter count
  trainable_pct          — percentage of parameters that are trainable

Run this before finetune_chatnt.py to confirm --target-modules.

Usage:
    uv run python scripts/spike_chatnt_finetune.py
    uv run python scripts/spike_chatnt_finetune.py --dry-run
    uv run python scripts/spike_chatnt_finetune.py --output data/model/chatnt_spike.json
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, PreTrainedModel

# Common attention/MLP projection suffixes across LLM families
_LORA_PATTERNS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value",
    "query_key_value", "dense",
    "fc1", "fc2",
    "gate_proj", "up_proj", "down_proj",
)


def probe(dry_run: bool) -> dict:
    report: dict = {}

    if dry_run:
        report["dry_run"] = True
        report["message"] = "Skipped model load in dry-run mode."
        return report

    print("Loading ChatNT from HuggingFace (may download ~8 GB on first run)...", file=sys.stderr)
    model = AutoModel.from_pretrained(
        "InstaDeepAI/ChatNT",
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="cpu",
    )

    # Class hierarchy
    report["model_class"] = type(model).__name__
    report["model_mro"] = [c.__name__ for c in type(model).__mro__]
    report["is_pretrained_model"] = isinstance(model, PreTrainedModel)

    # Forward signature
    try:
        sig = inspect.signature(model.forward)
        report["forward_params"] = list(sig.parameters.keys())
    except (ValueError, TypeError) as exc:
        report["forward_params_error"] = str(exc)

    # All named modules
    modules = [(name, type(mod).__name__) for name, mod in model.named_modules()]
    report["named_modules_count"] = len(modules)
    report["named_modules_sample"] = modules[:200]

    # LoRA candidates: module paths whose last segment matches a known pattern
    lora_candidates = [
        name for name, _ in modules
        if any(name.endswith(p) for p in _LORA_PATTERNS)
    ]
    report["lora_candidates"] = lora_candidates[:50]

    # Attempt PEFT attachment
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        # Unique leaf names from the first four candidates, or a safe default
        if lora_candidates:
            unique_targets = list(dict.fromkeys(
                name.split(".")[-1] for name in lora_candidates[:4]
            ))
        else:
            unique_targets = ["q_proj", "v_proj"]

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=unique_targets,
            lora_dropout=0.05,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_cfg)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())

        report["peft_attached"] = True
        report["peft_target_modules"] = unique_targets
        report["trainable_params"] = trainable
        report["total_params"] = total
        report["trainable_pct"] = round(100 * trainable / total, 3) if total else 0.0

    except Exception as exc:
        report["peft_attached"] = False
        report["peft_error"] = str(exc)

    return report


def main():
    parser = argparse.ArgumentParser(description="Probe ChatNT for fine-tunability.")
    parser.add_argument("--dry-run", action="store_true", help="Skip model download/load.")
    parser.add_argument("--output", help="Write JSON report to this file (default: stdout).")
    args = parser.parse_args()

    report = probe(args.dry_run)
    out = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
