#!/usr/bin/env bash
# Driver script for issue #56: fine-tune ChatNT on CRISPR-screen guides.
# Runs from the project root (/home/kellyl/lncFit).
# Steps:
#   1. Spike — probe ChatNT trainability (skipped if spike JSON already exists)
#   2. Build fine-tuning datasets (skipped if output files already exist)
#   3. Fine-tune with QLoRA
#   4. Evaluate fine-tuned model vs zero-shot ChatNT
set -euo pipefail

cd /home/kellyl/lncFit

LOG_DIR="results/issue56"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --------------------------------------------------------------------------- #
# Step 1: Spike (skip if already done)
# --------------------------------------------------------------------------- #
SPIKE_JSON="data/model/chatnt_spike.json"
if [ -f "$SPIKE_JSON" ]; then
    log "=== Step 1/4: Spike already complete (${SPIKE_JSON}), skipping ==="
else
    log "=== Step 1/4: Spike ChatNT trainability ==="
    mkdir -p data/model
    uv run python scripts/spike_chatnt_finetune.py --output "$SPIKE_JSON" \
        2>&1 | tee "$LOG_DIR/spike.log"
    log "Spike complete."
fi

# GPT decoder attention projections (confirmed from chatNT.py source inspection)
# These are the LoRA targets for fine-tuning the LLM decoder side of ChatNT.
TARGET_MODULES="query_linear key_linear value_linear out_linear"
log "LoRA target modules: $TARGET_MODULES"

# --------------------------------------------------------------------------- #
# Step 2: Build fine-tuning datasets (skip if already done)
# --------------------------------------------------------------------------- #
FINETUNE_TRAIN="data/processed/finetune_train.jsonl.gz"
if [ -f "$FINETUNE_TRAIN" ]; then
    log "=== Step 2/4: Fine-tuning datasets already built, skipping ==="
else
    log "=== Step 2/4: Build fine-tuning datasets ==="
    uv run python scripts/build_finetune_data.py \
        --train-glob "data/processed/train_chrom1.jsonl.gz" \
        --test "data/processed/test_chrom1.jsonl.gz" \
        --output-dir "data/processed" \
        2>&1 | tee "$LOG_DIR/build_data.log"
    log "Datasets written."
fi

# --------------------------------------------------------------------------- #
# Step 3: Fine-tune
# --------------------------------------------------------------------------- #
LORA_FINAL="data/model/chatnt_lora/final_checkpoint"
if [ -d "$LORA_FINAL" ]; then
    log "=== Step 3/4: Fine-tune already complete (${LORA_FINAL}), skipping ==="
else
    log "=== Step 3/4: QLoRA fine-tune ChatNT ==="
fi
if [ ! -d "$LORA_FINAL" ]; then
# shellcheck disable=SC2086
uv run python scripts/finetune_chatnt.py \
    --train "data/processed/finetune_train.jsonl.gz" \
    --val   "data/processed/finetune_val.jsonl.gz" \
    --target-modules $TARGET_MODULES \
    --lora-r 16 \
    --lora-alpha 32 \
    --epochs 3 \
    --max-steps 20000 \
    --batch-size 4 \
    --quantize 4bit \
    --gen-eval-examples 200 \
    --output-dir "data/model/chatnt_lora" \
    2>&1 | tee "$LOG_DIR/finetune.log"
fi

log "Fine-tuning complete. Checkpoints in data/model/chatnt_lora/"

# --------------------------------------------------------------------------- #
# Step 4: Evaluate
# --------------------------------------------------------------------------- #
log "=== Step 4/4: Evaluate ==="

log "  4a. Fine-tuned model..."
if [ -f "results/chatnt_finetuned/metrics.json" ]; then
    log "  Fine-tuned eval already complete, skipping."
else
    uv run python scripts/evaluate_chatnt.py \
        --mode finetuned \
        --model "data/model/chatnt_lora/best_checkpoint" \
        --test  "data/processed/finetune_test.jsonl.gz" \
        --max-examples 3000 \
        --output-dir "results" \
        2>&1 | tee "$LOG_DIR/eval_finetuned.log"
fi

log "  4b. Zero-shot baseline..."
if [ -f "results/chatnt_zero_shot/metrics.json" ]; then
    log "  Zero-shot eval already complete, skipping."
else
    uv run python scripts/evaluate_chatnt.py \
        --mode zero-shot \
        --test  "data/processed/finetune_test.jsonl.gz" \
        --max-examples 3000 \
        --output-dir "results" \
        2>&1 | tee "$LOG_DIR/eval_zero_shot.log"
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
log "=== Done ==="
log "Logs:       $LOG_DIR/"
log "Model:      data/model/chatnt_lora/best_checkpoint"
log "Metrics (Spearman rho / Pearson r / RMSE):"
for f in results/chatnt_finetuned/metrics.json results/chatnt_zero_shot/metrics.json; do
    [ -f "$f" ] && python3 -c "
import json
m = json.load(open('$f'))
print(f\"  {m['split']:<22}  n={m['n']:>6,}  rho={m['spearman_rho']:.4f}  r={m['pearson_r']:.4f}  RMSE={m['rmse']:.4f}  parse_fail={m.get('n_failed',0)}\")
"
done
# XGBoost baseline for comparison.
# Guide k-mers only (k=3, no body sequences) — same sequence input as ChatNT.
# train_chrom1.jsonl.gz / test_chrom1.jsonl.gz, reg:squarederror, no leakage.
XGBOOST_BEST="results/final_eval_20260615_132231/metrics.csv"
if [ -f "$XGBOOST_BEST" ]; then
    python3 -c "
import csv
with open('$XGBOOST_BEST') as f:
    rows = list(csv.DictReader(f))
overall = next(r for r in rows if r['split'] == 'Overall')
print(f\"  {'XGBoost (baseline)':<22}  n={int(overall['n']):>6,}  rho={float(overall['spearman_rho']):.4f}  r={float(overall['pearson_r']):.4f}  RMSE={float(overall['rmse']):.4f}\")
"
fi
