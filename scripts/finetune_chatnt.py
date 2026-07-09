"""QLoRA fine-tune ChatNT on CRISPR-screen guides for log2FC prediction.

Architecture notes (from spike_chatnt_finetune.py + chatNT.py inspection):
  - Model: TorchMultiOmicsModel (PreTrainedModel subclass, trust_remote_code)
  - forward() takes:
      multi_omics_tokens_ids  = (english_token_ids, bio_token_ids)
      projection_english_tokens_ids = english_token_ids  (same tensor)
  - forward() returns dict {"logits": ..., "projected_bio_embeddings": ...}
    logits shape: (batch, seq_len, vocab_size)  — no .loss attribute
  - Generation: pipeline fills pad positions autoregressively
  - Tokenizers: two separate subfolders in the HF repo
      english_tokenizer → subfolder="english_tokenizer"
      bio_tokenizer     → subfolder="bio_tokenizer"
  - LoRA task_type must be FEATURE_EXTRACTION (model has no prepare_inputs_for_generation)
  - LoRA targets (GPT decoder attention): query_linear, key_linear, value_linear, out_linear

Prerequisites:
  Run build_finetune_data.py first.

Usage:
    uv run python scripts/finetune_chatnt.py \\
        --train data/processed/finetune_train.jsonl.gz \\
        --val   data/processed/finetune_val.jsonl.gz \\
        --output-dir data/model/chatnt_lora

    # 8-bit / no quantization
    uv run python scripts/finetune_chatnt.py --quantize 8bit ...
    uv run python scripts/finetune_chatnt.py --quantize none ...
"""
import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Chat context prepended by the pipeline before the user prompt
_CONTEXT = (
    "A chat between a curious user and an artificial intelligence assistant "
    "that can handle bio sequences. The assistant gives helpful, detailed, "
    "and polite answers to the user's questions. USER: "
)
_ASSISTANT_TAG = " ASSISTANT:"


class ScreenDataset(Dataset):
    def __init__(self, path: str | Path):
        self.examples = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ChatNTCollator:
    """Tokenize (prompt, dna_sequences, target) examples for ChatNT.

    Builds two tensors:
      english_tokens  — full padded sequence for the GPT decoder:
                        [context + prompt + ASSISTANT: + target + EOS] padded to max_length
      bio_tokens      — DNA tokens: shape (batch, num_sequences, bio_max_length).
                        num_sequences is read from the first example and assumed uniform
                        across the batch (issue #56 body-sequence redesign: every example
                        in a given dataset file has either 1 sequence (guide-only fallback)
                        or 2 (guide + transcript body) — build_finetune_data.py drops
                        targets missing a transcript rather than mixing counts within a
                        file, so a batch is never mixed).
      labels          — same length as english_tokens, -100 everywhere except target tokens

    The model generates autoregressively, predicting token[i+1] from logits[i].
    So loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100).
    """

    def __init__(
        self,
        english_tokenizer,
        bio_tokenizer,
        max_length: int = 512,
        bio_max_length: int = 512,
    ):
        self.english_tokenizer = english_tokenizer
        self.bio_tokenizer = bio_tokenizer
        self.max_length = max_length
        self.bio_max_length = bio_max_length
        self.pad_id = english_tokenizer.pad_token_id
        self.eos_id = english_tokenizer.eos_token_id

    def _tokenize_english(self, text: str) -> list[int]:
        return self.english_tokenizer(
            text, truncation=True, max_length=self.max_length
        )["input_ids"]

    def __call__(self, batch):
        english_input_list = []
        labels_list = []
        num_sequences = len(batch[0]["dna_sequences"])
        dna_raw_lists = [[] for _ in range(num_sequences)]

        for ex in batch:
            prefix_text = _CONTEXT + ex["prompt"] + _ASSISTANT_TAG
            prefix_ids = self._tokenize_english(prefix_text)

            target_text = " " + ex["target"]
            target_ids = self._tokenize_english(target_text)
            # Drop BOS if tokenizer adds one (avoid double BOS from prefix)
            if target_ids and target_ids[0] == self.english_tokenizer.bos_token_id:
                target_ids = target_ids[1:]

            # Full sequence: prefix + target + EOS, truncated to max_length
            full_ids = (prefix_ids + target_ids + [self.eos_id])[: self.max_length]

            # Pad to max_length
            pad_len = self.max_length - len(full_ids)
            full_ids_padded = full_ids + [self.pad_id] * pad_len

            # Labels: -100 for prefix and padding, real ids for target + EOS
            prefix_len = min(len(prefix_ids), self.max_length)
            target_end = min(len(prefix_ids) + len(target_ids) + 1, self.max_length)
            labels = [-100] * self.max_length
            for pos in range(prefix_len, target_end):
                labels[pos] = full_ids_padded[pos]

            english_input_list.append(full_ids_padded)
            labels_list.append(labels)
            assert len(ex["dna_sequences"]) == num_sequences, (
                "Mixed dna_sequences counts within a batch — build_finetune_data.py "
                "should drop or pad targets missing a transcript before writing the file."
            )
            for i, seq in enumerate(ex["dna_sequences"]):
                dna_raw_lists[i].append(seq)

        # Tokenize each DNA sequence position across the batch independently — ChatNT
        # NT-encodes dna_sequences[i] separately and merges it at its own <DNA>
        # placeholder (see chatNT.py's insert_embeddings loop over bio_seq_num), so each
        # position gets its own bio_max_length token budget, not a shared one.
        bio_tensors = []
        for raw_list in dna_raw_lists:
            bio_enc = self.bio_tokenizer(
                raw_list,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.bio_max_length,
            )
            bio_tensors.append(bio_enc.input_ids)
        bio_tokens = torch.stack(bio_tensors, dim=1)  # (batch, num_sequences, bio_max_length)

        return {
            "english_tokens": torch.tensor(english_input_list, dtype=torch.long),
            "bio_tokens": bio_tokens,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }


def compute_loss(logits, labels):
    """CE loss on next-token positions, ignoring -100 labels."""
    vocab_size = logits.size(-1)
    # logits[:, :-1] predicts labels[:, 1:]
    shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)


def _call_model(model, english, bio):
    """Call TorchMultiOmicsModel with correct kwargs, bypassing PeftModel's wrapper.

    PeftModelForFeatureExtraction.forward() hardcodes 'input_ids', which
    TorchMultiOmicsModel does not accept.  Calling model.base_model() goes
    directly to BaseTuner.forward() → model.forward(*args, **kwargs), which
    passes our custom kwargs through unchanged.  LoRA adapters are active at
    the module level so gradients still flow through them.
    """
    return model.base_model(
        multi_omics_tokens_ids=(english, bio),
        projection_english_tokens_ids=english,
    )


def train_epoch(model, loader, optimizer, scheduler, device, log_interval=50,
                global_step=0, max_steps=None):
    model.train()
    total_loss, steps = 0.0, 0
    for step, batch in enumerate(loader):
        if max_steps is not None and global_step >= max_steps:
            break
        english = batch["english_tokens"].to(device)
        bio = batch["bio_tokens"].to(device)
        labels = batch["labels"].to(device)

        outs = _call_model(model, english, bio)
        loss = compute_loss(outs["logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        steps += 1
        global_step += 1
        if step % log_interval == 0:
            logger.info("  step=%d  loss=%.4f", step, loss.item())
    return total_loss / max(steps, 1), global_step


@torch.no_grad()
def eval_epoch(model, loader, device, max_val_steps=500):
    model.eval()
    total_loss, steps = 0.0, 0
    for batch in loader:
        if steps >= max_val_steps:
            break
        english = batch["english_tokens"].to(device)
        bio = batch["bio_tokens"].to(device)
        labels = batch["labels"].to(device)
        outs = _call_model(model, english, bio)
        total_loss += compute_loss(outs["logits"], labels).item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def generation_eval(model, val_dataset, english_tokenizer, bio_tokenizer,
                    device, n_examples=200, max_decode_tokens=20,
                    max_length=512, bio_max_length=512):
    """Spearman ρ on a fixed val subset via autoregressive generation.

    Replicates the ChatNT pipeline generation loop without loading a second
    model copy.  Returns (rho, n_failed, n_total); rho is None if fewer than
    10 predictions parsed successfully.
    """
    import random as _random
    from lncfit.parsers import parse_log2fc
    from scipy.stats import spearmanr

    model.eval()
    rng = _random.Random(42)
    examples = rng.sample(val_dataset.examples, min(n_examples, len(val_dataset.examples)))

    pad_id = english_tokenizer.pad_token_id
    eos_id = english_tokenizer.eos_token_id
    y_true, y_pred_list, n_failed = [], [], 0

    for ex in examples:
        eng_tokens = english_tokenizer(
            _CONTEXT + ex["prompt"] + _ASSISTANT_TAG,
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_length,
        ).input_ids.to(device)

        bio_tokens = bio_tokenizer(
            ex["dna_sequences"], return_tensors="pt",
            padding="max_length", max_length=bio_max_length, truncation=True,
        ).input_ids.unsqueeze(0).to(device)  # (1, num_sequences, bio_max_length)

        projected_bio_embeddings = None
        for _ in range(max_decode_tokens):
            pad_positions = (eng_tokens[0] == pad_id).nonzero(as_tuple=True)[0]
            if len(pad_positions) == 0:
                break
            outs = model.base_model(
                multi_omics_tokens_ids=(eng_tokens, bio_tokens),
                projection_english_tokens_ids=eng_tokens,
                projected_bio_embeddings=projected_bio_embeddings,
            )
            projected_bio_embeddings = outs["projected_bio_embeddings"]
            first_pad = pad_positions[0].item()
            predicted = outs["logits"][0, first_pad - 1].argmax().item()
            if predicted == eos_id:
                break
            eng_tokens[0, first_pad] = predicted

        decoded = english_tokenizer.decode(eng_tokens[0], skip_special_tokens=True)
        generated = decoded.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in decoded else decoded
        pred = parse_log2fc(generated)
        if pred is not None:
            y_true.append(float(ex["target"]))
            y_pred_list.append(pred)
        else:
            n_failed += 1

    if len(y_pred_list) < 10:
        return None, n_failed, len(examples)
    rho, _ = spearmanr(y_true, y_pred_list)
    return float(rho), n_failed, len(examples)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune ChatNT on CRISPR-screen log2FC."
    )
    parser.add_argument("--train", default="data/processed/finetune_train.jsonl.gz")
    parser.add_argument("--val", default="data/processed/finetune_val.jsonl.gz")
    parser.add_argument("--output-dir", default="data/model/chatnt_lora")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["query_linear", "key_linear", "value_linear", "out_linear"],
        help="LoRA target module leaf names in the GPT decoder attention layers",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--quantize",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="4bit = QLoRA (default), 8bit = INT8, none = fp16",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Stop after this many total optimizer steps (useful for long datasets). "
             "None = run all epochs fully.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--bio-max-length", type=int, default=2048,
        help="Bio-tokenizer truncation length per DNA sequence (6bp/token). ChatNT's NT "
             "encoder (nt_config.max_positions) supports up to 2048 -- the architectural "
             "ceiling, not an arbitrary choice. At 2048 (~12kb), ~99.5%% of lncRNA "
             "transcript bodies fit untruncated (issue #56 body-sequence redesign); the "
             "old default of 512 (~3kb) truncated ~10%%. Going above 2048 would exceed "
             "what the encoder's position embeddings were built for.",
    )
    parser.add_argument(
        "--gen-eval-examples", type=int, default=200,
        help="Val examples for generation-based Spearman ρ per epoch. 0 = disable.",
    )
    parser.add_argument(
        "--max-val-steps", type=int, default=500,
        help="Cap on token-level val batches per epoch (eval_epoch). At bio_max_length=2048 "
             "each val batch costs roughly as much as a train step, so 500 batches is not "
             "free -- lower this for quick/exploratory runs.",
    )
    parser.add_argument(
        "--max-decode-tokens", type=int, default=20,
        help="Max tokens to generate per example during generation eval.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers from subfolders (not via AutoTokenizer root — that fails for ChatNT)
    logger.info("Loading tokenizers...")
    english_tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/ChatNT", subfolder="english_tokenizer"
    )
    bio_tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/ChatNT", subfolder="bio_tokenizer"
    )

    # Build quantization config.
    # Use bfloat16 for non-quantized layers: it has the same exponent range as
    # float32 so ChatNT's internal -1e30 attention masks don't overflow (they
    # would overflow in float16, which has a max of ~65504).
    load_kwargs: dict = {"trust_remote_code": True, "device_map": {"": 0}}
    if args.quantize == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quantize == "8bit":
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["dtype"] = torch.bfloat16

    logger.info("Loading ChatNT (quantize=%s)...", args.quantize)
    model = AutoModel.from_pretrained("InstaDeepAI/ChatNT", **load_kwargs)

    # prepare_model_for_kbit_training requires get_input_embeddings(), which
    # TorchMultiOmicsModel does not implement. Do the essential parts manually:
    #   1. Cast norm layers to float32 so they don't underflow in bfloat16.
    #   2. Register a forward hook on the GPT decoder's token embedding so that
    #      its output carries requires_grad=True. This lets gradients flow back
    #      through the frozen layers to the LoRA adapter matrices.
    if args.quantize in ("4bit", "8bit"):
        # Hook the GPT decoder embedding so its output carries requires_grad=True.
        # This allows gradients to flow back through frozen layers to LoRA adapters.
        gpt_embed = model.biobrain_decoder.gpt_model.token_embed
        gpt_embed.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))

    # Attach LoRA — TaskType.FEATURE_EXTRACTION avoids prepare_inputs_for_generation check
    from peft import LoraConfig, TaskType, get_peft_model

    logger.info("Attaching LoRA (r=%d, target=%s)...", args.lora_r, args.target_modules)
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    device = next(p for p in model.parameters() if p.requires_grad).device

    # Datasets and loaders
    collator = ChatNTCollator(
        english_tokenizer, bio_tokenizer,
        max_length=args.max_length,
        bio_max_length=args.bio_max_length,
    )
    train_ds = ScreenDataset(args.train)
    val_ds = ScreenDataset(args.val)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    best_val_loss = float("inf")
    best_gen_rho = float("-inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if args.max_steps is not None and global_step >= args.max_steps:
            logger.info("Reached max_steps=%d, stopping early.", args.max_steps)
            break
        logger.info("=== Epoch %d / %d ===", epoch, args.epochs)
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            global_step=global_step, max_steps=args.max_steps,
        )
        val_loss = eval_epoch(model, val_loader, device, max_val_steps=args.max_val_steps)
        logger.info("  train_loss=%.4f  val_loss=%.4f  global_step=%d", train_loss, val_loss, global_step)

        # Generation-based Spearman ρ — the real signal for whether the model
        # is learning to predict log2FC values, not just generate valid text.
        save_reason = None
        if args.gen_eval_examples > 0:
            gen_rho, gen_failed, gen_n = generation_eval(
                model, val_ds, english_tokenizer, bio_tokenizer, device,
                n_examples=args.gen_eval_examples,
                max_decode_tokens=args.max_decode_tokens,
                max_length=args.max_length,
                bio_max_length=args.bio_max_length,
            )
            if gen_rho is not None:
                logger.info("  gen_rho=%.4f  (parsed %d/%d)", gen_rho, gen_n - gen_failed, gen_n)
                if gen_rho > best_gen_rho:
                    best_gen_rho = gen_rho
                    save_reason = f"gen_rho={gen_rho:.4f}"
            else:
                logger.info("  gen_rho=N/A  (%d/%d unparseable)", gen_failed, gen_n)

        # Fall back to CE val loss if gen_rho not yet meaningful
        if save_reason is None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_reason = f"val_loss={val_loss:.4f}"

        if save_reason:
            ckpt = out_dir / "best_checkpoint"
            model.save_pretrained(ckpt)
            english_tokenizer.save_pretrained(ckpt / "english_tokenizer")
            bio_tokenizer.save_pretrained(ckpt / "bio_tokenizer")
            logger.info("  Saved best -> %s (%s)", ckpt, save_reason)

    final = out_dir / "final_checkpoint"
    model.save_pretrained(final)
    english_tokenizer.save_pretrained(final / "english_tokenizer")
    bio_tokenizer.save_pretrained(final / "bio_tokenizer")
    logger.info("Final checkpoint -> %s", final)

    (out_dir / "train_config.json").write_text(
        json.dumps({**vars(args), "best_val_loss": best_val_loss, "best_gen_rho": best_gen_rho}, indent=2)
    )
    logger.info("Done. Best val loss: %.4f  Best gen_rho: %.4f", best_val_loss, best_gen_rho)


if __name__ == "__main__":
    main()
