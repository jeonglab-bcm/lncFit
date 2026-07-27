#!/usr/bin/env python3
"""Pre-compute DNABERT-2 embeddings for lncRNA body or guide sequences.

Produces a .npz file containing:
  embeddings  float32 array of shape (n_seqs, n_dims)
  index       JSON string mapping sequence id -> row in embeddings

Usage — body sequences (primary use case):
    python scripts/embed_sequences.py \\
        --source body \\
        --body-sequences data/processed/body_sequences_transcript.json \\
        --target-records data/processed/train_lncrna_day14_chrom1.jsonl.gz \\
                         data/processed/test_lncrna_day14_chrom1.jsonl.gz \\
        --output data/processed/dnabert2_body.npz

Usage — guide sequences (secondary / ablation):
    python scripts/embed_sequences.py \\
        --source guide \\
        --train data/processed/train_chrom1.jsonl.gz \\
        --output data/processed/dnabert2_guide.npz

For body sequences the --window flag selects which 1000 bp window to embed:
  first  embed the first 1000 bp of each transcript (default)
  last   embed the last 1000 bp
  mean   embed both and average the resulting vectors

Requires:  transformers, torch  (both already project dependencies)
DNABERT-2-117M uses custom model code; trust_remote_code=True is required.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DNABERT_MODEL = "zhihan1996/DNABERT-2-117M"
DNABERT_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
PINNED_REVISIONS = {
    DNABERT_MODEL: DNABERT_REVISION,
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species":
        "81b29e5786726d891dbf929404ef20adca5b36f1",
}


def _embed_batch(
    model, tokenizer, seqs: list[str], device: str, masked_lm: bool = False
) -> np.ndarray:
    """Mean-pool last hidden states over non-padding tokens -> (batch, n_dims)."""
    import torch

    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True) if masked_lm else model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    # DNABERT-2's BertModel returns a tuple; index 0 is last_hidden_state
    if masked_lm:
        hidden = out.hidden_states[-1]
    else:
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    vecs = (hidden * mask).sum(1) / mask.sum(1)
    return vecs.cpu().numpy().astype(np.float32)


def _embed_all(
    model,
    tokenizer,
    seqs: list[str],
    device: str,
    batch_size: int,
    masked_lm: bool = False,
) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        all_vecs.append(_embed_batch(model, tokenizer, batch, device, masked_lm))
        done = min(i + batch_size, len(seqs))
        if done % (batch_size * 10) == 0 or done == len(seqs):
            print(f"  {done:,}/{len(seqs):,}", flush=True)
    return np.concatenate(all_vecs, axis=0)


def _prepare_model_source(model: str, revision: str) -> str:
    """Create a no-Triton local snapshot when the accelerator is unavailable.

    DNABERT-2's ``bert_layers.py`` has an official PyTorch attention fallback,
    but Transformers' remote-code dependency scanner rejects the auxiliary
    ``flash_attn_triton.py`` before that fallback can run.  Replacing only that
    optional auxiliary module with an ImportError stub lets ``bert_layers`` take
    its existing fallback path.  Model weights and all numerical layer code are
    unchanged.
    """
    if importlib.util.find_spec("triton") is not None or Path(model).exists():
        return model

    from huggingface_hub import snapshot_download

    cache_root = Path(
        os.environ.get(
            "HF_HOME", str(Path(tempfile.gettempdir()) / "lncfit-huggingface")
        )
    )
    local_dir = cache_root / "dnabert2-no-triton" / revision
    print(f"Triton unavailable; preparing PyTorch-fallback snapshot at {local_dir}")
    snapshot_download(repo_id=model, revision=revision, local_dir=local_dir)
    triton_module = local_dir / "flash_attn_triton.py"
    if triton_module.exists():
        triton_module.write_text(
            'raise ImportError("Triton unavailable; use DNABERT-2 PyTorch attention fallback")\n'
        )
    return str(local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed DNA sequences with DNABERT-2 and save as .npz."
    )
    parser.add_argument(
        "--source", choices=["body", "guide"], required=True,
        help="'body' to embed lncRNA body windows; 'guide' to embed CRISPR spacer sequences.",
    )
    parser.add_argument(
        "--body-sequences",
        default="data/processed/body_sequences_transcript.json",
        help="Path to body sequences JSON produced by lncfit/sequence.py. "
             "Required when --source=body.",
    )
    parser.add_argument(
        "--target-records",
        nargs="+",
        default=None,
        help="Optional JSONL/JSONL.GZ record files whose target IDs define the "
             "body-sequence subset to embed. This avoids embedding unrelated "
             "transcripts from a genome-wide sequence file.",
    )
    parser.add_argument(
        "--train", default="data/processed/train_chrom1.jsonl.gz",
        help="Training JSONL used to collect unique guide sequences. "
             "Required when --source=guide.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .npz path (e.g. data/processed/dnabert2_body.npz).",
    )
    parser.add_argument(
        "--model", default=DNABERT_MODEL,
        help=f"HuggingFace model name or local checkpoint path. Default: {DNABERT_MODEL}.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HuggingFace revision. DNABERT-2 uses the project's pinned "
             "revision by default; other models use their repository default.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--masked-lm",
        action="store_true",
        help="Load AutoModelForMaskedLM and pool its final hidden state. Required "
             "for repositories that do not expose an AutoModel mapping.",
    )
    parser.add_argument(
        "--device", default=None,
        help="'cuda' or 'cpu'. Auto-detected when not specified.",
    )
    parser.add_argument(
        "--window", choices=["first", "last", "mean"], default="first",
        help="Which body window to embed (--source=body only). "
             "first/last use a single 1000 bp window; mean averages both. Default: first.",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading model {args.model!r} ...")
    # Pin revision so transformers does not re-download and overwrite the patched
    # flash_attn_triton.py (trans_b -> tl.trans() fix for Triton >= 2.3).
    revision = args.revision or PINNED_REVISIONS.get(args.model)
    model_source = (
        _prepare_model_source(args.model, revision)
        if args.model == DNABERT_MODEL
        else args.model
    )
    revision_kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_source, trust_remote_code=True, **revision_kwargs
    )
    model_class = AutoModelForMaskedLM if args.masked_lm else AutoModel
    model = model_class.from_pretrained(
        model_source, trust_remote_code=True, **revision_kwargs
    )
    model.eval().to(device)

    if args.source == "body":
        body_path = Path(args.body_sequences)
        if not body_path.exists():
            sys.exit(f"Body sequences file not found: {body_path}")
        print(f"Loading body sequences from {body_path} ...")
        with open(body_path) as fh:
            raw: dict[str, list[str]] = json.load(fh)
        if args.target_records:
            from lncfit.screen_data import LncRnaRecord, load_jsonl

            requested = {
                record.target
                for record_path in args.target_records
                for record in load_jsonl(Path(record_path), record_cls=LncRnaRecord)
            }
            missing = sorted(requested - raw.keys())
            if missing:
                preview = ", ".join(missing[:5])
                sys.exit(
                    f"{len(missing):,} requested target(s) are missing from "
                    f"{body_path}: {preview}"
                )
            gene_ids = sorted(requested)
        else:
            gene_ids = list(raw.keys())
        print(f"  {len(gene_ids):,} genes")

        if args.window == "mean":
            seqs_first = [raw[g][0] for g in gene_ids]
            seqs_last = [
                raw[g][1] if len(raw[g]) > 1 and raw[g][1] else raw[g][0][-1000:]
                for g in gene_ids
            ]
            print(f"Embedding first windows ...")
            emb_first = _embed_all(
                model, tokenizer, seqs_first, device, args.batch_size, args.masked_lm
            )
            print(f"Embedding last windows ...")
            emb_last = _embed_all(
                model, tokenizer, seqs_last, device, args.batch_size, args.masked_lm
            )
            embeddings = (emb_first + emb_last) / 2.0
        else:
            window_idx = 0 if args.window == "first" else 1
            if window_idx == 0:
                seqs = [raw[g][0] for g in gene_ids]
            else:
                seqs = [
                    raw[g][1] if len(raw[g]) > 1 and raw[g][1] else raw[g][0][-1000:]
                    for g in gene_ids
                ]
            print(f"Embedding {len(seqs):,} {args.window}-window sequences ...")
            embeddings = _embed_all(
                model, tokenizer, seqs, device, args.batch_size, args.masked_lm
            )

        index = {g: i for i, g in enumerate(gene_ids)}

    else:  # guide
        from lncfit.screen_data import load_jsonl

        train_path = Path(args.train)
        if not train_path.exists():
            sys.exit(f"Training JSONL not found: {train_path}")
        print(f"Loading guide sequences from {train_path} ...")
        records = load_jsonl(train_path)
        # Deduplicate by target_sequence; index key = target_sequence for lookup in features.py
        unique_seqs: list[str] = []
        seen: set[str] = set()
        for r in records:
            if r.target_sequence not in seen:
                unique_seqs.append(r.target_sequence)
                seen.add(r.target_sequence)
        print(f"  {len(unique_seqs):,} unique guide sequences (from {len(records):,} records)")
        print(f"Embedding {len(unique_seqs):,} guide sequences ...")
        embeddings = _embed_all(
            model, tokenizer, unique_seqs, device, args.batch_size, args.masked_lm
        )
        index = {s: i for i, s in enumerate(unique_seqs)}

    print(f"\nEmbedding matrix: {embeddings.shape}  dtype={embeddings.dtype}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        embeddings=embeddings,
        index=np.array(json.dumps(index)),
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
