#!/usr/bin/env python3
"""Pre-compute DNABERT-2 embeddings for lncRNA body or guide sequences.

Produces a .npz file containing:
  embeddings  float32 array of shape (n_seqs, n_dims)
  index       JSON string mapping sequence id -> row in embeddings

Usage — body sequences (primary use case):
    python scripts/embed_sequences.py \\
        --source body \\
        --body-sequences data/processed/body_sequences_transcript.json \\
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
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def _embed_batch(model, tokenizer, seqs: list[str], device: str) -> np.ndarray:
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
        out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    # DNABERT-2's BertModel returns a tuple; index 0 is last_hidden_state
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    vecs = (hidden * mask).sum(1) / mask.sum(1)
    return vecs.cpu().numpy().astype(np.float32)


def _embed_all(model, tokenizer, seqs: list[str], device: str, batch_size: int) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        all_vecs.append(_embed_batch(model, tokenizer, batch, device))
        done = min(i + batch_size, len(seqs))
        if done % (batch_size * 10) == 0 or done == len(seqs):
            print(f"  {done:,}/{len(seqs):,}", flush=True)
    return np.concatenate(all_vecs, axis=0)


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
        "--train", default="data/processed/train_chrom1.jsonl.gz",
        help="Training JSONL used to collect unique guide sequences. "
             "Required when --source=guide.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .npz path (e.g. data/processed/dnabert2_body.npz).",
    )
    parser.add_argument(
        "--model", default="zhihan1996/DNABERT-2-117M",
        help="HuggingFace model name or local checkpoint path. Default: zhihan1996/DNABERT-2-117M.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
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
    from transformers import AutoModel, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model {args.model!r} ...")
    # Pin revision so transformers does not re-download and overwrite the patched
    # flash_attn_triton.py (trans_b -> tl.trans() fix for Triton >= 2.3).
    _revision = "7bce263b15377fc15361f52cfab88f8b586abda0"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, revision=_revision)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, revision=_revision)
    model.eval().to(device)

    if args.source == "body":
        body_path = Path(args.body_sequences)
        if not body_path.exists():
            sys.exit(f"Body sequences file not found: {body_path}")
        print(f"Loading body sequences from {body_path} ...")
        with open(body_path) as fh:
            raw: dict[str, list[str]] = json.load(fh)
        gene_ids = list(raw.keys())
        print(f"  {len(gene_ids):,} genes")

        if args.window == "mean":
            seqs_first = [raw[g][0] for g in gene_ids]
            seqs_last  = [raw[g][1] for g in gene_ids]
            print(f"Embedding first windows ...")
            emb_first = _embed_all(model, tokenizer, seqs_first, device, args.batch_size)
            print(f"Embedding last windows ...")
            emb_last  = _embed_all(model, tokenizer, seqs_last,  device, args.batch_size)
            embeddings = (emb_first + emb_last) / 2.0
        else:
            window_idx = 0 if args.window == "first" else 1
            seqs = [raw[g][window_idx] for g in gene_ids]
            print(f"Embedding {len(seqs):,} {args.window}-window sequences ...")
            embeddings = _embed_all(model, tokenizer, seqs, device, args.batch_size)

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
        embeddings = _embed_all(model, tokenizer, unique_seqs, device, args.batch_size)
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
