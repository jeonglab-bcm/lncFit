"""Spike: does passing [guide, transcript body] as dna_sequences work end-to-end?

PR #57 diagnosed that fine-tuning ChatNT on the 23bp guide spacer alone can't
teach essentiality signal the guide-only XGBoost baseline doesn't already
capture -- the lncRNA transcript body is where the correlating signal lives
(issue #65). The PR flagged a risk before committing to a full retrain: ChatNT's
bio_tokens_max_length=512 default and unclear multi-sequence attribution.

Checking against the actual ChatNT pipeline code
(text_generation.py / chatNT.py, cached under
~/.cache/huggingface/modules/transformers_modules/InstaDeepAI/ChatNT/) shows
this concern was based on a misreading: each dna_sequences[i] is tokenized to
its OWN bio_tokens_max_length budget (not a shared budget across the list),
NT-encoded separately, and inserted at its own <DNA> placeholder in order
(bio_seq_num=0 -> first <DNA>, bio_seq_num=1 -> second <DNA>) -- see
chatNT.py's ChatNTMultimodalGptDecoder.forward's `insert_embeddings` loop.
So [guide, body] with a two-<DNA> prompt is architecturally well-supported,
not string-concatenated-without-boundary as feared.

This script confirms that empirically: run a real record's [guide, body] pair
through pipe.preprocess + one forward pass and check shapes / that both
placeholders. Also checks what fraction of transcripts exceed the default
~3kb (512 * 6bp) truncation budget.

Usage:
  uv run python scripts/spike_multidna_body_sequence.py --n 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default="data/processed/train_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--bio-tokens-max-length", type=int, default=512)
    args = parser.parse_args()

    records = load_jsonl(args.train)
    with open(args.transcript_sequences) as fh:
        raw = json.load(fh)
    transcripts = {gene_id: seq for gene_id, (seq, _) in raw.items()}

    picked = [r for r in records if r.target in transcripts][: args.n]
    print(f"Picked {len(picked)} records with both guide + transcript body.\n")

    print("Loading ChatNT pipeline ...")
    from transformers import pipeline
    pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True)
    print("  Model loaded.\n")

    from lncfit.prompts import build_essentiality_prompt

    for rec in picked:
        guide = rec.target_sequence
        body = transcripts[rec.target]
        prompt = build_essentiality_prompt(rec.cell_line, sequence_count=2)

        model_inputs = pipe.preprocess(
            {"english_sequence": prompt, "dna_sequences": [guide, body]},
            bio_tokens_max_length=args.bio_tokens_max_length,
        )
        english_tokens = model_inputs["english_tokens"]
        bio_tokens = model_inputs["bio_tokens"]

        print(f"=== {rec.target} / {rec.guide_id} / {rec.cell_line} (true log2FC={rec.fold_change:.3f}) ===")
        print(f"  guide length      : {len(guide)} bp")
        print(f"  body length       : {len(body)} bp  (truncated in bio_tokenizer if > ~{args.bio_tokens_max_length * 6} bp)")
        print(f"  english_tokens    : {tuple(english_tokens.shape)}")
        print(f"  bio_tokens        : {tuple(bio_tokens.shape)}  (batch, num_sequences, tokens_per_sequence)")

        device = next(pipe.model.parameters()).device
        english_tokens_d = english_tokens.to(device)
        bio_tokens_d = bio_tokens.to(device)

        outs = pipe.model(
            multi_omics_tokens_ids=(english_tokens_d, bio_tokens_d),
            projection_english_tokens_ids=english_tokens_d,
            projected_bio_embeddings=None,
        )
        logits = outs["logits"]
        print(f"  forward OK        : logits shape {tuple(logits.shape)}")

        # Sanity: does swapping in a DIFFERENT transcript body change the answer-position logits?
        # If not, the model is ignoring the second <DNA> sequence entirely.
        other_target = next(t for t in transcripts if t != rec.target)
        other_body = transcripts[other_target]
        alt_inputs = pipe.preprocess(
            {"english_sequence": prompt, "dna_sequences": [guide, other_body]},
            bio_tokens_max_length=args.bio_tokens_max_length,
        )
        alt_bio_tokens_d = alt_inputs["bio_tokens"].to(device)
        alt_outs = pipe.model(
            multi_omics_tokens_ids=(english_tokens_d, alt_bio_tokens_d),
            projection_english_tokens_ids=english_tokens_d,
            projected_bio_embeddings=None,
        )
        import torch
        diff = (logits - alt_outs["logits"]).abs().max().item()
        print(f"  max |logit delta| swapping body-sequence only: {diff:.4f} "
              f"({'sensitive to body content' if diff > 1e-3 else 'INSENSITIVE -- second <DNA> may be ignored'})")
        print()


if __name__ == "__main__":
    main()
