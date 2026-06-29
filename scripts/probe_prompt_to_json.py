"""
Probe the prompt → ChatNT → JSON pipeline for lncRNA essentiality prediction.

Investigation only: validates that prompt/IO plumbing is sound before any
fine-tuning. No model training is performed.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from lncfit.screen_data import ScreenRecord, load_jsonl
from lncfit.prompts import build_training_example


def build_probe_record(
    record: ScreenRecord,
    prompt: str,
    raw_response: str | None = None,
    parsed_log2fc: float | None = None,
) -> dict:
    return {
        "cell_line": record.cell_line,
        "guide_sequence": record.target_sequence,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_log2fc": parsed_log2fc,
        "expected_log2fc": record.fold_change,
        "parse_ok": parsed_log2fc is not None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe prompt → ChatNT → JSON pipeline for lncRNA essentiality."
    )
    parser.add_argument("--data", required=True, help="Path to screen_records.jsonl.gz")
    parser.add_argument("--n", type=int, default=5, help="Number of records to probe (default: 5)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build JSON without loading or running the model",
    )
    parser.add_argument(
        "--out-dir",
        default="results/prompt_probe",
        help="Output directory for probe JSON files (default: results/prompt_probe)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    records = load_jsonl(args.data)[: args.n]
    if not records:
        print("Error: no records loaded from --data.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parse_ok_count = 0
    results = []

    for i, record in enumerate(records):
        prompt, dna_sequences, _ = build_training_example(record)

        if args.dry_run:
            probe = build_probe_record(record, prompt)
        else:
            from lncfit.inference import run_chatnt_inference_full
            raw_response, parsed = run_chatnt_inference_full(prompt, dna_sequences)
            probe = build_probe_record(record, prompt, raw_response, parsed)

        results.append(probe)
        if probe["parse_ok"]:
            parse_ok_count += 1

        out_path = out_dir / f"probe_{i:04d}.json"
        out_path.write_text(json.dumps(probe, indent=2))
        logger.info("Wrote %s", out_path)

    print(f"\n=== Probe summary ({len(results)} records) ===")
    if args.dry_run:
        print("[dry-run] Model not called. JSON structure validated.")
    else:
        print(f"Parse success rate : {parse_ok_count}/{len(results)}")
        parsed_vals = [r["parsed_log2fc"] for r in results if r["parse_ok"]]
        if parsed_vals:
            print(f"log2FC range       : [{min(parsed_vals):.3f}, {max(parsed_vals):.3f}]")
        else:
            print("log2FC range       : (no successful parses)")


if __name__ == "__main__":
    main()
