"""Full held-out evaluation of the ChatNT zero-shot lncRNA essentiality classifier.

PR #71 shipped the zero-shot classifier wrapper and spot-checked it on 2 real
lncRNAs (n=2 — plumbing-only, no discrimination signal). The review flagged the
natural follow-up: run the full chr1 test set (2,470 records) and compute a real
AUROC/AUPRC, comparable to the existing XGBoost evaluation
(scripts/run_lncrna_classifier.py / results/lncrna_rra_day14/roc_curves.png).

One forward pass per record (with_reason=False — probability only; the PR's
rationale spot-check already showed the free-text reason is unreliable and
doubling the cost buys nothing for the AUROC number). The model is loaded once
and reused across all records.

Checkpointed: predictions are appended to predictions.csv as they're produced
and already-scored targets are skipped on restart, so an interrupted multi-hour
run resumes instead of restarting.

Usage:
  uv run python scripts/evaluate_chatnt_classifier.py
  uv run python scripts/evaluate_chatnt_classifier.py --limit 200   # subsample sanity check
  uv run python scripts/evaluate_chatnt_classifier.py --output-dir results/lncrna_rra_day14/chatnt_eval/run_full
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.io import git_commit
from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

PRED_FIELDS = ["target", "cell_line", "y_true", "y_pred_proba"]


def _load_transcript_sequences(path: str) -> dict[str, str]:
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def _load_existing_predictions(csv_path: Path) -> dict[tuple[str, str], float]:
    """(target, cell_line) -> y_pred_proba for rows already written, so a resumed run skips them."""
    if not csv_path.exists():
        return {}
    done = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            done[(row["target"], row["cell_line"])] = float(row["y_pred_proba"])
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="data/processed/test_lncrna_day14_chrom1.jsonl.gz")
    parser.add_argument("--transcript-sequences", default="data/processed/body_sequences_transcript.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N test records (sanity check).")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to results/lncrna_rra_day14/chatnt_eval/run_<timestamp>, "
                             "or pass an existing run dir to resume it.")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"Loading test records from {args.test} ...")
    test_records = load_jsonl(args.test, record_cls=LncRnaRecord)
    if args.limit:
        test_records = test_records[: args.limit]
    print(f"  {len(test_records):,} records")

    print(f"Loading transcript sequences from {args.transcript_sequences} ...")
    transcript_sequences = _load_transcript_sequences(args.transcript_sequences)
    print(f"  {len(transcript_sequences):,} lncRNAs")

    scorable = [r for r in test_records if r.target in transcript_sequences]
    missing = len(test_records) - len(scorable)
    if missing:
        print(f"  {missing} test records have no transcript sequence — skipped.")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results/lncrna_rra_day14/chatnt_eval") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.csv"

    done = _load_existing_predictions(preds_path)
    if done:
        print(f"Resuming: {len(done)} predictions already on disk at {preds_path}")

    remaining = [r for r in scorable if (r.target, r.cell_line) not in done]
    print(f"  {len(remaining)} records left to score ({len(scorable) - len(remaining)} already done)\n")

    if remaining:
        print("Loading ChatNT model ...")
        from lncfit.inference import run_chatnt_zeroshot_classifier
        from transformers import pipeline
        pipe = pipeline(model="InstaDeepAI/ChatNT", trust_remote_code=True)
        print("  Model loaded.\n")

        write_header = not preds_path.exists()
        with open(preds_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=PRED_FIELDS)
            if write_header:
                writer.writeheader()

            start = time.time()
            for i, rec in enumerate(remaining, 1):
                seq = transcript_sequences[rec.target]
                result = run_chatnt_zeroshot_classifier(
                    cell_line=rec.cell_line,
                    dna_sequences=[seq],
                    with_reason=False,
                    threshold=args.threshold,
                    pipe=pipe,
                )
                proba = result["essentiality_probability"]
                writer.writerow({
                    "target": rec.target,
                    "cell_line": rec.cell_line,
                    "y_true": float(rec.label),
                    "y_pred_proba": proba,
                })
                fh.flush()

                if i % args.log_every == 0 or i == len(remaining):
                    elapsed = time.time() - start
                    rate = elapsed / i
                    eta_min = rate * (len(remaining) - i) / 60
                    print(f"  [{i}/{len(remaining)}] {rec.target} {rec.cell_line} "
                          f"P(essential)={proba:.3f} ({rate:.1f}s/rec, ETA {eta_min:.0f}m)")

    print(f"\nAll predictions written -> {preds_path}")

    all_preds = _load_existing_predictions(preds_path)
    lookup = {(r.target, r.cell_line): r for r in scorable}
    ordered = [(t, cl) for (t, cl) in all_preds if (t, cl) in lookup]
    eval_records = [lookup[key] for key in ordered]
    y_true = [lookup[key].label for key in ordered]
    y_pred_proba = [all_preds[key] for key in ordered]

    metrics_rows = evaluate_lncrna_by_group(eval_records, y_true, y_pred_proba)
    metrics_path = out_dir / "metrics.csv"
    import pandas as pd
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"Metrics CSV -> {metrics_path}")
    for row in metrics_rows:
        print(f"  {row}")

    run_info = {
        "model": "InstaDeepAI/ChatNT",
        "eval": "zero-shot classifier, full chr1 held-out test set (PR #71 follow-up)",
        "with_reason": False,
        "threshold": args.threshold,
        "test_file": str(args.test),
        "n_test_total": len(test_records),
        "n_missing_sequence": missing,
        "n_scored": len(ordered),
        "git_commit": git_commit(),
    }
    run_info_path = out_dir / "run_info.json"
    with open(run_info_path, "w") as fh:
        json.dump(run_info, fh, indent=2, default=str)
        fh.write("\n")
    print(f"Run info JSON -> {run_info_path}")


if __name__ == "__main__":
    main()
