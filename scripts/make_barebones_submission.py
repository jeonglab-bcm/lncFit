#!/usr/bin/env python3
"""Write a complete, valid leaderboard submission. Standard library only.

The smallest thing that is a real entry: no genome download, no sequences, no
embeddings, no scikit-learn, no XGBoost, nothing to install. Reads the two
committed challenge files and writes the two files a submission needs.

The model is one line of arithmetic. For each lncRNA, average its knockout
fold-change across the three TRAINING cell lines and negate it, so genes whose
knockout depleted other cell lines rank highest:

    score(gene) = -mean(fold_change of that gene in HAP1, K562, MDA-MB-231)

That is a pan-essentiality prior: a gene that matters everywhere else probably
matters in THP1 too. It uses only training-cell-line columns, never THP1's.

Do not mistake it for a weak baseline. It scores AUROC 0.7085 / AUPRC 0.2000,
which at the time of writing beats four of the five real submissions on the
board -- including every model built on DNABERT-2 embeddings and tuned XGBoost.
If your model can't clear this, it hasn't learned anything a one-liner doesn't
already know. That is the point of shipping it.

Note that the test set is a single cell line, so any cell-line-level feature
(one-hot, Celligner coordinates) is constant across every row being scored and
cannot change the ranking at all. Only gene-level signal moves the metric here.

Usage:
  python scripts/make_barebones_submission.py \\
      --submitter your-github-handle \\
      --out results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/your-handle-barebones
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

_TRAIN = "data/holdout_thp1/train_thp1_holdout.jsonl.gz"
_TEST = "data/holdout_thp1/holdout_thp1_features.jsonl.gz"


def _read_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submitter", required=True, help="Your GitHub handle.")
    parser.add_argument("--out", required=True, help="Submission directory to create.")
    parser.add_argument("--train", default=_TRAIN)
    parser.add_argument("--test", default=_TEST)
    args = parser.parse_args()

    train = _read_jsonl_gz(args.train)
    test = _read_jsonl_gz(args.test)

    # Group each gene's fold-changes across the training cell lines.
    by_target: dict[str, list[float]] = defaultdict(list)
    for row in train:
        by_target[row["target"]].append(row["fold_change"])

    # Genes absent from training get 0.0 -- a neutral score, ranked mid-pack. There
    # are none today (the two files cover the same 5,496 genes), but a submission
    # that silently dropped rows would fail CI for a confusing reason.
    missing = 0
    out_rows = []
    for row in test:
        fold_changes = by_target.get(row["target"])
        if fold_changes:
            score = -sum(fold_changes) / len(fold_changes)
        else:
            score, missing = 0.0, missing + 1
        out_rows.append({"target": row["target"], "cell_line": row["cell_line"],
                         "y_pred_proba": round(score, 6)})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["target", "cell_line", "y_pred_proba"])
        writer.writeheader()
        writer.writerows(out_rows)

    # y_pred_proba is a ranking score, not a calibrated probability, and the
    # leaderboard only reads the ranking (AUROC and AUPRC are both rank metrics).
    # Say so in the description rather than fake a probability by squashing it.
    with open(out_dir / "submission.yaml", "w") as fh:
        fh.write(
            f"submitter: {args.submitter}\n"
            'model: "barebones: -mean(training fold_change) per gene"\n'
            "description: >\n"
            "  Zero-setup baseline from scripts/make_barebones_submission.py. For each\n"
            "  lncRNA, the negated mean knockout fold-change across the three training\n"
            "  cell lines -- a pan-essentiality prior with no learned parameters, no\n"
            "  sequence features and no cell-line features. Uses only columns from the\n"
            "  training file. y_pred_proba is an uncalibrated ranking score.\n"
        )

    print(f"Wrote {out_dir}/predictions.csv  ({len(out_rows):,} rows)")
    print(f"Wrote {out_dir}/submission.yaml")
    if missing:
        print(f"  note: {missing:,} test gene(s) absent from training, scored 0.0")
    print("\nThis is a valid submission as-is. It is also the bar to beat -- see "
          "docs/PARTICIPATE.md.")


if __name__ == "__main__":
    main()
