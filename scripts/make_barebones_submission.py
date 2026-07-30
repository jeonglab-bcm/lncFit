#!/usr/bin/env python3
"""The banned shortcut, kept as a demonstration. INELIGIBLE for ranking.

**This does not produce a rankable entry any more, and it is not a template.**
As of 2026-07-28 the rules forbid measured knockdown depletion as an input
feature, from any cell line (see docs/PARTICIPATE.md). Measured depletion is
this script's *only* feature, so what it writes is filed under "Ineligible" on
the board. It is retained because it is the clearest possible evidence for why
that rule was needed.

The model is one line of arithmetic. For each lncRNA, average its knockout
fold-change across the three TRAINING cell lines and negate it, so genes whose
knockout depleted other cell lines rank highest:

    score(gene) = -mean(fold_change of that gene in HAP1, K562, MDA-MB-231)

That is a pan-essentiality prior: a gene that matters everywhere else probably
matters in THP1 too. It never touches THP1's columns -- the objection to it is
not leakage, it is that the question becomes trivial and sequence-blind.

Here is the problem it exposes. It scores AUROC 0.7085 / AUPRC 0.2000 with no
learned parameters, while the best *sequence-only* entry on the board manages
0.1696 and a tuned DNABERT-2 + Optuna model manages 0.1268. A one-liner that
reads three columns beat every model that tried to learn biology from sequence.
That gap measures a shortcut in the task design, not progress on the science --
which is exactly what the feature ban removes.

For a compliant starting point, build from transcript sequence, guide design and
static annotation instead; see the quickstart in docs/PARTICIPATE.md.

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
            # This IS measured depletion as a feature -- it is the entire model. Declared
            # true so the board files it under Ineligible, which is where it belongs.
            "uses_measured_depletion: true\n"
            "description: >\n"
            "  Zero-setup reference from scripts/make_barebones_submission.py. For each\n"
            "  lncRNA, the negated mean knockout fold-change across the three training\n"
            "  cell lines -- a pan-essentiality prior with no learned parameters, no\n"
            "  sequence features and no cell-line features. INELIGIBLE by design: it uses\n"
            "  measured depletion as its only feature, which the rules no longer permit.\n"
            "  Kept as the demonstration of why that rule exists.\n"
        )

    print(f"Wrote {out_dir}/predictions.csv  ({len(out_rows):,} rows)")
    print(f"Wrote {out_dir}/submission.yaml")
    if missing:
        print(f"  note: {missing:,} test gene(s) absent from training, scored 0.0")
    print("\nNOTE: this submission is INELIGIBLE for ranking -- measured depletion is its\n"
          "only feature, which docs/PARTICIPATE.md now bans. It is a demonstration, not a\n"
          "template: it scores 0.2000 AUPRC while the best sequence-only entry manages\n"
          "0.1696, and that gap is the whole reason the rule exists. Build from sequence.")


if __name__ == "__main__":
    main()
