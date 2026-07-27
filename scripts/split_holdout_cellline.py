#!/usr/bin/env python3
"""Split the day14 dataset into a public training set and a private label set.

The leaderboard's held-out cell line is a *blind* test: submitters get its rows
(so they know what to predict) but not its labels. Those labels live in a
separate private repo and are only ever read by CI.

Writes three files:

  <out>/train_<public cell lines>.jsonl.gz   public. The non-held-out cell lines,
                                             labels included -- this is what
                                             submitters train on.
  <out>/holdout_<line>_features.jsonl.gz     public. The held-out cell line's rows
                                             with `label` forced to -1, so nobody
                                             can read the answer out of the file
                                             they're told to predict.
  <out>/holdout_<line>_labels.jsonl.gz       PRIVATE. The same rows with real
                                             labels. Push this to the private
                                             ground-truth repo, never to the
                                             public one.

Usage:
  python scripts/split_holdout_cellline.py --holdout THP1 --out data/holdout_thp1
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import LncRnaRecord, load_jsonl, save_jsonl

_DEFAULT_DATA = "data/processed/lncrna_rra_day14.jsonl.gz"
# HEK293FT is excluded from the challenge entirely (not a real cancer line, no
# Celligner data) -- see results/*/leaderboard/challenge.yaml.
_ALWAYS_EXCLUDE = {"HEK293FT"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout", required=True, help="Cell line to hold out as the blind test set.")
    parser.add_argument("--data", default=_DEFAULT_DATA)
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    records = load_jsonl(args.data, record_cls=LncRnaRecord)
    records = [r for r in records if r.cell_line not in _ALWAYS_EXCLUDE]

    cell_lines = sorted({r.cell_line for r in records})
    if args.holdout not in cell_lines:
        sys.exit(f"--holdout {args.holdout!r} not found. Available: {cell_lines}")

    public = [r for r in records if r.cell_line != args.holdout]
    holdout = [r for r in records if r.cell_line == args.holdout]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / f"train_{args.holdout.lower()}_holdout.jsonl.gz"
    features_path = out / f"holdout_{args.holdout.lower()}_features.jsonl.gz"
    labels_path = out / f"holdout_{args.holdout.lower()}_labels.jsonl.gz"

    save_jsonl(public, train_path)
    # Blank the label so the public file cannot be used to recover the answer.
    # -1 rather than 0: a wrong-but-plausible 0 would silently score as a real
    # label if anything ever read this file as ground truth by mistake.
    blinded = [dataclasses.replace(r, label=-1) for r in holdout]
    save_jsonl(blinded, features_path)
    save_jsonl(holdout, labels_path)

    n_pos = sum(r.label for r in holdout)
    print(f"public train : {train_path}  n={len(public):,}  "
          f"cell lines={sorted({r.cell_line for r in public})}")
    print(f"public blind : {features_path}  n={len(blinded):,}  labels blanked to -1")
    print(f"PRIVATE truth: {labels_path}  n={len(holdout):,}  positives={n_pos} "
          f"({n_pos / len(holdout):.1%})")
    print()
    print(f"Push ONLY {labels_path.name} to the private ground-truth repo.")


if __name__ == "__main__":
    main()
