#!/usr/bin/env python3
"""Split the day14 dataset into a training set and a label-withheld features set.

This is NOT a blind test and cannot be made one: `label` is derived from the
*published* supplementary tables of the source screen paper, so the held-out cell
line's answers are downloadable by anyone. The board runs on trust. What these
files do is make the honest path the easy one -- follow the instructions and you
never have the answer key in hand.

For that to be true the features file must omit the label AND the two columns the
label is computed from. It previously shipped `rra_pvalue` and `fold_change` with
only `label` blanked to -1, which withheld nothing: `label` is exactly
`rra_pvalue < 0.05 and fold_change < 0`, so one line of pandas recovered all 202
THP1 positives at 100% agreement.

Writes three files:

  <out>/train_<line>_holdout.jsonl.gz        The non-held-out cell lines, labels
                                             included -- this is what you train on.
  <out>/holdout_<line>_features.jsonl.gz     The held-out cell line's rows with
                                             label, rra_pvalue and fold_change all
                                             omitted -- this is what you predict.
  <out>/holdout_<line>_labels.jsonl.gz       The same rows with real labels. Not
                                             committed; scoring reads the held-out
                                             rows from the full processed dataset
                                             instead. Kept for the case where a
                                             future challenge holds out genuinely
                                             unpublished data.

Usage:
  python scripts/split_holdout_cellline.py --holdout THP1 --out data/holdout_thp1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import LncRnaRecord, load_jsonl, save_jsonl

_DEFAULT_DATA = "data/processed/lncrna_rra_day14.jsonl.gz"
# HEK293FT is excluded from the challenge entirely (not a real cancer line, no
# Celligner data) -- see results/*/leaderboard/challenge.yaml.
_ALWAYS_EXCLUDE = {"HEK293FT"}

# label == (rra_pvalue < 0.05 and fold_change < 0), so all three must go together.
# Withholding `label` alone leaves the answer key in the file.
_WITHHELD_FIELDS = ("label", "rra_pvalue", "fold_change")


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
    # Drop the label AND its two ingredients. Omitted rather than blanked: a reader
    # that wants the answer gets a KeyError, not a plausible-looking -1.
    save_jsonl(holdout, features_path, drop_fields=_WITHHELD_FIELDS)
    save_jsonl(holdout, labels_path)

    n_pos = sum(r.label for r in holdout)
    print(f"train    : {train_path}  n={len(public):,}  "
          f"cell lines={sorted({r.cell_line for r in public})}")
    print(f"features : {features_path}  n={len(holdout):,}  "
          f"withheld columns={list(_WITHHELD_FIELDS)}")
    print(f"labels   : {labels_path}  n={len(holdout):,}  positives={n_pos} "
          f"({n_pos / len(holdout):.1%})  -- do not commit")


if __name__ == "__main__":
    main()
