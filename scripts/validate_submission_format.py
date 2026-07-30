"""Check a submission's format without reading a single held-out label.

scripts/score_submission.py runs the same structural checks, but it also prints AUROC and
AUPRC -- so using it to catch a stray comma spends your one honest look at the answer key on
a formatting mistake. This does the structural half alone.

It can do that because data/holdout_thp1/holdout_thp1_features.jsonl.gz lists every
(target, cell_line) pair you must predict and contains no label, rra_pvalue or fold_change
column (see docs/PARTICIPATE.md -- an earlier version of that file did ship those columns,
which is the leak commit ada1b59 fixed). Row coverage can therefore be verified exactly
while the answers stay sealed.

Deliberately NOT done here: computing any metric, or opening data/processed/
lncrna_rra_day14.jsonl.gz, data/raw/mmc3.xlsx, or anything else carrying THP1 outcomes. Run
scripts/score_submission.py once, at the end, when the format is already known good.

Checks, mirroring the failure table in docs/PARTICIPATE.md:
  * predictions.csv and submission.yaml both present
  * submission.yaml has submitter and model
  * submitter looks like a GitHub handle
  * predictions.csv has exactly target, cell_line, y_pred_proba
  * no duplicate (target, cell_line) rows
  * every required row present, no unexpected extras
  * every y_pred_proba is a finite number

Usage:
  uv run python scripts/validate_submission_format.py \\
      results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/erica286-barebones
"""
import argparse
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path

import yaml

_FEATURES = "data/holdout_thp1/holdout_thp1_features.jsonl.gz"
_REQUIRED_COLUMNS = ["target", "cell_line", "y_pred_proba"]
_REQUIRED_YAML_FIELDS = ["submitter", "model"]
# GitHub handles: alphanumeric, single (not repeated or trailing) hyphens, <= 39 chars.
_HANDLE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")

# Columns that would mean the features file still carries the answer key. Their presence is
# not the submission's fault, but it is worth shouting about rather than reading past.
_LEAK_COLUMNS = {"label", "rra_pvalue", "fold_change"}


def load_required_keys(path: str) -> tuple[set[tuple[str, str]], set[str]]:
    """Return the required (target, cell_line) keys, and any leak columns spotted."""
    keys: set[tuple[str, str]] = set()
    leaked: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            leaked |= _LEAK_COLUMNS & row.keys()
            keys.add((str(row["target"]), str(row["cell_line"])))
    return keys, leaked


def _excluded_cell_lines(sub_dir: Path) -> set[str]:
    """Cell lines challenge.yaml excludes; rows for these are tolerated, not errors."""
    challenge = sub_dir.parent.parent / "challenge.yaml"
    if not challenge.exists():
        return set()
    with open(challenge) as fh:
        config = yaml.safe_load(fh) or {}
    return set(config.get("exclude_cell_lines") or [])


def validate(sub_dir: Path, features_path: str) -> list[str]:
    """Return a list of problems; empty means the format is good."""
    problems: list[str] = []

    predictions = sub_dir / "predictions.csv"
    metadata = sub_dir / "submission.yaml"
    if not predictions.exists():
        problems.append(f"missing predictions.csv (expected at {predictions})")
    if not metadata.exists():
        problems.append(f"missing submission.yaml (expected at {metadata})")
    if problems:
        return problems

    with open(metadata) as fh:
        meta = yaml.safe_load(fh) or {}
    missing = [f for f in _REQUIRED_YAML_FIELDS if not meta.get(f)]
    if missing:
        problems.append(f"submission.yaml missing field(s): {missing}")
    submitter = str(meta.get("submitter", "")).strip()
    if submitter and not _HANDLE.match(submitter):
        problems.append(
            f"submitter {submitter!r} doesn't look like a GitHub handle "
            "(letters, digits, single hyphens, <= 39 chars)")

    with open(predictions, newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        missing_cols = [c for c in _REQUIRED_COLUMNS if c not in columns]
        if missing_cols:
            problems.append(f"predictions.csv missing column(s): {missing_cols}")
            return problems
        rows = list(reader)

    required, leaked = load_required_keys(features_path)
    if leaked:
        problems.append(
            f"WARNING (not your submission's fault): {features_path} still carries "
            f"{sorted(leaked)} -- that is the answer key. Re-pull; see docs/PARTICIPATE.md.")

    excluded = _excluded_cell_lines(sub_dir)
    seen: dict[tuple[str, str], int] = {}
    bad_scores = 0
    for row in rows:
        key = (str(row["target"]).strip(), str(row["cell_line"]).strip())
        seen[key] = seen.get(key, 0) + 1
        raw = str(row["y_pred_proba"]).strip()
        try:
            if not math.isfinite(float(raw)):
                bad_scores += 1
        except (TypeError, ValueError):
            bad_scores += 1

    duplicates = {k: n for k, n in seen.items() if n > 1}
    if duplicates:
        example = ", ".join(f"{t}/{c} x{n}" for (t, c), n in list(duplicates.items())[:3])
        problems.append(f"{len(duplicates)} duplicate (target, cell_line) row(s): {example}")

    absent = required - set(seen)
    if absent:
        example = ", ".join(f"{t}/{c}" for t, c in sorted(absent)[:3])
        problems.append(f"missing {len(absent):,} row(s) required by the test set: {example}")

    extra = {k for k in seen if k not in required and k[1] not in excluded}
    if extra:
        example = ", ".join(f"{t}/{c}" for t, c in sorted(extra)[:3])
        problems.append(f"{len(extra):,} row(s) not in the test set: {example}")

    if bad_scores:
        problems.append(f"{bad_scores:,} row(s) with a non-numeric or non-finite "
                        "y_pred_proba (blank, NaN or inf)")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", help="Path to the submission directory.")
    parser.add_argument("--features", default=_FEATURES,
                        help="Label-free features file defining the required rows.")
    args = parser.parse_args()

    sub_dir = Path(args.submission)
    if not sub_dir.is_dir():
        sys.exit(f"Not a directory: {sub_dir}")
    if not Path(args.features).exists():
        sys.exit(f"Features file not found: {args.features} (run `git lfs pull`?)")

    problems = validate(sub_dir, args.features)

    print(f"{sub_dir.name}")
    if problems:
        print("\nFORMAT PROBLEMS -- CI would fail this submission:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    n_rows = sum(1 for _ in open(sub_dir / "predictions.csv")) - 1
    print(f"  predictions.csv  {n_rows:,} rows, columns and coverage OK")
    print(f"  submission.yaml  submitter and model present")
    if not (sub_dir / "config.yaml").exists():
        print("  note: no config.yaml -- optional, but it's how someone reproduces you.")
    print("\nFormat OK. No labels were read -- run scripts/score_submission.py once, "
          "at the end, for the actual score.")


if __name__ == "__main__":
    main()
