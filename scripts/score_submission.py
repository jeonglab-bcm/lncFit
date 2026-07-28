#!/usr/bin/env python3
"""Score one submission directory locally -- the same AUROC/AUPRC CI will report.

Without this, the only way to learn your score is to open a PR and wait for CI,
which also means every failed format check costs a round trip. This runs the
*same* two functions the leaderboard builder runs (`_load_truth` and
`_score_submission` from scripts/build_leaderboard.py) against the same
challenge.yaml, so a green run here is a green run there and the numbers match
exactly. It is a checker, not a second implementation.

    python scripts/score_submission.py results/lncrna_rra_day14_thp1_holdout/leaderboard/submissions/my-handle-barebones

Note what this necessarily does: to compute a score it reads the held-out
labels. That is fine for a sanity check -- the labels are published data and this
board is explicitly honour-system -- but repeatedly tweaking a model until this
number goes up is exactly the leaderboard-tuning that docs/PARTICIPATE.md warns
against. With 202 positives, gaps under ~0.02 AUPRC are noise. Select your model
with CV on the training cell lines; use this to confirm the file is valid and the
score is in the range you expected.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_leaderboard import SubmissionError, _load_truth, _score_submission


def _challenge_dir(sub_dir: Path) -> Path:
    """A submission lives at <challenge>/leaderboard/submissions/<name>, so the
    challenge config is two levels up. Inferring it beats making the user pass a
    --challenge flag that has to agree with the path they just typed."""
    return sub_dir.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("submission", help="Path to the submission directory.")
    parser.add_argument(
        "--test-path", default=None,
        help="Labeled test set. Defaults to the challenge.yaml test_path.",
    )
    args = parser.parse_args()

    sub_dir = Path(args.submission)
    if not sub_dir.is_dir():
        sys.exit(f"Not a directory: {sub_dir}")

    challenge_config_path = _challenge_dir(sub_dir) / "challenge.yaml"
    if not challenge_config_path.exists():
        sys.exit(
            f"No challenge.yaml at {challenge_config_path} -- is the submission in "
            "results/<challenge>/leaderboard/submissions/<name>/ ?"
        )
    with open(challenge_config_path) as fh:
        config = yaml.safe_load(fh) or {}

    test_path = args.test_path or config.get("test_path")
    if not test_path:
        sys.exit(f"No test_path in {challenge_config_path} -- pass --test-path.")

    records, truth, excluded_keys = _load_truth(
        test_path,
        set(config.get("exclude_cell_lines") or []),
        set(config.get("only_cell_lines") or []),
    )

    try:
        # evaluate_lncrna_by_group prints a per-group table on the way through. Useful
        # in a training run, noise here -- swallow it so the only output is the verdict.
        with contextlib.redirect_stdout(io.StringIO()):
            result = _score_submission(sub_dir, records, truth, excluded_keys)
    except SubmissionError as e:
        # Same message CI prints, so searching the docs' failure table works either way.
        print(f"INVALID -- CI would fail this submission:\n  {e}", file=sys.stderr)
        raise SystemExit(1)

    n_pos = sum(1 for r in records if truth[(r.target, r.cell_line)] == 1)
    base_rate = n_pos / len(records)

    print(f"{result['name']}  (submitter: {result['submitter']})")
    print(f"  model:  {result['model']}")
    print(f"  scored: {len(records):,} rows, {n_pos} positives ({base_rate:.4f} base rate)")
    print()
    print(f"  AUROC   {result['auroc']:.4f}")
    print(f"  AUPRC   {result['auprc']:.4f}   ({result['auprc'] / base_rate:.1f}x base rate)")
    print()
    if not result["has_config"]:
        print("  note: no config.yaml -- optional, but it's how someone reproduces you.")
    print("Valid. These are the numbers CI will publish.")


if __name__ == "__main__":
    main()
