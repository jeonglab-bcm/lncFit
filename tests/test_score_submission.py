"""Tests for scripts/score_submission.py.

This script exists so participants can get their real score without opening a PR,
and its whole value rests on one promise: **the number it prints is the number CI
publishes.** If it ever drifts from scripts/build_leaderboard.py, it becomes worse
than useless -- people would tune against a score the board disagrees with.

So the tests here check that promise directly (same inputs -> same metrics as the
leaderboard builder) plus the two contract details a CLI needs: non-zero exit on an
invalid submission, and inferring challenge.yaml from the submission path.
"""
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

_REPO_ROOT = Path(__file__).parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "score_submission.py"


@pytest.fixture
def challenge(tmp_path):
    """A miniature challenge laid out exactly as the real one is, since the script
    infers challenge.yaml by walking up from the submission directory."""
    leaderboard = tmp_path / "results" / "tiny" / "leaderboard"
    submissions = leaderboard / "submissions"
    submissions.mkdir(parents=True)

    truth = [
        {"target": f"G{i}", "cell_line": "THP1", "day": 14, "rra_pvalue": 0.01,
         "fold_change": -2.0, "label": 1 if i < 2 else 0, "chrom": "1"}
        for i in range(6)
    ]
    # A row for a cell line the challenge doesn't score, to exercise only_cell_lines.
    truth.append({"target": "G0", "cell_line": "HAP1", "day": 14, "rra_pvalue": 0.9,
                  "fold_change": 0.1, "label": 0, "chrom": "1"})

    test_path = leaderboard / "truth.jsonl.gz"
    with gzip.open(test_path, "wt", encoding="utf-8") as fh:
        for row in truth:
            fh.write(json.dumps(row) + "\n")

    (leaderboard / "challenge.yaml").write_text(yaml.safe_dump({
        "test_path": str(test_path),
        "only_cell_lines": ["THP1"],
        "title": "tiny",
    }))
    return {"leaderboard": leaderboard, "submissions": submissions}


def _make_submission(submissions_dir, name="octocat-test", scores=None):
    sub = submissions_dir / name
    sub.mkdir()
    scores = scores or {"G0": 0.9, "G1": 0.8, "G2": 0.3, "G3": 0.2, "G4": 0.1, "G5": 0.0}
    with open(sub / "predictions.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["target", "cell_line", "y_pred_proba"])
        writer.writeheader()
        for target, score in scores.items():
            writer.writerow({"target": target, "cell_line": "THP1", "y_pred_proba": score})
    (sub / "submission.yaml").write_text("submitter: octocat\nmodel: test model\n")
    return sub


def _run(sub_dir):
    return subprocess.run([sys.executable, str(_SCRIPT), str(sub_dir)],
                          capture_output=True, text=True, cwd=_REPO_ROOT)


def test_reports_the_same_metrics_as_the_leaderboard_builder(challenge):
    """The core promise. Scored via the CLI and via build_leaderboard's own functions;
    if these ever disagree, the number shown to participants is a lie."""
    from scripts.build_leaderboard import _load_truth, _score_submission

    sub = _make_submission(challenge["submissions"])
    result = _run(sub)
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load((challenge["leaderboard"] / "challenge.yaml").read_text())
    records, truth, excluded = _load_truth(
        config["test_path"], set(), set(config["only_cell_lines"]))
    expected = _score_submission(sub, records, truth, excluded)

    assert f"{expected['auroc']:.4f}" in result.stdout
    assert f"{expected['auprc']:.4f}" in result.stdout
    assert "Valid." in result.stdout


def test_exits_non_zero_and_names_the_problem_when_invalid(challenge):
    """It's meant to be usable as a pre-push check, so a bad submission must fail
    the process, not just print something unhappy."""
    sub = challenge["submissions"] / "octocat-broken"
    sub.mkdir()
    (sub / "submission.yaml").write_text("submitter: octocat\nmodel: m\n")
    # predictions.csv missing entirely.
    result = _run(sub)

    assert result.returncode == 1
    assert "missing predictions.csv" in result.stderr
    assert "INVALID" in result.stderr


def test_incomplete_coverage_is_rejected(challenge):
    """The failure mode most likely to reach the board: a run that only predicted
    some rows. Catching it locally is the whole point of the script."""
    sub = _make_submission(challenge["submissions"], scores={"G0": 0.9, "G1": 0.8})
    result = _run(sub)

    assert result.returncode == 1
    assert "missing 4 row(s)" in result.stderr


def test_fails_clearly_when_not_run_from_a_challenge_directory(tmp_path):
    """Pointing it at an arbitrary directory should explain the expected layout
    rather than raise a traceback about a missing file."""
    stray = tmp_path / "somewhere"
    stray.mkdir()
    result = _run(stray)

    assert result.returncode != 0
    assert "challenge.yaml" in result.stderr
