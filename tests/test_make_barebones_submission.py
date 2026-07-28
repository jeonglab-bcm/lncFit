"""Tests for scripts/make_barebones_submission.py.

This script is documented as the zero-setup entry point, so the thing worth
guarding is that its output is *actually submittable* -- the exact columns, one row
per test gene, no dependency on anything a fresh clone lacks. A subtly malformed
baseline is worse than none: it sends newcomers to a red CI check on their first try.
"""
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "make_barebones_submission.py"


def _write_jsonl_gz(rows, path):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture
def tiny_challenge(tmp_path):
    """4 genes x 3 training cell lines, plus a 4-gene held-out set with no labels."""
    train = []
    for i, fold_changes in enumerate([(-3.0, -3.0, -3.0), (-1.0, 0.0, -0.5),
                                      (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)]):
        for cell_line, fc in zip(["HAP1", "K562", "MDA-MB-231"], fold_changes):
            train.append({"target": f"G{i}", "cell_line": cell_line, "day": 14,
                          "rra_pvalue": 0.01 if fc < -0.5 else 0.9,
                          "fold_change": fc, "label": int(fc < -0.5)})
    # The features file omits label/rra_pvalue/fold_change, as the real one does.
    test = [{"target": f"G{i}", "cell_line": "THP1", "day": 14, "chrom": "1"}
            for i in range(4)]

    train_path, test_path = tmp_path / "train.jsonl.gz", tmp_path / "test.jsonl.gz"
    _write_jsonl_gz(train, train_path)
    _write_jsonl_gz(test, test_path)
    return {"train": str(train_path), "test": str(test_path)}


def _run(tiny_challenge, out_dir, submitter="octocat"):
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--submitter", submitter, "--out", str(out_dir),
         "--train", tiny_challenge["train"], "--test", tiny_challenge["test"]],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_output_matches_the_submission_format(tiny_challenge, tmp_path):
    out = tmp_path / "sub"
    _run(tiny_challenge, out)

    with open(out / "predictions.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == ["target", "cell_line", "y_pred_proba"]
    assert len(rows) == 4
    assert {r["cell_line"] for r in rows} == {"THP1"}

    meta = yaml.safe_load((out / "submission.yaml").read_text())
    assert meta["submitter"] == "octocat"
    assert meta["model"] and meta["description"]


def test_ranks_pan_essential_genes_highest(tiny_challenge, tmp_path):
    """The one substantive claim: genes depleted across the training cell lines must
    outrank genes that weren't. If this inverts, the baseline is anti-predictive."""
    out = tmp_path / "sub"
    _run(tiny_challenge, out)
    with open(out / "predictions.csv") as fh:
        scores = {r["target"]: float(r["y_pred_proba"]) for r in csv.DictReader(fh)}

    # G0 depleted hard everywhere, G1 mildly, G2 enriched, G3 flat.
    assert scores["G0"] > scores["G1"] > scores["G3"] > scores["G2"]
    assert scores["G0"] == pytest.approx(3.0)  # -mean(-3, -3, -3)


def test_genes_absent_from_training_get_a_neutral_score_not_a_dropped_row(tmp_path):
    """Dropping the row would fail CI with 'missing N row(s)' -- a confusing error to
    hand someone on their first submission."""
    train = [{"target": "G0", "cell_line": c, "day": 14, "rra_pvalue": 0.01,
              "fold_change": -2.0, "label": 1} for c in ["HAP1", "K562", "MDA-MB-231"]]
    test = [{"target": "G0", "cell_line": "THP1"}, {"target": "UNSEEN", "cell_line": "THP1"}]
    train_path, test_path = tmp_path / "tr.jsonl.gz", tmp_path / "te.jsonl.gz"
    _write_jsonl_gz(train, train_path)
    _write_jsonl_gz(test, test_path)

    out = tmp_path / "sub"
    result = _run({"train": str(train_path), "test": str(test_path)}, out)

    with open(out / "predictions.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2, "every test row must be present"
    assert float(next(r["y_pred_proba"] for r in rows if r["target"] == "UNSEEN")) == 0.0
    assert "absent from training" in result.stdout


def test_runs_without_the_project_dependencies():
    """Documented as needing nothing installed, so it must not import pandas, numpy,
    sklearn or lncfit -- a fresh clone has none of them until `uv sync`."""
    source = _SCRIPT.read_text()
    for banned in ("pandas", "numpy", "sklearn", "xgboost", "import lncfit", "from lncfit"):
        assert banned not in source, f"barebones script must not depend on {banned}"


def test_does_not_read_the_held_out_labels():
    """It must reach the answer only through the training file. Hard-coding a path to
    the full dataset would make the baseline a cheat."""
    source = _SCRIPT.read_text()
    assert "lncrna_rra_day14.jsonl.gz" not in source
    assert "_labels.jsonl.gz" not in source
    assert "mmc3" not in source
