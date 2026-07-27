"""Tests for scripts/split_holdout_cellline.py.

The point of these is one specific past failure. The features file published for
the leaderboard used to withhold only `label`, blanking it to -1, while still
shipping `rra_pvalue` and `fold_change`. But `label` is *defined* as
`rra_pvalue < 0.05 and fold_change < 0`, so the answer key was still in the file:
recomputing that expression on the published THP1 features recovered all 202
positives with 100% agreement. Withholding one derived column while publishing its
inputs withholds nothing.
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import LncRnaRecord, load_jsonl, save_jsonl

_REPO_ROOT = Path(__file__).parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "split_holdout_cellline.py"
_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]


def _records(n_targets=12):
    records = []
    for i in range(n_targets):
        is_hit = i % 4 == 0
        for cell_line in _CELL_LINES:
            records.append(LncRnaRecord(
                target=f"T{i}", cell_line=cell_line, day=14,
                rra_pvalue=0.01 if is_hit else 0.5,
                fold_change=-1.5 if is_hit else 0.3,
                label=int(is_hit), chrom=str(i % 3),
            ))
    return records


@pytest.fixture
def split(tmp_path):
    data_path = tmp_path / "day14.jsonl.gz"
    save_jsonl(_records(), data_path)
    out = tmp_path / "holdout"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--holdout", "THP1",
         "--data", str(data_path), "--out", str(out)],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return out


def _raw_rows(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_features_file_omits_the_label_and_both_of_its_ingredients(split):
    rows = _raw_rows(split / "holdout_thp1_features.jsonl.gz")
    assert rows
    for row in rows:
        for withheld in ("label", "rra_pvalue", "fold_change"):
            assert withheld not in row, f"{withheld} must not be published"


def test_labels_are_not_recoverable_from_the_features_file(split):
    """The exact attack that used to work: apply the label's own definition to the
    columns the features file ships."""
    for row in _raw_rows(split / "holdout_thp1_features.jsonl.gz"):
        with pytest.raises(KeyError):
            int(row["rra_pvalue"] < 0.05 and row["fold_change"] < 0)


def test_features_file_keeps_everything_needed_to_predict(split):
    """Withholding must not go so far that submitters can't build features or
    join their predictions back: target/cell_line are the leaderboard's key."""
    rows = _raw_rows(split / "holdout_thp1_features.jsonl.gz")
    for row in rows:
        assert row["cell_line"] == "THP1"
        assert row["target"]
        assert "chrom" in row and "distance_to_closest_pc_gene" in row


def test_withheld_features_load_as_label_minus_one(split):
    """A file with no label column must come back as the -1 sentinel, which is what
    the pipeline keys on to skip scoring."""
    records = load_jsonl(split / "holdout_thp1_features.jsonl.gz", record_cls=LncRnaRecord)
    assert records
    assert all(r.label == -1 for r in records)


def test_training_file_excludes_the_holdout_line_and_hek293ft(split):
    records = load_jsonl(split / "train_thp1_holdout.jsonl.gz", record_cls=LncRnaRecord)
    lines = {r.cell_line for r in records}
    assert lines == {"HAP1", "K562", "MDA-MB-231"}
    # Training labels are the point of the training file -- these stay.
    assert any(r.label == 1 for r in records)
    assert all(r.label in (0, 1) for r in records)


def test_labels_file_holds_the_real_answers(split):
    """The answer key still gets written -- it's just gitignored, for the case where
    a future challenge holds out genuinely unpublished data."""
    records = load_jsonl(split / "holdout_thp1_labels.jsonl.gz", record_cls=LncRnaRecord)
    assert {r.cell_line for r in records} == {"THP1"}
    assert sum(r.label for r in records) == 3  # targets 0, 4, 8 of 12


def test_features_and_labels_cover_the_same_rows_in_the_same_order(split):
    """Scoring joins on (target, cell_line); a divergence here would misalign
    every submission."""
    features = load_jsonl(split / "holdout_thp1_features.jsonl.gz", record_cls=LncRnaRecord)
    labels = load_jsonl(split / "holdout_thp1_labels.jsonl.gz", record_cls=LncRnaRecord)
    assert [(r.target, r.cell_line) for r in features] == [(r.target, r.cell_line) for r in labels]


def test_unknown_holdout_line_fails_loudly(tmp_path):
    data_path = tmp_path / "day14.jsonl.gz"
    save_jsonl(_records(), data_path)
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--holdout", "NOPE",
         "--data", str(data_path), "--out", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_committed_thp1_features_file_has_no_answer_key():
    """Belt and braces on the real published artifact, not just a synthetic one.
    If someone regenerates it with an older script, this fails."""
    path = _REPO_ROOT / "data" / "holdout_thp1" / "holdout_thp1_features.jsonl.gz"
    if not path.exists():
        pytest.skip("holdout features file not present")
    rows = _raw_rows(path)
    assert len(rows) == 5496
    leaked = {k for row in rows for k in ("label", "rra_pvalue", "fold_change") if k in row}
    assert not leaked, f"published features file leaks {sorted(leaked)}"
