import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_leaderboard import (
    SubmissionError, _load_truth, _render_leaderboard, _score_submission,
)
from lncfit.screen_data import LncRnaRecord, save_jsonl

_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]


def _synthetic_test_records(n_targets=4):
    records = []
    for i in range(n_targets):
        label = 1 if i == 0 else 0
        for cell_line in _CELL_LINES:
            records.append(LncRnaRecord(
                target=f"T{i}", cell_line=cell_line, day=14,
                rra_pvalue=0.01 if label else 0.5, fold_change=-1.0 if label else 0.1,
                label=label, chrom="1",
            ))
    return records


def _write_submission(tmp_path, name, preds_df, submission_meta):
    """uses_measured_depletion defaults to False so tests about *other* validation
    don't all have to restate it; pass it explicitly to exercise the rule itself."""
    sub_dir = tmp_path / name
    sub_dir.mkdir()
    preds_df.to_csv(sub_dir / "predictions.csv", index=False)
    meta = {"uses_measured_depletion": False, **submission_meta}
    with open(sub_dir / "submission.yaml", "w") as fh:
        yaml.safe_dump(meta, fh)
    return sub_dir


def _all_keys_df(records, proba_fn):
    return pd.DataFrame([
        {"target": r.target, "cell_line": r.cell_line, "y_pred_proba": proba_fn(r)}
        for r in records
    ])


def test_valid_submission_scores_correctly(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.9 if r.label == 1 else 0.1)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "perfect"})

    result = _score_submission(sub_dir, records, truth_map, excluded_keys)
    assert result["submitter"] == "alice"
    assert result["model"] == "perfect"
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)


def test_missing_submission_yaml_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _all_keys_df(records, lambda r: 0.5).to_csv(sub_dir / "predictions.csv", index=False)

    with pytest.raises(SubmissionError, match="missing submission.yaml"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_missing_required_field_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice"})  # missing "model"

    with pytest.raises(SubmissionError, match="missing field"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_non_github_handle_submitter_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    sub_dir = _write_submission(
        tmp_path, "sub", preds, {"submitter": "Jane Doe's Team!", "model": "m"}
    )

    with pytest.raises(SubmissionError, match="doesn't look like a GitHub handle"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_missing_rows_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5).iloc[:-1]  # drop one row
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="missing 1 row"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_extra_rows_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    extra = pd.DataFrame([{"target": "NOT_REAL", "cell_line": "HAP1", "y_pred_proba": 0.5}])
    preds = pd.concat([preds, extra], ignore_index=True)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="not in the held-out test set"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_duplicate_rows_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    preds = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="duplicate"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_missing_column_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map, excluded_keys = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5).drop(columns=["y_pred_proba"])
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="missing column"):
        _score_submission(sub_dir, records, truth_map, excluded_keys)


def test_excluded_cell_line_dropped_from_scoring_but_tolerated_in_predictions(tmp_path):
    all_records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(all_records, test_path)

    records, truth_map, excluded_keys = _load_truth(str(test_path), exclude_cell_lines={"HEK293FT"})
    assert all(r.cell_line != "HEK293FT" for r in records)
    assert all(cl == "HEK293FT" for _, cl in excluded_keys)

    # predictions.csv still covers ALL 5 cell lines (as copied straight from a
    # pipeline run) -- HEK293FT rows should be tolerated, not flagged as "extra".
    preds = _all_keys_df(all_records, lambda r: 0.9 if r.label == 1 else 0.1)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    result = _score_submission(sub_dir, records, truth_map, excluded_keys)
    assert result["auroc"] == pytest.approx(1.0)
    assert all(m["split"] != "HEK293FT" for m in result["metrics_rows"])


def test_missing_hek293ft_predictions_still_valid_when_excluded(tmp_path):
    all_records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(all_records, test_path)

    records, truth_map, excluded_keys = _load_truth(str(test_path), exclude_cell_lines={"HEK293FT"})

    # Submitter only predicts the 4 required cell lines, omitting HEK293FT entirely.
    non_excluded = [r for r in all_records if r.cell_line != "HEK293FT"]
    preds = _all_keys_df(non_excluded, lambda r: 0.9 if r.label == 1 else 0.1)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    result = _score_submission(sub_dir, records, truth_map, excluded_keys)
    assert result["auroc"] == pytest.approx(1.0)


def test_only_cell_lines_scores_just_that_line(tmp_path):
    """A single-held-out-cell-line challenge reuses the full dataset and narrows to
    one line, rather than shipping a separate answers file."""
    all_records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(all_records, test_path)

    records, truth, excluded = _load_truth(str(test_path), only_cell_lines={"THP1"})
    assert records, "expected THP1 rows to be scored"
    assert all(r.cell_line == "THP1" for r in records)
    # everything else must land in excluded_keys so extra prediction rows are tolerated
    assert all(cl != "THP1" for _, cl in excluded)

    preds = _all_keys_df(all_records, lambda r: 0.9 if r.label == 1 else 0.1)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})
    result = _score_submission(sub_dir, records, truth, excluded)
    assert result["auroc"] == pytest.approx(1.0)
    assert [m["split"] for m in result["metrics_rows"]] == ["Overall", "THP1"]


def test_only_and_exclude_combined_leaving_nothing_raises():
    # A config that filters everything out should fail loudly, not silently score 0 rows.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl.gz")
        save_jsonl(_synthetic_test_records(), p)
        with pytest.raises(ValueError, match="no rows left to score"):
            _load_truth(p, exclude_cell_lines={"THP1"}, only_cell_lines={"THP1"})


def test_uses_measured_depletion_is_required_and_must_be_boolean(tmp_path):
    """The no-measured-depletion rule is unenforceable by inspection -- we can't see a
    submitter's feature matrix. The one thing the board *can* enforce is that everyone
    answers the question, on the record. A missing or fuzzy value must fail, not
    default to eligible, or the rule quietly becomes opt-in.
    """
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth, excluded = _load_truth(str(test_path))
    preds = _all_keys_df(records, lambda r: 0.9 if r.label == 1 else 0.1)

    # Missing entirely.
    bare = tmp_path / "bare"
    bare.mkdir()
    preds.to_csv(bare / "predictions.csv", index=False)
    with open(bare / "submission.yaml", "w") as fh:
        yaml.safe_dump({"submitter": "alice", "model": "m"}, fh)
    with pytest.raises(SubmissionError, match="uses_measured_depletion"):
        _score_submission(bare, records, truth, excluded)

    # Present but not a bool -- "no"/"false"/"" are the plausible mistakes, and a
    # truthy string would silently mark an honest entry ineligible (or vice versa).
    for bad in ["false", "no", "", 0]:
        sub = _write_submission(
            tmp_path, f"str-{bad!r}", preds,
            {"submitter": "alice", "model": "m", "uses_measured_depletion": bad},
        )
        with pytest.raises(SubmissionError, match="must be true or false"):
            _score_submission(sub, records, truth, excluded)


def test_declaring_measured_depletion_marks_ineligible_but_still_scores(tmp_path):
    """An ineligible entry keeps its real score -- it's moved out of the ranking, not
    deleted or zeroed. Erasing the number would destroy the comparison that motivates
    the rule in the first place (the shortcut scores *higher*, which is the point)."""
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth, excluded = _load_truth(str(test_path))
    preds = _all_keys_df(records, lambda r: 0.9 if r.label == 1 else 0.1)

    clean = _write_submission(tmp_path, "clean", preds,
                              {"submitter": "alice", "model": "seq-only"})
    dirty = _write_submission(tmp_path, "dirty", preds,
                              {"submitter": "bob", "model": "depletion-transfer",
                               "uses_measured_depletion": True})

    r_clean = _score_submission(clean, records, truth, excluded)
    r_dirty = _score_submission(dirty, records, truth, excluded)
    assert r_clean["ineligible"] is False
    assert r_dirty["ineligible"] is True
    # Same predictions -> same score. Ineligibility changes placement, not scoring.
    assert r_dirty["auprc"] == pytest.approx(r_clean["auprc"])

    md = _render_leaderboard("ch", "", [r_dirty, r_clean], 0)
    assert "Ineligible" in md
    # The eligible one is ranked 1 even though the ineligible one was passed first.
    assert "| 1 | [@alice]" in md
    assert "| 2 |" not in md, "ineligible entries must not receive a rank"


def test_auprc_ci_brackets_the_point_estimate_and_is_deterministic(tmp_path):
    """The CI is published next to the score, so it has to be reproducible: CI
    regenerates this file on every submission PR and a wandering interval would read
    as a scoring change."""
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth, excluded = _load_truth(str(test_path))
    # Imperfect predictions -- a perfect ranking gives a degenerate interval at 1.0.
    preds = _all_keys_df(records, lambda r: 0.6 if r.label == 1 else 0.4)
    sub = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    first = _score_submission(sub, records, truth, excluded)
    second = _score_submission(sub, records, truth, excluded)
    lo, hi = first["auprc_ci"]
    assert lo <= first["auprc"] <= hi
    assert first["auprc_ci"] == second["auprc_ci"], "same predictions must give same CI"
