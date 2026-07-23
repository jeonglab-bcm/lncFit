import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_leaderboard import SubmissionError, _load_truth, _score_submission
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
    sub_dir = tmp_path / name
    sub_dir.mkdir()
    preds_df.to_csv(sub_dir / "predictions.csv", index=False)
    with open(sub_dir / "submission.yaml", "w") as fh:
        yaml.safe_dump(submission_meta, fh)
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
    records, truth_map = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.9 if r.label == 1 else 0.1)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "perfect"})

    result = _score_submission(sub_dir, records, truth_map)
    assert result["submitter"] == "alice"
    assert result["model"] == "perfect"
    assert result["auroc"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)


def test_missing_submission_yaml_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map = _load_truth(str(test_path))

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    _all_keys_df(records, lambda r: 0.5).to_csv(sub_dir / "predictions.csv", index=False)

    with pytest.raises(SubmissionError, match="missing submission.yaml"):
        _score_submission(sub_dir, records, truth_map)


def test_missing_required_field_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice"})  # missing "model"

    with pytest.raises(SubmissionError, match="missing field"):
        _score_submission(sub_dir, records, truth_map)


def test_missing_rows_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5).iloc[:-1]  # drop one row
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="missing 1 row"):
        _score_submission(sub_dir, records, truth_map)


def test_extra_rows_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    extra = pd.DataFrame([{"target": "NOT_REAL", "cell_line": "HAP1", "y_pred_proba": 0.5}])
    preds = pd.concat([preds, extra], ignore_index=True)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="not in the held-out test set"):
        _score_submission(sub_dir, records, truth_map)


def test_duplicate_rows_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5)
    preds = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="duplicate"):
        _score_submission(sub_dir, records, truth_map)


def test_missing_column_raises(tmp_path):
    records = _synthetic_test_records()
    test_path = tmp_path / "test.jsonl.gz"
    save_jsonl(records, test_path)
    records, truth_map = _load_truth(str(test_path))

    preds = _all_keys_df(records, lambda r: 0.5).drop(columns=["y_pred_proba"])
    sub_dir = _write_submission(tmp_path, "sub", preds, {"submitter": "alice", "model": "m"})

    with pytest.raises(SubmissionError, match="missing column"):
        _score_submission(sub_dir, records, truth_map)
