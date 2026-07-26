import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from ensemble_predictions import rank_average


def _frame(scores, targets=None, cell_line="HAP1"):
    targets = targets or [f"T{i}" for i in range(len(scores))]
    return pd.DataFrame({
        "target": targets,
        "cell_line": [cell_line] * len(scores),
        "y_pred_proba": scores,
    })


def test_rank_average_is_scale_free():
    """The whole reason for ranking before averaging: inputs are on incomparable
    scales (xgboost emits ~[0, 0.5] probabilities, the SVM wrapper emits an
    uncalibrated sigmoid of its margin). A plain mean would let the wider-spread
    model dominate; equal ranks must contribute equally."""
    a = _frame([0.1, 0.2, 0.3, 0.4])
    # Same ORDER as `a`, but on a 1000x scale.
    b_same_order = _frame([100.0, 200.0, 300.0, 400.0])
    out = rank_average([a, b_same_order])
    # Identical rankings -> the blend must reproduce that ranking exactly.
    assert list(out["y_pred_proba"].rank()) == [1.0, 2.0, 3.0, 4.0]


def test_rank_average_blends_disagreeing_inputs():
    a = _frame([0.1, 0.2, 0.3, 0.4])
    b_reversed = _frame([0.4, 0.3, 0.2, 0.1])
    out = rank_average([a, b_reversed])
    # Exactly opposing rankings average to a tie across the board.
    assert out["y_pred_proba"].nunique() == 1


def test_weights_are_honoured():
    a = _frame([0.1, 0.2, 0.3, 0.4])
    b_reversed = _frame([0.4, 0.3, 0.2, 0.1])
    out = rank_average([a, b_reversed], weights=[1.0, 0.0])
    # All weight on `a` -> `a`'s ranking survives intact.
    assert list(out["y_pred_proba"].rank()) == [1.0, 2.0, 3.0, 4.0]


def test_output_preserves_keys_and_row_count():
    a = _frame([0.1, 0.2, 0.3])
    b = _frame([0.3, 0.1, 0.2])
    out = rank_average([a, b])
    assert list(out.columns) == ["target", "cell_line", "y_pred_proba"]
    assert len(out) == 3
    assert set(out["target"]) == {"T0", "T1", "T2"}


def test_mismatched_weight_count_raises():
    a = _frame([0.1, 0.2])
    with pytest.raises(ValueError, match="weights"):
        rank_average([a, a], weights=[1.0])


def test_missing_score_column_raises():
    a = _frame([0.1, 0.2])
    bad = a.drop(columns=["y_pred_proba"])
    with pytest.raises(ValueError, match="missing column"):
        rank_average([a, bad])


def test_duplicate_rows_raise():
    a = _frame([0.1, 0.2])
    dup = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        rank_average([a, dup])


def test_inputs_covering_different_rows_raise():
    """Silently intersecting would drop rows the leaderboard requires, producing a
    submission that fails validation for a non-obvious reason."""
    a = _frame([0.1, 0.2, 0.3])
    b = _frame([0.1, 0.2], targets=["T0", "T1"])
    with pytest.raises(ValueError, match="disagree on which rows"):
        rank_average([a, b])
