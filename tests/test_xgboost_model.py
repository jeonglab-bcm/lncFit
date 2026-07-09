import numpy as np

from lncfit.screen_data import LncRnaRecord
from lncfit.xgboost_model import evaluate_lncrna_by_group


def _rec(cell_line):
    return LncRnaRecord(
        target="T1", cell_line=cell_line, day=14, rra_pvalue=0.5, fold_change=0.0, label=0,
    )


def test_evaluate_lncrna_by_group_overall_plus_per_cell_line():
    records = [_rec("HAP1"), _rec("HAP1"), _rec("K562"), _rec("K562")]
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    rows = evaluate_lncrna_by_group(records, y_true, y_pred, cell_lines=["HAP1", "K562"])
    labels = {r["split"] for r in rows}
    assert labels == {"Overall", "HAP1", "K562"}


def test_evaluate_lncrna_by_group_skips_absent_cell_lines():
    records = [_rec("HAP1"), _rec("HAP1")]
    y_true = np.array([0, 1])
    y_pred = np.array([0.1, 0.9])
    rows = evaluate_lncrna_by_group(records, y_true, y_pred, cell_lines=["HAP1", "THP1"])
    labels = {r["split"] for r in rows}
    assert labels == {"Overall", "HAP1"}
