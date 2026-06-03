import sys
import os

import openpyxl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lncfit.screen_data import ScreenRecord, load_targets, load_screen, to_dataframe


_FC_HEADERS = [
    "Day 7 Replicate 1 (Fold-change)",
    "Day 7 Replicate 2 (Fold-change)",
    "Day 14 Replicate 1 (Fold-change)",
    "Day 14 Replicate 2 (Fold-change)",
]

_S2_SHEETS = ["S2A", "S2B", "S2C", "S2D", "S2E"]


def _make_mmc2(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("S1B")
    ws.append(["ID", "target", "sequence"])
    ws.append(["gL_000001", "Hum_GENE1", "AAACCCGGGTTT"])
    ws.append(["gL_000002", "Hum_GENE2", "TTTAGCGCGCGC"])
    path = tmp_path / "mmc2.xlsx"
    wb.save(path)
    return path


def _make_mmc3(tmp_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for sheet in _S2_SHEETS:
        ws = wb.create_sheet(sheet)
        ws.append(["ID"] + _FC_HEADERS)
        ws.append(["gL_000001", 0.5, 0.6, -0.3, -0.4])
        ws.append(["gL_000002", 1.2, 1.1, 0.8, 0.9])
    path = tmp_path / "mmc3.xlsx"
    wb.save(path)
    return path


def test_load_targets(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    assert targets["gL_000001"] == ("Hum_GENE1", "AAACCCGGGTTT")
    assert targets["gL_000002"] == ("Hum_GENE2", "TTTAGCGCGCGC")


def test_melt_produces_4_rows_per_guide_per_cell_line(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    records = load_screen(_make_mmc3(tmp_path), targets)
    subset = [r for r in records if r.guide_id == "gL_000001" and r.cell_line == "HAP1"]
    assert len(subset) == 4


def test_total_row_count(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    records = load_screen(_make_mmc3(tmp_path), targets)
    # 2 guides × 5 cell lines × 4 FC columns = 40
    assert len(records) == 40


def test_day_and_replicate_parsed(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    records = load_screen(_make_mmc3(tmp_path), targets)
    assert {r.day for r in records} == {7, 14}
    assert {r.replicate for r in records} == {1, 2}


def test_negative_fold_changes_preserved(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    records = load_screen(_make_mmc3(tmp_path), targets)
    assert any(r.fold_change < 0 for r in records)


def test_s1_s2_join(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    records = load_screen(_make_mmc3(tmp_path), targets)
    r = next(r for r in records if r.guide_id == "gL_000001")
    assert r.target == "Hum_GENE1"
    assert r.target_sequence == "AAACCCGGGTTT"


def test_to_dataframe_schema(tmp_path):
    targets = load_targets(_make_mmc2(tmp_path))
    records = load_screen(_make_mmc3(tmp_path), targets)
    df = to_dataframe(records)
    assert set(df.columns) == {
        "guide_id", "target", "target_sequence",
        "cell_line", "day", "replicate", "fold_change",
    }
    assert len(df) == 40
