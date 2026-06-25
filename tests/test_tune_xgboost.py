import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lncfit.screen_data import ScreenRecord
from tune_xgboost import filter_records, obj_tag_for


def _rec(cell_line="HAP1", day=7):
    return ScreenRecord(
        guide_id="g1",
        target="T1",
        target_sequence="ACGT",
        cell_line=cell_line,
        day=day,
        replicate=1,
        fold_change=1.0,
    )


@pytest.fixture
def records():
    return [
        _rec("HAP1", 7),
        _rec("HAP1", 14),
        _rec("K562", 7),
        _rec("K562", 14),
        _rec("THP1", 7),
    ]


class TestFilterRecords:
    def test_no_filter_returns_all(self, records):
        assert filter_records(records) == records

    def test_cell_line_filter(self, records):
        out = filter_records(records, cell_line="K562")
        assert all(r.cell_line == "K562" for r in out)
        assert len(out) == 2

    def test_day_filter(self, records):
        out = filter_records(records, day=14)
        assert all(r.day == 14 for r in out)
        assert len(out) == 2

    def test_cell_line_and_day_filter(self, records):
        out = filter_records(records, cell_line="K562", day=7)
        assert len(out) == 1
        assert out[0].cell_line == "K562"
        assert out[0].day == 7

    def test_unknown_cell_line_yields_empty(self, records):
        assert filter_records(records, cell_line="NOPE") == []


class TestObjTagFor:
    def test_pooled_mse(self):
        assert obj_tag_for("reg:squarederror") == "mse"

    def test_pooled_huber(self):
        assert obj_tag_for("reg:pseudohubererror") == "huber"

    def test_cell_line_suffix(self):
        assert obj_tag_for("reg:squarederror", cell_line="K562") == "mse_K562"

    def test_day_suffix(self):
        assert obj_tag_for("reg:squarederror", day=14) == "mse_d14"

    def test_cell_line_and_day_suffix(self):
        assert (
            obj_tag_for("reg:pseudohubererror", cell_line="MDA-MB-231", day=7)
            == "huber_MDA-MB-231_d7"
        )

    def test_pooled_tag_has_no_suffix(self):
        assert "_" not in obj_tag_for("reg:squarederror")
