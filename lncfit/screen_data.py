from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ScreenRecord:
    guide_id: str
    target: str
    target_sequence: str
    cell_line: str
    day: int
    replicate: int
    fold_change: float


_SHEET_TO_CELL_LINE: dict[str, str] = {
    "S2A": "HAP1",
    "S2B": "HEK293FT",
    "S2C": "K562",
    "S2D": "MDA-MB-231",
    "S2E": "THP1",
}

_FC_HEADER_RE = re.compile(
    r"[Dd]ay\s*(\d+).*?[Rr]ep(?:licate)?\s*(\d+).*?\(Fold-change\)", re.IGNORECASE
)


def _find_header_row(path, sheet_name: str) -> int:
    """Return the 0-indexed row where the first cell is 'ID' (skips title/blank rows)."""
    probe = pd.read_excel(path, sheet_name=sheet_name, header=None, usecols=[0], dtype=str)
    for i, val in enumerate(probe.iloc[:, 0]):
        if str(val).strip() == "ID":
            return i
    return 0


def load_targets(path: Path | str) -> dict[str, tuple[str, str]]:
    """Parse S1B sheet from mmc2.xlsx. Returns {guide_id: (target, target_sequence)}."""
    df = pd.read_excel(path, sheet_name="S1B", header=_find_header_row(path, "S1B"), dtype=str)
    id_col, target_col, seq_col = df.columns[0], df.columns[1], df.columns[2]
    return {
        row[id_col]: (row[target_col], row[seq_col])
        for _, row in df.iterrows()
        if pd.notna(row[id_col]) and str(row[id_col]).strip()
    }


def load_screen(
    s2_path: Path | str,
    targets: dict[str, tuple[str, str]],
) -> list[ScreenRecord]:
    """Parse all S2A-S2E sheets from mmc3.xlsx, melt FC columns, join with targets."""
    records: list[ScreenRecord] = []
    xl = pd.ExcelFile(s2_path)
    for sheet_name, cell_line in _SHEET_TO_CELL_LINE.items():
        if sheet_name not in xl.sheet_names:
            continue
        df = pd.read_excel(xl, sheet_name=sheet_name, header=_find_header_row(xl, sheet_name), dtype=str)
        id_col = df.columns[0]
        fc_cols: list[tuple[str, int, int]] = []
        for col in df.columns[1:]:
            m = _FC_HEADER_RE.search(str(col))
            if m:
                fc_cols.append((col, int(m.group(1)), int(m.group(2))))
        for _, row in df.iterrows():
            gid = str(row[id_col]).strip()
            if not gid or gid.lower() == "nan":
                continue
            t, seq = targets.get(gid, ("", ""))
            for col, day, rep in fc_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                records.append(
                    ScreenRecord(
                        guide_id=gid,
                        target=t,
                        target_sequence=seq,
                        cell_line=cell_line,
                        day=day,
                        replicate=rep,
                        fold_change=float(val),
                    )
                )
    return records


def to_dataframe(records: list[ScreenRecord]) -> pd.DataFrame:
    """Convert a list of ScreenRecord to a tidy DataFrame."""
    return pd.DataFrame(
        [
            {
                "guide_id": r.guide_id,
                "target": r.target,
                "target_sequence": r.target_sequence,
                "cell_line": r.cell_line,
                "day": r.day,
                "replicate": r.replicate,
                "fold_change": r.fold_change,
            }
            for r in records
        ]
    )
