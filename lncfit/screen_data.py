from __future__ import annotations

import dataclasses
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ScreenRecord:
    guide_id: str
    target: str
    target_sequence: str
    cell_line: str
    day: int
    replicate: int
    fold_change: float
    chrom: str = ""
    strand: str = ""
    closest_pc_gene: str = ""
    distance_to_closest_pc_gene: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> ScreenRecord:
        """Construct from a dict, ignoring unknown keys and applying defaults for missing fields."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        # Coerce numeric fields that JSON may deserialise as float
        for name in ("day", "replicate"):
            if name in filtered and filtered[name] is not None:
                filtered[name] = int(filtered[name])
        if filtered.get("distance_to_closest_pc_gene") is not None:
            filtered["distance_to_closest_pc_gene"] = int(filtered["distance_to_closest_pc_gene"])
        return cls(**filtered)


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

_ANNOT_COLS = {
    "Chr": "chrom",
    "Strand": "strand",
    "Closest protein-coding gene symbol": "closest_pc_gene",
    "Distance to closest protein-coding gene": "distance_to_closest_pc_gene",
}


def _find_header_row(path, sheet_name: str, marker: str = "ID") -> int:
    """Return the 0-indexed row where the first cell equals marker (skips title/blank rows)."""
    probe = pd.read_excel(path, sheet_name=sheet_name, header=None, usecols=[0], dtype=str)
    for i, val in enumerate(probe.iloc[:, 0]):
        if str(val).strip() == marker:
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


def load_annotations(
    path: Path | str,
) -> dict[str, tuple[str, str, str, int | None]]:
    """Parse S1A sheet from mmc2.xlsx. Returns {lncRNA_id: (chrom, strand, closest_pc_gene, distance)}."""
    df = pd.read_excel(
        path, sheet_name="S1A", header=_find_header_row(path, "S1A", marker="lncRNA"), dtype=str
    )
    lncrna_col = df.columns[0]
    result: dict[str, tuple[str, str, str, int | None]] = {}
    for _, row in df.iterrows():
        lnc_id = str(row[lncrna_col]).strip()
        if not lnc_id or lnc_id.lower() == "nan":
            continue
        chrom = str(row.get("Chr", "")).strip()
        strand = str(row.get("Strand", "")).strip()
        closest = str(row.get("Closest protein-coding gene symbol", "")).strip()
        dist_raw = row.get("Distance to closest protein-coding gene", None)
        if chrom == "nan":
            chrom = ""
        if strand == "nan":
            strand = ""
        if closest == "nan":
            closest = ""
        dist: int | None = None
        if dist_raw is not None and str(dist_raw).strip() not in ("", "nan"):
            try:
                dist = int(float(str(dist_raw)))
            except ValueError:
                pass
        result[lnc_id] = (chrom, strand, closest, dist)
    return result


def load_screen(
    s2_path: Path | str,
    targets: dict[str, tuple[str, str]],
    annotations: dict[str, tuple[str, str, str, int | None]] | None = None,
) -> list[ScreenRecord]:
    """Parse all S2A-S2E sheets from mmc3.xlsx, melt FC columns, join with targets and annotations."""
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
            chrom, strand, closest, dist = ("", "", "", None)
            if annotations is not None:
                chrom, strand, closest, dist = annotations.get(t, ("", "", "", None))
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
                        chrom=chrom,
                        strand=strand,
                        closest_pc_gene=closest,
                        distance_to_closest_pc_gene=dist,
                    )
                )
    return records


def save_jsonl(records: list[ScreenRecord], path: Path | str) -> None:
    """Write records to a gzip-compressed JSONL file, one JSON object per line, stamped with schema version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            d = dataclasses.asdict(r)
            d["_v"] = SCHEMA_VERSION
            f.write(json.dumps(d) + "\n")


def load_jsonl(path: Path | str) -> list[ScreenRecord]:
    """Load records from a gzip-compressed JSONL file produced by save_jsonl."""
    records: list[ScreenRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ScreenRecord.from_dict(json.loads(line)))
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
                "chrom": r.chrom,
                "strand": r.strand,
                "closest_pc_gene": r.closest_pc_gene,
                "distance_to_closest_pc_gene": r.distance_to_closest_pc_gene,
            }
            for r in records
        ]
    )
