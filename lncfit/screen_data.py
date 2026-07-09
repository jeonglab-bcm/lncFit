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

_RRA_SHEET_TO_CELL_LINE: dict[str, str] = {
    "S2F": "HAP1",
    "S2G": "HEK293FT",
    "S2H": "K562",
    "S2I": "MDA-MB-231",
    "S2J": "THP1",
}

LNCRNA_TARGET_GROUP = "long non-coding RNA"


@dataclass(frozen=True, slots=True)
class LncRnaRecord:
    """One lncRNA x cell_line x day RRA result.

    label is 1 when the lncRNA is a significant depletion hit for this cell line/day
    (rra_pvalue < 0.05 and fold_change < 0), else 0. Has no sequence of its own —
    feature builders look up r.target in a separate {target: sequence} mapping (the
    lncRNA's own transcript/genomic sequence, e.g. from lncfit.sequence), not guide
    spacer sequences (see issue #65: guide sequences are not the lncRNA's sequence).
    """

    target: str
    cell_line: str
    day: int
    rra_pvalue: float
    fold_change: float
    label: int
    chrom: str = ""
    strand: str = ""
    closest_pc_gene: str = ""
    distance_to_closest_pc_gene: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> LncRnaRecord:
        """Construct from a dict, ignoring unknown keys and applying defaults for missing fields."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        for name in ("day", "label"):
            if name in filtered and filtered[name] is not None:
                filtered[name] = int(filtered[name])
        if filtered.get("distance_to_closest_pc_gene") is not None:
            filtered["distance_to_closest_pc_gene"] = int(filtered["distance_to_closest_pc_gene"])
        return cls(**filtered)


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


def load_target_groups(path: Path | str) -> dict[str, str]:
    """Parse S1B sheet from mmc2.xlsx. Returns {target: target_group}.

    target_group is one of "long non-coding RNA", "protein-coding gene",
    "essential protein-coding gene", "non-targeting".
    """
    df = pd.read_excel(path, sheet_name="S1B", header=_find_header_row(path, "S1B"), dtype=str)
    if df.shape[1] < 4:
        return {}
    target_col, group_col = df.columns[1], df.columns[3]
    result: dict[str, str] = {}
    for _, row in df.iterrows():
        t = str(row[target_col]).strip()
        if not t or t.lower() == "nan":
            continue
        result[t] = str(row[group_col]).strip()
    return result


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


def load_rra(
    s2_path: Path | str,
    day: int,
    target_groups: dict[str, str],
    annotations: dict[str, tuple[str, str, str, int | None]] | None = None,
) -> list[LncRnaRecord]:
    """Parse RRA sheets S2F-S2J from mmc3.xlsx into lncRNA-level hit records for one day.

    Restricts to targets whose target_groups entry is "long non-coding RNA" (see
    LNCRNA_TARGET_GROUP) — the RRA sheets' Gene column mixes lncRNA loci with
    protein-coding gene and control rows, same as the guide-level S2A-S2E sheets.
    A record's label is 1 when rra_pvalue < 0.05 and fold_change < 0 (a significant
    depletion hit), else 0. Records carry no sequence of their own — see
    LncRnaRecord's docstring.
    """
    records: list[LncRnaRecord] = []
    xl = pd.ExcelFile(s2_path)
    for sheet_name, cell_line in _RRA_SHEET_TO_CELL_LINE.items():
        if sheet_name not in xl.sheet_names:
            continue
        df = pd.read_excel(
            xl, sheet_name=sheet_name, header=_find_header_row(xl, sheet_name, marker="Gene"), dtype=str
        )
        pval_col = f"Day {day} - P value"
        fc_col = f"Day {day} - Fold-change (log2)"
        if pval_col not in df.columns or fc_col not in df.columns:
            continue
        for _, row in df.iterrows():
            gene = str(row["Gene"]).strip()
            if not gene or gene.lower() == "nan":
                continue
            if target_groups.get(gene) != LNCRNA_TARGET_GROUP:
                continue
            pval_raw, fc_raw = row[pval_col], row[fc_col]
            if pd.isna(pval_raw) or pd.isna(fc_raw):
                continue
            pval, fc = float(pval_raw), float(fc_raw)
            chrom, strand, closest, dist = ("", "", "", None)
            if annotations is not None:
                chrom, strand, closest, dist = annotations.get(gene, ("", "", "", None))
            records.append(
                LncRnaRecord(
                    target=gene,
                    cell_line=cell_line,
                    day=day,
                    rra_pvalue=pval,
                    fold_change=fc,
                    label=int(pval < 0.05 and fc < 0),
                    chrom=chrom,
                    strand=strand,
                    closest_pc_gene=closest,
                    distance_to_closest_pc_gene=dist,
                )
            )
    return records


def save_jsonl(records: list, path: Path | str) -> None:
    """Write records to a gzip-compressed JSONL file, one JSON object per line, stamped with schema version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            d = dataclasses.asdict(r)
            d["_v"] = SCHEMA_VERSION
            f.write(json.dumps(d) + "\n")


def load_jsonl(path: Path | str, record_cls: type = ScreenRecord) -> list:
    """Load records from a gzip-compressed JSONL file produced by save_jsonl.

    record_cls must implement a from_dict classmethod (ScreenRecord and LncRnaRecord both do).
    """
    records: list = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(record_cls.from_dict(json.loads(line)))
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
