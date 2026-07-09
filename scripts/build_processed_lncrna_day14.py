"""Build data/processed/lncrna_rra_day14.jsonl.gz from raw Excel files (issue #60).

Unlike screen_records.jsonl.gz (one row per guide x day x replicate), this dataset is
one row per lncRNA x cell_line for Day 14 only, labeled by MAGeCK-RRA significance
(RRA P value < 0.05 and log2 fold-change < 0 = significant depletion hit).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_annotations, load_rra, load_target_groups, save_jsonl

RAW = Path("data/raw")
OUT = Path("data/processed/lncrna_rra_day14.jsonl.gz")
DAY = 14


def main() -> None:
    print("Loading target groups...")
    target_groups = load_target_groups(RAW / "mmc2.xlsx")
    print(f"  {len(target_groups):,} targets")

    print("Loading annotations...")
    annots = load_annotations(RAW / "mmc2.xlsx")
    print(f"  {len(annots):,} lncRNA annotations")

    print(f"Loading Day {DAY} RRA records...")
    records = load_rra(
        RAW / "mmc3.xlsx", day=DAY, target_groups=target_groups, annotations=annots,
    )
    print(f"  {len(records):,} records")
    n_pos = sum(r.label for r in records)
    print(f"  {n_pos:,} significant hits ({n_pos / len(records):.1%})")

    print(f"Writing {OUT} ...")
    save_jsonl(records, OUT)
    print("Done.")


if __name__ == "__main__":
    main()
