"""Build data/processed/screen_records.jsonl from raw Excel files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_annotations, load_screen, load_targets, save_jsonl

RAW = Path("data/raw")
OUT = Path("data/processed/screen_records.jsonl")


def main() -> None:
    print("Loading targets...")
    targets = load_targets(RAW / "mmc2.xlsx")
    print(f"  {len(targets):,} guides")

    print("Loading annotations...")
    annots = load_annotations(RAW / "mmc2.xlsx")
    print(f"  {len(annots):,} lncRNA annotations")

    print("Loading screen records...")
    records = load_screen(RAW / "mmc3.xlsx", targets, annotations=annots)
    print(f"  {len(records):,} records")

    print(f"Writing {OUT} ...")
    save_jsonl(records, OUT)
    print("Done.")


if __name__ == "__main__":
    main()
