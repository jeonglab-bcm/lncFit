"""Build train/test split files from processed screen records.

Produces four files in data/processed/:
  train_chrom1.jsonl.gz  / test_chrom1.jsonl.gz   -- chromosome-1 hold-out
  train_THP1.jsonl.gz    / test_THP1.jsonl.gz      -- THP1 cell-line hold-out
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import load_jsonl, save_jsonl
from lncfit.splits import split_by_chrom, split_by_cell_line

IN = Path("data/processed/screen_records.jsonl.gz")
OUT = Path("data/processed")

TEST_CHROM = "1"
TEST_CELL_LINE = "THP1"


def main() -> None:
    print(f"Loading {IN} ...")
    records = load_jsonl(IN)
    print(f"  {len(records):,} records")

    print(f"\nSplitting by chromosome (test = chr{TEST_CHROM}) ...")
    train_chrom, test_chrom = split_by_chrom(records, TEST_CHROM)
    print(f"  train: {len(train_chrom):,}  test: {len(test_chrom):,}")
    save_jsonl(train_chrom, OUT / f"train_chrom{TEST_CHROM}.jsonl.gz")
    save_jsonl(test_chrom, OUT / f"test_chrom{TEST_CHROM}.jsonl.gz")
    print(f"  Wrote train_chrom{TEST_CHROM}.jsonl.gz and test_chrom{TEST_CHROM}.jsonl.gz")

    print(f"\nSplitting by cell line (test = {TEST_CELL_LINE}) ...")
    train_cell, test_cell = split_by_cell_line(records, TEST_CELL_LINE)
    print(f"  train: {len(train_cell):,}  test: {len(test_cell):,}")
    save_jsonl(train_cell, OUT / f"train_{TEST_CELL_LINE}.jsonl.gz")
    save_jsonl(test_cell, OUT / f"test_{TEST_CELL_LINE}.jsonl.gz")
    print(f"  Wrote train_{TEST_CELL_LINE}.jsonl.gz and test_{TEST_CELL_LINE}.jsonl.gz")

    print("\nDone.")


if __name__ == "__main__":
    main()
