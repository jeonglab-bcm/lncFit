"""Build train/test split files from processed lncRNA RRA records (issue #60).

Produces in data/processed/:
  train_lncrna_day14_chrom1.jsonl.gz / test_lncrna_day14_chrom1.jsonl.gz -- chromosome-1 hold-out
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import LncRnaRecord, load_jsonl, save_jsonl
from lncfit.splits import split_by_chrom

IN = Path("data/processed/lncrna_rra_day14.jsonl.gz")
OUT = Path("data/processed")
TEST_CHROM = "1"


def main() -> None:
    print(f"Loading {IN} ...")
    records = load_jsonl(IN, record_cls=LncRnaRecord)
    print(f"  {len(records):,} records")

    print(f"\nSplitting by chromosome (test = chr{TEST_CHROM}) ...")
    train, test = split_by_chrom(records, TEST_CHROM)
    print(f"  train: {len(train):,}  test: {len(test):,}")
    save_jsonl(train, OUT / f"train_lncrna_day14_chrom{TEST_CHROM}.jsonl.gz")
    save_jsonl(test, OUT / f"test_lncrna_day14_chrom{TEST_CHROM}.jsonl.gz")
    print(f"  Wrote train_lncrna_day14_chrom{TEST_CHROM}.jsonl.gz and test_lncrna_day14_chrom{TEST_CHROM}.jsonl.gz")

    print("\nDone.")


if __name__ == "__main__":
    main()
