"""Profile peak memory usage of build_features() for k=3 and/or k=6.

Run before and after the fix in lncfit/features.py to verify the improvement.

Usage:
    uv run python scripts/profile_memory.py --n-records 50000 --k both
    uv run python scripts/profile_memory.py --n-records 10000 --k 6

Logs written to logs/memory_profile_<timestamp>.txt and .json.
"""
import argparse
import json
import sys
import tracemalloc
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.features import build_features, all_kmers
from lncfit.screen_data import ScreenRecord

_SEQ = "ACGTACGTACGTACGTACGTACG"
_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]


def _synthetic_records(n: int) -> list[ScreenRecord]:
    return [
        ScreenRecord(
            guide_id=f"g{i}",
            target=f"T{i}",
            target_sequence=_SEQ,
            cell_line=_CELL_LINES[i % len(_CELL_LINES)],
            day=7 if i % 2 == 0 else 14,
            replicate=1,
            fold_change=float(i % 10 - 5) / 5,
            distance_to_closest_pc_gene=None,
        )
        for i in range(n)
    ]


def profile_one(records: list, k: int) -> dict:
    n = len(records)
    n_features = len(all_kmers(k)) + 2 + 5  # kmer + days + cell lines
    theoretical_bytes = n * n_features * 4   # float32 = 4 bytes
    theoretical_mb = theoretical_bytes / (1024 ** 2)
    projected_1m_gb = theoretical_bytes * (1_000_000 / n) / (1024 ** 3)

    tracemalloc.start()
    X, y = build_features(records, k=k)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "k": k,
        "n_records": n,
        "n_features": X.shape[1],
        "theoretical_float32_array_mb": round(theoretical_mb, 1),
        "theoretical_float32_array_gb_at_1M": round(projected_1m_gb, 2),
        "tracemalloc_peak_mb": round(peak_bytes / (1024 ** 2), 1),
        "tracemalloc_peak_gb_at_1M": round(peak_bytes * (1_000_000 / n) / (1024 ** 3), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile build_features() peak RAM.")
    parser.add_argument("--n-records", type=int, default=50_000,
                        help="Number of synthetic records to build (default: 50000)")
    parser.add_argument("--k", choices=["3", "6", "both"], default="both",
                        help="k-mer size to profile (default: both)")
    args = parser.parse_args()

    ks = [3, 6] if args.k == "both" else [int(args.k)]
    records = _synthetic_records(args.n_records)

    print(f"Profiling build_features() with {args.n_records:,} synthetic records\n")
    results = []
    for k in ks:
        print(f"  k={k} ({len(all_kmers(k)):,} k-mer features) ...", flush=True)
        result = profile_one(records, k)
        results.append(result)
        print(f"    float32 array size:           {result['theoretical_float32_array_mb']:>8.1f} MB  "
              f"  ({result['theoretical_float32_array_gb_at_1M']:.1f} GB projected @ 1M records)")
        print(f"    tracemalloc peak:             {result['tracemalloc_peak_mb']:>8.1f} MB  "
              f"  ({result['tracemalloc_peak_gb_at_1M']:.1f} GB projected @ 1M records)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    txt_path = logs_dir / f"memory_profile_{timestamp}.txt"
    json_path = logs_dir / f"memory_profile_{timestamp}.json"

    with open(txt_path, "w") as fh:
        fh.write(f"build_features() memory profile  {timestamp}\n")
        fh.write(f"n_records = {args.n_records:,}\n\n")
        fh.write(f"{'k':<6}{'features':<12}{'array_mb':<14}{'array_gb@1M':<16}{'peak_mb':<14}{'peak_gb@1M'}\n")
        for r in results:
            fh.write(f"{r['k']:<6}{r['n_features']:<12}"
                     f"{r['theoretical_float32_array_mb']:<14.1f}"
                     f"{r['theoretical_float32_array_gb_at_1M']:<16.2f}"
                     f"{r['tracemalloc_peak_mb']:<14.1f}"
                     f"{r['tracemalloc_peak_gb_at_1M']:.2f}\n")

    with open(json_path, "w") as fh:
        json.dump({"timestamp": timestamp, "n_records": args.n_records, "results": results}, fh, indent=2)

    print(f"\nLogs written to {txt_path} and {json_path}")


if __name__ == "__main__":
    main()
