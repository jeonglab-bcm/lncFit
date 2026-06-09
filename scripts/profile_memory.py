"""Profile peak memory usage of build_features() for k=3 and/or k=6.

Usage:
    uv run python scripts/profile_memory.py --n-records 50000 --k both
    uv run python scripts/profile_memory.py --n-records 10000 --k 6

Logs written to logs/memory_profile_<timestamp>.txt and .json.
"""
import argparse
import gc
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
    import numpy as np

    n = len(records)
    n_features = len(all_kmers(k)) + 2 + 5  # kmer + days + cell lines
    mb = lambda b: round(b / (1024 ** 2), 1)
    gb1m = lambda b: round(b * (1_000_000 / n) / (1024 ** 3), 2)

    def _peak(fn):
        gc.collect()
        tracemalloc.start()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak

    # ── float32 baseline ──────────────────────────────────────────────────────
    (X32, y32, cols), peak_build32 = _peak(lambda: build_features(records, k=k, dtype=np.float32))
    gc.collect()

    mask = np.zeros(n, dtype=bool)
    mask[:max(1, n * 4 // 5)] = True  # ~80% train split (worst-case fold)
    _, peak_fold32 = _peak(lambda: X32[mask].astype(np.float32))

    # ── float16 (this fix) ────────────────────────────────────────────────────
    (X16, y16, _), peak_build16 = _peak(lambda: build_features(records, k=k, dtype=np.float16))
    gc.collect()

    # Fold slice: float16 base → float32 copy (what XGBoost receives)
    _, peak_fold16 = _peak(lambda: X16[mask].astype(np.float32))

    steady32 = X32.nbytes + y32.nbytes
    steady16 = X16.nbytes + y16.nbytes

    return {
        "k": k,
        "n_records": n,
        "n_features": n_features,
        "steady_state_float32_mb": mb(steady32),
        "steady_state_float16_mb": mb(steady16),
        "steady_state_saving_mb": mb(steady32 - steady16),
        "steady_state_float32_gb_at_1M": gb1m(steady32),
        "steady_state_float16_gb_at_1M": gb1m(steady16),
        "steady_state_saving_gb_at_1M": gb1m(steady32 - steady16),
        "peak_build_float32_mb": mb(peak_build32),
        "peak_build_float16_mb": mb(peak_build16),
        "peak_fold_float32_mb": mb(peak_fold32),
        "peak_fold_float16_mb": mb(peak_fold16),
        "peak_fold_saving_mb": mb(peak_fold32 - peak_fold16),
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
        print(f"    Steady-state  float32: {result['steady_state_float32_mb']:>8.1f} MB"
              f"  ({result['steady_state_float32_gb_at_1M']:.2f} GB @ 1M records)")
        print(f"    Steady-state  float16: {result['steady_state_float16_mb']:>8.1f} MB"
              f"  ({result['steady_state_float16_gb_at_1M']:.2f} GB @ 1M records)")
        print(f"    Steady-state  SAVING:  {result['steady_state_saving_mb']:>8.1f} MB"
              f"  ({result['steady_state_saving_gb_at_1M']:.2f} GB @ 1M records)  ✓")
        print(f"    Build peak    float32: {result['peak_build_float32_mb']:>8.1f} MB")
        print(f"    Build peak    float16: {result['peak_build_float16_mb']:>8.1f} MB")
        print(f"    Fold peak     float32: {result['peak_fold_float32_mb']:>8.1f} MB")
        print(f"    Fold peak     float16: {result['peak_fold_float16_mb']:>8.1f} MB")
        print(f"    Fold peak     SAVING:  {result['peak_fold_saving_mb']:>8.1f} MB  ✓")
        print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    txt_path = logs_dir / f"memory_profile_{timestamp}.txt"
    json_path = logs_dir / f"memory_profile_{timestamp}.json"

    header = (f"{'k':<5} {'features':<10} {'ss_f32_gb@1M':<16} {'ss_f16_gb@1M':<16} "
              f"{'ss_saving_gb@1M':<18} {'fold_saving_gb@1M'}")
    with open(txt_path, "w") as fh:
        fh.write(f"build_features() float32 vs float16 memory profile  {timestamp}\n")
        fh.write(f"n_records = {args.n_records:,}\n\n")
        fh.write(header + "\n")
        for r in results:
            fh.write(
                f"{r['k']:<5} {r['n_features']:<10} "
                f"{r['steady_state_float32_gb_at_1M']:<16.2f} "
                f"{r['steady_state_float16_gb_at_1M']:<16.2f} "
                f"{r['steady_state_saving_gb_at_1M']:<18.2f} "
                f"{r['peak_fold_saving_mb'] / 1024 * (1_000_000 / args.n_records):.2f}\n"
            )

    with open(json_path, "w") as fh:
        json.dump({"timestamp": timestamp, "n_records": args.n_records, "results": results}, fh, indent=2)

    print(f"Logs written to {txt_path} and {json_path}")


if __name__ == "__main__":
    main()
