"""Build the per-cell-line vs pooled comparison table for issue #49.

Scans results/final_eval_*/run_info.json to find the matched pooled baseline and
each per-cell-line run (same k, objective, day=null), then produces a side-by-side
Spearman rho comparison table.

Outputs:
  results/per_cell_line_comparison.csv   the acceptance-criteria table
  results/per_cell_line_comparison.md    human-readable summary + decision prompt
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_run(eval_dir: Path) -> dict | None:
    info_path = eval_dir / "run_info.json"
    metrics_path = eval_dir / "metrics.csv"
    if not info_path.exists() or not metrics_path.exists():
        return None
    with open(info_path) as fh:
        info = json.load(fh)
    metrics = pd.read_csv(metrics_path)
    return {"dir": eval_dir, "info": info, "metrics": metrics}


def _matches(info: dict, k: int, objective: str, cell_line: str | None,
             signed_overlap: bool | None = None) -> bool:
    if info.get("k") != k or info.get("objective") != objective:
        return False
    if info.get("cell_line", None) != cell_line or info.get("day", None) is not None:
        return False
    if signed_overlap is not None and info.get("signed_overlap") != signed_overlap:
        return False
    return True


def _rho_for(metrics: pd.DataFrame, split: str) -> tuple[float, int]:
    row = metrics[metrics["split"] == split]
    if row.empty:
        return float("nan"), 0
    return float(row.iloc[0]["spearman_rho"]), int(row.iloc[0]["n"])


def main():
    parser = argparse.ArgumentParser(description="Build per-cell-line vs pooled comparison (issue #49).")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--objective", default="reg:squarederror")
    parser.add_argument("--signed-overlap", default=None, type=lambda x: x.lower() == "true",
                        help="Match only runs with this signed_overlap setting. "
                             "Default: don't filter on signed_overlap.")
    parser.add_argument("--pooled-baseline", default=None,
                        help="Explicit path to the pooled baseline eval dir. "
                             "Skips auto-discovery when provided (issue #49).")
    parser.add_argument("--cell-lines", nargs="*",
                        default=["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    pooled_run = None
    if args.pooled_baseline:
        pooled_dir = Path(args.pooled_baseline)
        pooled_run = _load_run(pooled_dir)
        if pooled_run is None:
            print(f"Could not load pooled baseline at {pooled_dir}")
            sys.exit(1)
        print(f"Pooled baseline (explicit): {pooled_run['dir'].name}")
    else:
        eval_dirs = sorted(results_dir.glob("final_eval_*"))
        runs = [_load_run(d) for d in eval_dirs]
        runs = [r for r in runs if r is not None]
        for r in runs:
            cl = r["info"].get("cell_line", None)
            if not _matches(r["info"], args.k, args.objective, cl, args.signed_overlap):
                continue
            if cl is None and (pooled_run is None or r["dir"].name > pooled_run["dir"].name):
                pooled_run = r
        if pooled_run is None:
            print("No matching pooled baseline found. Run without --cell-line first, "
                  "or pass --pooled-baseline PATH.")
            sys.exit(1)
        print(f"Pooled baseline (auto): {pooled_run['dir'].name}")

    # Discover per-cell-line runs (most recent match per cell line).
    per_cl_runs: dict[str, dict] = {}
    eval_dirs = sorted(results_dir.glob("final_eval_*"))
    for d in eval_dirs:
        r = _load_run(d)
        if r is None:
            continue
        cl = r["info"].get("cell_line", None)
        if cl is None:
            continue
        if not _matches(r["info"], args.k, args.objective, cl, args.signed_overlap):
            continue
        if cl not in per_cl_runs or r["dir"].name > per_cl_runs[cl]["dir"].name:
            per_cl_runs[cl] = r

    rows = []
    for cl in args.cell_lines:
        pooled_rho, pooled_n = _rho_for(pooled_run["metrics"], cl)
        pooled_d7, _ = _rho_for(pooled_run["metrics"], f"{cl} Day 7")
        pooled_d14, _ = _rho_for(pooled_run["metrics"], f"{cl} Day 14")

        if cl in per_cl_runs:
            per = per_cl_runs[cl]
            per_rho, per_n = _rho_for(per["metrics"], cl)
            per_d7, _ = _rho_for(per["metrics"], f"{cl} Day 7")
            per_d14, _ = _rho_for(per["metrics"], f"{cl} Day 14")
            per_dir = per["dir"].name
        else:
            per_rho = per_d7 = per_d14 = float("nan")
            per_n = 0
            per_dir = "(missing)"

        rows.append({
            "cell_line": cl,
            "n_test": pooled_n,
            "pooled_rho": round(pooled_rho, 4),
            "per_cell_line_rho": round(per_rho, 4) if per_rho == per_rho else float("nan"),
            "delta_rho": round(per_rho - pooled_rho, 4) if per_rho == per_rho else float("nan"),
            "pooled_day7": round(pooled_d7, 4),
            "per_day7": round(per_d7, 4) if per_d7 == per_d7 else float("nan"),
            "pooled_day14": round(pooled_d14, 4),
            "per_day14": round(per_d14, 4) if per_d14 == per_d14 else float("nan"),
            "per_cell_line_run": per_dir,
        })

    df = pd.DataFrame(rows)
    out_csv = results_dir / "per_cell_line_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nComparison table -> {out_csv}")
    print(df.to_string(index=False))

    # Human-readable summary with decision prompt (issue acceptance criteria)
    md_path = results_dir / "per_cell_line_comparison.md"
    with open(md_path, "w") as fh:
        fh.write("# Per-cell-line vs Pooled Comparison (issue #49)\n\n")
        fh.write(f"Config: k={args.k}, objective={args.objective}, day=both\n")
        fh.write(f"Pooled baseline: {pooled_run['dir'].name}\n\n")
        # Manual markdown table (avoids the tabulate dependency)
        cols = list(df.columns)
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float) and v != v:
                    cells.append("")
                else:
                    cells.append(str(v))
            fh.write("| " + " | ".join(cells) + " |\n")
        fh.write("\n## Interpretation\n\n")
        fh.write("- Positive `delta_rho` => per-cell-line model beats pooled (pooling was diluting).\n")
        fh.write("- If K562 per-cell-line > pooled K562 => adopt per-cell-line models.\n")
        fh.write("- If HAP1/THP1/MDA stay ~0 alone => low-reliability labels, not a modeling problem.\n\n")
        fh.write("### How to read the K562 result (training-set confound)\n\n")
        fh.write("- If per-cell-line K562 beats pooled K562 **despite 5x less data** -> specialization dominates, strong signal to adopt per-cell-line.\n")
        fh.write("- If per-cell-line K562 is worse -> inconclusive (could be data volume, not pooling). A size-matched pooled control (210K random records across all 5 cell lines) would disentangle this.\n")
    print(f"Markdown summary  -> {md_path}")


if __name__ == "__main__":
    main()
