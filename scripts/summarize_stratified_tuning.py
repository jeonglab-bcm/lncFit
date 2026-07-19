"""Summarize scripts/tune_lncrna_stratified.py runs into one comparison table.

Scans results/lncrna_rra_day14/tune_stratified/<model>_k<K>_cw<on|off>/ for the
most recent final_eval_*/{run_info.json,metrics.csv} in each run directory and
prints/saves the Overall + per-cell-line chr1 held-out metrics side by side, so
the stratified-CV numbers can be compared directly against the chromosome-LOCO
numbers in results/lncrna_rra_day14/README.md.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def _latest_eval_dir(run_dir: Path) -> Path | None:
    eval_dirs = sorted(run_dir.glob("final_eval_*"))
    return eval_dirs[-1] if eval_dirs else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-dir", default="results/lncrna_rra_day14/tune_stratified")
    parser.add_argument("--output", default="results/lncrna_rra_day14/tune_stratified/summary.csv")
    args = parser.parse_args()

    tune_root = Path(args.tune_dir)
    rows = []
    for run_dir in sorted(tune_root.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_dir = _latest_eval_dir(run_dir)
        if eval_dir is None:
            print(f"  (skipping {run_dir.name}: no final_eval_* found)")
            continue

        run_info_path = eval_dir / "run_info.json"
        metrics_path = eval_dir / "metrics.csv"
        if not run_info_path.exists() or not metrics_path.exists():
            print(f"  (skipping {run_dir.name}: incomplete eval dir {eval_dir.name})")
            continue

        with open(run_info_path) as fh:
            run_info = json.load(fh)
        metrics = pd.read_csv(metrics_path).set_index("split")

        row = {
            "run": run_dir.name,
            "model": run_info["model"],
            "class_weight": run_info["class_weight"],
            "k": run_info["k"],
            "cv_mean_auprc": run_info["cv_mean_auprc"],
            "cv_std_auprc": run_info["cv_std_auprc"],
            "test_auroc_overall": metrics.loc["Overall", "auroc"],
            "test_auprc_overall": metrics.loc["Overall", "auprc"],
        }
        for cl in ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]:
            if cl in metrics.index:
                row[f"auroc_{cl}"] = metrics.loc[cl, "auroc"]
                row[f"auprc_{cl}"] = metrics.loc[cl, "auprc"]
        rows.append(row)

    if not rows:
        print("No completed runs found.")
        return

    df = pd.DataFrame(rows).sort_values(["model", "k", "class_weight"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(df[["run", "cv_mean_auprc", "cv_std_auprc", "test_auroc_overall", "test_auprc_overall"]]
          .to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
