"""Plot untuned vs tuned AUROC/AUPRC across k for the lncRNA RRA-hit classifier (issues #61/#62)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/lncrna_rra_day14")
OUT_PATH = RESULTS_DIR / "auroc_auprc_sweep.png"
KS = [3, 4, 5, 6]

COLOR_UNTUNED = "#2a78d6"  # categorical slot 1 (blue)
COLOR_TUNED = "#1baf7a"    # categorical slot 2 (aqua)
COLOR_BASELINE = "#8a8a86"


def _overall_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    return df[df["split"] == "Overall"].iloc[0]


def _tuned_final_eval_csv(k: int) -> Path:
    (metrics_csv,) = sorted((RESULTS_DIR / f"tune_k{k}").glob("final_eval_*/metrics.csv"))
    return metrics_csv


def main() -> None:
    untuned = [_overall_row(RESULTS_DIR / f"metrics_k{k}.csv") for k in KS]
    tuned = [_overall_row(_tuned_final_eval_csv(k)) for k in KS]

    test_n = int(untuned[0]["n"])
    test_pos = int(untuned[0]["n_pos"])
    base_rate = test_pos / test_n

    x = range(len(KS))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle(
        "lncRNA RRA-hit classifier: untuned (#61) vs tuned (#62), held-out chr1 test set",
        fontsize=11,
    )

    series_handles = None
    for ax, metric, baseline, ylabel in [
        (axes[0], "auroc", 0.5, "AUROC"),
        (axes[1], "auprc", base_rate, "AUPRC"),
    ]:
        untuned_vals = [row[metric] for row in untuned]
        tuned_vals = [row[metric] for row in tuned]
        top = max(max(untuned_vals), max(tuned_vals))

        bars_u = ax.bar(
            [xi - width / 2 - 0.01 for xi in x], untuned_vals, width,
            color=COLOR_UNTUNED, label="Untuned", zorder=3,
        )
        bars_t = ax.bar(
            [xi + width / 2 + 0.01 for xi in x], tuned_vals, width,
            color=COLOR_TUNED, label="Tuned", zorder=3,
        )
        # Alternate label height for the two series so adjacent close values
        # (e.g. k=5's 0.702 vs 0.697) don't merge into one another.
        for bars, dy in [(bars_u, 10), (bars_t, 3)]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                )
        if series_handles is None:
            series_handles = [bars_u, bars_t]

        ax.axhline(baseline, color=COLOR_BASELINE, linestyle="--", linewidth=1.2, zorder=2)
        ax.annotate(
            f"trivial baseline ({baseline:.3f})", (len(KS) - 1 + width / 2 + 0.05, baseline),
            textcoords="offset points", xytext=(0, 4), ha="right", fontsize=7.5, color=COLOR_BASELINE,
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"k={k}" for k in KS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, top * 1.25)
        ax.set_title(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.legend(
        series_handles, ["Untuned (#61)", "Tuned (#62)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=9, frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
