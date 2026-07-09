"""Plot ROC curves for the tuned lncRNA RRA-hit classifiers (issue #65 corrected features).

Uses the held-out chr1 test set predictions saved by scripts/tune_lncrna_xgboost.py
(results/lncrna_rra_day14/tune_k<K>/final_eval_*/predictions.csv). Only k values with a
completed tuning run are plotted — k=6's sweep was stopped early by request and has no
held-out predictions (see results/lncrna_rra_day14/README.md).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results/lncrna_rra_day14")
OUT_PATH = RESULTS_DIR / "roc_curves.png"
KS = [3, 4, 5, 6]

# Categorical palette, fixed order (blue/aqua/yellow/violet) — one hue per k, never cycled.
COLORS = {3: "#2a78d6", 4: "#1baf7a", 5: "#eda100", 6: "#4a3aa7"}
COLOR_CHANCE = "#8a8a86"


def _predictions_csv(k: int) -> Path | None:
    matches = sorted((RESULTS_DIR / f"tune_k{k}").glob("final_eval_*/predictions.csv"))
    return matches[-1] if matches else None


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    skipped = []
    for k in KS:
        csv_path = _predictions_csv(k)
        if csv_path is None:
            skipped.append(k)
            continue
        df = pd.read_csv(csv_path)
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_pred_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[k], linewidth=2, label=f"k={k} tuned (AUC={roc_auc:.3f})")

    if skipped:
        print(f"Skipped k={skipped} — no completed tuning run (missing predictions.csv).")

    ax.plot([0, 1], [0, 1], color=COLOR_CHANCE, linestyle="--", linewidth=1.2, label="Chance (AUC=0.500)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.set_title(
        "lncRNA RRA-hit classifier: ROC, held-out chr1 test set\n"
        "(transcript-sequence features, issue #65)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
