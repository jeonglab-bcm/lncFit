"""Plot ROC and PR curves for the feature x model comparison predictions.

Reads results/lncrna_rra_day14/feature_model_comparison/predictions_<features>_<model>.csv
(written by scripts/run_lncrna_feature_model_comparison.py) and produces a 2x2 grid:
rows = ROC / PR, columns = kmer / dnabert2 features, one line per model.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

FEATURE_SETS = ["kmer", "dnabert2"]
MODELS_BY_FEATURE = {
    "kmer": ["xgboost", "randomforest", "logreg", "knn"],
    "dnabert2": ["xgboost", "randomforest", "logreg", "knn", "mlp"],
}
FEATURE_LABELS = {"kmer": "k-mer (k=5) + cell one-hot", "dnabert2": "DNABERT-2 embedding + cell one-hot"}
MODEL_LABELS = {
    "xgboost": "XGBoost", "randomforest": "Random Forest", "logreg": "Logistic Regression",
    "knn": "kNN", "mlp": "MLP head",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/lncrna_rra_day14/feature_model_comparison")
    parser.add_argument("--output", default="results/lncrna_rra_day14/feature_model_comparison/roc_pr_curves.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for col, features in enumerate(FEATURE_SETS):
        ax_roc = axes[0, col]
        ax_pr = axes[1, col]
        pos_rate = None
        for model in MODELS_BY_FEATURE[features]:
            df = pd.read_csv(results_dir / f"predictions_{features}_{model}.csv")
            y_true, y_pred = df["y_true"].to_numpy(), df["y_pred_proba"].to_numpy()
            pos_rate = y_true.mean()

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{MODEL_LABELS[model]} (AUROC={roc_auc:.3f})")

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(recall, precision)
            ax_pr.plot(recall, precision, label=f"{MODEL_LABELS[model]} (AUPRC={pr_auc:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
        ax_roc.set_title(FEATURE_LABELS[features])
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.legend(fontsize=8, loc="lower right")

        ax_pr.axhline(pos_rate, color="k", linestyle="--", linewidth=0.8, label=f"chance ({pos_rate:.3f})")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=8, loc="upper right")

    fig.suptitle("lncRNA Day-14 RRA-hit classification: chr1 held-out test set")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
