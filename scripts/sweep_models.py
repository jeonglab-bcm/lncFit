"""Swap the classifier, hold everything else fixed: does any model beat tuned XGBoost?

Last untried option. Features are exhausted -- of eleven information sources only expression
and guide design ever helped -- and tuning XGBoost's max_depth was worth more than every
feature combined. This asks whether the algorithm itself is the limit.

Same features, same leave-one-cell-line-out folds, same scoring; only the estimator changes.

Two feature sets per model, because they are not equally affected by the inert sequence
block. The 256 k-mer columns are harmless to trees, which simply never split on them, but
they are actively damaging to distance-based models: kNN measuring distance across 256
meaningless dimensions drowns the ~42 useful ones. Reporting both separates "this model is
weak here" from "this model was handicapped by noise columns".
  with_kmer   298 cols -- k-mer + one-hot + TPM + S1A + guide9 (the architecture-compatible set)
  no_kmer      42 cols -- the same minus k-mers

Scale-sensitive models (SVMs, kNN, neural net, naive Bayes, QDA) get StandardScaler fitted
on the training fold only. Tree models get raw features, which they are indifferent to.

Class imbalance is handled where the estimator supports it (class_weight='balanced', or
scale_pos_weight for XGBoost). kNN, GaussianNB and QDA have no equivalent, so at a 3.7%
positive rate they are structurally disadvantaged here -- worth stating rather than reading
their scores as a fair verdict on the algorithm.

Not run, because they cannot be:
  Gaussian Process -- O(n^3) in the 10,992 training rows, with an n x n kernel matrix.
  QDA on with_kmer -- estimates a covariance matrix per class, which needs more samples than
    features; 298 columns against ~200 positives is singular. Run on no_kmer only.
Exact RBF SVM is replaced by a Nystroem approximation, which the repo measured as 52x faster
and better-scoring on this data.

The comparison is tilted toward XGBoost and should be read that way: it is the only estimator
here that received a tuning pass (max_depth=3, scale_pos_weight=1, colsample_bytree=0.3,
worth +0.023). Others get sensible settings, not tuned ones. A close XGBoost win is not
evidence that XGBoost is better -- only that it is better tuned.

Usage:
  uv run python scripts/sweep_models.py
  uv run python scripts/sweep_models.py --seeds 3
"""
import argparse
import csv
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from sweep_gene_level_priors import load_gene_table
from sweep_prescreen_features import _GUIDE, _KMER, _ONEHOT, _S1A, _TPM, build, load_guides, load_s1a
from sweep_tpm_features import _TRAIN, kmer_matrix, load_tpm

_WITH_KMER = [_KMER, _ONEHOT, _TPM, _S1A, _GUIDE]
_NO_KMER = [_ONEHOT, _TPM, _S1A, _GUIDE]


def build_estimator(name: str, seed: int, spw: float):
    """Return (estimator, needs_scaling, is_stochastic)."""
    if name == "xgboost (tuned)":
        return xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.3, tree_method="hist", objective="binary:logistic",
            eval_metric="aucpr", scale_pos_weight=1.0, random_state=seed, n_jobs=8), False, True
    if name == "logistic":
        return LogisticRegression(class_weight="balanced", max_iter=3000,
                                  random_state=seed), True, False
    if name == "linear_svm":
        return LinearSVC(class_weight="balanced", C=0.05, max_iter=5000,
                         random_state=seed), True, False
    if name == "rbf_svm_nystroem":
        return make_pipeline(
            Nystroem(n_components=1000, random_state=seed),
            LinearSVC(class_weight="balanced", C=0.05, max_iter=5000, random_state=seed),
        ), True, True
    if name == "knn":
        # No class_weight equivalent; distance weighting is the closest lever.
        return KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=8), True, False
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                      random_state=seed), False, False
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                      class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=8), False, True
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5,
                                    class_weight="balanced_subsample",
                                    random_state=seed, n_jobs=8), False, True
    if name == "adaboost":
        return AdaBoostClassifier(n_estimators=300, learning_rate=0.5,
                                  random_state=seed), False, True
    if name == "gaussian_nb":
        return GaussianNB(), True, False
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=400,
                             early_stopping=True, random_state=seed), True, True
    if name == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=0.5), True, False
    raise ValueError(name)


MODELS = ["xgboost (tuned)", "logistic", "linear_svm", "rbf_svm_nystroem", "knn",
          "decision_tree", "random_forest", "extra_trees", "adaboost", "gaussian_nb",
          "mlp", "qda"]

# QDA needs more samples than features per class; 298 columns is singular.
_NO_KMER_ONLY = {"qda"}


def scores_from(est, X):
    """Continuous score for ranking. AUPRC reads order only, so decision_function is fine."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seeds for stochastic estimators. Deterministic ones run once.")
    parser.add_argument("--out", default="results/model_sweep.csv")
    args = parser.parse_args()

    table = load_gene_table(_TRAIN)
    genes = sorted(table)
    train_cells = sorted({c for obs in table.values() for c in obs})
    labels = {c: np.asarray([table[g][c]["label"] for g in genes], dtype=int)
              for c in train_cells}
    print(f"{len(genes):,} genes, folds: {', '.join(train_cells)}")

    total, mrna = load_tpm()
    s1a, s1a_ext, _, _ = load_s1a(genes)
    guide, _ = load_guides(genes)
    kmer_X = kmer_matrix(genes, args.k)

    feature_sets = {
        "with_kmer": {c: build(genes, c, _WITH_KMER, kmer_X, total, mrna, s1a, s1a_ext, guide)
                      for c in train_cells},
        "no_kmer": {c: build(genes, c, _NO_KMER, kmer_X, total, mrna, s1a, s1a_ext, guide)
                    for c in train_cells},
    }
    for tag, mats in feature_sets.items():
        print(f"  {tag}: {next(iter(mats.values())).shape[1]} columns")

    rows = []
    results: dict[tuple[str, str], list[float]] = defaultdict(list)

    for tag, mats in feature_sets.items():
        print(f"\n########## feature set: {tag} ##########")
        for name in MODELS:
            if tag == "with_kmer" and name in _NO_KMER_ONLY:
                print(f"  {name:<18} skipped on {tag} (singular: more features than "
                      "samples per class)")
                continue
            _, needs_scaling, stochastic = build_estimator(name, 0, 1.0)
            n_seeds = args.seeds if stochastic else 1
            t0 = time.time()
            fold_scores = []
            failed = None
            for holdout in train_cells:
                y_eval = labels[holdout]
                X_eval = mats[holdout]
                X_train = np.vstack([mats[c] for c in train_cells if c != holdout])
                y_train = np.concatenate([labels[c] for c in train_cells if c != holdout])
                spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

                if needs_scaling:
                    scaler = StandardScaler().fit(X_train)
                    Xtr, Xev = scaler.transform(X_train), scaler.transform(X_eval)
                else:
                    Xtr, Xev = X_train, X_eval

                for seed in range(n_seeds):
                    est, _, _ = build_estimator(name, seed, spw)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            est.fit(Xtr, y_train)
                            p = scores_from(est, Xev)
                        auprc = float(average_precision_score(y_eval, p))
                        auroc = float(roc_auc_score(y_eval, p))
                    except Exception as e:  # noqa: BLE001 - report, do not abort the sweep
                        failed = f"{type(e).__name__}: {e}"
                        break
                    fold_scores.append(auprc)
                    results[(name, tag)].append(auprc)
                    rows.append({"feature_set": tag, "model": name, "holdout": holdout,
                                 "seed": seed, "n_features": Xev.shape[1],
                                 "auroc": round(auroc, 4), "auprc": round(auprc, 4)})
                if failed:
                    break
            if failed:
                print(f"  {name:<18} FAILED -- {failed[:90]}")
                continue
            print(f"  {name:<18} AUPRC {np.mean(fold_scores):.4f} "
                  f"+/- {np.std(fold_scores):.4f}   ({time.time() - t0:.0f}s, "
                  f"{n_seeds} seed{'s' if n_seeds > 1 else ''})", flush=True)

    ref = float(np.mean(results[("xgboost (tuned)", "with_kmer")]))
    print(f"\n=== all results, sorted by AUPRC (reference: tuned XGBoost with_kmer "
          f"{ref:.4f}) ===")
    print(f"{'model':<18} {'features':<10} {'AUPRC':>7} {'sd':>7} {'vs xgb':>12}")
    for (name, tag), vals in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
        mean = float(np.mean(vals))
        d = mean - ref
        note = "" if (name, tag) == ("xgboost (tuned)", "with_kmer") else (
            f"{d:+.4f}" + (" (noise)" if abs(d) < 0.02 else ""))
        print(f"{name:<18} {tag:<10} {mean:>7.4f} {float(np.std(vals)):>7.4f} {note:>12}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature_set", "model", "holdout", "seed",
                                           "n_features", "auroc", "auprc"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-fold/per-seed -> {out}")
    print("Only XGBoost was tuned. Read a narrow XGBoost win as a tuning advantage, "
          "not an algorithmic one.")


if __name__ == "__main__":
    main()
