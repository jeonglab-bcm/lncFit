"""Step 2: can any fitted model beat sorting genes by their own mRNA expression?

Step 1 found the plain sort (0.115 AUPRC) beat logistic regression on the two expression
columns (0.089). This tries the obvious alternatives on the same two columns, nothing else:

  no fit     mRNA sort (the step-1 bar), mean rank of total + mRNA, max of the two
  linear     logistic regression on mRNA only
  trees      XGBoost (depth 3) and random forest on both columns; XGBoost on mRNA alone,
             unconstrained and with a monotone-increasing constraint
  smooth     logistic regression on cubic-spline expansions of both columns

and prints, for the monotone XGBoost on mRNA alone, how many distinct scores it gives the
top 10% most-expressed genes -- the diagnostic for why trees lose to the sort.

Usage:
  uv run python scripts/step02_expression_models.py
"""
import argparse

import numpy as np
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from stepwise_common import (fitted, fixed, load_labels, lr, own_expression,
                             predict_cell_and_gene_holdout, run)


def _xgb(**kw):
    return lambda: xgb.XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                                     tree_method="hist", n_jobs=8, random_state=0, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/step02_expression_models.csv")
    args = ap.parse_args()

    genes, cells, y = load_labels()
    E = own_expression(genes, cells)
    mrna = {c: E[c][:, 1:2] for c in cells}

    rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=8,
                                        random_state=0)
    spline = lambda: make_pipeline(SplineTransformer(n_knots=6, degree=3), StandardScaler(),
                                   LogisticRegression(max_iter=4000))
    candidates = [
        fixed("mRNA sort (step-1 bar)", lambda h: E[h][:, 1]),
        fixed("mean rank of total + mRNA", lambda h: rankdata(E[h][:, 0]) + rankdata(E[h][:, 1])),
        fixed("max of total, mRNA", lambda h: np.maximum(E[h][:, 0], E[h][:, 1])),
        fitted("LR, mRNA only", mrna, lr),
        fitted("LR, both", E, lr),
        fitted("spline LR, both", E, spline),
        fitted("XGBoost d3, both", E, _xgb()),
        fitted("random forest, both", E, rf),
        fitted("XGBoost d3, mRNA only", mrna, _xgb()),
        fitted("XGBoost d3 monotone, mRNA only", mrna, _xgb(monotone_constraints=(1,))),
    ]
    run(candidates, y, cells, args.seeds, args.out)

    print("\nDistinct scores among the top 10% most-expressed genes, "
          "monotone XGBoost on mRNA alone (seed 0):")
    for h in cells:
        p = predict_cell_and_gene_holdout(mrna, y, h, _xgb(monotone_constraints=(1,)), seed=0)
        top = E[h][:, 1] >= np.quantile(E[h][:, 1], 0.9)
        print(f"  {h:<11} {len(np.unique(np.round(p[top], 8)))} distinct / {top.sum()} genes")
    print("\nHit rate by mRNA decile among expressed genes (zero-mRNA genes separately):")
    for c in cells:
        m, nz = E[c][:, 1], E[c][:, 1] > 0
        edges = np.quantile(m[nz], np.linspace(0, 1, 11))[1:-1]
        bins = np.digitize(m[nz], edges)
        rates = " ".join(f"{y[c][nz][bins == i].mean():.3f}" for i in range(10))
        print(f"  {c:<11} zero {y[c][~nz].mean():.3f} | {rates}")


if __name__ == "__main__":
    main()
