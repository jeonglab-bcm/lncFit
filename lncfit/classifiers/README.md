# lncfit.classifiers

Pluggable classifier layer for the lncRNA RRA-hit task, so training + evaluation
can be run systematically across models by swapping one name. Structure follows
the [teddy](https://github.com/LiuzLab/teddy) project's model layer
(`teddy/model/`: shared abstract base + a registry so architectures are drop-in),
adapted from teddy's PyTorch `nn.Module`s to sklearn-style binary classifiers
(`fit` / `predict_proba` on feature matrices). Different project, different domain —
only the plug-in *pattern* is borrowed.

## Layout

| File | Role |
|------|------|
| `base.py` | `ClassifierModel` ABC — `model_type`, `fit(X, y)`, `predict_proba(X) -> P(pos), shape (n,)` |
| `registry.py` | `CLASSIFIER_REGISTRY`, `@register_classifier(name)`, `build_classifier(name, **params)`, `available_classifiers()` |
| `null.py` | `NullClassifier` — constant training base-rate prediction (AUROC 0.5 floor) |
| `logreg.py` | `LogRegClassifier` — L2 logistic regression, balanced class weights |
| `xgboost_clf.py` | `XGBoostClassifier` — gradient-boosted trees, auto `scale_pos_weight` |
| `__init__.py` | imports the wrappers to trigger registration |

## Usage

```python
from lncfit.classifiers import build_classifier, available_classifiers

available_classifiers()               # ['logreg', 'null', 'xgboost']
model = build_classifier("xgboost", max_depth=4)
model.fit(X_train, y_train)           # X from lncfit.features.build_lncrna_features
proba = model.predict_proba(X_test)   # P(hit) per row, shape (n,)
```

Run the full train + chr1-holdout eval + results bundle for any model:

```bash
uv run python scripts/run_lncrna_classifier.py --model xgboost --k 3
uv run python scripts/run_lncrna_classifier.py --model logreg  --k 3
uv run python scripts/run_lncrna_classifier.py --model null
# extra hyperparameters: --param max_depth=4 --param learning_rate=0.03
```

## Adding a model

1. New module in this package, e.g. `randomforest.py`.
2. Subclass `ClassifierModel`, set `model_type`, implement `fit` / `predict_proba`,
   decorate with `@register_classifier("randomforest")`.
3. Import it in `__init__.py` so registration fires on package import.
4. It's now available to `build_classifier(...)` and `--model randomforest`.

## Notes

- `predict_proba` returns a **1-D** positive-class probability array (not sklearn's
  2-column form) — downstream eval and plots only ever want that column.
- The runner builds features **dense** (`sparse=False`): XGBoost treats a CSR
  matrix's implicit zeros as *missing*, but a k-mer frequency of zero is a real,
  informative value ("k-mer absent"), not missing. Dense also reproduces
  `scripts/train_lncrna_xgboost.py` exactly. (Note: `tune_lncrna_xgboost.py` and
  `lncfit.cv` currently use sparse — a pre-existing inconsistency worth revisiting
  separately, out of scope for this layer.)
- Class-imbalance handling (~5% positive rate) lives inside each wrapper's `fit`
  (XGBoost `scale_pos_weight`, logreg `class_weight="balanced"`), not on callers.
