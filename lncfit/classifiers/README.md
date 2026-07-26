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
| `randomforest.py` | `RandomForestClassifier` — bagged trees, `class_weight="balanced_subsample"` |
| `knn.py` | `KNNClassifier` — distance-weighted k-nearest-neighbors |
| `mlp.py` | `MLPClassifier` — one-hidden-layer network, Adam + internal early stopping |
| `xgboost_clf.py` | `XGBoostClassifier` — gradient-boosted trees, auto `scale_pos_weight` |
| `histgb.py` | `HistGradientBoostingClassifier` — sklearn histogram GBM, `class_weight="balanced"`, internal early stopping. Dense X only |
| `balanced_bagging.py` | `BalancedBaggingClassifier` — trees each fit on all positives + an equal-size majority subsample (undersampling ensemble, not loss reweighting) |
| `__init__.py` | imports the wrappers to trigger registration |

## Usage

```python
from lncfit.classifiers import build_classifier, available_classifiers

available_classifiers()               # ['knn', 'logreg', 'mlp', 'null', 'randomforest', 'xgboost']
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

Or, for the full model x features x cell-embedding x tuning x CV pipeline, set
`model.name` in a YAML config and run `scripts/run_pipeline.py --config <file>.yaml`
(see `configs/README.md`) — any model registered here is automatically available
there too, nothing extra to wire up.

## Adding a model

1. **Add a new module** in this package, e.g. `svm.py`. Subclass `ClassifierModel`
   and implement its three required pieces:

   ```python
   from lncfit.classifiers.base import ClassifierModel
   from lncfit.classifiers.registry import register_classifier

   @register_classifier("svm")
   class SVMClassifier(ClassifierModel):
       model_type = "svm"

       def __init__(self, C: float = 1.0, seed: int = 42, **params) -> None:
           super().__init__(C=C, seed=seed, **params)  # stash hyperparams on self.params
           self._model = None

       def fit(self, X, y) -> "SVMClassifier":
           # handle the ~5% positive rate yourself (e.g. class_weight="balanced")
           self._model = ...fit on (X, y)...
           return self

       def predict_proba(self, X):
           # 1-D P(hit) per row, NOT sklearn's 2-column form -- see Notes below
           return self._model.predict_proba(X)[:, 1]
   ```

2. **Register it for import** — add `from lncfit.classifiers.svm import SVMClassifier`
   to `__init__.py` (and to its `__all__` list) so the `@register_classifier`
   decorator actually fires when the package is imported. Forgetting this step is
   the most common way a new model "doesn't show up."
3. **Done** — it's now available everywhere by name, no other file needs to change:
   - `build_classifier("svm", ...)` / `available_classifiers()`
   - `scripts/run_lncrna_classifier.py --model svm`
   - `model.name: svm` in a `scripts/run_pipeline.py` config
4. **Optional — make it tunable via grid/Optuna**: add
   `configs/search_spaces/svm.yaml` listing its hyperparameters (see the existing
   files there for the format — each parameter can carry a `grid:` list, an
   Optuna `low`/`high` range, or both). Without this file, `svm` still works fine
   with `tuning.method: fixed`; you only need it for `grid`/`optuna`.
5. **Add a test** in `tests/test_classifiers.py` — the existing
   `test_classifier_wrappers_share_the_fit_predict_contract` loops over a list of
   model names; add `"svm"` to it and it's covered by the same shared contract
   checks (predict_proba shape, [0, 1] range, dense/sparse handling) for free.

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
