# Pipeline configuration (issue #78 follow-up)

Full flow -- raw data through feature-matrix construction (with a worked
numeric example) to training, tuning, CV, and evaluation. Source:
[`docs/diagrams/pipeline.d2`](../docs/diagrams/pipeline.d2).

![lncFit pipeline diagram](../docs/diagrams/pipeline.svg)

One YAML file describes an entire training run; `scripts/run_pipeline.py` reads it
and drives `lncfit.pipeline.LncRnaPipeline`. Every axis that used to mean picking a
different script is now a field in the config:

| You want to choose... | Config section | Options |
|---|---|---|
| Which model | `model.name` | `xgboost`, `logreg`, `randomforest`, `knn`, `mlp`, `null` |
| Which sequence features | `features.type` | `kmer`, `dnabert2` |
| Which cell-line embedding | `features.cell_embedding_dim` | `0` (one-hot only), `2` (Celligner UMAP), `10`/`70` (Celligner PCA) |
| How to pick hyperparameters | `tuning.method` | `fixed`, `grid`, `optuna` |
| How to cross-validate | `cv.strategy` | `none`, `chrom`, `stratified` |

```bash
uv run python scripts/run_pipeline.py --config configs/pipeline/xgboost_kmer_fixed.yaml
```

Four ready-to-run examples live in `configs/pipeline/` (see the table at the
bottom). Copy one and edit it rather than writing a config from scratch.

## Config schema

```yaml
data:
  train: data/processed/train_lncrna_day14_chrom1.jsonl.gz
  test: data/processed/test_lncrna_day14_chrom1.jsonl.gz
  # Required only when features.type is "kmer" -- {target: [spliced_seq, ""]} JSON
  # from `python -m lncfit.sequence --sequence-type transcript`.
  transcript_sequences: data/processed/body_sequences_transcript.json

features:
  type: kmer                 # "kmer" or "dnabert2"
  k: 5                       # kmer only: k-mer length (3-6)
  embeddings: ...            # dnabert2 only: path to an .npz from scripts/embed_sequences.py
  include_distance: false    # append distance_to_closest_pc_gene as a feature
  cell_embedding_dim: 0      # 0 = cell-line one-hot only (5 columns)
                             # 2 = + Celligner 2-D UMAP (data/external/celligner_cell_line_umap.csv)
                             # up to 70 = + that many Celligner pre-UMAP PCA columns
                             # (data/external/celligner_cell_line_pca.csv). See
                             # data/external/README.md -- bigger is not automatically
                             # better, dim=2 has the best AUPRC in our own sweep.

model:
  name: xgboost              # any name from lncfit.classifiers.available_classifiers()
  params: {}                 # fixed hyperparameters. Used as-is when tuning.method
                             # is "fixed"; ignored otherwise (tuning picks these instead).
  resample:                  # optional training-set resampling (lncfit.resample).
    method: none             # none | random_over | random_under | smote | smote_tomek
    ratio: auto              # "auto" = balance 1:1; a float = minority:majority
                             # target (0.3 => partial rebalancing).
                             # Applied to TRAINING splits only -- see below.

tuning:
  method: fixed              # "fixed" (use model.params as-is), "grid", or "optuna"
  search_space: configs/search_spaces/xgboost.yaml   # required for grid/optuna
  n_trials: 50               # optuna only: number of TPE trials
  metric: auprc              # "auprc" or "auroc" -- objective grid/optuna optimize,
                              # and the metric reported for a "fixed" run's CV score

cv:
  strategy: none             # "none", "chrom" (chromosome LOCO), or "stratified" (K-fold)
  n_splits: 5                # stratified only

seed: 42
output_dir: results/lncrna_rra_day14/pipeline_runs
```

`tuning.method: grid` or `optuna` **requires** `cv.strategy` to be `chrom` or
`stratified` -- tuning needs a validation split to score candidate hyperparameters
against; the pipeline raises immediately if this is missing rather than silently
guessing.

## Choosing a model (`model.name`)

Backed by `lncfit.classifiers` (see `lncfit/classifiers/README.md`):

| name | what it is |
|---|---|
| `xgboost` | gradient-boosted trees (the project's usual best performer) |
| `logreg` | L2 logistic regression, `class_weight="balanced"` |
| `randomforest` | bagged trees, `class_weight="balanced_subsample"` |
| `knn` | distance-weighted k-nearest-neighbors |
| `mlp` | one-hidden-layer network, trained with Adam + internal early stopping |
| `null` | constant base-rate baseline (the floor everything else must clear) |

Adding a new model means adding a new wrapper to `lncfit/classifiers/` (see that
package's README) -- it becomes available here automatically, no pipeline changes
needed.

**Note on early stopping:** unlike the older `scripts/tune_lncrna_xgboost.py` /
`scripts/tune_lncrna_stratified.py`, the `xgboost` wrapper here fits a flat
`n_estimators` trees with no `eval_set`/early stopping (same as
`scripts/run_lncrna_classifier.py`) -- keeps `model.fit(X, y)` uniform across every
registered model. If you need early-stopped XGBoost specifically, those older
scripts are still the place for that.

## Choosing features (`features.type`)

- **`kmer`** (default): k-mer frequency vector of the lncRNA's own spliced
  transcript sequence (`lncfit.features.build_lncrna_features`). Fast, no GPU,
  no precomputation beyond `lncfit/sequence.py`.
- **`dnabert2`**: a frozen DNABERT-2 embedding per lncRNA
  (`lncfit.features.build_lncrna_embedding_features`). Requires precomputing
  embeddings first:
  ```bash
  uv run python scripts/embed_sequences.py --sequence-type transcript \
      --output data/processed/dnabert2_transcript_full.npz
  ```
  then pointing `features.embeddings` at that file.

Both feature types accept `include_distance` and `cell_embedding_dim` identically,
so a model/feature/cell-embedding comparison is apples-to-apples.

### `features.embedding_pca` (dnabert2 only)

`0`/absent (default) uses the raw 768 embedding dims. Set it to a positive
integer to PCA-reduce the embedding to that many components first
(`lncfit.embeddings.reduce_embeddings_pca`) -- the motivation being that
mean-pooled transformer dims are highly correlated, which is awkward for trees
(`colsample_bytree` keeps sampling redundant columns).

**In our own sweep this did not help** -- PCA at 64/128 components was
consistently *worse* than the raw 768 dims on the chr1-held-out task (e.g.
`balanced_bagging` AUPRC 0.1441 raw vs 0.1236 at 128 components). Kept as an
option because it's cheap to try and may behave differently with other
models/embeddings, but don't reach for it expecting a win.

Leakage note: PCA (and its standardization) is fit on **training targets only**
in `lncfit.pipeline`, then applied to every row -- fitting on the full matrix
would let held-out chr1 genes' embedding distribution inform the projection.
`scripts/run_cellline_loco.py` deliberately fits on all targets instead, because
its folds partition by cell line and every lncRNA appears in all of them, so
there is no held-out gene to protect.

## Choosing a cell-line embedding (`features.cell_embedding_dim`)

See `data/external/README.md` for the full methodology and validation. Short
version: `0` is a plain 5-column cell-line one-hot; `2`/`10`/`70` add real
transcriptomic-similarity coordinates from a from-scratch Celligner realignment.
More dimensions is **not** automatically better -- our own sweep
(`results/lncrna_rra_day14/celligner_embedding_comparison/summary.csv`) found
AUPRC peaks at `dim=2` and drops at `dim=10`/`70` even though AUROC keeps
climbing. Try more than one value; don't assume 70 beats 2.

## Handling class imbalance (`model.resample`, and imbalance-aware models)

The task is ~4.5% positive. Two orthogonal levers: resample the training set
(`model.resample`, any model), or pick a model that reweights its own loss
(`histgb`, `randomforest`, `logreg`) or ensembles over balanced subsamples
(`balanced_bagging`).

Resampling is applied to **training splits only** -- both in CV folds and the
final fit, never to validation/test. Resampling a held-out split would change
the class balance you measure against and silently invalidate the metrics;
`tests/test_pipeline.py::test_resample_applies_to_training_split_only` pins this.

### Measured results (chr1-held-out, HEK293FT-excluded, 3 seeds each)

| config | AUPRC mean ± sd | AUROC mean | F1 mean |
|---|---|---|---|
| **`svm` rbf, Nystroem(2000), C=0.1** | **0.1813 ± 0.0088** | 0.6982 | 0.1834 |
| `svm` rbf, Nystroem(1000), C=1.0 | 0.1772 ± 0.0154 | 0.6872 | 0.1774 |
| `svm` rbf, Nystroem(1000), C=0.1 | 0.1661 ± 0.0055 | 0.6994 | 0.1810 |
| xgboost + `random_over` | 0.1602 ± 0.0019 | 0.6815 | 0.0412 |
| **xgboost, no resampling** | **0.1593 ± 0.0123** | **0.6940** | 0.0000 |
| `balanced_bagging` ratio=3 | 0.1504 ± 0.0052 | 0.6989 | 0.1895 |
| xgboost + `random_under` | 0.1485 ± 0.0201 | 0.6624 | 0.1604 |
| `balanced_bagging` ratio=5 | 0.1451 ± 0.0029 | 0.6947 | 0.1369 |
| `balanced_bagging` ratio=1 | 0.1414 ± 0.0051 | 0.6994 | 0.1874 |
| `randomforest` | 0.1379 ± 0.0058 | 0.6952 | 0.0000 |
| `histgb` | 0.1315 ± 0.0100 | 0.6703 | 0.2133 |
| xgboost + `smote` ratio=0.3 | 0.1155 ± 0.0045 | 0.6505 | 0.0354 |
| xgboost + `smote` | 0.1074 ± 0.0025 | 0.6400 | 0.1810 |
| xgboost + `smote_tomek` | 0.1063 ± 0.0027 | 0.6386 | 0.1859 |

What this says:

- **A Nystroem-approximated RBF SVM is the current best model on this task**
  (AUPRC 0.1813 ± 0.0088 vs XGBoost's 0.1593 ± 0.0123 -- its *mean* clears the
  incumbent's luckiest single run). Switching model *family* helped where
  resampling didn't. Notes:
  - Use `kernel_approx`, not the exact kernel: exact RBF took ~198s per fit and
    scored *worse* (0.1581). The low-rank approximation regularizes a kernel that
    was overfitting 844 correlated dims -- 52x faster and better.
  - Performance plateaus around 2000 components (3000: 0.1804-0.1886, 4000:
    0.1847), so there's no reason to pay for more.
  - Low `C` (0.1) beats 1.0, and with tighter variance.
  - `kernel: linear` is *bad* here (0.0983) -- the nonlinearity is doing the work.
  - **This does not transfer to the cell-line-LOCO task**, where the same model
    gets the best AUROC seen (0.6021) but by far the worst AUPRC (0.0835 vs
    0.1435): it orders the bulk reasonably but is poor at the top of the ranking,
    which is what AUPRC weights. Don't assume a chr1 win generalizes.
- **No resampling strategy beats plain XGBoost on ranking metrics.** Imbalance
  handling was not the bottleneck here; the model family was.
- **SMOTE is actively harmful** (0.106-0.116 vs 0.159), and consistently so
  across seeds. Interpolating between neighbours in a 768-dim frozen
  transformer-embedding space evidently manufactures points that don't lie on
  the real data manifold. Don't reach for it on embedding features.
- **`random_over` matches the baseline's mean AUPRC with ~6x lower variance**
  (sd 0.0019 vs 0.0123). It won't produce a lucky high score, but it also won't
  produce a bad one -- worth preferring when you want a number you can trust
  rather than a leaderboard-topping one.
- **`balanced_bagging`'s `sampling_ratio` matters**: 3.0 clearly beats the 1.0
  default (0.1504 vs 0.1414), i.e. fully-balanced 1:1 undersampling is too
  aggressive and throws away too much majority data per estimator.
- **The reweighting/undersampling models are the only ones that make hard
  positive calls**: F1 0.14-0.21 vs exactly 0.0000 for unweighted xgboost and
  randomforest, which never cross the 0.5 threshold. If you need hit/no-hit
  labels rather than a ranking, use one of those even though AUPRC is lower.

## Ensembling runs (`scripts/ensemble_predictions.py`)

Rank-averages several runs' `predictions.csv` into one. Ranks rather than raw
scores because the inputs are on incomparable scales (XGBoost emits ~[0, 0.5]
probabilities; the `svm` wrapper emits an uncalibrated `sigmoid(margin)`), so a
plain mean would let the wider-spread model dominate. Weights are equal by
default on purpose -- fitting blend weights to maximize the held-out score is
leaderboard overfitting, not a result.

**Measured: ensembling reliably improves AUROC and reliably *hurts* AUPRC here**,
so it does not help on either leaderboard (both rank by AUPRC):

| task | combination | AUROC | AUPRC |
|---|---|---|---|
| chr1 | SVM alone (current #1) | 0.7005 | **0.1851** |
| chr1 | XGBoost alone | 0.7010 | 0.1733 |
| chr1 | SVM + XGBoost | **0.7074** | 0.1697 |
| LOCO | XGBoost alone (current #1) | 0.5956 | **0.1435** |
| LOCO | SVM alone | 0.6021 | 0.0835 |
| LOCO | XGBoost + SVM | **0.6041** | 0.1187 |
| LOCO | XGBoost x 5 seeds | 0.5877 | 0.1424 |

The direction is consistent and has a clear reason: AUPRC is dominated by the
very top of the ranking, and averaging a strong model with a weaker one dilutes
exactly the confident top-of-list calls that AUPRC rewards, while the extra
robustness shows up in the bulk ordering that AUROC measures. **If a future
challenge ranks by AUROC, ensemble; while it ranks by AUPRC, don't.**

Seed-averaging the same model (LOCO x5) behaves like the variance reduction it
is: 0.1424 beats the 5-seed *mean* of 0.1408 but not the luckiest single seed
(0.1435, which is the currently-submitted run). Worth knowing that the LOCO #1
is a best-of-5 rather than a typical result.

## Choosing a tuning method (`tuning.method`)

- **`fixed`**: use `model.params` as-is, no search. If `cv.strategy` is also set,
  still runs that CV once with the fixed params purely to report a comparable
  score alongside the held-out test metrics.
- **`grid`**: exhaustive grid over every parameter in the search-space file that
  has a `grid:` list (`configs/search_spaces/<model>.yaml`), scored by mean
  `tuning.metric` across CV folds.
- **`optuna`**: TPE-sampled search (`tuning.n_trials` trials) over every parameter
  in the search-space file that has `type`/`low`/`high` (or `type: categorical` +
  `choices`), same CV-scored objective as grid.

Search-space files live in `configs/search_spaces/`, one per model
(`xgboost.yaml`, `logreg.yaml`, `randomforest.yaml`, `knn.yaml`, `mlp.yaml`), and
can define both a `grid:` list and a `low`/`high` range for the same parameter so
one file serves both tuning methods. To add tuning support for a new model,
add a matching `configs/search_spaces/<name>.yaml`.

## Choosing cross-validation (`cv.strategy`)

- **`none`**: no CV. Only valid with `tuning.method: fixed`.
- **`chrom`**: chromosome leave-one-chromosome-out, one fold per chromosome with
  enough records (matches `lncfit.cv.build_lncrna_folds`'s grouping). Leak-free
  with respect to the lncRNA's own sequence.
- **`stratified`**: plain `StratifiedKFold` over the binary label, ignoring
  chromosome. **Not leak-free** for k-mer features -- every cell-line row of a
  given lncRNA shares one k-mer vector, so the same lncRNA can appear in both a
  fold's train and validation split via its other cell-line rows (same caveat
  already documented in `scripts/tune_lncrna_stratified.py`). Kept because it's
  useful for direct comparison against that script's numbers, not because it's
  leakage-free.

Both strategies here fit the k-mer vocabulary **once**, on all training records,
rather than refitting it per fold -- a deliberate simplification so the same CV
code path (`lncfit.cv.make_cv_splits`) works for k-mer and DNABERT-2 features
alike. This is slightly more optimistic than `lncfit.cv.build_lncrna_folds`'s
per-fold vocab refit; use that older function directly if you need the stricter
version.

## What a run writes

Under `output_dir/run_<model>_<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | exact config used (echoed back, for reproducibility) |
| `best_params.json` | hyperparameters the final model was fit with |
| `cv_scores.csv` | per-fold (fixed) or per-combo/per-trial (grid/optuna) CV scores |
| `metrics.csv` | held-out chr1 test AUROC/AUPRC/etc., overall + per cell line |
| `predictions.csv` | target, cell_line, y_true, y_pred_proba for every test row |
| `run_info.json` | model, best params, feature/CV/tuning choices, git commit |

## Example configs (`configs/pipeline/`)

| File | model | features | cell embedding | tuning | cv |
|---|---|---|---|---|---|
| `xgboost_kmer_fixed.yaml` | xgboost | kmer (k=5) | off | fixed | stratified (report only) |
| `xgboost_kmer_optuna.yaml` | xgboost | kmer (k=5) | UMAP (2) | optuna, 50 trials | chrom |
| `logreg_kmer_grid.yaml` | logreg | kmer (k=3) | off | grid | stratified |
| `mlp_dnabert2_fixed.yaml` | mlp | dnabert2 | PCA (10) | fixed | none |
