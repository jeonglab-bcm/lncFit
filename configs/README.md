# Pipeline configuration (issue #78 follow-up)

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

## Choosing a cell-line embedding (`features.cell_embedding_dim`)

See `data/external/README.md` for the full methodology and validation. Short
version: `0` is a plain 5-column cell-line one-hot; `2`/`10`/`70` add real
transcriptomic-similarity coordinates from a from-scratch Celligner realignment.
More dimensions is **not** automatically better -- our own sweep
(`results/lncrna_rra_day14/celligner_embedding_comparison/summary.csv`) found
AUPRC peaks at `dim=2` and drops at `dim=10`/`70` even though AUROC keeps
climbing. Try more than one value; don't assume 70 beats 2.

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
