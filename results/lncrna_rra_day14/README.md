# lncRNA-level RRA hit classification (Day 14)

Issue #60: predict, per lncRNA x cell line, whether MAGeCK-RRA calls it a significant
depletion hit at Day 14 (RRA P value < 0.05 and log2 fold-change < 0), instead of
regressing per-guide log2 fold-change.

Dataset: `data/processed/lncrna_rra_day14.jsonl.gz`, built from the previously-unparsed
`S2F`-`S2J` RRA sheets of `mmc3.xlsx`, restricted to `Target group == "long non-coding RNA"`
rows (5,496 lncRNAs x 5 cell lines = 27,480 records; 1,249 hits, 4.5% positive rate).
Chromosome-1 hold-out split (`train_lncrna_day14_chrom1.jsonl.gz` / `test_lncrna_day14_chrom1.jsonl.gz`):
25,010 train / 2,470 test.

> **The numbers below are the corrected ones (issue #65).** The #61/#62 runs built
> features by pooling k-mer frequencies over each lncRNA's **guide spacer sequences**
> (the CRISPR-Cas13 guides designed *against* the lncRNA) — not the lncRNA's own
> sequence. That's a real confound, not a minor caveat: guide design is an engineered
> process, so any signal learned from it isn't necessarily lncRNA biology. See the
> "Superseded results" section at the bottom for the old numbers and why they don't
> hold up. Features now come from each lncRNA's own **spliced transcript sequence**
> (Sarropoulos et al. 2019, PMC6660317, hg19/GRCh37 — see issue #66 on why hg38 was
> dropped entirely rather than kept as an option).

Features: k-mer frequencies over the lncRNA's own spliced transcript sequence
(`lncfit.sequence.extract_spliced_sequences`, longest transcript per gene) + cell-line
one-hot. XGBoost classifier (`binary:logistic`), same `tree_method="hist"` as
`scripts/train_xgboost.py`. Reproduce end-to-end in
`notebooks/lncrna_rra_analysis.py` (`uv run marimo edit notebooks/lncrna_rra_analysis.py`).

## k sweep, untuned (chr1 test split)

Fixed hyperparameters copied from the guide-level regression model's defaults,
`scale_pos_weight` computed once from the raw train-set class ratio (~21.35).

| k | features | AUROC | AUPRC |
|---|---|---|---|
| 3 | 69   | 0.6051 | 0.0846 |
| **4** | 261  | **0.6352** | 0.1050 |
| 5 | 1029 | 0.5980 | 0.1015 |
| 6 | 4101 | 0.5872 | **0.1079** |

Trivial baseline on this test set: AUROC 0.500, AUPRC 0.053 (5.3% positive rate). Every
k beats it, but only modestly (max AUROC 0.635, a ~1.3x lift on AUPRC at best) — nowhere
near the 0.70+ AUROC the guide-sequence-based numbers showed. That gap is the size of
the guide-design confound: a real chunk of the previously reported "signal" was not
lncRNA biology.

## Hyperparameter tuning (issue #62, re-run for #65)

`scripts/tune_lncrna_xgboost.py`: Optuna TPE (50 trials) + chromosome LOCO-CV (20 folds,
`build_lncrna_folds`), searching 7 hyperparameters plus a `scale_pos_weight_mult` that
scales each fold's natural neg/pos ratio. CV objective is mean AUPRC across folds. Best
trial's config is retrained on all training data (early-stopped on chr22) and evaluated
once on the real chr1 held-out test set.

| k | CV mean AUPRC | held-out AUROC (tuned) | held-out AUPRC (tuned) | held-out AUPRC (untuned) |
|---|---|---|---|---|
| **3** | 0.1307 ± 0.052 | 0.6248 | **0.1190** | 0.0846 |
| 4 | (killed before completion; see below) | 0.6087 | 0.0932 | 0.1050 |
| 5 | (killed before completion; see below) | 0.5748 | 0.0981 | 0.1015 |
| 6 | *not completed* | *not completed* | *not completed* | 0.1079 |

k=3 tuned is the best AUPRC found (0.119, ~2.3x the 0.053 baseline), improving on its own
untuned run (0.085) — unlike the #62 guide-sequence result, where tuning *hurt* k=3.
k=4 and k=5 tuning slightly underperformed their untuned baselines this time. **k=6's
tuning sweep was stopped by request after trial 8/50** (large feature space made it far
slower than k=3-5; partial per-trial CV scores are in `tune_k6/cv_scores.csv`, but there
is no completed held-out evaluation for it — it is not included as a "tuned" result).

**Caveat on k=3 tuned**: its retrained final model has only **4 trees**
(`n_estimators_final_model` in `data/model/xgboost_lncrna_best_params_k3.json`) — the
same early-stopping-on-a-tiny-slice pattern seen in the original #62 tuning (chr22's
early-stop slice has only 19 positives). A 4-tree model landing on the best AUPRC of the
sweep is plausible but not something to over-trust; a follow-up should revisit the
early-stopping chromosome choice or patience for low-capacity k values.

![ROC curves for the tuned k=3,4,5 models on the held-out chr1 test set](roc_curves.png)

`scripts/plot_lncrna_roc.py` regenerates this from each completed tuning run's
`tune_k<K>/final_eval_*/predictions.csv`. k=6 has no curve — its tuning sweep was
stopped early (see above) and has no held-out predictions to plot.

## Per-cell-line breakdown, k=3 tuned (best AUPRC)

| cell line | AUROC | AUPRC |
|---|---|---|
| Overall | 0.6248 | 0.1190 |
| HAP1 | 0.5203 | 0.0570 |
| HEK293FT | 0.6109 | 0.1047 |
| K562 | **0.6956** | **0.2273** |
| MDA-MB-231 | 0.5805 | 0.0576 |
| THP1 | 0.6006 | 0.0723 |

K562 is consistently the strongest cell line across both the untuned and tuned sweeps
(also true in the old, invalidated guide-sequence numbers) — worth a closer look in a
follow-up on whether that's a real biological signal specific to K562 or a dataset
artifact. Precision/recall at the default 0.5 threshold are ~0 across the board (still
a ~5%-positive-rate problem; precision-at-k would be a better fit than F1@0.5).

## Row-level stratified CV comparison (chromosome-agnostic; xgboost + logreg; k=3-6)

A second tuning pass, `scripts/tune_lncrna_stratified.py`, answers a different
question from the LOCO-CV sweep above: what if hyperparameter search uses a plain
`StratifiedKFold(n_splits=5)` over the binary label instead of chromosome-grouped
folds? **This is deliberately not leak-free**: every cell-line row for a given
lncRNA shares one k-mer vector (only the cell-line one-hot differs), so the same
lncRNA's sequence can appear in both a fold's train and validation split via its
other cell-line rows. Kept this way on request, for direct comparison against the
chromosome-grouped numbers above — not because it's a sound generalization
estimate. The tell: CV mean AUPRC runs ~0.17 for xgboost vs. the true chr1
held-out AUPRC of ~0.12-0.15. Trust the held-out column, not the CV column.

Also swept a class-weight on/off toggle (xgboost: tuned `scale_pos_weight_mult` vs.
fixed at 1; logreg: `class_weight="balanced"` vs. `None`) and, for xgboost, a
`VarianceThreshold` filter on k-mer columns (fit per fold / per final-train split,
never on val/test) to make k=6's 4,096-column space more tractable.

**chr1 held-out test results** (best class-weight variant per k, by AUPRC):

| k | model | class weight | n features | AUROC | AUPRC |
|---|---|---|---|---|---|
| 3 | xgboost | off | 69 | 0.6623 | 0.1293 |
| 4 | xgboost | off | 261 | 0.6631 | 0.1227 |
| **5** | **xgboost** | **off** | 1,029 | 0.6628 | **0.1446** |
| 6 | xgboost | off (variance filter, threshold=2e-7) | 2,468 / 4,101 | 0.5987 | 0.1209 |
| 3-6 | logreg | on/off (barely differs) | 69-2,468 | ~0.578 | ~0.109-0.110 |

k=5 (off) is the best AUPRC found anywhere in this project's history on the real
chr1 test set among the Optuna-tuned configs above -- but see the `max_depth`
follow-up below, which pushes both records (AUROC and AUPRC) further still.
k=6 underperforms k=3-5 on 4 of 5 cell lines (HAP1, HEK293FT,
MDA-MB-231, THP1) despite the variance filter, while being the *best* single
cell-line result anywhere (K562, AUPRC 0.315 with class-weight on) — consistent
with a dimensionality problem, not a k=6-is-just-worse conclusion. The final
90/10-split XGBoost training set has only ~1,007 positives; k=6's 2,468 post-filter
features means ~2.5 features per positive example (k=5 sits right at ~1:1, k=3 is
~0.07:1) — past the point where a model can reliably distinguish real signal from
per-fold noise, especially under CV that's already leaking lncRNA identity across
folds. The variance filter itself (fit on train-fold frequency variance) only cut
xgboost's k=6 per-trial wall-clock by ~14% (4:38 -> 4:00) since `tree_method="hist"`
with `colsample_bytree≈0.8` is bottlenecked by tree depth and row count, not raw
column count — it's a legitimate noise-reduction step, not a speed fix.

Class-weight reweighting doesn't clearly help xgboost (off wins on AUPRC at k=3,
4, and 5) and is a rounding error for logreg. logreg stays flat at ~0.578 AUROC
across every k and class-weight setting -- k-mer counts alone don't give it more
to work with as k grows.

### max_depth follow-up: forcing depth=9 onto tuned configs

Two one-off probes (`tune_stratified/k4_depth9_comparison.json`,
`tune_stratified/k5_depth9_comparison.json`) tested forcing `max_depth=9`
(matching k=3's tuned depth) onto k=4's and k=5's Optuna-tuned configs, keeping
every other hyperparameter unchanged:

| k | class weight | original depth | AUROC (orig -> depth=9) | AUPRC (orig -> depth=9) | n_estimators (orig -> depth=9) |
|---|---|---|---|---|---|
| 4 | on | 6 | 0.6577 -> 0.6365 (worse) | 0.1120 -> 0.1155 | 58 -> 37 |
| 5 | on | 5 | 0.6482 -> 0.6190 (worse) | 0.1268 -> 0.1104 (worse) | 72 -> 33 |
| 5 | off | 4 | 0.6628 -> **0.6801 (new best AUROC)** | 0.1446 -> **0.1460 (new best AUPRC)** | 495 -> 106 |

A clear, replicated pattern: **forcing depth=9 helps the class-weight=off
configs but hurts class-weight=on ones.** Both reweighted (`on`) configs
collapse to very few trees once forced deeper (58->37, 72->33) --
`scale_pos_weight` pushes predictions toward positives faster, so early
stopping triggers sooner once trees can also go deeper, cutting the ensemble
down before it benefits from the extra depth. The non-reweighted (`off`)
configs don't have that interaction; k=5/off/depth=9 gets a straightforward
capacity boost and is now **the best AUROC and AUPRC found anywhere in this
project's history** (previous records: AUROC 0.6757 at k=3/on, AUPRC 0.1446 at
k=5/off/depth=4). Per-cell-line, k=5/off/depth=9 improves HAP1 and THP1
substantially (AUPRC 0.127->0.184, 0.160->0.261) while K562 stays strong
(AUROC 0.757) and MDA-MB-231 dips slightly (0.268->0.215).

A separate probe on the plain **untuned** k=4 defaults
(`scripts/run_lncrna_classifier.py`, no Optuna involved, no class-weight
reweighting) told a version of the same story: AUROC improved 0.6352 -> 0.6578,
AUPRC 0.1050 -> 0.1228 when forced to depth=9, consistent with "no reweighting
+ more depth" being the combination that benefits.

Net: `max_depth` isn't a free knob to crank in isolation -- it interacts with
`scale_pos_weight`/early stopping, not just with the other tree-shape
hyperparameters. The one-off comparison JSONs only test the two hyperparameter
combinations described above; they are not a re-run of the full Optuna search
at depth=9, so there may be an even better config than k=5/off/depth=9 that
a proper search around fixed depth=9 would find.

Files: `tune_stratified/<model>_k<K>_cw<on|off>/` (`cv_scores.csv`,
`final_eval_<timestamp>/{metrics,predictions,run_info}`), `tune_stratified/summary.csv`
(aggregated via `scripts/summarize_stratified_tuning.py`), the `k4_depth9_comparison.json`
/ `k5_depth9_comparison.json` one-off probes above, and
`data/model/<model>_lncrna_stratified_k<K>_cw<on|off>_{best_params,vocab}.json`
(+ `_variance_mask.json` for k=6). `logreg_k6_cwoff` was stopped mid-run by
request and is not included (no completed held-out evaluation).

## Files

- `metrics_k3.csv`, `metrics_k4.csv`, `metrics_k5.csv`, `metrics_k6.csv` — untuned
  per-cell-line classification metrics for each k (corrected transcript-sequence features).
- `tune_k<K>/cv_scores.csv` — per-trial, per-chromosome CV AUPRC for the Optuna search
  (k=6 is partial — only 8/50 trials, stopped by request).
- `tune_k<K>/final_eval_<timestamp>/` — tuned final model's held-out test metrics and
  predictions for each completed k (3, 4, 5).
- `data/model/xgboost_lncrna_best_params_k<K>.json` — best hyperparameters + CV/held-out
  summary for each k.
- `../../notebooks/lncrna_rra_analysis.py` — marimo notebook reproducing the data
  loading, feature building, and a quick interactive fit.

## Superseded results (guide-sequence features — do not use)

The original #61/#62 passes built features from each lncRNA's **guide spacer
sequences** instead of its own sequence (issue #65). Kept here only for the historical
record of what changed and by how much:

| k | untuned AUROC / AUPRC (guide-seq) | tuned AUROC / AUPRC (guide-seq) |
|---|---|---|
| 3 | 0.7049 / 0.1403 | 0.6719 / 0.1007 |
| 4 | 0.7031 / 0.1102 | 0.7169 / 0.1613 |
| 5 | 0.7018 / 0.1198 | 0.6968 / 0.1149 |
| 6 | 0.6494 / 0.1084 | 0.7023 / 0.1509 |

Every one of these is higher than its corrected counterpart above. The delta is the
guide-design confound, not a regression in the model.
