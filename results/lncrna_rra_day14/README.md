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

![Untuned vs tuned AUROC/AUPRC across k](auroc_auprc_sweep.png)

`scripts/plot_lncrna_auc_sweep.py` regenerates this from the committed metrics CSVs
(`metrics_k<K>.csv` for untuned, `tune_k<K>/final_eval_*/metrics.csv` for tuned — a k
with no completed tuned run, like k=6 here, is labeled "not completed" instead of
plotted).

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
