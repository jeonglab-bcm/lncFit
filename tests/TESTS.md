# Test suite inventory

What's in `tests/`, file by file, and why each test exists. Written after a full
line-by-line review (see PR that introduced this file) that removed 4 genuinely
redundant tests out of the original 132; this document covers the 128 that
remain, plus the one file that can't currently collect in this environment.

**Scale for context**: `lncfit/` (the reusable library) is ~2,020 lines;
`scripts/` (one-off tuning/training CLIs) is ~3,624 lines; these tests are
~1,400 lines. Coverage is concentrated almost entirely on `lncfit/` — the
scripts get two small helper-function tests each and are otherwise untested by
design, since a bug there just makes one analysis run wrong, not something
that silently corrupts every downstream use the way a library bug would.

---

## test_screen_data.py — guide-level screen data loading pipeline

Covers `load_targets`, `load_annotations`, `load_screen` (the S1/S2 sheet melt
+ join that turns `mmc2.xlsx`/`mmc3.xlsx` into per-guide fold-change records),
`to_dataframe`, and `ScreenRecord`'s jsonl round-trip / schema versioning /
`from_dict` defaulting.

| Test | Verifies |
|---|---|
| `test_load_targets` | `load_targets` maps guide ID -> (target gene, sequence) correctly from the S1B sheet. |
| `test_load_annotations` | `load_annotations` maps target -> (chrom, strand, closest PC gene, distance) from S1A. |
| `test_load_annotations_blank_distance_is_none` | A blank distance cell becomes `None`, not `0` or `""`. |
| `test_melt_produces_4_rows_per_guide_per_cell_line` | The wide fold-change table (4 day/replicate columns) melts into 4 long-format rows per guide per cell line. |
| `test_total_row_count` | End-to-end row count matches `guides x cell_lines x fc_columns` exactly (2x5x4=40). |
| `test_day_and_replicate_parsed` | Day (7/14) and replicate (1/2) are correctly parsed out of the FC column headers. |
| `test_negative_fold_changes_preserved` | Negative fold-change values survive the melt (not accidentally clipped/abs'd). |
| `test_s1_s2_join` | A record's `target`/`target_sequence` correctly join from the S1B guide-to-target table. |
| `test_annotations_enriched_in_records` | When annotations are supplied, chrom/strand/closest-gene/distance land on the record. |
| `test_missing_annotation_falls_back_to_defaults` | Passing an explicit empty `{}` for annotations still yields safe defaults (`""`/`None`) via the dict `.get()` fallback path. |
| `test_annotations_optional` | *Omitting* the annotations argument entirely (not passing `{}`) also yields safe defaults — a different code branch (`if annotations is not None`) than the test above, so both are needed. |
| `test_to_dataframe_schema` | `to_dataframe` produces exactly the expected column set and row count. |
| `test_save_load_jsonl_round_trip` | A record written to `.jsonl.gz` and reloaded is byte-for-byte equal to the original. |
| `test_jsonl_stamped_with_schema_version` | Every saved record carries the current `SCHEMA_VERSION` tag. |
| `test_from_dict_ignores_unknown_keys` | `ScreenRecord.from_dict` tolerates/drops keys it doesn't recognize (forward compatibility). |
| `test_from_dict_missing_optional_fields_use_defaults` | Optional fields absent from a dict get their documented defaults, not a `KeyError`. |

## test_lncrna_rra.py — lncRNA RRA-hit data loading pipeline

Covers `load_target_groups`, `load_rra` (issue #60's Day-14 lncRNA hit-calling
loader, reading the S2F-S2J RRA sheets), and `LncRnaRecord`'s jsonl/schema/
`from_dict` behavior. This is a materially different code path from
`load_screen` above (hit-labeling logic, day selection, lncRNA-only filtering)
despite superficially similar test scaffolding.

| Test | Verifies |
|---|---|
| `test_load_target_groups` | Maps target -> group string (`"long non-coding RNA"` / `"protein-coding gene"` / `"non-targeting"`). |
| `test_load_rra_filters_to_lncrna_only` | Only `"long non-coding RNA"`-group rows survive; protein-coding rows (e.g. TP53) are dropped entirely, not just relabeled. |
| `test_load_rra_label_significant_and_negative_is_hit` | `p < 0.05 AND log2FC < 0` -> `label=1` (one branch of the compound hit condition). |
| `test_load_rra_label_not_significant_is_non_hit` | `p >= 0.05` -> `label=0` regardless of fold-change (second branch). |
| `test_load_rra_significant_but_positive_fc_is_non_hit` | `p < 0.05` but `log2FC > 0` -> `label=0` (third branch — significant enrichment isn't a depletion hit). All three branches of the AND are independently exercised since flipping either half wrong is a classic bug. |
| `test_load_rra_day_selection_reads_correct_columns` | `day=7` vs `day=14` reads distinct p-value/fold-change columns, not the same ones twice. |
| `test_load_rra_joins_annotations` | Chrom/strand/closest-gene/distance land on the record when annotations are supplied. |
| `test_load_rra_missing_annotation_defaults` | Empty annotations dict -> safe defaults. |
| `test_load_rra_skips_missing_pvalue_or_fc` | A row with a blank p-value or fold-change cell is skipped, not coerced to `0`/`NaN`-as-hit. |
| `test_save_load_lncrna_jsonl_round_trip` | Round-trip equality through `.jsonl.gz`. |
| `test_lncrna_jsonl_stamped_with_schema_version` | Schema version tag present on save. |
| `test_lncrna_from_dict_ignores_unknown_keys` | Forward-compatible `from_dict`. |
| `test_lncrna_from_dict_missing_optional_fields_use_defaults` | Optional field defaults. |

## test_features.py — guide-level feature engineering

Covers `all_kmers`, `kmer_freq_vector`, `fit_vocab`, and `build_features` (the
k-mer + day + cell-line + optional-distance + optional-signed-overlap feature
matrix builder used by the guide-level regression models).

| Test | Verifies |
|---|---|
| `test_k3_length` / `test_k6_length` | `all_kmers(k)` returns exactly `4**k` entries. |
| `test_sorted_order` | The k-mer vocabulary is lexicographically sorted (implies "AAA" first / "TTT" last for any k — those two used to be separate tests and were removed as redundant). |
| `test_sums_to_one` | A k-mer frequency vector sums to 1 (it's a real frequency distribution). |
| `test_correct_counts` | A specific homopolymer sequence produces the exact expected count at the right vocab index. |
| `test_skips_non_acgt` | Windows containing a non-ACGT base (e.g. `N`) are excluded from the count, not treated as a phantom k-mer. |
| `test_all_non_acgt_returns_zeros` | An all-`N` sequence returns an all-zero vector rather than dividing by zero. |
| `test_x_shape_k3` / `test_x_shape_k6` | Feature matrix column count matches the `k-mers + day-onehot + cell-onehot` formula at two different k values. |
| `test_y_values` | The label vector is the raw fold-change, unmodified. |
| `test_day_onehot_columns_present` / `test_cell_line_onehot_columns_present` | The categorical one-hot columns exist and are named as expected. |
| `test_include_distance_true` / `test_include_distance_false` | The distance column is present/absent exactly per the `include_distance` flag. |
| `test_include_distance_none_becomes_sentinel` | A missing distance value becomes the `-1` sentinel, not `NaN` or `0`. |
| `test_sparse_matches_dense` | Sparse and dense builds produce identical columns and values — this codebase has previously had real sparse-vs-dense semantics bugs (XGBoost treats sparse implicit zeros as *missing*), so this equivalence check is load-bearing, not decorative. |
| `test_custom_vocab_reduces_columns` | Passing a truncated vocab shrinks the feature matrix accordingly. |
| `test_holdout_unseen_kmer_silently_dropped` | A k-mer present in holdout data but absent from the training vocab is silently ignored (zero contribution), not an error or a leaked extra column. |
| `test_custom_vocab_sparse_matches_dense` | The sparse/dense equivalence still holds under a restricted vocab (distinct from the full-vocab case above — vocab restriction and sparse encoding must compose correctly together). |
| `TestSignedOverlap` (2 tests) | Reverse-complement overlap between a guide and its target body correctly flips the sign of shared k-mers (and only those), with sparse/dense equivalence re-checked for this separate code path. |
| `TestFitVocab` (5 tests) | `fit_vocab` returns only observed k-mers, sorted, skipping non-ACGT k-mers, always a subset of the full alphabet, and empty input yields an empty vocab. |

## test_lncrna_features.py — lncRNA-level feature engineering

Covers `build_lncrna_features`, the classifier-task counterpart to
`build_features` above — key difference: features come from the lncRNA's own
transcript sequence (issue #65), shared identically across that lncRNA's rows
for every cell line, with a binary hit label instead of continuous fold-change.

| Test | Verifies |
|---|---|
| `test_shape_matches_vocab_plus_cell_columns` | Column count is `vocab + 5 cell-line one-hot` (no day dimension — Day-14-only task). |
| `test_y_is_binary_label_not_fold_change` | The label vector is the binary hit/non-hit flag, not a regression target. |
| `test_kmer_freq_computed_from_own_transcript_sequence` | Frequencies are computed from the lncRNA's own sequence, normalized correctly over a restricted vocab. |
| `test_same_target_shares_feature_vector_across_cell_lines` | Two rows for the same lncRNA (different cell lines) get identical k-mer features — the whole point of issue #65's fix. |
| `test_cell_line_one_hot` | Cell-line one-hot columns are correctly set. |
| `test_no_day_column_present` | No `day_*` column exists (would be a regression from the guide-level builder's schema). |
| `test_include_distance_uses_negative_one_sentinel_when_missing` | Same `-1` sentinel convention as the guide-level builder. |
| `test_target_missing_from_transcript_sequences_gets_zero_kmer_vector` | A target with no known sequence gets an all-zero k-mer vector rather than crashing or silently vanishing. |
| `test_sparse_matches_dense` | Same sparse/dense equivalence guarantee as the guide-level builder, checked independently since it's a separate function. |

## test_classifiers.py — pluggable classifier registry (`lncfit/classifiers/`)

Covers the `null`/`logreg`/`xgboost` model wrappers and the
`@register_classifier` / `build_classifier` registry added for the systematic
model-comparison runner.

| Test | Verifies |
|---|---|
| `test_registry_has_expected_models` | Exactly `{null, logreg, xgboost}` are registered. |
| `test_build_unknown_raises` | Requesting an unregistered model name raises a clear error. |
| `test_duplicate_registration_raises` | Re-registering an existing model name is rejected, not silently overwritten. |
| `test_wrapper_fit_returns_self_and_proba_shape_range` (x3 models) | `fit()` returns `self` (chainable) and `predict_proba` returns a `[0,1]`-bounded 1-D array of the right length, for every registered model. |
| `test_wrapper_accepts_sparse_and_dense` (x2 models) | logreg/xgboost give identical predictions from sparse or dense input — the same real risk area as the feature-builder sparse/dense tests above, checked at the model layer too. |
| `test_null_predicts_training_base_rate` | The null baseline predicts exactly the training positive rate for every row, by construction. |
| `test_model_type_matches_registry_key` | Every registered class's `model_type` attribute matches its registry key (a consistency invariant, not behavior). |
| `test_params_stored_and_forwarded` | Constructor kwargs are captured on `.params` and actually used. |
| `test_xgboost_auto_scale_pos_weight_runs` | Leaving `scale_pos_weight=None` triggers the auto-compute-from-labels path inside `fit()` without erroring — this is the exact behavior this session relied on throughout the stratified-CV tuning work. |

## test_cv.py — chromosome LOCO-CV fold construction (`lncfit/cv.py`)

Covers `build_lncrna_folds`, the chromosome-grouped CV fold builder used by
`scripts/tune_lncrna_xgboost.py`. (Its stratified, chromosome-agnostic sibling,
`build_lncrna_stratified_folds`, added in the k=3-6 class-weight sweep, has no
dedicated tests yet — see "Gaps" below.)

| Test | Verifies |
|---|---|
| `test_excludes_chroms_below_min_fold_records` | A chromosome with fewer than `MIN_FOLD_RECORDS` rows never becomes a validation fold (this is exactly why chrY/chr21/chrX were excluded from the real tuning runs this session). |
| `test_val_and_es_and_train_partition_without_overlap` | Train/validation/early-stop splits are a strict partition of all records — no row appears in two splits, no row is dropped. The core leakage-prevention invariant. |
| `test_feature_columns_consistent_across_folds` | *(weak — see note below)* Only checks that some feature columns exist, not that they're actually consistent across folds despite the test's name. |

## test_splits.py — simple train/test split helpers (`lncfit/splits.py`)

Covers `split_by_chrom` and `split_by_cell_line`, the basic held-out-group
splitters (distinct from the CV fold builders above).

| Test | Verifies |
|---|---|
| `test_test_set_contains_only_target_chrom` / `..._cell_line` | The test split contains only the requested chromosome/cell-line. |
| `test_train_set_contains_no_target_chrom` / `..._cell_line` | The train split contains none of it (no leakage back into train). |
| `test_partition_is_complete` (x2) | train + test together reconstruct the original record set exactly. |
| `test_unknown_chrom_yields_empty_test` / `..._cell_line` | An unrecognized split key yields an empty test set and the full set as train, rather than erroring. |

## test_metrics.py — regression + classification metrics (`lncfit/metrics.py`)

| Test | Verifies |
|---|---|
| `test_perfect_predictions` | Perfect regression predictions give `pearson_r=spearman_rho=1, rmse=mae=0, r2=1`. |
| `test_returns_correct_keys` | `compute_metrics` returns exactly the documented key set. |
| `test_label_stored_in_split` | The `split` label passed in comes back unchanged in the result. |
| `test_n_matches_input_length` | Row count is reported correctly. |
| `test_rmse_known_value` / `test_mae_known_value` | RMSE/MAE match a hand-computed value for a simple known input. |
| `test_constant_target_r2_is_nan` | A constant true-value array gives `r2=NaN` rather than a divide-by-zero crash or a misleading `0`. |
| `test_classification_perfect_predictions` | Perfectly-separated classes give AUROC=AUPRC=F1=accuracy=1. |
| `test_classification_returns_correct_keys` | Documented key set for classification metrics. |
| `test_classification_n_pos_and_pos_rate` | Positive count and rate are computed correctly. |
| `test_classification_single_class_auc_is_nan` | All-negative (or all-positive) `y_true` gives AUROC/AUPRC=NaN instead of crashing — this exact edge case shows up for real in the per-cell-line breakdowns throughout this project's results. |
| `test_classification_threshold_applied` | Changing the decision threshold actually changes recall. |

## test_xgboost_model.py — `evaluate_lncrna_by_group`

| Test | Verifies |
|---|---|
| `test_evaluate_lncrna_by_group_overall_plus_per_cell_line` | Returns one row for "Overall" plus one per requested cell line present in the data. |
| `test_evaluate_lncrna_by_group_skips_absent_cell_lines` | A requested cell line with zero matching records is silently skipped, not reported as a spurious all-NaN row. |

## test_tune_lncrna_xgboost.py — helpers from `scripts/tune_lncrna_xgboost.py`

| Test | Verifies |
|---|---|
| `test_natural_ratio_matches_neg_over_pos` | `_natural_ratio` computes `n_neg/n_pos` correctly. |
| `test_natural_ratio_no_positives_falls_back_to_one` | Zero positives -> ratio of `1.0`, not a `ZeroDivisionError`. |
| `test_classifier_kwargs_includes_scale_pos_weight_and_binary_objective` | The XGBoost kwargs dict built for tuning trials has the right fixed objective/eval-metric and correctly threads through the trial's `scale_pos_weight`. |

## test_tune_xgboost.py — helpers from `scripts/tune_xgboost.py` — ⚠️ cannot currently collect

`scripts/tune_xgboost.py` imports `polars`, which is not installed in this
environment and not listed in `pyproject.toml`. Pytest fails to *collect* this
file (`ModuleNotFoundError`), so these tests silently don't run at all here —
worth fixing (either install polars or gate the import) since right now the
file just doesn't execute rather than reporting a failure.

| Test | Verifies |
|---|---|
| `TestFilterRecords` (5 tests) | `filter_records` correctly filters by cell line, day, both together, no filter, or an unknown value yielding empty. |
| `TestObjTagFor` (6 tests) | `obj_tag_for` builds the right pooled/per-cell-line/per-day/combined objective tag string for MSE and Huber losses. |

## test_chatnt.py — ChatNT log2FC regression pipeline

Covers the *original* (non-classifier) ChatNT path: `build_essentiality_prompt`,
`parse_log2fc`, `run_chatnt_inference`.

| Test | Verifies |
|---|---|
| `test_single_placeholder` / `test_multiple_placeholders` | The prompt template contains exactly N `<DNA>` placeholders for N sequences. |
| `test_cell_line_as_plain_text` | The cell-line name is inserted as plain text, not accidentally wrapped in template markup. |
| `test_parse_log2fc_positive` / `_negative` / `_integer` / `_scientific_notation` / `_no_value` | The log2FC-extracting regex correctly handles 4 distinct numeric formats plus the no-match case — each a realistic way a model's free-text output could vary. |
| `test_dry_run` | The CLI's `--dry-run` flag prints the built prompt without invoking the model. |
| `test_run_chatnt_inference_returns_float` / `_no_numeric` | The inference wrapper returns a parsed float on success, `None` when the model's output has no extractable number. |

## test_chatnt_classifier.py — ChatNT zero-shot essentiality classifier

Covers the newer probability-based classifier path: prompt builders,
`essentiality_probability_from_logits`, `resolve_yes_no_token_ids`, and the
fully-mocked end-to-end `run_chatnt_zeroshot_classifier`.

| Test | Verifies |
|---|---|
| `test_classification_prompt_is_yes_no_and_has_placeholders` / `test_rationale_prompt_asks_for_explanation` | The two distinct prompt builders (yes/no question vs. free-text rationale) are shaped correctly. |
| `test_probability_symmetric_logits_is_half` | Equal yes/no logits give exactly `p=0.5`. |
| `test_probability_yes_dominates` / `test_probability_no_dominates` | Large logit gaps saturate toward `1`/`0` — specifically guards against overflow/underflow bugs at the numerical extremes, a distinct failure mode from the symmetric midpoint case. |
| `test_probability_in_unit_interval_random` | 50 random logit draws all produce a valid `[0,1]` probability. |
| `test_probability_renormalizes_over_yes_no_only` | A huge unrelated dominant token doesn't skew the yes-vs-no ratio — the core correctness property of the whole function. |
| `test_probability_nan_when_no_yes_no_mass` | All-mass-elsewhere degenerate input gives `NaN` rather than a silent wrong number. |
| `test_resolve_yes_no_token_ids_takes_first_token_per_variant` | Token resolution correctly covers `Yes/yes/YES` and space-prefixed variants. |
| `test_classifier_returns_probability_and_reason` | End-to-end: yes-dominant logits -> high probability, `"essential"` label, rationale text present. |
| `test_classifier_label_flips_below_threshold` | No-dominant logits -> low probability, `"non-essential"` label. |
| `test_classifier_no_reason_skips_generation` | `with_reason=False` genuinely skips the second (rationale) generation pass (`pipe.assert_not_called()`), not just omits it from the result. |
| `test_classifier_result_is_json_serialisable` | The result dict survives `json.dumps` — required for the CLI's `--output` flag to work at all. |

---

## Gaps (not fixed by this PR — noted for follow-up)

- **`lncfit/cv.py`'s `build_lncrna_stratified_folds`** (row-level stratified CV,
  added for the class-weight/k=3-6 sweep) and **`VarianceThreshold` integration**
  have no dedicated tests — `test_cv.py` only covers the chromosome-LOCO builder.
- **`scripts/tune_lncrna_stratified.py`** and **`scripts/summarize_stratified_tuning.py`**
  have no tests at all (consistent with the project's existing convention of not
  testing `scripts/`, but flagging since they're recent additions).
- **`test_feature_columns_consistent_across_folds`** (test_cv.py) is weak, not
  redundant — its assertion (`len(feature_cols) > 0`) doesn't actually verify
  the cross-fold consistency its name promises.
- **`test_tune_xgboost.py`** can't collect in this environment (missing `polars`).
