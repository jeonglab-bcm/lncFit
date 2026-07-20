# Test suite inventory

What's in `tests/`, file by file, and why each test exists.

This went through two review passes. The first pass (132 -> 128 tests) only
cut tests that were *logically implied* by another test in the same file. On
feedback that this was too conservative for a research codebase — most of
those 128 could still be justified with a "covers a technically different
branch" argument, which is true but not the same as "guards a realistic silent
failure" — the second pass (128 -> 112 tests) applied a sharper bar:

- **Cut** tests that only guard against a failure that would be loud and
  obvious the first time anyone ran anything for real (a registry rejecting an
  unknown model name, a sorted list starting with its smallest element, a
  string round-tripping through a constructor).
- **Consolidated** near-duplicate variations (e.g. 5 separate log2FC parsing
  formats, or the same 4 split invariants tested once per split function) into
  single parametrized tests — same coverage, less function-count bloat.
- **Kept**, and in some cases now comment *why*, tests that guard against
  genuinely silent failure modes: wrong scientific labels, sparse-vs-dense
  numerical divergence (a proven past bug in this codebase), NaN/degenerate-
  input handling that shows up for real in this project's per-cell-line
  results, and mislabeling that would corrupt a `run_info.json` nobody would
  think to double-check.
- **Found one real gap while reviewing**, not previously known: see
  "A finding worth knowing about" below.

**Scale for context**: `lncfit/` (the reusable library) is ~2,020 lines;
`scripts/` (one-off tuning/training CLIs) is ~3,624 lines; these tests are now
~900 lines / 89 tests. Coverage is concentrated on `lncfit/` — the scripts get
two small helper-function tests each and are otherwise untested by design,
since a bug there just makes one analysis run wrong, not something that
silently corrupts every downstream use the way a library bug would.

`test_chatnt.py` and `test_chatnt_classifier.py` (the two ChatNT-pipeline test
files, 23 tests between them) were removed outright on request rather than
trimmed — not a redundancy finding, a scope call.

## A finding worth knowing about

While deciding whether `test_cv.py`'s `test_feature_columns_consistent_across_folds`
was worth keeping (it only asserted `len(feature_cols) > 0`, not real
consistency), I checked empirically whether `build_lncrna_folds`'s returned
`feature_cols` is actually consistent across folds. **It isn't, when
chromosomes have genuinely different vocabularies** — `feature_cols` is only
ever set from whichever fold is processed first; other folds' real feature
matrices can have both a different width *and* different column identities.
This is a real latent gap, not a false alarm. It currently has zero practical
impact because **no script actually reads the returned `feature_cols`** —
`tune_lncrna_xgboost.py`, `tune_lncrna_stratified.py`, and `tune_xgboost.py`
all capture it into a variable and never use it (confirmed by grep). So the
test was asserting something about a dead return value. I cut the test rather
than strengthen it — testing an unused value isn't "high quality" coverage,
it's coverage of nothing at risk. If a future script starts actually using
per-fold `feature_cols` (e.g. for feature importances), a real test should be
added at that point, alongside a fix to `build_lncrna_folds`/
`build_lncrna_stratified_folds` to return it per-fold instead of once globally.

---

## test_screen_data.py — guide-level screen data loading pipeline (15 tests)

`load_targets`, `load_annotations`, `load_screen` (the S1/S2 sheet melt + join
that turns `mmc2.xlsx`/`mmc3.xlsx` into per-guide fold-change records),
`to_dataframe`, and `ScreenRecord`'s jsonl round-trip / schema versioning /
`from_dict` defaulting. Kept almost entirely intact — spreadsheet parsing and
joins are real, easy-to-silently-break logic, not incidental plumbing.

| Test | Verifies |
|---|---|
| `test_load_targets` | Guide ID -> (target, sequence) mapping from S1B. |
| `test_load_annotations` | Target -> (chrom, strand, closest PC gene, distance) from S1A. |
| `test_load_annotations_blank_distance_is_none` | A blank distance cell becomes `None`, not `0` — would silently corrupt the `include_distance` feature otherwise. |
| `test_melt_produces_4_rows_per_guide_per_cell_line` | The wide fold-change table melts into exactly 4 long-format rows per guide per cell line. |
| `test_total_row_count` | All 5 cell-line sheets are found and parsed (distinct from the row-per-guide check, which only inspects one sheet). |
| `test_day_and_replicate_parsed` | Day/replicate correctly regex-parsed out of FC column headers. |
| `test_negative_fold_changes_preserved` | Negative fold-change survives the melt — a silent `abs()` bug here would corrupt every "hit" call downstream, since hit-calling is sign-dependent. |
| `test_s1_s2_join` | Guide-to-target join is correct. |
| `test_annotations_enriched_in_records` | Annotations actually land on final records, not just in the intermediate dict. |
| `test_missing_annotation_falls_back_to_defaults` | No matching annotation -> safe defaults, not a `KeyError`. (Its sibling, testing the *omitted*-argument code path, was cut: every real call site in `scripts/` always passes real annotations, so that branch is currently untested by any actual usage either way — keeping one is enough.) |
| `test_to_dataframe_schema` | `to_dataframe` produces the expected column set and row count. |
| `test_save_load_jsonl_round_trip` | Round-trip equality through `.jsonl.gz`. |
| `test_jsonl_stamped_with_schema_version` | Every saved record carries the current schema version. |
| `test_from_dict_ignores_unknown_keys` | Forward-compatible `from_dict`. |
| `test_from_dict_missing_optional_fields_use_defaults` | Backward-compatible `from_dict`. |

## test_lncrna_rra.py — lncRNA RRA-hit data loading pipeline (13 tests)

`load_target_groups`, `load_rra` (issue #60's Day-14 hit-calling loader,
reading the S2F-S2J RRA sheets), and `LncRnaRecord`'s jsonl/schema/`from_dict`
behavior. Untouched by the second pass — this is the highest-stakes file in
the suite: a wrong label here is a wrong scientific conclusion, not a crash.

| Test | Verifies |
|---|---|
| `test_load_target_groups` | Target -> group string mapping. |
| `test_load_rra_filters_to_lncrna_only` | Protein-coding rows are dropped entirely, not relabeled. |
| `test_load_rra_label_significant_and_negative_is_hit` | `p<0.05 AND log2FC<0` -> hit (one branch of the compound condition). |
| `test_load_rra_label_not_significant_is_non_hit` | `p>=0.05` -> non-hit regardless of fold-change (second branch). |
| `test_load_rra_significant_but_positive_fc_is_non_hit` | Significant *enrichment* isn't a depletion hit (third branch). All three are independently tested since flipping either half of the AND is a classic bug. |
| `test_load_rra_day_selection_reads_correct_columns` | Day 7 vs Day 14 read distinct columns. |
| `test_load_rra_joins_annotations` | Annotation join lands on records. |
| `test_load_rra_missing_annotation_defaults` | Missing annotation -> safe defaults. |
| `test_load_rra_skips_missing_pvalue_or_fc` | A blank p-value/FC cell is skipped, not coerced into a spurious hit/non-hit. |
| `test_save_load_lncrna_jsonl_round_trip` | Round-trip equality. |
| `test_lncrna_jsonl_stamped_with_schema_version` | Schema version tag present. |
| `test_lncrna_from_dict_ignores_unknown_keys` | Forward-compatible `from_dict`. |
| `test_lncrna_from_dict_missing_optional_fields_use_defaults` | Backward-compatible `from_dict`. |

## test_features.py — guide-level feature engineering (23 tests)

`all_kmers`, `kmer_freq_vector`, `fit_vocab`, `build_features`. Heavily
consolidated in the second pass via `@pytest.mark.parametrize`.

| Test | Verifies |
|---|---|
| `test_length_is_4_pow_k[3/6]` | `all_kmers(k)` returns `4**k` entries at two k values (was 2 separate functions). |
| `test_sorted_order` | Vocabulary is lexicographically sorted. |
| `test_sums_to_one` | A frequency vector sums to 1. |
| `test_correct_counts` | A homopolymer sequence gives the exact expected count at the right index. |
| `test_non_acgt_windows_excluded[partial/all]` | Windows with a non-ACGT base are excluded, whether some or all windows are affected (was 2 separate functions). |
| `test_x_shape_matches_vocab_plus_day_plus_cell_columns[3/6]` | Feature matrix shape formula holds at two k values (was 2 separate functions). |
| `test_y_values_are_raw_fold_change` | Label vector is the raw fold-change. |
| `test_day_and_cell_line_onehot_columns_present` | Both categorical column sets exist (was 2 separate functions). |
| `test_include_distance[3 cases]` | Present+correct-value, missing->sentinel, and disabled->absent, in one parametrized test (was 3 separate functions). |
| `test_sparse_matches_dense[full/restricted vocab]` | Sparse and dense builds are numerically identical, at full and restricted vocab (was 2 separate functions) — this equivalence is load-bearing: XGBoost treats sparse implicit zeros as *missing*, not the real "k-mer absent" value a dense zero represents. |
| `test_custom_vocab_reduces_columns_and_preserves_order` | Restricted vocab shrinks the matrix and keeps column order. |
| `test_holdout_unseen_kmer_silently_dropped` | A k-mer seen only in holdout data is silently zero, not an error or a misaligned extra column — the exact scenario every chr1-holdout evaluation in this project relies on. |
| `TestSignedOverlap` (2 tests) | Reverse-complement overlap correctly negates shared k-mers when present, and correctly does *not* when absent — both branches of the conditional are independently tested since they're independent bugs, not mirror views of one property. |
| `TestFitVocab` (3 tests) | Returns only observed (and sorted) k-mers, skips non-ACGT, and empty input yields empty output. |

## test_lncrna_features.py — lncRNA-level feature engineering (8 tests)

`build_lncrna_features` — the classifier-task counterpart to `build_features`,
where features come from the lncRNA's own transcript sequence (issue #65) and
are shared across that lncRNA's cell-line rows.

| Test | Verifies |
|---|---|
| `test_shape_matches_vocab_plus_cell_columns` | Column count formula (no day dimension — Day-14-only task). |
| `test_y_is_binary_label_not_fold_change` | Label is the binary hit flag, not a regression target. |
| `test_kmer_freq_computed_from_own_transcript_sequence` | Frequencies computed and normalized from the lncRNA's own sequence. |
| `test_same_target_shares_feature_vector_across_cell_lines` | **The** property issue #65 fixed: two rows for the same lncRNA get identical k-mer features regardless of cell line. |
| `test_no_day_column_present` | No `day_*` column — guards against copy-paste regression from the sibling guide-level builder. |
| `test_include_distance_uses_negative_one_sentinel_when_missing` | Same sentinel convention, independently implemented in this function. |
| `test_target_missing_from_transcript_sequences_gets_zero_kmer_vector` | Incomplete sequence data -> zero vector, not a crash. |
| `test_sparse_matches_dense` | Same sparse/dense equivalence guarantee, checked independently since it's a separate function. |

## test_classifiers.py — pluggable classifier registry (8 tests)

`lncfit/classifiers/`'s `null`/`logreg`/`xgboost` wrappers and registry. Cut
the pure registry-mechanics tests (unknown-model-raises, duplicate-
registration-raises, params-stored-and-forwarded) — all three would fail
loudly and immediately on first real use, not silently.

| Test | Verifies |
|---|---|
| `test_wrapper_fit_returns_self_and_proba_shape_range[3 models]` | `fit()` returns `self`; `predict_proba` gives a `[0,1]`-bounded array of the right length — the shared contract every registered model must satisfy. |
| `test_wrapper_accepts_sparse_and_dense[2 models]` | Identical predictions from sparse or dense input, at the model layer — same proven risk area as the feature-builder checks. |
| `test_null_predicts_training_base_rate` | The null baseline predicts exactly the training positive rate (a behavioral guarantee, not mechanics). |
| `test_model_type_matches_registry_key` | Kept deliberately: a mismatch wouldn't crash anything, it would silently mislabel `"model"` in every `run_info.json` this project's tuning scripts write — exactly the field this session's result tables were built from. |
| `test_xgboost_auto_scale_pos_weight_runs` | The auto-compute-when-`None` path this whole project's tuning relies on doesn't error. |

## test_cv.py — chromosome LOCO-CV fold construction (2 tests)

`build_lncrna_folds`. Down to 2 tests after this pass — see "A finding worth
knowing about" above for why the third was cut rather than fixed.

| Test | Verifies |
|---|---|
| `test_excludes_chroms_below_min_fold_records` | A chromosome under `MIN_FOLD_RECORDS` never becomes a validation fold — exactly why chrY/chr21/chrX were excluded from this session's real tuning runs. |
| `test_val_and_es_and_train_partition_without_overlap` | Train/val/early-stop splits strictly partition all records — the core leakage-prevention invariant. |

## test_splits.py — simple train/test split helpers (8 tests)

`split_by_chrom` and `split_by_cell_line`. Restructured as one parametrized
class instead of two near-identical test classes — same 4 invariants, tested
once per function via `@pytest.mark.parametrize` rather than duplicated.

| Test | Verifies (for both `split_by_chrom` and `split_by_cell_line`) |
|---|---|
| `test_test_set_contains_only_target` | Test split contains only the requested value. |
| `test_train_set_excludes_target` | Train split contains none of it. |
| `test_partition_is_complete` | train + test reconstruct the original set exactly. |
| `test_unknown_value_yields_empty_test` | An unrecognized key yields empty test + full train, not an error. |

## test_metrics.py — regression + classification metrics (7 tests)

Cut the tests that mostly re-verified scipy/sklearn's own correctness
(`pearsonr(y,y)==1`, key-set checks) as standalone tests and merged them into
the "perfect predictions" tests instead. Kept everything that's custom logic
or a real degenerate-input path.

| Test | Verifies |
|---|---|
| `test_perfect_predictions_and_correct_keys` | Correct key set + all metrics correct for a trivial perfect-prediction case, in one test. |
| `test_rmse_known_value` / `test_mae_known_value` | Two different custom formulas (not sklearn wrappers) against hand-computed values. |
| `test_constant_target_r2_is_nan` | Constant target -> NaN, not a crash or misleading 0/1. |
| `test_classification_perfect_predictions_and_correct_keys` | Same consolidation for the classification metrics. |
| `test_classification_single_class_auc_is_nan` | All-one-class `y_true` -> NaN, not a crash — the exact scenario that shows up for real in this project's per-cell-line breakdowns (e.g. chr18/THP1's zero positives). |
| `test_classification_threshold_applied` | Changing the decision threshold actually changes recall (guards against a silently-ignored parameter). |

## test_xgboost_model.py — `evaluate_lncrna_by_group` (2 tests)

Untouched — already minimal, both tests distinct.

| Test | Verifies |
|---|---|
| `test_evaluate_lncrna_by_group_overall_plus_per_cell_line` | One row for "Overall" plus one per present cell line. |
| `test_evaluate_lncrna_by_group_skips_absent_cell_lines` | A requested but absent cell line is silently skipped, not a spurious all-NaN row. |

## test_tune_lncrna_xgboost.py — helpers from `scripts/tune_lncrna_xgboost.py` (3 tests)

Untouched — already minimal.

| Test | Verifies |
|---|---|
| `test_natural_ratio_matches_neg_over_pos` | `_natural_ratio` = `n_neg/n_pos`. |
| `test_natural_ratio_no_positives_falls_back_to_one` | Zero positives -> `1.0`, not `ZeroDivisionError`. |
| `test_classifier_kwargs_includes_scale_pos_weight_and_binary_objective` | Tuning-trial kwargs correctly thread through `scale_pos_weight` and fixed objective/eval-metric. |

## test_tune_xgboost.py — helpers from `scripts/tune_xgboost.py` — ⚠️ cannot currently collect

`scripts/tune_xgboost.py` imports `polars`, which is not installed in this
environment and not listed in `pyproject.toml`. Pytest fails to *collect* this
file, so these 11 tests silently don't run here at all. Not touched by this
pass (a dependency-environment issue, not a redundancy one) — worth fixing
separately (install polars or gate the import) since right now the file just
doesn't execute rather than reporting a failure.
