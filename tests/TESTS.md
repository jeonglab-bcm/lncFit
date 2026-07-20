# Test suite inventory

What's in `tests/`, file by file, and why each test exists.

This went through four review rounds (132 -> 128 -> 89 -> 86 -> 62 tests):

1. **Conservative pass**: cut 4 tests strictly implied by another test in the
   same file (e.g. "sorted vocabulary starts with its smallest element" is a
   mathematical certainty once you know the vocabulary is sorted and full-length).
2. **Sharper bar**: cut tests that only guard against loud, obvious failures
   (a registry rejecting an unknown model name, params round-tripping through
   a constructor) — would be caught immediately on first real use, not
   silently. Consolidated near-duplicate variations via `@pytest.mark.parametrize`.
   Removed `test_chatnt.py`/`test_chatnt_classifier.py` entirely on request
   (23 tests — a scope call, not a redundancy finding).
3. **Real call-graph dependencies**: traced actual function call relationships
   (not just similar-looking tests) for cases where a "downstream" test's
   assertions already exercise an "upstream" one's code path. Found one real
   hit — `kmer_freq_vector` was dead code, called by nothing except its own
   test class — and cut it, folding its coverage into `build_features`'s tests
   instead (which exercise the same shared `_count_kmers` logic for real).
4. **Aggressive consolidation** (this round, on continued feedback that the
   suite was still too large): merged tests that check different properties of
   the *same* function call into one function body, cut one case out of
   parametrized pairs that varied by magnitude rather than by branch (e.g.
   k=3/k=6 shape checks — same formula, not a different code path), and
   deliberately dropped some genuine edge-case coverage rather than just
   deduplicating. This round trades real coverage for size, unlike rounds 1-3
   — see "What this round actually gave up" below.

**Scale for context**: `lncfit/` (the reusable library) is ~2,015 lines;
`scripts/` (one-off tuning/training CLIs) is ~3,624 lines; these tests are now
~650 lines / 62 tests.

## What this round actually gave up

Unlike the first three rounds (pure deduplication, zero coverage lost), this
round cut real distinct checks to hit a smaller number. Worth knowing what's
no longer directly tested:

- **Shape formula only checked at k=3**, not also k=6 — a k-specific bug
  (unlikely, since the formula is `4**k + constant`) wouldn't be caught until
  a real k=6 run.
- **Sparse/dense equivalence** dropped the restricted-vocab variant in
  `test_features.py` (kept full-vocab only) and dropped the sparse/dense
  re-check inside `TestSignedOverlap`'s two tests (general equivalence is
  still covered once, elsewhere in the same file).
- **`test_classification_threshold_applied`** was folded into the
  perfect-predictions test rather than kept as an independent scenario —
  still checked, just no longer isolated.
- Several **data-pipeline tests were merged** (e.g. annotation-join +
  missing-annotation-defaults, jsonl round-trip + schema-version stamp,
  `from_dict` unknown-keys + missing-fields) into single test functions with
  multiple assertions. Coverage is the same, but a failure now points at a
  merged test name instead of a narrowly-named one — slightly worse failure
  localization in exchange for fewer functions.

None of this touches the highest-stakes file (`test_lncrna_rra.py`'s hit-
labeling branch tests are still 3 separate tests — collapsing those would mean
not testing 2 of the 3 branches of the compound condition at all).

---

## test_screen_data.py — guide-level screen data loading pipeline (10 tests)

`load_targets`, `load_annotations`, `load_screen`, `to_dataframe`, and
`ScreenRecord`'s jsonl round-trip / schema versioning / `from_dict` defaulting.

| Test | Verifies |
|---|---|
| `test_load_targets` | Guide ID -> (target, sequence) mapping. |
| `test_load_annotations` | Target -> (chrom, strand, closest PC gene, distance) mapping, including that a blank distance cell becomes `None` not `0`. |
| `test_melt_produces_correct_row_counts` | Total row count (all 5 sheets found) and per-guide-per-cell-line count (melt is correct), in one test. |
| `test_day_and_replicate_parsed` | Day/replicate correctly regex-parsed from FC column headers. |
| `test_negative_fold_changes_preserved` | Negative fold-change survives the melt — sign-dependent hit-calling downstream would silently break otherwise. |
| `test_s1_s2_join_and_annotation_enrichment` | Guide-to-target join and annotation enrichment both land on final records, in one `load_screen` call. |
| `test_missing_annotation_falls_back_to_defaults` | No matching annotation -> safe defaults, not `KeyError`. |
| `test_to_dataframe_schema` | Expected column set and row count. |
| `test_jsonl_round_trip_and_schema_version` | Round-trip equality and schema version stamp, in one test. |
| `test_from_dict_unknown_and_missing_fields` | Forward- and backward-compatible `from_dict`, in one test. |

## test_lncrna_rra.py — lncRNA RRA-hit data loading pipeline (10 tests)

`load_target_groups`, `load_rra` (issue #60's Day-14 hit-calling loader), and
`LncRnaRecord`'s jsonl/schema/`from_dict` behavior. The highest-stakes file —
a wrong label here is a wrong scientific conclusion, not a crash — so the 3
hit-labeling branch tests were left untouched even in the aggressive round.

| Test | Verifies |
|---|---|
| `test_load_target_groups` | Target -> group string mapping. |
| `test_load_rra_filters_to_lncrna_only` | Protein-coding rows dropped entirely, not relabeled. |
| `test_load_rra_label_significant_and_negative_is_hit` | `p<0.05 AND log2FC<0` -> hit. |
| `test_load_rra_label_not_significant_is_non_hit` | `p>=0.05` -> non-hit regardless of fold-change. |
| `test_load_rra_significant_but_positive_fc_is_non_hit` | Significant *enrichment* isn't a depletion hit. All 3 independently tested since flipping either half of the AND is a classic bug. |
| `test_load_rra_day_selection_reads_correct_columns` | Day 7 vs Day 14 read distinct columns. |
| `test_load_rra_annotation_join_and_missing_defaults` | Annotation join and missing-annotation defaults, in one test. |
| `test_load_rra_skips_missing_pvalue_or_fc` | Blank p-value/FC -> skipped, not coerced into a spurious hit/non-hit. |
| `test_lncrna_jsonl_round_trip_and_schema_version` | Round-trip equality and schema version stamp. |
| `test_lncrna_from_dict_unknown_and_missing_fields` | Forward- and backward-compatible `from_dict`. |

## test_features.py — guide-level feature engineering (14 tests)

`all_kmers`, `fit_vocab`, `build_features`.

| Test | Verifies |
|---|---|
| `test_all_kmers_sorted_order` | Vocabulary is lexicographically sorted (length checked via the shape test below instead of standalone). |
| `test_shape_label_and_exact_kmer_value` | Shape formula, raw fold-change passthrough, and an exact k-mer frequency value, from one `build_features` call — merged from 3 separate tests. |
| `test_non_acgt_windows_excluded[partial/all]` | Non-ACGT windows excluded, whether partially or entirely present — real branch coverage, kept as 2 cases. |
| `test_day_and_cell_line_onehot_columns_present` | Both categorical column sets exist. |
| `test_include_distance[3 cases]` | Present+correct-value, missing->sentinel, disabled->absent — real branch coverage, kept as 3 cases. |
| `test_sparse_matches_dense` | Sparse and dense builds are numerically identical (full vocab only now — see "what this round gave up") — this equivalence is load-bearing: XGBoost treats sparse implicit zeros as *missing*. |
| `test_custom_vocab_reduces_columns_and_preserves_order` | Restricted vocab shrinks the matrix and keeps column order. |
| `test_holdout_unseen_kmer_silently_dropped` | A k-mer seen only in holdout data is silently zero — the exact scenario every chr1-holdout evaluation in this project relies on. |
| `TestSignedOverlap` (2 tests) | Reverse-complement overlap correctly negates shared k-mers when present, and correctly doesn't when absent — independent bugs, both branches kept. |
| `test_fit_vocab_observed_sorted_and_edge_cases` | Only-observed, sorted, non-ACGT-skipped, and empty-input-safe, all in one test (merged from 3). |

## test_lncrna_features.py — lncRNA-level feature engineering (6 tests)

`build_lncrna_features` — features come from the lncRNA's own transcript
sequence (issue #65), shared across that lncRNA's cell-line rows.

| Test | Verifies |
|---|---|
| `test_shape_and_binary_label` | Column count formula and binary (not continuous) label, merged into one test. |
| `test_kmer_freq_computed_from_own_transcript_sequence` | Frequencies computed and normalized from the lncRNA's own sequence. |
| `test_same_target_shares_feature_vector_across_cell_lines` | **The** property issue #65 fixed. |
| `test_no_day_column_and_distance_sentinel` | No `day_*` column (Day-14-only task) and the `-1` distance sentinel, merged into one test. |
| `test_target_missing_from_transcript_sequences_gets_zero_kmer_vector` | Incomplete sequence data -> zero vector, not a crash. |
| `test_sparse_matches_dense` | Same sparse/dense equivalence, checked independently since it's a separate function. |

## test_classifiers.py — pluggable classifier registry (5 tests)

| Test | Verifies |
|---|---|
| `test_wrapper_fit_returns_self_and_proba_shape_range[3 models]` | Shared contract (fit returns self, proba in `[0,1]`); the `xgboost` case also covers the auto-`scale_pos_weight` path (it's left at its default here), and the `null` case additionally checks it predicts exactly the training base rate. Three previously-separate tests folded into this one parametrized test. |
| `test_xgboost_accepts_sparse_and_dense` | The one proven bug class in this codebase — logreg isn't tested here too since it has no such special-casing. |
| `test_model_type_matches_registry_key` | Kept deliberately: a mismatch would silently mislabel `"model"` in every `run_info.json` this project's tuning scripts write. |

## test_cv.py — chromosome LOCO-CV fold construction (2 tests)

Already minimal — see the third-pass note in earlier history about a real
finding here (`build_lncrna_folds`'s `feature_cols` return value is dead code,
documented rather than tested).

| Test | Verifies |
|---|---|
| `test_excludes_chroms_below_min_fold_records` | A chromosome under `MIN_FOLD_RECORDS` never becomes a validation fold. |
| `test_val_and_es_and_train_partition_without_overlap` | Train/val/early-stop splits strictly partition all records. |

## test_splits.py — simple train/test split helpers (4 tests)

`split_by_chrom` and `split_by_cell_line`, parametrized across both rather
than duplicated. Merged from 4 properties to 2 per function this round.

| Test | Verifies (for both functions) |
|---|---|
| `test_split_is_correct_and_complete` | Test split contains only the target value, train excludes it, and train+test reconstruct the original set — merged into one test. |
| `test_unknown_value_yields_empty_test` | An unrecognized key yields empty test + full train. |

## test_metrics.py — regression + classification metrics (6 tests)

| Test | Verifies |
|---|---|
| `test_perfect_predictions_and_correct_keys` | Correct key set + all metrics correct for a trivial perfect-prediction case. |
| `test_known_value_formulas[rmse/mae]` | Two custom (non-sklearn-wrapper) formulas against hand-computed values. |
| `test_constant_target_r2_is_nan` | Constant target -> NaN, not a crash. |
| `test_classification_perfect_predictions_correct_keys_and_threshold` | Correct keys/values for perfect predictions, plus that raising the threshold actually changes recall — merged from 2 tests. |
| `test_classification_single_class_auc_is_nan` | All-one-class `y_true` -> NaN — the exact scenario that shows up for real in this project's per-cell-line breakdowns. |

## test_xgboost_model.py — `evaluate_lncrna_by_group` (2 tests)

Untouched across all 4 rounds — already minimal, both tests distinct.

## test_tune_lncrna_xgboost.py — helpers from `scripts/tune_lncrna_xgboost.py` (3 tests)

| Test | Verifies |
|---|---|
| `test_natural_ratio[2 cases]` | `_natural_ratio` = `n_neg/n_pos`, or `1.0` fallback with zero positives — merged into one parametrized test. |
| `test_classifier_kwargs_includes_scale_pos_weight_and_binary_objective` | Tuning-trial kwargs correctly thread through `scale_pos_weight` and fixed objective/eval-metric. |

## test_tune_xgboost.py — helpers from `scripts/tune_xgboost.py` — ⚠️ cannot currently collect

`scripts/tune_xgboost.py` imports `polars`, not installed in this environment
and not in `pyproject.toml`. Untouched across all rounds — a dependency-
environment issue, not a redundancy one.
