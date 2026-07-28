# Test suite inventory

What's in `tests/`, file by file, and why each test exists.

This went through five review rounds (132 -> 128 -> 89 -> 86 -> 62 -> 20 tests):

1. **Conservative pass**: cut 4 tests strictly implied by another test in the
   same file.
2. **Sharper bar**: cut tests that only guard against loud, obvious failures.
   Consolidated near-duplicate variations via `@pytest.mark.parametrize`.
   Removed `test_chatnt.py`/`test_chatnt_classifier.py` entirely on request
   (23 tests — a scope call, not a redundancy finding).
3. **Real call-graph dependencies**: traced actual function call relationships.
   Found `kmer_freq_vector` was dead code (no caller anywhere except its own
   tests) — deleted it from `lncfit/features.py`, folded its coverage into
   `build_features`'s tests instead.
4. **Aggressive consolidation**: merged tests checking different properties of
   the same call into one function, dropped magnitude-only parametrize
   variants. First round to trade away *some* real coverage for size.
5. **Smoke-test level** (this round, target: ~20): one test per module,
   almost always a single function per file with several sequential
   assertions instead of many small functions. `@pytest.mark.parametrize` is
   gone everywhere except where it was already collapsed away — every
   remaining "case" lives as a plain assertion inside one function body,
   because a parametrized case still counts as a separate collected test.

**Scale for context**: `lncfit/` (the reusable library) is ~2,015 lines;
`scripts/` (one-off tuning/training CLIs) is ~3,624 lines; these tests are now
~350 lines / 20 tests.

## What this level actually means

At 20 tests, this is no longer "does every branch and edge case work" — it's
"does the core contract of each module hold." Concretely, what's *inside* each
test is mostly unchanged from round 4 (same assertions, same scenarios), but
they're now packed into one function per file/concern rather than many. The
main real loss versus round 4: failures now point at one merged test name per
file instead of a narrowly-named one, so pinpointing *which* assertion inside
a large test failed takes reading the traceback line number instead of the
test name. If a test in, say, `test_lncrna_rra.py` starts failing, you'll know
"something in the RRA loading pipeline broke," not immediately which of the
~8 assertions packed into that function.

The one thing that did **not** get compressed away: the 3 branches of the RRA
hit-labeling compound condition (`p<0.05 AND log2FC<0`) are still all checked,
just inside one function (`test_load_target_groups_filter_and_hit_labeling`)
instead of 3. That's the highest-stakes logic in this codebase — a wrong label
is a wrong scientific conclusion — so it was consolidated in structure, not
dropped in substance.

---

## File-by-file (1 test each unless noted)

> **This table covers the round-5 core only.** The suite is now **95 tests**: the
> consolidation above was a one-off exercise, and everything added since has been
> written to a normal bar rather than the ~20-test target. Not in the table below:
> `test_pipeline.py`, `test_build_leaderboard.py`, `test_score_submission.py`,
> `test_make_barebones_submission.py`, `test_split_holdout_cellline.py`,
> `test_resample.py`, `test_embeddings_pca.py`, `test_ensemble_predictions.py`,
> `test_build_day14_loco_ensemble.py`, `test_day14_cellline_loco_guide.py`,
> `test_day14_compliant_multimodal.py`. Each of those files documents its own
> rationale in a module docstring; the leaderboard and hold-out-split tests in
> particular guard scoring integrity and are worth reading before touching either.

| File | Test | Covers |
|---|---|---|
| `test_cv.py` | `test_excludes_chroms_below_min_fold_records` | A chromosome under `MIN_FOLD_RECORDS` never becomes a validation fold. |
| `test_cv.py` | `test_val_and_es_and_train_partition_without_overlap` | Train/val/early-stop splits strictly partition all records (2 tests kept — already minimal from round 3, both are irreducible core invariants). |
| `test_features.py` | `test_build_features_core_contract` | k-mer vocab sort order, shape formula, fold-change passthrough, exact k-mer frequency value, non-ACGT window exclusion, day/cell-line one-hot columns, and `include_distance` behavior — all from a handful of `build_features` calls in one function. |
| `test_features.py` | `test_sparse_dense_and_vocab_handling` | Sparse/dense equivalence (the one proven bug class — XGBoost treats sparse implicit zeros as missing), restricted-vocab column reduction, and holdout-unseen-k-mer handling. |
| `test_features.py` | `test_signed_overlap_negates_shared_kmers_only_when_present` | Reverse-complement overlap sign-flipping, both branches (negate when present, don't when absent). |
| `test_features.py` | `test_fit_vocab_observed_sorted_and_edge_cases` | Only-observed, sorted, non-ACGT-skipped, empty-input-safe. |
| `test_lncrna_features.py` | `test_build_lncrna_features_core_contract` | Shape, binary label, own-transcript-sequence frequency computation, the same-target-shares-vector property (issue #65's fix), distance sentinel, missing-sequence zero vector. |
| `test_lncrna_features.py` | `test_sparse_matches_dense` | Same equivalence guarantee, checked independently since it's a separate function from `build_features`. |
| `test_lncrna_rra.py` | `test_load_target_groups_filter_and_hit_labeling` | Target-group loading, lncRNA-only filtering, and all 3 branches of the compound hit-labeling condition. |
| `test_lncrna_rra.py` | `test_load_rra_day_selection_annotations_and_skip_missing` | Day 7 vs 14 column selection, annotation join + missing-annotation defaults, skipping rows with blank p-value/FC. |
| `test_lncrna_rra.py` | `test_lncrna_jsonl_round_trip_schema_version_and_from_dict` | jsonl round-trip, schema version stamp, `from_dict` forward/backward compatibility. |
| `test_screen_data.py` | `test_load_targets_and_annotations` | Guide-to-target mapping; annotation mapping including the blank-distance-is-`None` edge case. |
| `test_screen_data.py` | `test_load_screen_pipeline` | Melt row counts, day/replicate parsing, negative-fold-change preservation, S1/S2 join, annotation enrichment, missing-annotation defaults. |
| `test_screen_data.py` | `test_to_dataframe_jsonl_and_from_dict` | DataFrame schema, jsonl round-trip + schema stamp, `from_dict` compatibility. |
| `test_metrics.py` | `test_compute_metrics` | Regression metrics: perfect-prediction values, correct key set, custom rmse/mae formulas, constant-target NaN handling. |
| `test_metrics.py` | `test_compute_classification_metrics` | Classification metrics: perfect predictions, threshold sensitivity, all-one-class NaN handling — the exact scenario this project's per-cell-line breakdowns hit for real. |
| `test_classifiers.py` | `test_classifier_wrappers_share_the_fit_predict_contract` | The shared fit/predict_proba contract across all 3 registered models, null's base-rate behavior, xgboost's sparse/dense equivalence, and registry `model_type` consistency (silent mislabeling would corrupt every `run_info.json` this project's result tables are built from). |
| `test_splits.py` | `test_split_by_chrom_and_cell_line` | Both split functions' invariants (test-only-target, train-excludes-target, complete partition, unknown-value-yields-empty), looped inline over both functions. |
| `test_tune_lncrna_xgboost.py` | `test_natural_ratio_and_classifier_kwargs` | `_natural_ratio`'s two branches and `_classifier_kwargs`'s fixed objective/eval-metric threading. |
| `test_xgboost_model.py` | `test_evaluate_lncrna_by_group` | Overall + per-present-cell-line rows, absent cell line silently skipped. |

## test_tune_xgboost.py — ⚠️ cannot currently collect

`scripts/tune_xgboost.py` imports `polars`, not installed in this environment
and not in `pyproject.toml`. Untouched across all 5 rounds — a dependency-
environment issue, not a redundancy one.
