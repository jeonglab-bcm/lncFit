# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All scripts must be run from the project root (`/home/kellyl/lncFit`). The `--output-dir` argument defaults to `.`, so outputs land relative to wherever you run from.

```bash
# Run tests
uv run pytest
uv run pytest tests/test_features.py::TestBuildFeatures::test_x_shape_k6   # single test

# Hyperparameter tuning (the main experiment script)
uv run python scripts/tune_xgboost.py --k 6 --objective reg:squarederror
uv run python scripts/tune_xgboost.py --k 6 --objective reg:pseudohubererror

# Per-cell-line models (issue #49): one model per cell line, namespaced outputs.
# Reuses the same chromosome LOCO-CV splits. Run sequentially (see memory notes).
for cl in HAP1 HEK293FT K562 MDA-MB-231 THP1; do
  uv run python scripts/tune_xgboost.py --k 6 --cell-line $cl --objective reg:pseudohubererror
done
# Optional per-cell-line x day models (Day 14 carries more signal than Day 7):
uv run python scripts/tune_xgboost.py --k 6 --cell-line K562 --day 14 --objective reg:pseudohubererror

# Train a single model with fixed hyperparameters
uv run python scripts/train_xgboost.py --k 6 --objective reg:pseudohubererror

# Evaluate a trained model on the held-out test set
uv run python scripts/evaluate_model.py --model data/model/xgboost_k6_tuned_mse.ubj --k 6

# Memory profiling for build_features()
uv run python scripts/profile_memory.py
```

## Architecture

**Data flow:**
1. Raw screen data → `scripts/build_processed.py` → `data/processed/*.jsonl.gz` (ScreenRecord objects serialized as JSONL)
2. `scripts/build_splits.py` → `data/processed/train_chrom1.jsonl.gz` + `test_chrom1.jsonl.gz` (chromosome 1 held out as test)
3. `scripts/tune_xgboost.py` → Optuna LOCO-CV over training chromosomes → `data/model/xgboost_best_params_k<K>_<obj>.json` + final model `.ubj` + `results/final_eval_<timestamp>/`

**Core library (`lncfit/`):**
- `screen_data.py` — `ScreenRecord` dataclass and `load_jsonl()`. Every script starts here.
- `features.py` — `build_features()` converts a list of ScreenRecords into a feature matrix: 4^k k-mer frequencies + day one-hot (7/14) + cell-line one-hot (5 lines). Returns `(X, y, columns)` as bare numpy arrays. Accepts a `dtype` parameter — pass `np.float16` to halve memory (k=6 steady-state: ~8 GB vs ~16 GB for float32). This is the performance-critical path.
- `metrics.py` — `compute_metrics()` returns Spearman rho, Pearson r, R² for a split label.

**Body sequence files (`data/processed/`):**
Three variants exist — they are NOT interchangeable:
- `body_sequences.json` — first and last 1000 bp windows of the genomic locus (used in early runs; incomplete representation)
- `body_sequences_genomic_full.json` — full genomic span including introns
- `body_sequences_transcript.json` — spliced exonic sequence only (preferred for sequence-level signal)

Pass via `--body-sequences <path>`. The signed_overlap encoding (`--signed-overlap`) is a failed experiment — it dropped K562 Spearman from ~0.31 to ~0.29 and should not be used.

**`tune_xgboost.py` internals:**
- Builds the full feature matrix once, then slices numpy arrays per fold — `build_features` is NOT called per trial.
- `--cell-line` / `--day` filter records in-memory after loading (issue #49). No new split files are generated; the existing chromosome splits are reused. Outputs are namespaced: `xgboost_k6_tuned_mse_K562.ubj`, `xgboost_best_params_k6_mse_K562.json`, etc. When filtered, the cell-line/day one-hot columns become constant — harmless, XGBoost ignores them.
- LOCO-CV: for each chromosome in training data, that chromosome is the val set; the next chromosome (rotating) is the early-stop set; everything else trains.
- After Optuna finishes, retrains a final model on all training data and evaluates on `data/processed/test_chrom1.jsonl.gz`.
- CV scores written incrementally to `results/cv/cv_scores.csv` after each trial.
- Global matrix stored as float16 (`dtype=np.float16`); fold slices are cast to float32 before XGBoost. `gc.collect()` is called after feature build and after each fold to release slice memory promptly.

## Known memory constraints

- k=6 on the full training set (~1M records): steady-state float16 matrix ~8 GB, peak during `build_features` ~16 GB. Running two k=6 experiments simultaneously has triggered the OOM killer before — run them sequentially.
- Use `tmux` to run experiments in the background so SSH disconnection doesn't kill jobs.
