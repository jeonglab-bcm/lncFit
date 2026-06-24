#!/usr/bin/env bash
# Run per-cell-line vs pooled comparison for issue #49.
# Launches 6 runs IN PARALLEL: 1 pooled baseline + 5 per-cell-line models.
# All use identical settings: k=3, reg:squarederror, transcript body sequences,
# NO --signed-overlap (clean transcript-body baseline).
#
# Memory: ~13 GB pooled + ~2.5 GB x5 per-cell-line = ~25 GB total.
# CPU: --nthread 3 x6 = 18 threads on 20 cores (leaves headroom for other users).
set -euo pipefail

cd "$(dirname "$0")/.."

K=3
OBJ=reg:squarederror
N_TRIALS=50
NTHREAD=3
BODY_SEQ=data/processed/body_sequences_transcript.json
OUT_DIR=results/transcript_overlap
LOG_DIR=logs
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Labels for each run: "pooled" has no --cell-line flag
declare -A RUN_FLAGS
RUN_FLAGS[pooled]=""
RUN_FLAGS[HAP1]="--cell-line HAP1"
RUN_FLAGS[HEK293FT]="--cell-line HEK293FT"
RUN_FLAGS[K562]="--cell-line K562"
RUN_FLAGS[MDA-MB-231]="--cell-line MDA-MB-231"
RUN_FLAGS[THP1]="--cell-line THP1"

ORDER=(pooled HAP1 HEK293FT K562 MDA-MB-231 THP1)
PIDS=()

for label in "${ORDER[@]}"; do
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/${label}_${stamp}.log"
  echo "=== [$label] starting $(date) -> $log (nthread=$NTHREAD) ==="
  PYTHONUNBUFFERED=1 uv run python scripts/tune_xgboost.py \
    --k "$K" --objective "$OBJ" --n-trials "$N_TRIALS" \
    --nthread "$NTHREAD" \
    --body-sequences "$BODY_SEQ" \
    --output-dir "$OUT_DIR" \
    ${RUN_FLAGS[$label]} \
    >"$log" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "=== Launched ${#PIDS[@]} parallel runs: PIDs ${PIDS[*]} ==="
echo "=== Waiting for all to finish ... ==="
echo ""

FAILED=0
for i in "${!ORDER[@]}"; do
  label="${ORDER[$i]}"
  pid="${PIDS[$i]}"
  if wait "$pid"; then
    echo "=== [$label] PID $pid completed successfully $(date) ==="
  else
    echo "=== [$label] PID $pid FAILED (exit $?) $(date) ==="
    FAILED=$((FAILED+1))
  fi
done

if [ "$FAILED" -gt 0 ]; then
  echo "=== WARNING: $FAILED run(s) failed. Check logs/ for details. ==="
fi

# Build the comparison table against the clean pooled baseline (no --signed-overlap)
echo ""
echo "=== Building comparison table ==="
uv run python scripts/build_comparison_table.py \
  --results-dir "$OUT_DIR/results" \
  --k "$K" --objective "$OBJ" \
  --signed-overlap false

echo ""
echo "=== ALL DONE $(date) ==="
