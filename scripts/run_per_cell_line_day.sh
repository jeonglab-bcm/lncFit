#!/usr/bin/env bash
# Phase 2: per-cell-line x day models for issue #49.
# Waits for the 6 phase-1 runs (pooled + 5 per-cell-line) to finish, then
# launches 10 per-cell-line x day models in parallel.
#
# Memory: ~1.3 GB x10 = ~13 GB. CPU: --nthread 2 x10 = 20 threads (all cores,
# other users' jobs are low-CPU so contention is acceptable).
set -euo pipefail

cd "$(dirname "$0")/.."

K=3
OBJ=reg:squarederror
N_TRIALS=50
NTHREAD=2
BODY_SEQ=data/processed/body_sequences_transcript.json
OUT_DIR=results/transcript_overlap
LOG_DIR=logs
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Wait for phase-1 runs to finish (no tune_xgboost with no --day flag running)
echo "=== Waiting for phase-1 runs (pooled + per-cell-line) to finish ... ==="
while true; do
  n=$(ps -u kellyl -o args= 2>/dev/null | grep "tune_xgboost.py" | grep -vc "wait_then\|run_per_cell\|build_comparison" || true)
  if [ "$n" -eq 0 ]; then
    echo "=== All phase-1 runs finished. Free RAM: ==="
    free -h | head -2
    break
  fi
  echo "  ... $n phase-1 run(s) still active ($(date +%H:%M:%S))"
  sleep 60
done

CELL_LINES=(HAP1 HEK293FT K562 MDA-MB-231 THP1)
DAYS=(7 14)
PIDS=()
LABELS=()

for cl in "${CELL_LINES[@]}"; do
  for day in "${DAYS[@]}"; do
    label="${cl}_d${day}"
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="$LOG_DIR/${label}_${stamp}.log"
    echo "=== [$label] starting $(date) -> $log (nthread=$NTHREAD) ==="
    PYTHONUNBUFFERED=1 uv run python scripts/tune_xgboost.py \
      --k "$K" --objective "$OBJ" --n-trials "$N_TRIALS" \
      --nthread "$NTHREAD" \
      --body-sequences "$BODY_SEQ" \
      --output-dir "$OUT_DIR" \
      --cell-line "$cl" --day "$day" \
      >"$log" 2>&1 &
    PIDS+=($!)
    LABELS+=("$label")
  done
done

echo ""
echo "=== Launched ${#PIDS[@]} parallel runs: PIDs ${PIDS[*]} ==="
echo "=== Waiting for all to finish ... ==="
echo ""

FAILED=0
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
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

# Build the comparison table (includes both per-cell-line and per-cell-line x day)
echo ""
echo "=== Building comparison table ==="
uv run python scripts/build_comparison_table.py \
  --results-dir "$OUT_DIR/results" \
  --k "$K" --objective "$OBJ" \
  --signed-overlap false

echo ""
echo "=== ALL DONE $(date) ==="
