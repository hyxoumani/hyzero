#!/usr/bin/env bash
# Smoke test for dual-model evaluation ladder.
#
# Runs selfplay for 120s with HYZERO_PROMOTION_THRESHOLD=0.0 to force a promotion
# regardless of win rate, then greps the log for the "[eval] promoted" line.
#
# Usage:
#   bash scripts/smoke_dual_eval.sh
#
# Exit codes:
#   0 = success (promotion line found)
#   1 = failure (no promotion line or build failed)

set -euo pipefail
set +m

DURATION=${SMOKE_DURATION:-120}
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/smoke_dual_eval_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=== Smoke Test: dual-model evaluation ladder ==="
echo "Duration: ${DURATION}s"
echo "Threshold: 0.0 (force promotion)"
echo "Log: ${LOG_FILE}"
echo ""

# Build release binary (required for correct Dirichlet noise timing)
echo "[1/3] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# Run selfplay with forced promotion threshold
echo "[2/3] Running selfplay for ${DURATION}s..."
HYZERO_GAMES=5 \
HYZERO_SIMS=40 \
HYZERO_GAMES_PER_SIDE=2 \
HYZERO_PROMOTION_THRESHOLD=0.0 \
HYZERO_EVAL_SIMS=25 \
HYZERO_CHAMPION_SCORE_WEIGHT=2.0 \
target/release/selfplay > "$LOG_FILE" 2>&1 &
PID=$!

sleep "$DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 2
kill -KILL $PID 2>/dev/null || true
set +e
wait $PID 2>/dev/null
set -e

# Grep for the promotion line
echo "[3/3] Checking for '[eval] promoted' log line..."
echo ""

if grep -q '\[eval\] promoted' "$LOG_FILE"; then
    echo "PASS: Found [eval] promoted line"
    echo ""
    echo "Matching lines:"
    grep '\[eval\] promoted' "$LOG_FILE"
    echo ""
    echo "All eval lines:"
    grep '\[eval\]' "$LOG_FILE" || true
    echo ""
    echo "SMOKE TEST PASSED"
    exit 0
else
    echo "FAIL: No '[eval] promoted' line found in log"
    echo ""
    echo "All eval lines (if any):"
    grep '\[eval\]' "$LOG_FILE" || echo "  (none)"
    echo ""
    echo "Last 30 lines of log:"
    tail -30 "$LOG_FILE"
    echo ""
    echo "SMOKE TEST FAILED"
    exit 1
fi
