#!/usr/bin/env bash
set -euo pipefail
# Suppress job control messages so kill doesn't print "Terminated: 15"
set +m

# Configuration
DURATION=${1:-180}  # seconds to run
MIN_GAMES=1         # minimum games that must complete
MIN_TRAIN_STEPS=1   # minimum training steps
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/e2e_${TIMESTAMP}.log"
METRICS_FILE="${LOG_DIR}/e2e_${TIMESTAMP}_metrics.txt"

mkdir -p "$LOG_DIR"

echo "=== hyzero End-to-End Test ==="
echo "Duration: ${DURATION}s"
echo "Log: ${LOG_FILE}"

# Ensure Python package is installed
python3 -c "import hyzero" 2>/dev/null || pip install -e python/ -q 2>&1 | tail -1

# Build
echo "[1/4] Building..."
cargo build --release --bin selfplay 2>&1 | tail -1

# Run selfplay with timeout — run binary directly (not cargo run) to avoid cargo's
# output buffering layer; SIGTERM + 2s grace period lets Rust flush line buffers.
echo "[2/4] Running selfplay for ${DURATION}s..."
target/release/selfplay > "$LOG_FILE" 2>&1 &
PID=$!
sleep "$DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 2
kill -KILL $PID 2>/dev/null || true
set +e
wait $PID 2>/dev/null
set -e

# Extract metrics — use awk for counting (always exits 0, avoids pipefail on grep no-match)
# Log format: "[py_training] Game received: N steps, buffer: M games / T total steps, model vX"
#             "[py_training] step N: total=X.XXXX policy=... value=... reward=... (vX)"
#             "[py_training] Checkpoint saved: checkpoints/model_vX.pt"
echo "[3/4] Extracting metrics..."
GAMES=$(awk '/\[py_training\].*Game received/{n++} END{print n+0}' "$LOG_FILE" 2>/dev/null)
TRAIN_STEPS=$(awk '/\[py_training\].*step [0-9]/{n++} END{print n+0}' "$LOG_FILE" 2>/dev/null)
# Loss extraction: use grep+sed (macOS awk lacks 3-arg match(); grep returns 1 on no-match
# so wrap with || true to satisfy set -e).
FIRST_LOSS=$(grep '\[py_training\].*step [0-9]' "$LOG_FILE" 2>/dev/null | head -1 | sed 's/.*total=\([0-9.]*\).*/\1/' || true)
LAST_LOSS=$(grep '\[py_training\].*step [0-9]' "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*total=\([0-9.]*\).*/\1/' || true)
ERRORS=$(awk 'tolower($0) ~ /error|panic/{n++} END{print n+0}' "$LOG_FILE" 2>/dev/null)
CHECKPOINTS=$(awk '/\[py_training\].*Checkpoint saved/{n++} END{print n+0}' "$LOG_FILE" 2>/dev/null)

# Extract average game lengths: "[py_training] Game received: 188 steps, ..."
AVG_STEPS=$(awk '/\[py_training\].*Game received/{split($0,a,"received: "); split(a[2],b," "); sum+=b[1]; n++} END{if(n>0) printf "%.0f", sum/n; else print "N/A"}' "$LOG_FILE" 2>/dev/null)

# Ensure numeric defaults and non-empty loss values
GAMES=${GAMES:-0}
TRAIN_STEPS=${TRAIN_STEPS:-0}
ERRORS=${ERRORS:-0}
CHECKPOINTS=${CHECKPOINTS:-0}
FIRST_LOSS=${FIRST_LOSS:-N/A}
LAST_LOSS=${LAST_LOSS:-N/A}
AVG_STEPS=${AVG_STEPS:-N/A}

# Write metrics
cat > "$METRICS_FILE" << EOF
timestamp=$TIMESTAMP
duration_s=$DURATION
games_completed=$GAMES
training_steps=$TRAIN_STEPS
first_loss=$FIRST_LOSS
last_loss=$LAST_LOSS
avg_game_steps=$AVG_STEPS
errors=$ERRORS
checkpoints=$CHECKPOINTS
EOF

echo "  Games completed: $GAMES"
echo "  Training steps:  $TRAIN_STEPS"
echo "  First loss:      $FIRST_LOSS"
echo "  Last loss:       $LAST_LOSS"
echo "  Avg game length: $AVG_STEPS steps"
echo "  Errors:          $ERRORS"
echo "  Checkpoints:     $CHECKPOINTS"

# Validate
echo "[4/4] Validating..."
PASS=true

if [ "$GAMES" -lt "$MIN_GAMES" ]; then
    echo "  FAIL: Only $GAMES games (need $MIN_GAMES)"
    PASS=false
fi

if [ "$TRAIN_STEPS" -lt "$MIN_TRAIN_STEPS" ]; then
    echo "  FAIL: Only $TRAIN_STEPS train steps (need $MIN_TRAIN_STEPS)"
    PASS=false
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "  FAIL: $ERRORS errors found"
    PASS=false
fi

# Check loss decrease (if we have enough training steps)
if [ "$FIRST_LOSS" != "N/A" ] && [ "$LAST_LOSS" != "N/A" ]; then
    # Use awk for float comparison
    LOSS_DECREASED=$(awk "BEGIN {print ($LAST_LOSS < $FIRST_LOSS) ? 1 : 0}")
    if [ "$LOSS_DECREASED" -eq 0 ] && [ "$TRAIN_STEPS" -ge 5 ]; then
        echo "  WARN: Loss did not decrease ($FIRST_LOSS -> $LAST_LOSS)"
    fi
fi

echo ""
if [ "$PASS" = true ]; then
    echo "E2E TEST PASSED"
    echo "Metrics saved to: $METRICS_FILE"
    exit 0
else
    echo "E2E TEST FAILED"
    echo "Full log: $LOG_FILE"
    exit 1
fi
