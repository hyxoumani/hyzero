#!/usr/bin/env bash
set -euo pipefail
set +m

# ── Configuration ──────────────────────────────────────────────
DURATION=${1:-1800}          # 30 minutes default
EVAL_INTERVAL=${HYZERO_EVAL_INTERVAL:-50}
BASELINE_FILE="logs/baseline_score.json"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/baseline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=== hyzero Baseline Run ==="
echo "Duration: ${DURATION}s"
echo "Eval interval: every ${EVAL_INTERVAL} training steps"
echo "Log: ${LOG_FILE}"

# ── Build ──────────────────────────────────────────────────────
echo "[1/5] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Run ────────────────────────────────────────────────────────
echo "[2/5] Running selfplay for ${DURATION}s..."
HYZERO_EVAL_INTERVAL=$EVAL_INTERVAL target/release/selfplay > "$LOG_FILE" 2>&1 &
PID=$!
sleep "$DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 2
kill -KILL $PID 2>/dev/null || true
set +e
wait $PID 2>/dev/null
set -e

# ── Extract metrics ────────────────────────────────────────────
echo "[3/5] Extracting metrics..."

GAMES=$(awk '/\[py_training\].*Game received/{n++} END{print n+0}' "$LOG_FILE")
TRAIN_STEPS=$(awk '/\[py_training\].*step [0-9]/{n++} END{print n+0}' "$LOG_FILE")
FIRST_LOSS=$(grep '\[py_training\].*step [0-9]' "$LOG_FILE" | head -1 | sed 's/.*total=\([0-9.]*\).*/\1/' || echo "N/A")
LAST_LOSS=$(grep '\[py_training\].*step [0-9]' "$LOG_FILE" | tail -1 | sed 's/.*total=\([0-9.]*\).*/\1/' || echo "N/A")
LAST_POLICY=$(grep '\[py_training\].*step [0-9]' "$LOG_FILE" | tail -1 | sed 's/.*policy=\([0-9.]*\).*/\1/' || echo "N/A")
ERRORS=$(awk 'tolower($0) ~ /error|panic/{n++} END{print n+0}' "$LOG_FILE")
CHECKPOINTS=$(awk '/\[py_training\].*Checkpoint saved/{n++} END{print n+0}' "$LOG_FILE")
AVG_STEPS=$(awk '/\[py_training\].*Game received/{split($0,a,"received: "); split(a[2],b," "); sum+=b[1]; n++} END{if(n>0) printf "%.1f", sum/n; else print "0"}' "$LOG_FILE")

# Extract eval metrics (use last eval cycle if multiple)
EVAL_LINE=$(grep '\[eval\]' "$LOG_FILE" | tail -1)
if [ -n "$EVAL_LINE" ]; then
    DECISIVE_RATIO=$(echo "$EVAL_LINE" | sed 's/.*decisive_ratio=\([0-9.]*\).*/\1/')
    EVAL_AVG_LEN=$(echo "$EVAL_LINE" | sed 's/.*avg_length=\([0-9.]*\).*/\1/')
    EVAL_CYCLES=$(grep -c '\[eval\]' "$LOG_FILE")
else
    DECISIVE_RATIO="0.0"
    EVAL_AVG_LEN="300.0"
    EVAL_CYCLES=0
fi

# ── Compute composite score ────────────────────────────────────
# score = (initial_loss - final_policy_loss) + (decisive_ratio * 10) - (avg_game_length / 100)
# Higher is better. Rewards: fast loss decrease, decisive games, shorter games.
echo "[4/5] Computing score..."

SCORE=$(awk "BEGIN {
    init_loss = 8.55;
    policy_loss = $LAST_POLICY;
    decisive = $DECISIVE_RATIO;
    avg_len = $AVG_STEPS;
    score = (init_loss - policy_loss) + (decisive * 10) - (avg_len / 100);
    printf \"%.4f\", score
}")

GIT_COMMIT=$(git rev-parse --short HEAD)

echo ""
echo "=== Results ==="
echo "  Games:           $GAMES"
echo "  Training steps:  $TRAIN_STEPS"
echo "  Loss:            $FIRST_LOSS → $LAST_LOSS"
echo "  Policy loss:     $LAST_POLICY"
echo "  Avg game length: $AVG_STEPS"
echo "  Decisive ratio:  $DECISIVE_RATIO"
echo "  Eval cycles:     $EVAL_CYCLES"
echo "  Checkpoints:     $CHECKPOINTS"
echo "  Errors:          $ERRORS"
echo "  ────────────────────"
echo "  SCORE:           $SCORE"
echo ""

# ── Write baseline ─────────────────────────────────────────────
echo "[5/5] Writing baseline..."

# Compare with previous baseline if it exists
if [ -f "$BASELINE_FILE" ]; then
    PREV_SCORE=$(python3 -c "import json; print(json.load(open('$BASELINE_FILE'))['score'])" 2>/dev/null || echo "0")
    IMPROVED=$(awk "BEGIN {print ($SCORE > $PREV_SCORE) ? 1 : 0}")
    if [ "$IMPROVED" -eq 1 ]; then
        echo "  ↑ Improved from $PREV_SCORE → $SCORE"
    else
        echo "  ↓ Regressed from $PREV_SCORE → $SCORE"
        echo "  Previous baseline kept. New score saved to logs/."
    fi
fi

cat > "$BASELINE_FILE" << EOF
{
    "score": $SCORE,
    "timestamp": "$TIMESTAMP",
    "git_commit": "$GIT_COMMIT",
    "duration_s": $DURATION,
    "metrics": {
        "games_completed": $GAMES,
        "training_steps": $TRAIN_STEPS,
        "first_loss": $FIRST_LOSS,
        "last_loss": $LAST_LOSS,
        "last_policy_loss": $LAST_POLICY,
        "avg_game_length": $AVG_STEPS,
        "decisive_ratio": $DECISIVE_RATIO,
        "eval_cycles": $EVAL_CYCLES,
        "checkpoints": $CHECKPOINTS,
        "errors": $ERRORS
    },
    "config": {
        "eval_interval_steps": $EVAL_INTERVAL,
        "concurrent_games": ${HYZERO_GAMES:-4},
        "simulations": ${HYZERO_SIMS:-50}
    },
    "log_file": "$LOG_FILE"
}
EOF

echo "  Baseline written to: $BASELINE_FILE"
echo "  Log saved to: $LOG_FILE"

# ── Validate ───────────────────────────────────────────────────
if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "WARNING: $ERRORS errors detected — check log"
    exit 1
fi

echo ""
echo "BASELINE COMPLETE — score: $SCORE"
