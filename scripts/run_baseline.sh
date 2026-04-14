#!/usr/bin/env bash
set -euo pipefail
set +m

# ── Configuration ──────────────────────────────────────────────
DURATION=${1:-1800}          # 30 minutes default
EVAL_INTERVAL=${HYZERO_EVAL_INTERVAL:-25}
EVAL_GAMES=${HYZERO_EVAL_GAMES:-5}
EVAL_SIMS=${HYZERO_EVAL_SIMS:-25}
BASELINE_FILE="logs/baseline_score.json"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/baseline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=== hyzero Baseline Run ==="
echo "Duration: ${DURATION}s"
echo "Eval: every ${EVAL_INTERVAL} versions × ${EVAL_GAMES} games × ${EVAL_SIMS} sims"
echo "Log: ${LOG_FILE}"

# ── Build ──────────────────────────────────────────────────────
echo "[1/5] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Run ────────────────────────────────────────────────────────
echo "[2/5] Running selfplay for ${DURATION}s..."
HYZERO_EVAL_INTERVAL=$EVAL_INTERVAL \
HYZERO_EVAL_GAMES=$EVAL_GAMES \
HYZERO_EVAL_SIMS=$EVAL_SIMS \
target/release/selfplay > "$LOG_FILE" 2>&1 &
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
_FIRST_TRAIN_LINE=$(awk '/\[py_training\].*step [0-9]/{print; exit}' "$LOG_FILE")
_LAST_TRAIN_LINE=$(awk '/\[py_training\].*step [0-9]/{line=$0} END{print line}' "$LOG_FILE")
FIRST_LOSS=$(printf '%s\n' "$_FIRST_TRAIN_LINE" | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
FIRST_LOSS=${FIRST_LOSS:-0.0}
LAST_LOSS=$(printf '%s\n' "$_LAST_TRAIN_LINE" | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
LAST_LOSS=${LAST_LOSS:-0.0}
LAST_POLICY=$(printf '%s\n' "$_LAST_TRAIN_LINE" | sed -n 's/.*policy=\([0-9.]*\).*/\1/p')
LAST_POLICY=${LAST_POLICY:-0.0}
ERRORS=$(awk 'tolower($0) ~ /error|panic/{n++} END{print n+0}' "$LOG_FILE")
CHECKPOINTS=$(awk '/\[py_training\].*Checkpoint saved/{n++} END{print n+0}' "$LOG_FILE")
AVG_STEPS=$(awk '/\[py_training\].*Game received/{split($0,a,"received: "); split(a[2],b," "); sum+=b[1]; n++} END{if(n>0) printf "%.1f", sum/n; else print "0"}' "$LOG_FILE")

# Extract eval metrics (use MAX decisive_ratio across all eval cycles)
EVAL_CYCLES=$(awk '/\[eval\]/{n++} END{print n+0}' "$LOG_FILE")
if [ "$EVAL_CYCLES" -gt 0 ]; then

    # For each eval line, print cycle_number decisive_ratio avg_length
    EVAL_SUMMARY=$(awk '/\[eval\]/{
        cycle++
        dr = "0.0"; al = "0.0"
        for (i=1; i<=NF; i++) {
            if ($i ~ /^decisive_ratio=/) { split($i, a, "="); dr = a[2] }
            if ($i ~ /^avg_length=/)    { split($i, a, "="); al = a[2] }
        }
        print cycle, dr, al
    }' "$LOG_FILE")

    # Report all cycles
    echo "  Eval cycles detail:"
    echo "$EVAL_SUMMARY" | while read -r cyc dr al; do
        echo "    Cycle $cyc: decisive=${dr} avg_length=${al}"
    done

    # Pick the line with max decisive_ratio
    BEST_EVAL=$(echo "$EVAL_SUMMARY" | awk 'BEGIN{max=-1; best_dr="0.0"; best_al="0.0"} {
        if ($2+0 > max) { max=$2+0; best_dr=$2; best_al=$3 }
    } END{ print best_dr, best_al }')

    DECISIVE_RATIO=$(echo "$BEST_EVAL" | awk '{print $1}')
    EVAL_AVG_LEN=$(echo "$BEST_EVAL" | awk '{print $2}')
    echo "  Using MAX: decisive=${DECISIVE_RATIO} avg_length=${EVAL_AVG_LEN}"
else
    EVAL_CYCLES=0
    DECISIVE_RATIO="0.0"
    EVAL_AVG_LEN="300.0"
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
        "eval_games": $EVAL_GAMES,
        "eval_sims": $EVAL_SIMS,
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
