#!/usr/bin/env bash
set -euo pipefail
set +m

# ── Configuration ──────────────────────────────────────────────
DURATION=${1:-7200}          # 2 hours default
SIMS=${HYZERO_SIMS:-80}
EVAL_SIMS=${HYZERO_EVAL_SIMS:-50}
GAMES=${HYZERO_GAMES:-8}
VALUE_BETA=${HYZERO_VALUE_OUTCOME_BETA:-0.3}
PROMOTION_THRESHOLD=${HYZERO_PROMOTION_THRESHOLD:-0.55}
CHAMPION_SCORE_WEIGHT=${HYZERO_CHAMPION_SCORE_WEIGHT:-2.0}
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/training_${TIMESTAMP}.json"

mkdir -p "$LOG_DIR"

echo "=== hyzero Training Run ==="
echo "Duration: ${DURATION}s"
echo "SIMS=${SIMS}, EVAL_SIMS=${EVAL_SIMS}, GAMES=${GAMES}, beta=${VALUE_BETA}"
echo "Log: ${LOG_FILE}"
echo "(Checkpoints are preserved — this is keep-training mode)"

# ── Build ──────────────────────────────────────────────────────
echo "[1/4] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Run — NO checkpoint cleanup ────────────────────────────────
echo "[2/4] Running selfplay for ${DURATION}s (checkpoints preserved)..."
if [ -d checkpoints ]; then
    echo "  Existing checkpoints:"
    ls checkpoints/best*.pt 2>/dev/null || echo "  (none)"
fi

ENV_ARGS=(
    "HYZERO_SIMS=$SIMS"
    "HYZERO_EVAL_SIMS=$EVAL_SIMS"
    "HYZERO_GAMES=$GAMES"
    "HYZERO_VALUE_OUTCOME_BETA=$VALUE_BETA"
    "HYZERO_PROMOTION_THRESHOLD=$PROMOTION_THRESHOLD"
    "HYZERO_CHAMPION_SCORE_WEIGHT=$CHAMPION_SCORE_WEIGHT"
)
# Pass adjudication env vars through if set (for smoke testing with low thresholds)
if [ -n "${HYZERO_ADJ_THRESHOLD:-}" ]; then
    ENV_ARGS+=("HYZERO_ADJ_THRESHOLD=$HYZERO_ADJ_THRESHOLD")
fi
if [ -n "${HYZERO_ADJ_PLIES:-}" ]; then
    ENV_ARGS+=("HYZERO_ADJ_PLIES=$HYZERO_ADJ_PLIES")
fi

env "${ENV_ARGS[@]}" target/release/selfplay > "$LOG_FILE" 2>&1 &
PID=$!
sleep "$DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 2
kill -KILL $PID 2>/dev/null || true
set +e
wait $PID 2>/dev/null
set -e

# ── Extract metrics ────────────────────────────────────────────
echo "[3/4] Extracting metrics..."

GAMES_PLAYED=$(awk '/\[py_training\].*Game received/{n++} END{print n+0}' "$LOG_FILE")
TRAIN_STEPS=$(awk '/\[py_training\].*step [0-9]/{n++} END{print n+0}' "$LOG_FILE")
_LAST_TRAIN_LINE=$(awk '/\[py_training\].*step [0-9]/{line=$0} END{print line}' "$LOG_FILE")
LAST_POLICY=$(printf '%s\n' "$_LAST_TRAIN_LINE" | sed -n 's/.*policy=\([0-9.]*\).*/\1/p')
LAST_POLICY=${LAST_POLICY:-0.0}
LAST_LOSS=$(printf '%s\n' "$_LAST_TRAIN_LINE" | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
LAST_LOSS=${LAST_LOSS:-0.0}

PROMOTIONS=$(awk '/\[eval\].*promoted/{n++} END{print n+0}' "$LOG_FILE")
MAX_CHAMPION_VERSION=$(awk '/\[eval\].*promoted/{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^champion_version=/) { split($i, a, "="); v=a[2]+0; if(v>max) max=v }
    }
} END{print max+0}' "$LOG_FILE")

# Adjudication rate: adjudicated games / total games
ADJUDICATED=$(awk '/\[selfplay\] adjudicated/{n++} END{print n+0}' "$LOG_FILE")
if [ "$GAMES_PLAYED" -gt 0 ]; then
    ADJ_RATE=$(awk "BEGIN { printf \"%.4f\", $ADJUDICATED / $GAMES_PLAYED }")
else
    ADJ_RATE="0.0000"
fi

AVG_GAME_LEN=$(awk '/\[py_training\].*Game received/{split($0,a,"received: "); split(a[2],b," "); sum+=b[1]; n++} END{if(n>0) printf "%.1f", sum/n; else print "0"}' "$LOG_FILE")

ERRORS=$(awk 'tolower($0) ~ /error|panic/{n++} END{print n+0}' "$LOG_FILE")
GIT_COMMIT=$(git rev-parse --short HEAD)

echo ""
echo "=== Results ==="
echo "  Games played:        $GAMES_PLAYED"
echo "  Training steps:      $TRAIN_STEPS"
echo "  Final policy loss:   $LAST_POLICY"
echo "  Avg game length:     $AVG_GAME_LEN"
echo "  Promotions:          $PROMOTIONS"
echo "  Adjudicated games:   $ADJUDICATED (rate: $ADJ_RATE)"
echo "  Errors:              $ERRORS"

# ── Write JSON summary ─────────────────────────────────────────
echo "[4/4] Writing JSON summary..."
cat > "$JSON_FILE" << EOF
{
    "timestamp": "$TIMESTAMP",
    "git_commit": "$GIT_COMMIT",
    "duration_s": $DURATION,
    "metrics": {
        "games_played": $GAMES_PLAYED,
        "training_steps": $TRAIN_STEPS,
        "final_policy_loss": $LAST_POLICY,
        "final_total_loss": $LAST_LOSS,
        "avg_game_length": $AVG_GAME_LEN,
        "promotions": $PROMOTIONS,
        "max_champion_version": ${MAX_CHAMPION_VERSION:-0},
        "adjudicated_games": $ADJUDICATED,
        "adjudication_rate": $ADJ_RATE,
        "errors": $ERRORS
    },
    "config": {
        "sims": $SIMS,
        "eval_sims": $EVAL_SIMS,
        "concurrent_games": $GAMES,
        "value_outcome_beta": $VALUE_BETA,
        "promotion_threshold": $PROMOTION_THRESHOLD,
        "champion_score_weight": $CHAMPION_SCORE_WEIGHT
    },
    "log_file": "$LOG_FILE"
}
EOF

echo "  JSON written to: $JSON_FILE"
echo "  Log saved to: $LOG_FILE"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "WARNING: $ERRORS errors detected — check log"
fi

echo ""
echo "TRAINING RUN COMPLETE"
