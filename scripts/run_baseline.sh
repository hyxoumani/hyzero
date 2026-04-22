#!/usr/bin/env bash
set -euo pipefail
set +m

# ── Configuration ──────────────────────────────────────────────
DURATION=${1:-1800}          # 30 minutes default
SIMS=${HYZERO_SIMS:-200}
EVAL_SIMS=${HYZERO_EVAL_SIMS:-100}
GAMES=${HYZERO_GAMES:-9}              # total slots: 1 for eval + N-1 for selfplay
BATCH_SIZE=${HYZERO_BATCH_SIZE:-64}
GAMES_PER_SIDE=${HYZERO_GAMES_PER_SIDE:-4}
PROMOTION_THRESHOLD=${HYZERO_PROMOTION_THRESHOLD:-0.55}
CHAMPION_SCORE_WEIGHT=${HYZERO_CHAMPION_SCORE_WEIGHT:-2.0}
DEVICE=${HYZERO_DEVICE:-cuda}
BASELINE_FILE="logs/baseline_score.json"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/baseline_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=== hyzero Baseline Run ==="
echo "Duration: ${DURATION}s"
echo "Device: ${DEVICE}"
echo "Sims: selfplay=${SIMS}, eval=${EVAL_SIMS}"
echo "Concurrency: ${GAMES} total slots (${GAMES}-1 selfplay + 1 eval), batch_size=${BATCH_SIZE}"
echo "Eval: ${GAMES_PER_SIDE} games/side, threshold=${PROMOTION_THRESHOLD}, weight=${CHAMPION_SCORE_WEIGHT}"
echo "Log: ${LOG_FILE}"

# ── Build ──────────────────────────────────────────────────────
echo "[1/5] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Cleanup: remove ONLY training checkpoints, preserve champion files ─────
# model_vNNNNNN.pt = training checkpoints → delete
# best.pt and best_vNNN.pt = champion archive → PRESERVE across runs
echo "[1b/5] Cleaning up old training checkpoints (preserving champion files)..."
if [ -d checkpoints ]; then
    find checkpoints -maxdepth 1 -name 'model_v*.pt' -delete 2>/dev/null || true
    echo "  Retained champion files:"
    ls checkpoints/best*.pt 2>/dev/null || echo "  (none yet)"
fi

# ── Run ────────────────────────────────────────────────────────
echo "[2/5] Running selfplay for ${DURATION}s..."
HYZERO_DEVICE=$DEVICE \
HYZERO_SIMS=$SIMS \
HYZERO_EVAL_SIMS=$EVAL_SIMS \
HYZERO_GAMES=$GAMES \
HYZERO_BATCH_SIZE=$BATCH_SIZE \
HYZERO_GAMES_PER_SIDE=$GAMES_PER_SIDE \
HYZERO_PROMOTION_THRESHOLD=$PROMOTION_THRESHOLD \
HYZERO_CHAMPION_SCORE_WEIGHT=$CHAMPION_SCORE_WEIGHT \
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

# Extract eval metrics from ladder_match lines
# Format: [eval] v{v} cycle={c} ladder_wins={w} ... decisive_ratio is computed here
EVAL_CYCLES=$(awk '/\[eval\].*ladder_match/{n++} END{print n+0}' "$LOG_FILE")

# Extract promotions count and MAX champion_version
PROMOTIONS=$(awk '/\[eval\].*promoted/{n++} END{print n+0}' "$LOG_FILE")
MAX_CHAMPION_VERSION=$(awk '/\[eval\].*promoted/{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^champion_version=/) { split($i, a, "="); v=a[2]+0; if(v>max) max=v }
    }
} END{print max+0}' "$LOG_FILE")
MAX_CHAMPION_VERSION=${MAX_CHAMPION_VERSION:-0}

if [ "$EVAL_CYCLES" -gt 0 ]; then
    # Parse ladder_match lines: compute win_rate and avg game info from wins/draws/losses
    EVAL_SUMMARY=$(awk '/\[eval\].*ladder_match/{
        cycle++
        wr = "0.0"
        for (i=1; i<=NF; i++) {
            if ($i ~ /^win_rate=/) { split($i, a, "="); wr = a[2] }
        }
        print cycle, wr
    }' "$LOG_FILE")

    echo "  Eval cycles (ladder_match):"
    echo "$EVAL_SUMMARY" | while read -r cyc wr; do
        echo "    Cycle $cyc: win_rate=${wr}"
    done

    # Use the last win_rate as decisive proxy (higher win_rate = better)
    # We use win_rate rather than decisive_ratio for the new ladder
    LAST_WIN_RATE=$(echo "$EVAL_SUMMARY" | awk '{last=$2} END{print last+0}')
    DECISIVE_RATIO=${LAST_WIN_RATE:-0.0}
    # avg_length from game received logs
    EVAL_AVG_LEN=$AVG_STEPS
    echo "  Max champion_version: $MAX_CHAMPION_VERSION"
    echo "  Promotions: $PROMOTIONS"
    echo "  Last win_rate: $DECISIVE_RATIO"
else
    EVAL_CYCLES=0
    DECISIVE_RATIO="0.0"
    EVAL_AVG_LEN="300.0"
fi

# ── Compute composite score ────────────────────────────────────
# score = (8.55 - final_policy_loss) + (promotions * CHAMPION_SCORE_WEIGHT) - (avg_length / 100)
# Higher is better. Rewards: fast policy learning, promotion count (not version tag), shorter games.
# Note: max_champion_version is kept in JSON for debugging but NOT used in scoring.
echo "[4/5] Computing score..."

SCORE=$(awk "BEGIN {
    init_loss = 8.55;
    policy_loss = $LAST_POLICY;
    promotions = $PROMOTIONS;
    weight = $CHAMPION_SCORE_WEIGHT;
    avg_len = $AVG_STEPS;
    score = (init_loss - policy_loss) + (promotions * weight) - (avg_len / 100);
    printf \"%.4f\", score
}")

GIT_COMMIT=$(git rev-parse --short HEAD)

echo ""
echo "=== Results ==="
echo "  Games:               $GAMES"
echo "  Training steps:      $TRAIN_STEPS"
echo "  Loss:                $FIRST_LOSS → $LAST_LOSS"
echo "  Policy loss:         $LAST_POLICY"
echo "  Avg game length:     $AVG_STEPS"
echo "  Eval cycles:         $EVAL_CYCLES"
echo "  Promotions:          $PROMOTIONS"
echo "  Max champion ver:    $MAX_CHAMPION_VERSION"
echo "  Last win_rate:       $DECISIVE_RATIO"
echo "  Champion weight:     $CHAMPION_SCORE_WEIGHT"
echo "  Checkpoints:         $CHECKPOINTS"
echo "  Errors:              $ERRORS"
echo "  ────────────────────"
echo "  SCORE:               $SCORE"
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
        "last_win_rate": $DECISIVE_RATIO,
        "eval_cycles": $EVAL_CYCLES,
        "promotions": $PROMOTIONS,
        "max_champion_version": $MAX_CHAMPION_VERSION,
        "checkpoints": $CHECKPOINTS,
        "errors": $ERRORS
    },
    "config": {
        "games_per_side": $GAMES_PER_SIDE,
        "promotion_threshold": $PROMOTION_THRESHOLD,
        "champion_score_weight": $CHAMPION_SCORE_WEIGHT,
        "eval_sims": $EVAL_SIMS,
        "concurrent_games": $GAMES,
        "batch_size": $BATCH_SIZE,
        "simulations": $SIMS,
        "device": "$DEVICE"
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
