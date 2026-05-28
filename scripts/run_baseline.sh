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
ELO_SCORE_WEIGHT=${HYZERO_ELO_SCORE_WEIGHT:-0.05}
DEVICE=${HYZERO_DEVICE:-cuda}
# Supervision data sources (validated 2026-04-21 through 2026-04-23).
# Defaults are opt-in for the full stack; unset/override to disable.
STARTS_FILE=${HYZERO_STARTS_FILE:-data/starting_positions.txt}
TB_PATH=${HYZERO_TABLEBASE_PATH:-data/syzygy}
# Default supervision cache includes both Syzygy TB endgames AND Lichess
# mate-in-1 puzzles (built by scripts/build_merged_supervision_cache.py).
# Mate puzzles provide the explicit +1 reward signal that pure TB endgames
# and self-play lack — prevents "reward head dead" from mate-starvation.
TB_CACHE=${HYZERO_TABLEBASE_CACHE_PATH:-data/syzygy/cache_tb_plus_mates.pkl}
TB_FRAC=${HYZERO_TABLEBASE_FRAC:-0.45}
# Resume point — defaults to the mate-pretrained checkpoint. This gives every
# run a starting state where the reward head already recognizes mating moves
# (avoids the bootstrap failure where self-play never generates mates).
# If the file is missing, the block below auto-creates it from the pretrain
# dynamics checkpoint by running scripts/pretrain_on_mates.py.
RESUME_FROM=${HYZERO_RESUME_FROM:-checkpoints/mate_pretrained.pt}
MATE_PUZZLES=${HYZERO_MATE_PUZZLES:-data/lichess_mates.pkl}
MATE_PRETRAIN_STEPS=${HYZERO_MATE_PRETRAIN_STEPS:-4000}
MATE_PRETRAIN_POSITIONS=${HYZERO_MATE_PRETRAIN_POSITIONS:-100000}
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

# Auto-build the mate-pretrained checkpoint if the resume file is the default
# mate-pretrained path and doesn't exist yet. This is the "always insert the
# pretrained" behavior: every run starts from a network that already understands
# mate patterns. Skips this block if user overrode RESUME_FROM to something else.
if [ "$RESUME_FROM" = "checkpoints/mate_pretrained.pt" ] && [ ! -f "$RESUME_FROM" ]; then
    echo "[pre-run] mate-pretrained checkpoint missing — building it..."
    if [ ! -f "$MATE_PUZZLES" ]; then
        echo "  WARN: $MATE_PUZZLES missing — cannot auto-pretrain. Mine it via:"
        echo "    python3 scripts/mine_lichess_mate_in_1.py"
        echo "  Falling back to checkpoints/pretrain_dynamics.pt"
        RESUME_FROM="checkpoints/pretrain_dynamics.pt"
    elif [ ! -f "checkpoints/pretrain_dynamics.pt" ] && [ ! -f "checkpoints/best.pt" ]; then
        echo "  WARN: no base checkpoint (pretrain_dynamics.pt or best.pt) — skipping auto-pretrain"
        RESUME_FROM=""
    else
        BASE_CKPT="checkpoints/pretrain_dynamics.pt"
        [ ! -f "$BASE_CKPT" ] && BASE_CKPT="checkpoints/best.pt"
        echo "  source: $BASE_CKPT  puzzles: $MATE_PUZZLES"
        echo "  steps: $MATE_PRETRAIN_STEPS  positions: $MATE_PRETRAIN_POSITIONS"
        HYZERO_MATE_PUZZLES="$MATE_PUZZLES" python3 scripts/pretrain_on_mates.py \
            --in-ckpt "$BASE_CKPT" \
            --out-ckpt "checkpoints/mate_pretrained.pt" \
            --use-file \
            --n-positions "$MATE_PRETRAIN_POSITIONS" \
            --steps "$MATE_PRETRAIN_STEPS" \
            --batch-size 128 \
            --lr 3e-4 \
            --device cpu \
            2>&1 | tail -3
        if [ -f "checkpoints/mate_pretrained.pt" ]; then
            echo "  mate-pretrained checkpoint created."
        else
            echo "  WARN: auto-pretrain FAILED. Falling back to $BASE_CKPT"
            RESUME_FROM="$BASE_CKPT"
        fi
    fi
fi

# Warn (don't fail) on missing resume checkpoint — binary will fall back to RandomEvaluator.
if [ -n "$RESUME_FROM" ] && [ ! -f "$RESUME_FROM" ]; then
    echo "  WARN: HYZERO_RESUME_FROM=$RESUME_FROM missing — starting from random init"
    RESUME_FROM=""
fi
echo "Resume: ${RESUME_FROM:-random init (no checkpoint)}"

# Warn (don't fail) on missing supervision files — training falls back to pure self-play.
if [ -n "$STARTS_FILE" ] && [ ! -f "$STARTS_FILE" ]; then
    echo "  WARN: HYZERO_STARTS_FILE=$STARTS_FILE missing — diverse starts disabled"
    STARTS_FILE=""
fi
if [ -n "$TB_CACHE" ] && [ ! -f "$TB_CACHE" ]; then
    echo "  WARN: HYZERO_TABLEBASE_CACHE_PATH=$TB_CACHE missing — TB supervision disabled"
    TB_PATH=""
    TB_CACHE=""
fi
echo "Supervision: starts=$([ -n "$STARTS_FILE" ] && echo "$STARTS_FILE" || echo "off"), tb_frac=$([ -n "$TB_CACHE" ] && echo "$TB_FRAC ($TB_CACHE)" || echo "off")"
echo "Log: ${LOG_FILE}"

# ── Build ──────────────────────────────────────────────────────
echo "[1/5] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Cleanup: full-slate reset ──────────────────────────────────
# Since we resume from a fixed pretrain checkpoint by default, wipe both
# training checkpoints (model_v*.pt) AND any champion from a prior run
# (best*.pt) so the ladder starts from scratch. Override RESUME_FROM to
# checkpoints/best.pt if you want champion continuity instead.
echo "[1b/5] Cleaning up prior-run artifacts (training ckpts + champion archive)..."
if [ -d checkpoints ]; then
    find checkpoints -maxdepth 1 -name 'model_v*.pt' -delete 2>/dev/null || true
    # Never delete the resume-from file itself — only archive/training files.
    for f in checkpoints/best.pt checkpoints/best_v*.pt; do
        [ -f "$f" ] && [ "$(realpath "$f")" != "$(realpath "$RESUME_FROM" 2>/dev/null || echo /dev/null)" ] && rm -f "$f"
    done
    echo "  Remaining checkpoints:"
    ls checkpoints/*.pt 2>/dev/null | sed 's/^/    /' || echo "    (none)"
fi

# ── Run ────────────────────────────────────────────────────────
echo "[2/5] Running selfplay for ${DURATION}s..."
export HYZERO_POLICY_ENTROPY_WEIGHT=0.01
export HYZERO_LR_SCHEDULE=cosine
export HYZERO_LR_COSINE_T_MAX=7000
export HYZERO_LR_COSINE_ETA_MIN=1e-5
echo "[env] $(env | grep '^HYZERO_' | sort | tr '\n' ' ')"
HYZERO_DEVICE=$DEVICE \
HYZERO_SIMS=$SIMS \
HYZERO_EVAL_SIMS=$EVAL_SIMS \
HYZERO_GAMES=$GAMES \
HYZERO_BATCH_SIZE=$BATCH_SIZE \
HYZERO_GAMES_PER_SIDE=$GAMES_PER_SIDE \
HYZERO_PROMOTION_THRESHOLD=$PROMOTION_THRESHOLD \
HYZERO_CHAMPION_SCORE_WEIGHT=$CHAMPION_SCORE_WEIGHT \
HYZERO_STARTS_FILE=$STARTS_FILE \
HYZERO_TABLEBASE_PATH=$TB_PATH \
HYZERO_TABLEBASE_CACHE_PATH=$TB_CACHE \
HYZERO_TABLEBASE_FRAC=$TB_FRAC \
HYZERO_RESUME_FROM=$RESUME_FROM \
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

    # Extract per-cycle candidate Elo for the pool-based promotion gate.
    # Falls back to 1500.0 (initial) on cycles that pre-date the field.
    CANDIDATE_ELO_SUMMARY=$(awk '/\[eval\].*ladder_match/{
        cycle++
        elo = "1500.0"
        for (i=1; i<=NF; i++) {
            if ($i ~ /^candidate_elo=/) { split($i, a, "="); elo = a[2] }
        }
        print cycle, elo
    }' "$LOG_FILE")
    LAST_CANDIDATE_ELO=$(echo "$CANDIDATE_ELO_SUMMARY" | awk '{last=$2} END{print last+0}')
    LAST_CANDIDATE_ELO=${LAST_CANDIDATE_ELO:-1500.0}

    # avg_length from game received logs
    EVAL_AVG_LEN=$AVG_STEPS
    echo "  Max champion_version: $MAX_CHAMPION_VERSION"
    echo "  Promotions: $PROMOTIONS"
    echo "  Last win_rate: $DECISIVE_RATIO"
    echo "  Last candidate Elo:  $LAST_CANDIDATE_ELO"
else
    EVAL_CYCLES=0
    DECISIVE_RATIO="0.0"
    EVAL_AVG_LEN="300.0"
    LAST_CANDIDATE_ELO="1500.0"
fi

# ── Compute composite score ────────────────────────────────────
# score = (8.55 - final_policy_loss) + (promotions * CHAMPION_SCORE_WEIGHT)
#       - (avg_length / 100) + (last_candidate_elo - 1500.0) * ELO_SCORE_WEIGHT
# Higher is better. Rewards: fast policy learning, promotion count (not version
# tag), shorter games, and Elo progress against the archive pool. The Elo term
# is signed (gains contribute, regressions subtract).
# Note: max_champion_version is kept in JSON for debugging but NOT used in scoring.
echo "[4/5] Computing score..."

SCORE=$(awk "BEGIN {
    init_loss = 8.55;
    policy_loss = $LAST_POLICY;
    promotions = $PROMOTIONS;
    weight = $CHAMPION_SCORE_WEIGHT;
    avg_len = $AVG_STEPS;
    last_candidate_elo = $LAST_CANDIDATE_ELO;
    elo_score_weight = $ELO_SCORE_WEIGHT;
    score = (init_loss - policy_loss) + (promotions * weight) - (avg_len / 100) + (last_candidate_elo - 1500.0) * elo_score_weight;
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
echo "  Last candidate Elo:  $LAST_CANDIDATE_ELO"
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
        "last_candidate_elo": $LAST_CANDIDATE_ELO,
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
        "device": "$DEVICE",
        "resume_from": "$RESUME_FROM",
        "starts_file": "$STARTS_FILE",
        "tablebase_cache": "$TB_CACHE",
        "tablebase_frac": $TB_FRAC
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
