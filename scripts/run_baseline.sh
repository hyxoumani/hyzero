#!/usr/bin/env bash
set -euo pipefail
set +m

# ── Configuration ──────────────────────────────────────────────
DURATION=${1:-1800}          # 30 minutes default
# 300 (was 200): let small Q-differences re-concentrate visits past residual noise.
SIMS=${HYZERO_SIMS:-300}
EVAL_SIMS=${HYZERO_EVAL_SIMS:-100}
GAMES=${HYZERO_GAMES:-9}              # total slots: 1 for eval + N-1 for selfplay
BATCH_SIZE=${HYZERO_BATCH_SIZE:-64}
GAMES_PER_SIDE=${HYZERO_GAMES_PER_SIDE:-8}
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

# Rotate the append-only game PGNs out of the way so this run's writer creates a
# fresh file and the KQvK audit measures ONLY the current run's games. Both
# logs/selfplay_sample.pgn and logs/eval_games.pgn are opened create+append and
# never truncated (see src/selfplay/pgn.rs write_pgn_game), so without rotation
# the audit would score a month-old accumulator. Rotation (not deletion)
# preserves history for later inspection.
_ROTATE_STAMP=$(date +%s)
mv logs/selfplay_sample.pgn "logs/selfplay_sample_prev_${_ROTATE_STAMP}.pgn" 2>/dev/null || true
mv logs/eval_games.pgn "logs/eval_games_prev_${_ROTATE_STAMP}.pgn" 2>/dev/null || true

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

# Decisive-start curriculum: bias self-play toward decisively-winnable starts
# (white-absolute material |Δ| ≥ 3) to fight value-signal starvation (self-play
# is ~92.6% draws, ~63.5% of them by repetition). Generated from the base starts
# file into data/decisive_starts.txt (~55% imbalanced + ~25% original-distribution
# for diversity + ~20% near-mate conversion starts walked back from
# data/mate_puzzles_v2.pkl, picked up automatically via the generator's default
# --mate-puzzles path; if that file is absent the share is redistributed).
# Idempotent and cheap; deterministic given the fixed seed. The base
# data/starting_positions.txt is never modified. Only swap STARTS_FILE to the
# curriculum when generation succeeds — otherwise keep the original starts.
DECISIVE_STARTS_FILE="data/decisive_starts.txt"
if [ -n "$STARTS_FILE" ] && [ -f "$STARTS_FILE" ]; then
    echo "[pre-run] Building decisive-start curriculum -> $DECISIVE_STARTS_FILE"
    # Fail fast if the generator's own self-test (classify/mix/validate) regresses.
    if ! python3 scripts/make_decisive_starts.py --self-test; then
        echo "  ERROR: make_decisive_starts.py self-test failed — aborting"
        exit 1
    fi
    if python3 scripts/make_decisive_starts.py --in "$STARTS_FILE" --out "$DECISIVE_STARTS_FILE"; then
        STARTS_FILE="$DECISIVE_STARTS_FILE"
    else
        echo "  WARN: decisive-start generation failed — using $STARTS_FILE unchanged"
    fi
fi

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

# ── Tablebase WDL rescoring (HYZERO_TB_RESCORE, default 1) ──────
# lc0-style tail-rescoring: self-play value targets for positions covered by the
# Syzygy WDL export are overridden with the exact tablebase result (STM POV),
# superseding the outcome/bootstrap target only for those covered steps. Generate
# the normfen->wdl CSV from the supervision cache before launch — export_tb_wdl.py
# is idempotent (skips when the CSV is newer than the cache). If the cache is
# missing or the export fails, rescoring is turned off so behavior is unchanged.
TB_RESCORE=${HYZERO_TB_RESCORE:-1}
TB_WDL_PATH=${HYZERO_TB_WDL_PATH:-data/syzygy/tb_wdl.csv}
if [ "$TB_RESCORE" != "0" ] && [ -n "$TB_CACHE" ] && [ -f "$TB_CACHE" ]; then
    echo "[pre-run] Exporting tablebase WDL CSV -> $TB_WDL_PATH"
    if ! HYZERO_TABLEBASE_CACHE_PATH="$TB_CACHE" HYZERO_TB_WDL_PATH="$TB_WDL_PATH" \
        python3 scripts/export_tb_wdl.py; then
        echo "  WARN: tablebase WDL export failed — rescoring disabled"
        TB_RESCORE=0
    fi
else
    TB_RESCORE=0
fi
echo "Rescore: $([ "$TB_RESCORE" != "0" ] && echo "on ($TB_WDL_PATH)" || echo "off")"
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
    # Seed the eval pool with preserved champions from prior validation runs.
    # Pool scanner (src/selfplay/pool.rs) discovers checkpoints/best_v{NNN}.pt with
    # min-width-3 zero-padding (champion.rs writes via format! "best_v{:03}.pt").
    # Versions 3806/3905 already exceed 3 digits, so no extra leading zeros — these
    # names match exactly what a real promotion would have produced.
    if [ -f checkpoints/backup_champion_v3806_20260609.pt ]; then
        cp checkpoints/backup_champion_v3806_20260609.pt checkpoints/best_v3806.pt
    else
        echo "  WARN: checkpoints/backup_champion_v3806_20260609.pt missing — pool not seeded with v3806"
    fi
    if [ -f checkpoints/backup_champion_v3905_20260609.pt ]; then
        cp checkpoints/backup_champion_v3905_20260609.pt checkpoints/best_v3905.pt
    else
        echo "  WARN: checkpoints/backup_champion_v3905_20260609.pt missing — pool not seeded with v3905"
    fi
    echo "  Remaining checkpoints:"
    ls checkpoints/*.pt 2>/dev/null | sed 's/^/    /' || echo "    (none)"
fi

# ── Run ────────────────────────────────────────────────────────
echo "[2/5] Running selfplay for ${DURATION}s..."
# The trainer's policy-entropy term (trainer.py _policy_loss) is an entropy
# BONUS: minimizing the loss maximizes H(π), pushing the k0 policy toward
# uniform. It caused the k0 pred_entropy divergence in the 2026-06-09 runs at
# β=0.01 and β=0.003. In MuZero-style distillation exploration comes from MCTS
# Dirichlet noise + selfplay temperature, not from flattening the trained
# policy. Default 0.0 (off); env-overridable for deliberate experiments.
export HYZERO_POLICY_ENTROPY_WEIGHT=${HYZERO_POLICY_ENTROPY_WEIGHT:-0.0}
# Tablebase trajectory rows carry uniform-over-Syzygy-optimal policy targets
# (~48% of TB positions have >=2 optimal moves). At tb_frac 0.45 these flat
# targets dominate the policy CE and flatten the shared policy head — the
# 2026-06-09/10 "entropy divergence" (pred_entropy_legal 0.90->1.71, top1
# 0.40->0.125). Weight 0.0 disables the TB policy gradient for the controlled
# run; TB value/reward supervision is unaffected. Code default 1.0 = legacy.
export HYZERO_TB_POLICY_WEIGHT=${HYZERO_TB_POLICY_WEIGHT:-0.0}
export HYZERO_ANTISYM_LOSS_WEIGHT=0.01
# Material shaping is OFF (SOTA alignment): rule-draws (repetition, fifty-move,
# move-cap) return their true 0.0 terminal value instead of a tanh(Δmaterial)
# surrogate. SOTA board-game agents (MuZero/AlphaZero) train on exact game
# outcomes; shaped draw labels bias the value head away from the true objective.
# HYZERO_MATERIAL_SHAPING is intentionally unset here (code default = OFF).
# Value-target mode (SOTA alignment): `outcome` propagates the full game outcome
# to every step with γ=1 (MuZero board-game convention). `td` keeps the legacy
# n-step TD path (HYZERO_TD*). Default here is outcome; override to switch back.
export HYZERO_VALUE_TARGET_MODE=${HYZERO_VALUE_TARGET_MODE:-outcome}
# Mirrored (antithetic) eval start pairs: each eval-ladder slot samples ONE
# curriculum start and plays it twice with the challenger's color swapped, so
# both games share the identical position. Reduces win_rate/candidate_elo
# variance without changing the total game count or the promotion gates. Bench
# opts in here; the code default stays OFF.
export HYZERO_EVAL_MIRRORED_STARTS=${HYZERO_EVAL_MIRRORED_STARTS:-1}
export HYZERO_LR_SCHEDULE=cosine
# Cosine T_max tracks the run length so the LR completes one full decay over the
# run instead of a fixed 14000 steps. Throughput is ~18 trainer steps/min at the
# current settings, so T_max ≈ duration_minutes * 18 = DURATION/60 * 18. Integer
# arithmetic; floored at 100 to respect the trainer's lower clamp on short runs.
LR_COSINE_T_MAX=$(( DURATION / 60 * 18 ))
[ "$LR_COSINE_T_MAX" -lt 100 ] && LR_COSINE_T_MAX=100
export HYZERO_LR_COSINE_T_MAX=$LR_COSINE_T_MAX
export HYZERO_LR_COSINE_ETA_MIN=1e-5
# Measured 2026-06-10: root Dirichlet noise is baked into the stored MCTS visit
# targets, so ε=0.25 floors replay target entropy at ~2.0 nats in drawish play
# (value≈0 in-search → 200-sim PUCT can't re-concentrate past the noise),
# flattening the distilled policy. ε=0.10 lowers that floor; selection
# temperature still provides exploration diversity. α=0.3 = AlphaZero default.
export HYZERO_DIRICHLET_EPS=${HYZERO_DIRICHLET_EPS:-0.10}
export HYZERO_DIRICHLET_ALPHA=${HYZERO_DIRICHLET_ALPHA:-0.3}
# Shorten the self-play exploration window to 12: games seeded from midgame/
# endgame FENs anneal to exploitation faster (less random walking, more decisive
# play). When unset, the self-play window falls through to the legacy
# HYZERO_TEMP_MOVES/RunConfig-default-15 chain (bit-identical to pre-knob
# behavior); eval ladder is always unaffected (it uses HYZERO_TEMP_MOVES).
export HYZERO_TEMPERATURE_MOVES=${HYZERO_TEMPERATURE_MOVES:-12}
# High-threshold material adjudication for SELF-PLAY games: an otherwise-Ongoing
# position with a white-absolute material lead >= HYZERO_SELFPLAY_ADJ_MARGIN ends
# decisively for the leading side (scored ±1 into value/TD targets, like a real
# checkmate) instead of grinding to the move cap / repetition. Margin 12 is high
# so only overwhelmingly-decided positions adjudicate. Code default is OFF; bench
# opts in here. Eval games are unaffected (they use HYZERO_EVAL_ADJUDICATE).
export HYZERO_SELFPLAY_ADJUDICATE=${HYZERO_SELFPLAY_ADJUDICATE:-1}
export HYZERO_SELFPLAY_ADJ_MARGIN=${HYZERO_SELFPLAY_ADJ_MARGIN:-12}
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
HYZERO_TB_RESCORE=$TB_RESCORE \
HYZERO_TB_WDL_PATH=$TB_WDL_PATH \
HYZERO_RESUME_FROM=$RESUME_FROM \
HYZERO_PGN_SAMPLE_RATE=${HYZERO_PGN_SAMPLE_RATE:-1.0} \
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

# Extract value-antisymmetry metric from [antisym] lines (per-train_batch probe).
# Format: [antisym] step={v} mean_sum={f} corr={f} (N={n})
# mean_sum trending toward 0 = value head approaching POV-antisymmetry. Take the
# latest line as the run's standing. Falls back to 0.0 on runs predating the field.
LAST_ANTISYM_MEAN_SUM=$(awk '/\[antisym\]/{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^mean_sum=/) { split($i, a, "="); ms = a[2] }
    }
} END{print ms+0}' "$LOG_FILE")
LAST_ANTISYM_MEAN_SUM=${LAST_ANTISYM_MEAN_SUM:-0.0}

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

# ── KQvK conversion audit ──────────────────────────────────────
# Report how often self-play converts won K+Q vs K starts into actual mates
# (vs shuffling to repetition / move-cap). Robust: if the audit script is absent
# or fails, write null into the baseline JSON — never break the score run.
KQVK_JSON="null"
if [ -f scripts/kqvk_audit.py ]; then
    echo "[4b/5] Running KQvK conversion audit..."
    if _KQVK_OUT=$(python3 scripts/kqvk_audit.py logs/selfplay_sample.pgn 2>>"$LOG_FILE") \
        && [ -n "$_KQVK_OUT" ]; then
        KQVK_JSON="$_KQVK_OUT"
        echo "  kqvk: $KQVK_JSON"
        echo "[kqvk_audit] $KQVK_JSON" >> "$LOG_FILE"
    else
        echo "  WARN: kqvk audit failed — writing null"
        echo "[kqvk_audit] FAILED (null)" >> "$LOG_FILE"
    fi
fi

# ── Endgame conversion probe (opt-in) ──────────────────────────
# After the timed self-play window, replay the run's FINAL candidate net over
# the 120 fixed won-endgame starts and count how many it drives to an actual
# checkmate (vs shuffling to a draw). Self-play probe: the SAME checkpoint plays
# both sides via the arena tool, so conversion is measured by checkmate
# terminations regardless of which side delivers mate. Adjudication is forced
# OFF for the probe process only (HYZERO_EVAL_ADJUDICATE=0) so a material lead at
# the move cap does NOT count as a conversion — only real mates do. Robust: any
# failure writes null into the baseline JSON and never breaks the score run.
# Runs AFTER the timed window (adds ~5-8 min); gate off with HYZERO_CONVERSION_PROBE=0.
CONVERSION_PROBE=${HYZERO_CONVERSION_PROBE:-1}
PROBE_JSON="null"
if [ "$CONVERSION_PROBE" != "0" ]; then
    echo "[4c/5] Running endgame conversion probe..."
    # Stable probe-starts location; data/ is untracked, so copy from the campaign
    # runs/ path at runtime (existence-guarded) if it isn't there yet.
    PROBE_STARTS="data/probe_won_starts_120.txt"
    PROBE_STARTS_SRC="runs/auto-20260610-101529/probe_won_starts_120.txt"
    if [ ! -f "$PROBE_STARTS" ] && [ -f "$PROBE_STARTS_SRC" ]; then
        cp "$PROBE_STARTS_SRC" "$PROBE_STARTS"
    fi
    # Pick the run's FINAL candidate checkpoint: newest by mtime among the
    # training candidates (model_v*.pt) and champions (best_v*.pt) the run
    # produced. Fall back to the resume checkpoint if the run wrote none.
    PROBE_CKPT=$(find checkpoints -maxdepth 1 \( -name 'model_v*.pt' -o -name 'best_v*.pt' \) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [ -z "$PROBE_CKPT" ] && PROBE_CKPT="$RESUME_FROM"
    if [ ! -f "$PROBE_STARTS" ]; then
        echo "  WARN: probe starts file missing ($PROBE_STARTS) — writing null"
    elif [ -z "$PROBE_CKPT" ] || [ ! -f "$PROBE_CKPT" ]; then
        echo "  WARN: no candidate checkpoint for probe — writing null"
    else
        echo "  probe checkpoint: $PROBE_CKPT"
        cargo build --release --bin arena 2>&1 | tail -1
        _PROBE_LOG="${LOG_DIR}/probe_${TIMESTAMP}.log"
        # 120 starts × mirrored pairs = 240 games (each start played from both
        # colors by the same net). --games 240 makes the stride sampler select
        # every one of the 120 lines exactly once as a pair. EVAL_ADJUDICATE=0 is
        # scoped to this process only via the env prefix.
        set +e
        HYZERO_EVAL_ADJUDICATE=0 target/release/arena \
            --model-a "$PROBE_CKPT" --model-b "$PROBE_CKPT" \
            --games 240 --sims 100 --concurrency 8 \
            --starts "$PROBE_STARTS" --device "$DEVICE" \
            > "$_PROBE_LOG" 2>&1
        _PROBE_RC=$?
        set -e
        if [ "$_PROBE_RC" -eq 0 ]; then
            # Each played game emits one per-game line carrying `termination=<T>`;
            # the checkmate ones are a subset. awk counts (never exits nonzero, so
            # it is safe under set -e / pipefail).
            _PROBE_GAMES=$(awk '/termination=/{n++} END{print n+0}' "$_PROBE_LOG")
            _PROBE_MATES=$(awk '/termination=checkmate/{n++} END{print n+0}' "$_PROBE_LOG")
            if [ "$_PROBE_GAMES" -gt 0 ]; then
                PROBE_JSON=$(awk -v g="$_PROBE_GAMES" -v m="$_PROBE_MATES" \
                    -v ck="$(basename "$PROBE_CKPT")" \
                    'BEGIN{printf "{\"games\": %d, \"checkmates\": %d, \"rate\": %.4f, \"checkpoint\": \"%s\"}", g, m, m/g, ck}')
                echo "  probe: $PROBE_JSON"
                echo "[conversion_probe] $PROBE_JSON" >> "$LOG_FILE"
            else
                echo "  WARN: probe produced no games — writing null"
                echo "[conversion_probe] FAILED (no games)" >> "$LOG_FILE"
            fi
        else
            echo "  WARN: probe arena run failed (rc=$_PROBE_RC) — writing null"
            echo "[conversion_probe] FAILED (rc=$_PROBE_RC)" >> "$LOG_FILE"
        fi
    fi
fi

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
        "last_antisym_mean_sum": $LAST_ANTISYM_MEAN_SUM,
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
    "conversion": $KQVK_JSON,
    "probe": $PROBE_JSON,
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
