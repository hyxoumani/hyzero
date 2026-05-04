#!/usr/bin/env bash
# Continuous training loop. Runs run_baseline.sh repeatedly, resuming from
# checkpoints/best.pt each iteration. Runs until killed.
#
# Usage: nohup bash scripts/run_nonstop.sh > logs/nonstop_outer.log 2>&1 &
set -u

OUTER_LOG="logs/nonstop_outer.log"
mkdir -p logs

iter=0
while true; do
    iter=$((iter + 1))
    ts=$(date +%Y%m%d_%H%M%S)
    iter_log="logs/main_nonstop_iter${iter}_${ts}.log"

    echo "[nonstop iter=$iter @ $ts] starting 24h block, log=$iter_log" \
        | tee -a "$OUTER_LOG"

    if [ ! -f checkpoints/best.pt ]; then
        echo "[nonstop iter=$iter] no best.pt — falling back to mate_pretrained.pt" \
            | tee -a "$OUTER_LOG"
        RESUME=checkpoints/mate_pretrained.pt
    else
        RESUME=checkpoints/best.pt
    fi

    # Big training batch for GPU throughput, but supervision count is locked
    # absolute (HYZERO_TB_ABS_PER_BATCH) so TB/mate/midgame exposure stays
    # at the historical 115 rows/batch (was 0.45 * 256 in the prior baseline).
    # Sims raised to 800/400 to give Gumbel sequential halving more visit budget
    # per candidate (~12 sims/cand in round 1 with K=16).
    HYZERO_USE_GUMBEL=1 \
    HYZERO_SIMS=800 \
    HYZERO_EVAL_SIMS=400 \
    HYZERO_GAMES=32 \
    HYZERO_BATCH_SIZE=128 \
    HYZERO_TRAIN_BATCH_SIZE=1024 \
    HYZERO_TB_ABS_PER_BATCH=115 \
    HYZERO_RESUME_FROM="$RESUME" \
        bash scripts/run_baseline.sh 86400 > "$iter_log" 2>&1
    rc=$?

    end_ts=$(date +%Y%m%d_%H%M%S)
    score=$(python3 -c "import json; print(json.load(open('logs/baseline_score.json'))['score'])" 2>/dev/null || echo "?")
    echo "[nonstop iter=$iter @ $end_ts] block done rc=$rc score=$score" \
        | tee -a "$OUTER_LOG"

    # Brief pause before next iter so a stuck loop doesn't burn CPU instantly.
    sleep 5
done
