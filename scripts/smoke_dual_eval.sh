#!/usr/bin/env bash
# Smoke test for dual-model evaluation ladder and cross-run champion persistence.
#
# Stage 1 (60s): Run selfplay with HYZERO_PROMOTION_THRESHOLD=0.0 to force a
#   promotion. This creates checkpoints/best.pt and checkpoints/best_v001.pt.
#
# Stage 2 (30s): Restart selfplay and verify that the startup log line
#   "[selfplay] Loaded champion from checkpoints/best.pt (version=N)" appears,
#   confirming that the cross-run loading path ran.
#
# Between stages: model_v*.pt checkpoints are deleted; best.pt / best_vNNN.pt
#   are preserved so stage 2 can pick them up.
#
# Usage:
#   bash scripts/smoke_dual_eval.sh
#
# Exit codes:
#   0 = both stages passed
#   1 = build failed, stage 1 promotion not found, or stage 2 load line not found

set -euo pipefail
set +m

STAGE1_DURATION=${SMOKE_STAGE1_DURATION:-180}
STAGE2_DURATION=${SMOKE_STAGE2_DURATION:-30}
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_STAGE1="${LOG_DIR}/smoke_dual_eval_stage1_${TIMESTAMP}.log"
LOG_STAGE2="${LOG_DIR}/smoke_dual_eval_stage2_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "=== Smoke Test: dual-model evaluation ladder + cross-run persistence ==="
echo "Stage 1 duration: ${STAGE1_DURATION}s (force promotion threshold=0.0)"
echo "Stage 2 duration: ${STAGE2_DURATION}s (verify cross-run champion load)"
echo ""

# ── Build ──────────────────────────────────────────────────────────────────────
echo "[1/5] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Clean state (delete model_v*.pt only; preserve best*.pt) ──────────────────
echo "[2/5] Cleaning stale model checkpoints (preserving best.pt / best_vNNN.pt)..."
mkdir -p checkpoints
# Remove transient model_vNNN.pt files but leave best.pt and best_vNNN.pt alone.
find checkpoints -maxdepth 1 -name 'model_v*.pt' -delete 2>/dev/null || true
echo "    checkpoints/best.pt present: $([ -f checkpoints/best.pt ] && echo yes || echo no)"

# ── Stage 1: force a promotion ─────────────────────────────────────────────────
echo "[3/5] Stage 1: running selfplay for ${STAGE1_DURATION}s (threshold=0.0)..."
HYZERO_GAMES=5 \
HYZERO_SIMS=40 \
HYZERO_GAMES_PER_SIDE=2 \
HYZERO_PROMOTION_THRESHOLD=0.0 \
HYZERO_EVAL_SIMS=25 \
HYZERO_CHAMPION_SCORE_WEIGHT=2.0 \
target/release/selfplay > "$LOG_STAGE1" 2>&1 &
PID=$!

sleep "$STAGE1_DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 2
kill -KILL $PID 2>/dev/null || true
set +e
wait $PID 2>/dev/null
set -e

echo "    Stage 1 log: ${LOG_STAGE1}"

if ! grep -q '\[eval\] promoted' "$LOG_STAGE1"; then
    echo "FAIL: Stage 1 — no '[eval] promoted' line found in log"
    echo ""
    echo "All eval lines (if any):"
    grep '\[eval\]' "$LOG_STAGE1" || echo "  (none)"
    echo ""
    echo "Last 30 lines of stage 1 log:"
    tail -30 "$LOG_STAGE1"
    echo ""
    echo "SMOKE TEST FAILED (stage 1)"
    exit 1
fi

echo "    Stage 1 PASS: promotion detected"
grep '\[eval\] promoted' "$LOG_STAGE1"
echo ""
echo "    Checkpoint state after stage 1:"
ls -la checkpoints/ 2>/dev/null || echo "  (no checkpoints dir)"

# ── Stage 2: restart and verify cross-run load ────────────────────────────────
echo "[4/5] Stage 2: restarting selfplay for ${STAGE2_DURATION}s (verify champion load)..."

# Clean transient model checkpoints again so stage 2 starts fresh training.
find checkpoints -maxdepth 1 -name 'model_v*.pt' -delete 2>/dev/null || true

HYZERO_GAMES=5 \
HYZERO_SIMS=40 \
HYZERO_GAMES_PER_SIDE=2 \
HYZERO_PROMOTION_THRESHOLD=0.0 \
HYZERO_EVAL_SIMS=25 \
HYZERO_CHAMPION_SCORE_WEIGHT=2.0 \
target/release/selfplay > "$LOG_STAGE2" 2>&1 &
PID2=$!

sleep "$STAGE2_DURATION"
kill -TERM $PID2 2>/dev/null || true
sleep 2
kill -KILL $PID2 2>/dev/null || true
set +e
wait $PID2 2>/dev/null
set -e

echo "    Stage 2 log: ${LOG_STAGE2}"

# ── Check results ──────────────────────────────────────────────────────────────
echo "[5/5] Verifying stage 2 startup log for cross-run champion load..."
echo ""

if grep -q '\[selfplay\] Loaded champion from checkpoints/best.pt' "$LOG_STAGE2"; then
    echo "PASS: Stage 2 — cross-run champion load detected"
    echo ""
    echo "Matching startup line:"
    grep '\[selfplay\] Loaded champion from checkpoints/best.pt' "$LOG_STAGE2"
    echo ""
    echo "Stage 1 promotion line:"
    grep '\[eval\] promoted' "$LOG_STAGE1"
    echo ""
    echo "SMOKE TEST PASSED (both stages)"
    exit 0
else
    echo "FAIL: Stage 2 — no cross-run load line found"
    echo ""
    echo "Expected: '[selfplay] Loaded champion from checkpoints/best.pt (version=N)'"
    echo "Selfplay startup lines:"
    grep '\[selfplay\]' "$LOG_STAGE2" | head -20 || echo "  (none)"
    echo ""
    echo "Last 30 lines of stage 2 log:"
    tail -30 "$LOG_STAGE2"
    echo ""
    echo "SMOKE TEST FAILED (stage 2)"
    exit 1
fi
