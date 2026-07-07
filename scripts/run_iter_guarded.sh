#!/usr/bin/env bash
#
# Guarded per-iteration training launcher — wraps scripts/run_baseline.sh with a
# heartbeat watchdog and an automatic post-training conversion probe. Born as a
# redo of the silently-dead iter-43 from campaign auto-20260702-151405 (its
# selfplay child died ~step 2416 while run_baseline.sh sat in its timed
# `sleep $DURATION` with a dead child, so the run wedged and the probe never
# ran). All per-iteration config now comes from env/args (see below); no
# iteration-specific values are hardcoded.
#
# Wraps scripts/run_baseline.sh with:
#   (a) a heartbeat WATCHDOG — if the detailed training log stops growing for
#       $STALL_LIMIT seconds, the whole run's process group is killed, the line
#       "WATCHDOG: training stalled at step X" is printed, and the script exits
#       non-zero (3).
#   (b) if run_baseline itself exits non-zero, the line
#       "TRAINING_FAILED rc=X at step Y" is printed and the script exits with
#       that code WITHOUT running the probe (a probe on a half-trained
#       from-scratch checkpoint would pollute the verdict).
#   (c) on clean training completion, the conversion probe is run on the final
#       candidate checkpoint and a machine-readable line is printed:
#           CONVERSION_RESULT: <mates>/<games> = <pct>%
#   (d) tee-free logging: run_baseline's output is redirected straight into
#       $RUN_LOG (no `tee`, so no pipe can mask an exit code); control-plane
#       lines are appended with a plain echo/append helper.
#
# Pre-registered decision rule (evaluated by the orchestrator, NOT here — this
# script only emits the raw number):
#   conversion rate >= baseline+5pp (baseline 14.2%) => MLH confirmed
#   flat                                              => MLH exhausted
#
# Usage:
#   scripts/run_iter_guarded.sh [--dry-run] [--help]
#
# Per-iteration config (env; defaults below are the iter-2 config):
#   DURATION      training window in seconds          (default 86400 = 24h)
#   RESUME_CKPT   checkpoint to resume from; EMPTY ""  (default "" = from-scratch)
#                 means train from scratch (do NOT export HYZERO_RESUME_FROM)
#   HYZERO_MLH_CAP        moves-left-head normalization cap in plies (default 30)
#   RUN_LOG       output log path            (default runs/.../iter-2_mlhcap30.log)
#   HYZERO_PROBE_GAMES    conversion probe game count            (default 240)
#   HYZERO_PROBE_STARTS   probe start positions   (default data/probe_won_starts_120.txt)
#   EXTRA_ENV     space-separated KEY=VALUE assignments exported before launch
#   Other tunables: STALL_LIMIT, POLL, HYZERO_DEVICE, and any HYZERO_* the child
#   reads (all inherited by run_baseline). MLH search bonus is intentionally
#   left UNSET (under forensic investigation).
#
# ── iter-2 documented example invocation (from-scratch, 110-plane, 24h) ──────
#   HYZERO_SELFPLAY_ADJUDICATE=1 HYZERO_SELFPLAY_ADJ_MARGIN=12 \
#   HYZERO_VALUE_HEAD=categorical HYZERO_MOVES_LEFT_HEAD=1 HYZERO_MLH_CAP=30 \
#   DURATION=86400 RESUME_CKPT= \
#   RUN_LOG=runs/auto-20260706-100435/iter-2_mlhcap30.log \
#   scripts/run_iter_guarded.sh
set -uo pipefail

# ── Arg parsing (env is the primary config surface; flags are auxiliary) ────
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $arg (see --help)" >&2; exit 64 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Configuration (per-iter; defaults = iter-2, all overridable via env) ────
DURATION=${DURATION:-86400}                                   # 24h timed window
RESUME_CKPT=${RESUME_CKPT-}          # empty => from-scratch (no resume export)
RUN_LOG=${RUN_LOG:-runs/auto-20260706-100435/iter-2_mlhcap30.log}
STALL_LIMIT=${STALL_LIMIT:-900}   # 15 min with no detailed-log growth => stalled
POLL=${POLL:-60}                  # heartbeat poll interval (s)
DEVICE=${HYZERO_DEVICE:-cuda}
PROBE_STARTS=${HYZERO_PROBE_STARTS:-data/probe_won_starts_120.txt}
PROBE_GAMES=${HYZERO_PROBE_GAMES:-240}

mkdir -p "$(dirname "$RUN_LOG")"

# tee-free logger: emit to stdout AND append to $RUN_LOG (no `tee` process).
log() { local m="[guard] $*"; echo "$m"; echo "$m" >> "$RUN_LOG"; }

# ── Preflight: resume vs from-scratch ───────────────────────────────────────
# Empty RESUME_CKPT => train from scratch (do NOT export HYZERO_RESUME_FROM;
# force HYZERO_FROM_SCRATCH=1). 110-plane nets cannot load 102-plane ckpts, so
# iter-2 runs from scratch by default.
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        log "ERROR: resume checkpoint missing: $RESUME_CKPT"
        exit 2
    fi
    export HYZERO_RESUME_FROM="$RESUME_CKPT"
    export HYZERO_FROM_SCRATCH=${HYZERO_FROM_SCRATCH:-0}
    RESUME_DISPLAY="$RESUME_CKPT"
else
    export HYZERO_FROM_SCRATCH=1
    RESUME_DISPLAY="from-scratch"
fi

# ── Training env (MLH_CAP experiment config; each var overridable) ───────────
export HYZERO_ANTISYM_LOSS_WEIGHT=${HYZERO_ANTISYM_LOSS_WEIGHT:-0.01}
export HYZERO_DIRICHLET_ALPHA=${HYZERO_DIRICHLET_ALPHA:-0.3}
export HYZERO_DIRICHLET_EPS=${HYZERO_DIRICHLET_EPS:-0.10}
export HYZERO_EVAL_MIRRORED_STARTS=${HYZERO_EVAL_MIRRORED_STARTS:-1}
export HYZERO_LR_COSINE_ETA_MIN=${HYZERO_LR_COSINE_ETA_MIN:-1e-5}
export HYZERO_LR_SCHEDULE=${HYZERO_LR_SCHEDULE:-cosine}
export HYZERO_MLH_CAP=${HYZERO_MLH_CAP:-30}
export HYZERO_MOVES_LEFT_HEAD=${HYZERO_MOVES_LEFT_HEAD:-1}
export HYZERO_POLICY_ENTROPY_WEIGHT=${HYZERO_POLICY_ENTROPY_WEIGHT:-0.0}
export HYZERO_SELFPLAY_ADJ_MARGIN=${HYZERO_SELFPLAY_ADJ_MARGIN:-12}
export HYZERO_SELFPLAY_ADJUDICATE=${HYZERO_SELFPLAY_ADJUDICATE:-1}
export HYZERO_TB_POLICY_WEIGHT=${HYZERO_TB_POLICY_WEIGHT:-0.5}
export HYZERO_TB_SUPERVISION_GRADED=${HYZERO_TB_SUPERVISION_GRADED:-1}
export HYZERO_TEMPERATURE_MOVES=${HYZERO_TEMPERATURE_MOVES:-12}
export HYZERO_VALUE_HEAD=${HYZERO_VALUE_HEAD:-categorical}
export HYZERO_VALUE_TARGET_MODE=${HYZERO_VALUE_TARGET_MODE:-outcome}
# run_baseline recomputes T_MAX = DURATION/60*45 (=64800 at 86400s).
export HYZERO_LR_COSINE_T_MAX=${HYZERO_LR_COSINE_T_MAX:-$(( DURATION / 60 * 45 ))}
export HYZERO_DEVICE="$DEVICE"
# The wrapper runs its OWN probe with the machine-readable line below; disable
# run_baseline's internal [4c/5] probe to avoid a redundant 5-8 min pass.
export HYZERO_CONVERSION_PROBE=0

# Extra per-iter env passthrough: space-separated KEY=VALUE assignments.
if [ -n "${EXTRA_ENV:-}" ]; then
    for kv in $EXTRA_ENV; do
        export "$kv"
        log "extra env: $kv"
    done
fi

# ── Dry run: print resolved config + command, then exit without launching ───
if [ "$DRY_RUN" -eq 1 ]; then
    log "DRY RUN — resolved config (no training launched):"
    log "  DURATION=${DURATION}s  RUN_LOG=$RUN_LOG  DEVICE=$DEVICE"
    log "  resume=${RESUME_DISPLAY}  HYZERO_FROM_SCRATCH=${HYZERO_FROM_SCRATCH}"
    log "  probe: ${PROBE_GAMES} games on ${PROBE_STARTS}"
    log "  command: setsid bash scripts/run_baseline.sh ${DURATION}"
    env | grep -E '^HYZERO_' | sort | while IFS= read -r line; do log "  env $line"; done
    exit 0
fi

# ── Launch training in its own process group ───────────────────────────────
START_TS=$(date +%s)
log "launching run_baseline.sh ${DURATION}s (MLH_CAP=${HYZERO_MLH_CAP}, resume=${RESUME_DISPLAY}, device=${DEVICE})"
setsid bash scripts/run_baseline.sh "$DURATION" >> "$RUN_LOG" 2>&1 &
RB_PID=$!   # setsid makes RB_PID the process-group leader (PGID == RB_PID)
log "run_baseline pid/pgid=$RB_PID; heartbeat watchdog armed (stall limit ${STALL_LIMIT}s)"

# ── Heartbeat watchdog ─────────────────────────────────────────────────────
DETAIL_LOG=""
last_size=0
last_change=$START_TS
armed=0

while kill -0 "$RB_PID" 2>/dev/null; do
    sleep "$POLL"

    # Discover the detailed selfplay log (run_baseline prints "Log: <path>";
    # fall back to the newest logs/baseline_*.log created since launch).
    if [ -z "$DETAIL_LOG" ] || [ ! -f "$DETAIL_LOG" ]; then
        DETAIL_LOG=$(grep -m1 '^Log: ' "$RUN_LOG" 2>/dev/null | awk '{print $2}')
        if [ -z "$DETAIL_LOG" ] || [ ! -f "$DETAIL_LOG" ]; then
            DETAIL_LOG=$(find logs -maxdepth 1 -name 'baseline_*.log' \
                -newermt "@$((START_TS - 5))" -printf '%T@ %p\n' 2>/dev/null \
                | sort -rn | head -1 | cut -d' ' -f2-)
        fi
        if [ -z "$DETAIL_LOG" ] || [ ! -f "$DETAIL_LOG" ]; then
            # Still in pre-run build (curriculum/TB export); grace, reset clock.
            last_change=$(date +%s)
            continue
        fi
        log "monitoring training heartbeat: $DETAIL_LOG"
    fi

    now=$(date +%s)
    cur_size=$(stat -c %s "$DETAIL_LOG" 2>/dev/null || echo 0)
    if [ "$cur_size" -gt "$last_size" ]; then
        last_size=$cur_size
        last_change=$now
        armed=1
    fi

    idle=$(( now - last_change ))
    if [ "$armed" -eq 1 ] && [ "$idle" -ge "$STALL_LIMIT" ]; then
        step=$(grep -oE '\[py_training\] step [0-9]+' "$DETAIL_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        [ -z "$step" ] && step=$(grep -oE 'step [0-9]+' "$DETAIL_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        [ -z "$step" ] && step="unknown"
        log "detected stall: no growth in $DETAIL_LOG for ${idle}s (>= ${STALL_LIMIT}s) — killing pgid $RB_PID"
        kill -TERM -"$RB_PID" 2>/dev/null || true
        sleep 5
        kill -KILL -"$RB_PID" 2>/dev/null || true
        echo "WATCHDOG: training stalled at step $step"
        echo "WATCHDOG: training stalled at step $step" >> "$RUN_LOG"
        exit 3
    fi
done

wait "$RB_PID"
RB_RC=$?
log "run_baseline exited rc=$RB_RC after $(( $(date +%s) - START_TS ))s"

# ── Propagate a training failure WITHOUT probing ────────────────────────────
# A non-zero run_baseline exit means training did not complete cleanly. Running
# the conversion probe on a half-trained (possibly from-scratch) checkpoint
# would pollute the verdict, so emit the failure line and exit with rc.
if [ "$RB_RC" -ne 0 ]; then
    step=$(grep -oE '\[py_training\] step [0-9]+' "$DETAIL_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    [ -z "$step" ] && step=$(grep -oE 'step [0-9]+' "$DETAIL_LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
    [ -z "$step" ] && step="unknown"
    log "training failed (rc=$RB_RC) — skipping conversion probe"
    echo "TRAINING_FAILED rc=$RB_RC at step $step"
    echo "TRAINING_FAILED rc=$RB_RC at step $step" >> "$RUN_LOG"
    exit "$RB_RC"
fi

# ── Conversion probe on the final candidate checkpoint ─────────────────────
PROBE_CKPT=$(find checkpoints -maxdepth 1 \( -name 'model_v*.pt' -o -name 'best_v*.pt' \) \
    -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -z "$PROBE_CKPT" ] && PROBE_CKPT="$RESUME_CKPT"
if [ ! -f "$PROBE_CKPT" ]; then
    log "ERROR: no checkpoint available for conversion probe"
    echo "CONVERSION_RESULT: FAILED (no checkpoint)"
    echo "CONVERSION_RESULT: FAILED (no checkpoint)" >> "$RUN_LOG"
    exit 4
fi
log "running conversion probe on: $PROBE_CKPT (${PROBE_GAMES} games, device=${DEVICE})"

PROBE_OUT=$(HYZERO_VALUE_HEAD=categorical HYZERO_MOVES_LEFT_HEAD=1 \
    HYZERO_PROBE_GAMES="$PROBE_GAMES" HYZERO_PROBE_STARTS="$PROBE_STARTS" \
    scripts/diagnostics/conversion_probe.sh "$PROBE_CKPT" "$DEVICE" 2>>"$RUN_LOG")
PROBE_RC=$?
echo "$PROBE_OUT" >> "$RUN_LOG"

if [ "$PROBE_RC" -ne 0 ]; then
    log "ERROR: conversion probe failed rc=$PROBE_RC"
    echo "CONVERSION_RESULT: FAILED (probe rc=$PROBE_RC)"
    echo "CONVERSION_RESULT: FAILED (probe rc=$PROBE_RC)" >> "$RUN_LOG"
    exit 5
fi

MATES=$(printf '%s\n' "$PROBE_OUT" | grep -oE '"checkmates": [0-9]+' | grep -oE '[0-9]+' | tail -1)
GAMES_N=$(printf '%s\n' "$PROBE_OUT" | grep -oE '"games": [0-9]+' | grep -oE '[0-9]+' | tail -1)
MATES=${MATES:-0}
GAMES_N=${GAMES_N:-$PROBE_GAMES}
PCT=$(awk -v m="$MATES" -v g="$GAMES_N" 'BEGIN{ if(g==0){print "0.0"} else {printf "%.1f", 100*m/g} }')
RESULT="CONVERSION_RESULT: ${MATES}/${GAMES_N} = ${PCT}%"
echo "$RESULT"
echo "$RESULT" >> "$RUN_LOG"
exit 0
