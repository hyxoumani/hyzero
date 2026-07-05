#!/usr/bin/env bash
#
# Endgame conversion probe for a frozen checkpoint.
#
# Replays a checkpoint against ITSELF over the 120 fixed won-endgame starts
# (each start played from both colors -> 240 games) via the arena tool with
# adjudication forced OFF, and counts how many games reach an actual checkmate.
# A material lead at the move cap does NOT count — only real mates. Prints a
# single JSON line: {"games", "checkmates", "rate", "checkpoint"}.
#
# This is the standalone form of the probe block in scripts/run_baseline.sh so a
# frozen checkpoint can be measured without a full baseline run.
#
# Usage:
#   scripts/diagnostics/conversion_probe.sh <checkpoint.pt> [device]
#
# Env overrides:
#   HYZERO_PROBE_STARTS   starts file (default data/probe_won_starts_120.txt,
#                         copied from the campaign runs/ path if absent)
#   HYZERO_PROBE_GAMES    total games (default 240)
#   HYZERO_PROBE_SIMS     sims per move (default 100)
#   HYZERO_VALUE_HEAD / HYZERO_MOVES_LEFT_HEAD  head config matching the ckpt
set -euo pipefail

CKPT=${1:?usage: conversion_probe.sh <checkpoint.pt> [device]}
DEVICE=${2:-${HYZERO_DEVICE:-cpu}}
PROBE_GAMES=${HYZERO_PROBE_GAMES:-240}
PROBE_SIMS=${HYZERO_PROBE_SIMS:-100}
PROBE_STARTS=${HYZERO_PROBE_STARTS:-data/probe_won_starts_120.txt}
PROBE_STARTS_SRC="runs/auto-20260610-101529/probe_won_starts_120.txt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi
if [ ! -f "$PROBE_STARTS" ] && [ -f "$PROBE_STARTS_SRC" ]; then
    cp "$PROBE_STARTS_SRC" "$PROBE_STARTS"
fi
if [ ! -f "$PROBE_STARTS" ]; then
    echo "ERROR: probe starts file missing: $PROBE_STARTS" >&2
    exit 1
fi

cargo build --release --bin arena 2>&1 | tail -1

_PROBE_LOG=$(mktemp)
trap 'rm -f "$_PROBE_LOG"' EXIT

set +e
HYZERO_EVAL_ADJUDICATE=0 target/release/arena \
    --model-a "$CKPT" --model-b "$CKPT" \
    --games "$PROBE_GAMES" --sims "$PROBE_SIMS" --concurrency 8 \
    --starts "$PROBE_STARTS" --device "$DEVICE" \
    > "$_PROBE_LOG" 2>&1
_RC=$?
set -e
if [ "$_RC" -ne 0 ]; then
    echo "ERROR: arena run failed (rc=$_RC)" >&2
    tail -5 "$_PROBE_LOG" >&2
    exit "$_RC"
fi

_GAMES=$(awk '/termination=/{n++} END{print n+0}' "$_PROBE_LOG")
_MATES=$(awk '/termination=checkmate/{n++} END{print n+0}' "$_PROBE_LOG")
if [ "$_GAMES" -eq 0 ]; then
    echo "ERROR: probe produced no games" >&2
    exit 1
fi
awk -v g="$_GAMES" -v m="$_MATES" -v ck="$(basename "$CKPT")" \
    'BEGIN{printf "{\"games\": %d, \"checkmates\": %d, \"rate\": %.4f, \"checkpoint\": \"%s\"}\n", g, m, m/g, ck}'
