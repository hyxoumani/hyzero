#!/usr/bin/env bash
# Campaign-3 curriculum generation: run one gen_curriculum_endgame_10k.py process
# per endgame class in parallel (each writes its own shard + report), then merge
# all shards into a single de-duplicated starts file. Structurally near-drawn
# classes (KRvKB/KRvKN) simply deliver what fits within the wall-clock budget.
#
# Usage: run_curriculum_10k.sh <outdir> <outfile> [budget_s]
set -euo pipefail

OUTDIR="${1:?outdir required}"
OUTFILE="${2:?outfile required}"
BUDGET="${3:-18000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/gen_curriculum_endgame_10k.py"
SF="/home/devs/.local/bin/stockfish"
PROBE_WON="/home/devs/workspace/hyzero/data/probe_won_starts_120.txt"
PROBE_HOLD="/home/devs/workspace/hyzero/data/probe_holdout_starts_150.txt"

mkdir -p "$OUTDIR" "$(dirname "$OUTFILE")"

# class:seed:total  (per-class totals sum to 10000; 0.40 shallow / 0.60 deep)
CELLS=(
  "KQvK:20260711:1667"
  "KRvK:20260712:1667"
  "KQvKR:20260713:1666"
  "K2RvK:20260714:1667"
  "KRvKB:20260715:1667"
  "KRvKN:20260716:1666"
)

pids=()
for cell in "${CELLS[@]}"; do
  IFS=":" read -r cls seed total <<<"$cell"
  python3 "$GEN" "$OUTDIR/shard_${cls}.txt" \
    --classes "$cls" --seed "$seed" --total "$total" \
    --shallow-frac 0.40 --probe-ms 30 --budget-s "$BUDGET" \
    --flush-every 500 --stockfish-bin "$SF" \
    --exclude "$PROBE_WON" "$PROBE_HOLD" \
    --report "$OUTDIR/report_${cls}.md" \
    >"$OUTDIR/log_${cls}.txt" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

# Merge + defensive re-dedup (board+STM key) across shards and vs probe files.
python3 - "$OUTFILE" "$PROBE_WON" "$PROBE_HOLD" "$OUTDIR"/shard_*.txt <<'PY'
import sys
out_path, probe_won, probe_hold, *shards = sys.argv[1:]

def key(fen):
    p = fen.split()
    return f"{p[0]} {p[1]}"

seen = set()
for pf in (probe_won, probe_hold):
    try:
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(key(line))
    except FileNotFoundError:
        pass

kept = 0
dup_probe = 0
dup_cross = 0
with open(out_path, "w") as out:
    for shard in shards:
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k = key(line)
                if k in seen:
                    # already-seen probe key or cross-shard dup
                    dup_cross += 1
                    continue
                seen.add(k)
                out.write(line + "\n")
                kept += 1
print(f"[merge] wrote={kept} skipped_dupes={dup_cross} out={out_path}")
PY

# Combined report.
{
  echo "# Curriculum 10k combined report"
  echo
  for cell in "${CELLS[@]}"; do
    cls="${cell%%:*}"
    echo "## $cls"
    cat "$OUTDIR/report_${cls}.md" 2>/dev/null || echo "(missing)"
    echo
  done
  echo "total_lines: $(wc -l <"$OUTFILE")"
} >"$OUTDIR/combined_report.md"

echo "[run_curriculum_10k] done fail=$fail out=$OUTFILE report=$OUTDIR/combined_report.md"
exit "$fail"
