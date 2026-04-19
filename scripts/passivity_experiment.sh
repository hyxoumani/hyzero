#!/usr/bin/env bash
# Short-form training experiment targeting value-head collapse hypothesis.
# Usage: bash scripts/passivity_experiment.sh <label> <duration_s> [env_var_assignments...]
# Example: bash scripts/passivity_experiment.sh e2_beta07 600 HYZERO_VALUE_OUTCOME_BETA=0.7

set -u
set +m

LABEL="${1:?label required}"
DURATION="${2:-600}"
shift 2 || true

LOG_DIR="logs/experiments"
LOG_FILE="${LOG_DIR}/${LABEL}.log"
SUM_FILE="${LOG_DIR}/${LABEL}_summary.log"
RESULTS_FILE="${LOG_DIR}/${LABEL}_results.json"

mkdir -p "$LOG_DIR"

# Clean checkpoints + logs for a fresh start
rm -f checkpoints/best*.pt checkpoints/model_v*.pt
rm -f logs/mcts_summary.log logs/mcts_trace.log logs/selfplay_sample.pgn

echo "=== Experiment: $LABEL (${DURATION}s) ==="
echo "  Env: $* HYZERO_DEVICE=cpu HYZERO_GAMES=4 HYZERO_SIMS=50 HYZERO_MCTS_TRACE=1"

# Run selfplay with the requested env overrides
env "$@" \
    HYZERO_DEVICE=cpu \
    HYZERO_GAMES=4 \
    HYZERO_SIMS=50 \
    HYZERO_MCTS_TRACE=1 \
    target/release/selfplay > "$LOG_FILE" 2>&1 &
PID=$!
echo "  PID: $PID"

sleep "$DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 3
kill -KILL $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

# Keep a copy of the summary log for this experiment
cp -f logs/mcts_summary.log "$SUM_FILE" 2>/dev/null || true
cp -f logs/selfplay_sample.pgn "${LOG_DIR}/${LABEL}_selfplay.pgn" 2>/dev/null || true

# Extract metrics via python
python3 << PYEOF
import re, json, statistics
from collections import Counter

log = open("$LOG_FILE").read()

# Training losses
loss_pat = re.compile(r"\[py_training\] step (\d+): total=([\d.]+) policy=([\d.]+) value=([\d.]+) reward=([\d.]+) consistency=([\d.]+) \(v(\d+)\)")
losses = [m.groups() for m in loss_pat.finditer(log)]
last_v = int(losses[-1][6]) if losses else 0
last_policy = float(losses[-1][2]) if losses else float('nan')
last_value  = float(losses[-1][3]) if losses else float('nan')
first_policy = float(losses[0][2]) if losses else float('nan')
first_value  = float(losses[0][3]) if losses else float('nan')
n_steps = len(losses)

# Games
game_pat = re.compile(r"Game received: (\d+) steps")
game_lens = [int(m.group(1)) for m in game_pat.finditer(log)]
avg_len = statistics.mean(game_lens) if game_lens else 0

# Per-game outcome trace (added in this session)
outcome_pat = re.compile(r"\[game_outcome\] v=(\d+) len=(\d+) outcome=(-?[\d.]+) is_draw=(true|false)")
go = [(int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)=="true") for m in outcome_pat.finditer(log)]
n_games = len(go) if go else 0
# is_draw=true simply means "not a checkmate" — it includes material-tanh outcomes.
# Use |outcome|>0.5 as the operational decisive threshold (strong material advantage).
n_decisive = sum(1 for _,_,o,_ in go if abs(o) > 0.5) if go else 0
n_draws = n_games - n_decisive
decisive_ratio = (n_decisive/n_games) if n_games else 0.0
avg_abs_outcome = statistics.mean(abs(o) for _,_,o,_ in go) if go else 0.0
n_checkmates = sum(1 for _,_,_,d in go if not d) if go else 0
white_wins = sum(1 for _,_,o,_ in go if o > 0.5)
black_wins = sum(1 for _,_,o,_ in go if o < -0.5)

# Eval results
eval_pat = re.compile(r"\[eval\] v(\d+) cycle=(\d+) ladder_wins=(\d+) ladder_draws=(\d+) ladder_losses=(\d+) win_rate=([\d.]+)")
evals = [m.groups() for m in eval_pat.finditer(log)]

# Decisive ratio from self-play sample pgn (if any)
# Check outcomes from selfplay_sample pgn instead
pgn_txt = ""
try: pgn_txt = open("logs/selfplay_sample.pgn").read()
except: pass
pgn_results = re.findall(r'\[Result "([^"]+)"\]', pgn_txt)
# Only consider PGN entries written during THIS experiment (approximate: count all, or count by timing)
# We don't have per-experiment timestamps here; leave as-is
pgn_counter = Counter(pgn_results)

# MCTS summary aggregates
top_p_vals, entropy_vals, nvis_vals = [], [], []
try:
    with open("$SUM_FILE") as f:
        mcts_pat = re.compile(r"top_p=([\d.]+).*n_visited=(\d+) entropy=([\d.]+)")
        for line in f:
            m = mcts_pat.search(line)
            if m:
                top_p_vals.append(float(m.group(1)))
                nvis_vals.append(int(m.group(2)))
                entropy_vals.append(float(m.group(3)))
except: pass

# Promotions
promotions = len(re.findall(r"\[eval\].*promoted", log))

# Training score
SCORE_WEIGHT = float("${HYZERO_CHAMPION_SCORE_WEIGHT:-2.0}")
score = (8.55 - last_policy) + (promotions * SCORE_WEIGHT) - (avg_len / 100) if not (last_policy != last_policy) else None

result = {
    "label": "$LABEL",
    "duration_s": $DURATION,
    "last_version": last_v,
    "n_train_steps": n_steps,
    "policy_loss": {"first": first_policy, "last": last_policy},
    "value_loss": {"first": first_value, "last": last_value},
    "games_total": len(game_lens),
    "avg_game_length": round(avg_len, 2),
    "n_outcomes": n_games,
    "decisive_ratio": round(decisive_ratio, 4),
    "white_wins": white_wins,
    "black_wins": black_wins,
    "n_checkmates": n_checkmates,
    "avg_abs_outcome": round(avg_abs_outcome, 4),
    "eval_cycles": len(evals),
    "promotions": promotions,
    "pgn_outcomes": dict(pgn_counter),
    "mcts_n_calls": len(top_p_vals),
    "mcts_top_p_mean": round(statistics.mean(top_p_vals), 4) if top_p_vals else None,
    "mcts_top_p_p95": round(sorted(top_p_vals)[int(0.95*len(top_p_vals))], 4) if top_p_vals else None,
    "mcts_entropy_mean": round(statistics.mean(entropy_vals), 4) if entropy_vals else None,
    "mcts_nvisited_mean": round(statistics.mean(nvis_vals), 2) if nvis_vals else None,
    "training_score": round(score, 4) if score is not None else None,
}
with open("$RESULTS_FILE", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
PYEOF
