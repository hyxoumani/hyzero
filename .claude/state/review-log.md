# Scheduled Review Log

Each firing of the scheduled review task appends an entry below. The
baseline SHA in `reviewed-baseline.txt` is updated to the HEAD reviewed
at that firing, so the next firing only inspects commits added since.

Commits that only touch `.claude/state/` are excluded from "new" commits
(the state file itself is not review-worthy).

---

## 2026-07-22 — baseline set at bde4f9b

**Scope:** first firing. No prior state; baselined against the recent
elo-promotion feature series (commits 7b5dd87..9450e38) which was the
last substantial code change before HEAD.

**Findings (5, sorted by user impact):**

1. **[HIGH — real correctness]** `scripts/run_baseline.sh:249,251` — the
   baseline SCORE formula hard-codes 1500 as the reference elo, but the
   Rust side honors `HYZERO_OPPONENT_INITIAL_ELO`. Setting the env var
   to anything other than 1500 makes the shell tank SCORE with
   `(candidate_elo − 1500) × 0.05` even when nothing regressed. Fix:
   read the env var in the script with `${HYZERO_OPPONENT_INITIAL_ELO:-1500}`
   and substitute it into the formula.

2. **[HIGH — cosmetic]** `src/selfplay/evaluation.rs:422-428, 460-466`
   — PGN "Eval Cycle N Game 1" is written once per pool opponent, so
   with pool_size=3 and games_per_side=4 the same game numbers repeat
   three times in `logs/eval_games.pgn`. No data loss; per-opponent
   traceability requires reading the White/Black tags.

3. **[MEDIUM — noise]** `src/selfplay/evaluation.rs:277-280` — when
   resuming from a pretrained checkpoint with no `best_vNNN.pt`
   archive, `starting_version=1` but the pool is empty, so
   `[eval] WARN: pool empty despite champion_version=1 > 0` fires
   every eval cycle. Not classed as ERROR by the shell's awk filter,
   but misleading log noise.

4. **[MEDIUM]** `src/bin/selfplay.rs:183-192` — startup notice claims
   "one cycle = 2·K·g games" unconditionally. On the bootstrap cycle
   (empty pool) it's actually 2·g. A user calibrating
   `promotion_cooldown_games` from the printed formula will overshoot
   on the bootstrap cycle.

5. **[HIGH code path, contingent impact]** `src/selfplay/evaluation.rs:341-376`
   — fallback path (pool nonempty but `opponent_evaluator` /
   `opponent_server_handle` unset) logs and `continue`s without
   incrementing `total_games_since_last_promotion`. Dead in production
   because `bin/selfplay.rs:530` always attaches opponents, but
   fragile if that wiring is ever changed.

**Clean paths inspected:** `src/selfplay/elo.rs` (expected_score /
update_rating numerically clean), `src/selfplay/pool.rs` (no panics on
missing dir / malformed names, sorts+truncates+excludes correctly),
opponent-server plumbing (Py<PyAny> clone_ref'd before move; Mutex
scoped inside Python::attach; load_weights sequenced between awaits),
env-var parsing (silent fallback pattern throughout), startup notices
(no credential leaks).

---
