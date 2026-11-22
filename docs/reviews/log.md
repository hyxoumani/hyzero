# Automated Code Reviews

Running log of what the scheduled bug-review routine has covered.
Each entry: date, HEAD at review time, scope, findings summary.
Next run picks up from the last `reviewed_through` commit.

---

## 2026-11-22 — elo-promotion feature

- **reviewed_through**: `df794b3` (HEAD at review: `bde4f9b`)
- **branch**: `claude/modest-rubin-z0ln8i`
- **scope**: commits `7b53e5d..df794b3` — elo ladder eval feature (elo math, archive pool, per-opponent ladder, env vars, candidate_elo extraction)
- **focus**: bugs only

### Findings

1. **Bootstrap-twice bug** _(important)_ — `src/selfplay/evaluation.rs:247-251` + `src/selfplay/pool.rs:33-35`
   After promotion #1, `latest_archive_versions` excludes `v1` and returns an empty pool, so cycle 2 silently falls back to the win-rate bootstrap gate instead of the Elo ladder. Plan says bootstrap runs exactly once; actually it runs until promotion #2 lands. Runs that only achieve one promotion never activate the Elo gate.

2. **PGN game_num collision** _(important)_ — `src/selfplay/evaluation.rs:422-427,460-465`
   Each pool opponent independently writes games `1..2*gps`, so with `pool_size=3` a single cycle produces three PGN entries tagged "Eval Cycle N Game 1" (etc.), distinguished only by player-label strings. Breaks per-game PGN indexing within a cycle.

3. **Gate operator inconsistency** _(important, low impact)_ — `src/selfplay/evaluation.rs:531,533`
   Bootstrap gate uses `win_rate >= promotion_threshold`; Elo gate uses strict `candidate_elo > opp_initial + delta`. Asymmetric treatment of the equality case.

4. **Cooldown startup notice prints wrong formula on bootstrap** _(important, cosmetic)_ — `src/bin/selfplay.rs:183-191`
   Message says "one cycle = 2 × pool_size × games_per_side games", but on the bootstrap branch a cycle is only `2 × gps` games. User tuning cooldown off the printed formula will mis-set it during bootstrap cycles.

5. **NaN `game_outcome` silently classifies as draw** _(pre_existing)_ — `src/selfplay/evaluation.rs:302-306,329-333,430-439,468-478`
   All `>0.5` / `<-0.5` comparisons are false on NaN. Unlikely path, unguarded.

### Passed

Elo math (`elo.rs`), pool enumeration (`pool.rs`), env-var parsing in `run_config_from_env`, opponent `InferenceServer` serialization (single-writer channel + await — no race), and the `candidate_elo` awk extraction in `scripts/run_baseline.sh`.
