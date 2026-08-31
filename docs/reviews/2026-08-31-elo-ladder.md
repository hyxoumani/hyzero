# Elo-Ladder Eval — Bug Review (2026-08-31)

**Range reviewed:** `aff97fb..bde4f9b` (13 commits on `claude/modest-rubin-6whske`)
**Focus:** bugs only (correctness, crashes, races, edge cases)
**Reviewer:** scheduled routine (analyst subagent)

## Summary

3 MED, 3 LOW, no HIGH. The Elo math (`src/selfplay/elo.rs`) is textbook-correct. Bugs are in the surrounding wiring: PGN tag collisions across pool opponents, stale opponent labels when a pool load fails, and integer parsing that collapses `best_v001.pt` and `best_v1.pt` to the same version.

## Findings

### MED-1 — PGN game_num collides across pool opponents

**File:** `src/selfplay/evaluation.rs:424,462`

`game_num` resets to `1..gps` per pool opponent, so games from different opponents collide on the same "Cycle N Game K" tag.

_Scenario:_ pool_size=3, gps=4, cycle=5 → PGN gets three games titled "Eval Cycle 5 Game 1" (one per opponent). Any tool keying `(cycle, game_num)` as unique loses 2/3 of games.

_Suggested fix:_ use a running counter across the per-opponent loop, or embed `opponent_version` in the round tag.

### MED-2 — opponents label built before per-opponent load failures filter

**File:** `src/selfplay/evaluation.rs:379-381`

`opponents_label` is built from the entire pool before the per-opponent loop, so failed `load_weights` / `fs::read` skips (lines 409, 388) don't remove that opponent from the printed `opponents=` list.

_Scenario:_ pool = [v9, v8, v7]; reading v8 fails → 8 games play against v9 and 8 against v7, but log prints `opponents=v9,v8,v7` and `pool_size=3`, misattributing `candidate_elo`.

_Suggested fix:_ build the label after the loop from the actually-played opponents, or emit per-opponent lines.

### MED-3 — u64 archive-version parse accepts leading zeros

**File:** `src/bin/selfplay.rs:31` and `src/selfplay/pool.rs:32`

`u64::from_str` accepts leading zeros, so `best_v001.pt` and `best_v1.pt` both hash to `version=1`. `find_latest_archive_version` silently picks whichever wins the max collision, and `pool.rs` pushes duplicate entries.

_Scenario:_ mixed-format archives on disk → pool contains two paths at version=1, one is opened twice or a real opponent is displaced from the top-k truncation.

_Suggested fix:_ reject filenames that don't match a canonical `best_v(\d+)\.pt` regex with no leading-zero group, or normalize the parsed integer back to the canonical name and reject mismatches.

### LOW-1 — dead `total_games` in handle-unset branch

**File:** `src/selfplay/evaluation.rs:355,374`

`let total_games = 0usize;` followed by `let _ = total_games;` is dead in the handle-unset branch; the value is never read into the format string.

### LOW-2 — redundant cooldown disjunct

**File:** `src/selfplay/evaluation.rs:527`

`cooldown_ok = games >= cd || cd == 0` has a redundant disjunct (`0 >= 0` already covers `cd==0`). Harmless, but noise.

### LOW-3 — misleading PROMOTION_THRESHOLD notice

**File:** `src/bin/selfplay.rs:174`

The "PROMOTION_THRESHOLD only applies to bootstrap" notice fires only when the env var is explicitly set; users on defaults (0.55) don't see it and may still expect win-rate gating post-bootstrap.

## Test coverage gaps

- No test drives the pool-nonempty branch end-to-end (`opponent_load_weights_changes_root_setup_output` is `#[ignore]`d and covers only `load_weights`, not the loop).
- No test asserts PGN `game_num` uniqueness across opponents.
- No test covers a partial-failure pool (one opponent's `fs::read` errors).

These leave all three MED findings plausibly latent.

## Cleared (no findings)

Elo formula / K / draws / numerical range; empty-pool + `champion_version>0` warn path; sequential per-opponent await ordering (no in-flight-inference race on `load_weights`); env parse fallbacks (no panics); `watch::channel` version-wait semantics; `Mutex<Py<PyAny>>` poisoning (not reachable via `?` propagation).
