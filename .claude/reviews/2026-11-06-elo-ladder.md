# Code review — 2026-11-06

**Scope**: `7b53e5d^..bde4f9b` on `main`, Elo-ladder evaluation + candidate promotion. Docs / log-refresh / cargo-fmt commits excluded.

**Verdict**: no high-confidence correctness, concurrency, or resource bugs found. Three weak/plausible findings below.

## Plausible findings

### 1. PGN game-number collisions across pool opponents — `src/selfplay/evaluation.rs:422-427, 460-465`

In the pool loop, each opponent iteration uses `game_idx+1` (and `gps+game_idx+1` on the swapped half), so for `pool_size=3, games_per_side=4`, three games all get tagged "Eval Cycle N Game 1", three get "Game 2", etc. No running counter spans pool members.

- **Impact**: any PGN consumer that keys or dedupes by `(cycle, game_num)` silently loses `pool_size - 1` out of every `pool_size` eval games per cycle.
- `scripts/compare_peak_vs_end_games.py` streams sequentially so is unaffected; `scripts/analyze_run.py` and future keyed consumers would be.
- Not a correctness bug for Elo computation or the promotion gate.

### 2. Misleading bootstrap startup NOTE — `src/bin/selfplay.rs` (~line 185)

"With pool_size=K and games_per_side=g, one cycle = 2*K*g games." On the empty-pool (bootstrap) path only `2*g` games are played per cycle, so a user sizing `promotion_cooldown_games` from this hint gets a cooldown ~K× larger than intended during bootstrap.

- Informational only, not a correctness bug.

### 3. Spurious WARN after first bootstrap promotion — `src/selfplay/evaluation.rs:277-281`

"pool empty despite champion_version=N > 0; using win-rate fallback" fires legitimately on the first cycle after the first bootstrap promotion: the just-written `best_v{N}.pt` is the only archive and `pool.rs` excludes it. The WARN text frames this as "unexpected archive deletion" even though it's the normal first-post-bootstrap transition.

- Log noise, not incorrectness.

## Verified clean

- `elo.rs` `expected_score` / `update_rating` math and signs.
- Pool enumeration (parse, exclusion, newest-first sort, k-truncation, panic-free on missing dir).
- Env-var parsing (`.ok().and_then(parse)` guarded by defaults; `set_var` in tests is serialized by Mutex).
- Promotion gate direction (`> opp_initial + delta`; bootstrap retains `>= win_rate`).
- Challenger-perspective score flip on the swapped-color half of each ladder match.
- `load_weights` checkpoint-format compatibility: `best_v{N}.pt` contains h/g/f keys that `InferenceServer.load_weights` indexes (`python/hyzero/inference/server.py:137-139`).
- Concurrency around `opp_server_handle`: eval task is the sole holder; `play_game_dual` is awaited to completion between weight swaps; GIL serializes Python calls.
- Baseline awk extractor (`scripts/run_baseline.sh`): guarded by `EVAL_CYCLES>0`, pre-feature logs score neutrally via `elo="1500.0"` substitution.
