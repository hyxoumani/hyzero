# Review Log

**Reviewed-up-to:** `bde4f9b`

Scheduled "Review changes & give feedback" routine. Each entry below records a commit range reviewed and the findings. The cursor above advances when a new range is reviewed; the next run should review `<cursor>..HEAD` (or pick a scope if the branch is still clean against main).

---

## 2026-10-09 — Elo-promotion ladder stack

**Range:** `7b5dd87..bde4f9b` (9 code commits + 1 log refresh).

**Files touched:** `src/selfplay/elo.rs` (new), `src/selfplay/pool.rs` (new), `src/selfplay/evaluation.rs` (per-opponent ladder refactor), `src/bin/selfplay.rs` (env-var wiring), `scripts/run_baseline.sh` (candidate_elo in score).

**Verdict:** No critical or high-severity findings. Elo math, pool enumeration, env-var parsing, bootstrap-vs-Elo gate branching, cooldown logic, and the graceful-degradation paths all check out. Three low-severity items remain.

### Low-severity findings

1. **PGN game numbering collides across pool opponents** — `src/selfplay/evaluation.rs:422-428, 460-466`.
   With `pool_size=3` and `games_per_side=4`, each pool member restarts the numbering (`game_idx+1` for whites, `gps+game_idx+1` for blacks), so three different games per cycle share the same PGN Event tag ("Eval Cycle N Game 3"). Debugging/replay by game number is ambiguous. The bootstrap branch (single opponent) is unaffected.
   *Fix:* thread the pool-member index into the game_num, e.g. `pool_idx * 2 * gps + game_idx + 1`.

2. **Dead `let _ = total_games;` + near-duplicate format strings in the WARN branch** — `src/selfplay/evaluation.rs:355,374` vs the main emission at `506-521`.
   `total_games` is set to 0 and thrown away. The near-duplicate `ladder_match` format strings on the two paths will drift as one gets updated.
   *Fix:* factor a `format_ladder_match_line(...)` helper used by both branches.

3. **`opp_handle.lock().unwrap()` can panic on a poisoned mutex** — `src/selfplay/evaluation.rs:397, 972`.
   The code advertises "errors are logged and the opponent is skipped for that cycle" but a poisoned mutex panics instead of degrading gracefully. In production only the eval task locks the handle, so poisoning is unlikely — but the documented contract is still broken.
   *Fix:* `opp_handle.lock().unwrap_or_else(|p| p.into_inner())`.

### Pre-existing issues (noted, out of scope for this stack)

- `watch::channel(1u64)` triggers an immediate first eval cycle before training has produced a real version.
- `latest_checkpoint_path` can point to a later version than `challenger_version` because training and eval run concurrently, so `best_v{N}.pt` can contain `v{N+k}` bytes.
- `champion_backend` field held in `EvaluationTask` but never read.

### Explicitly ruled out (checked and clean)

- Elo formulas (`expected_score`, `update_rating`) match the standard; pinned by unit tests.
- Pool enumeration (`pool.rs`): missing dir, non-matching filenames, current-version exclusion, truncation, `k > len` — no panics.
- Env-var parsing in `src/bin/selfplay.rs:101-161` follows the pre-existing `.ok().and_then(parse).unwrap_or(default)` pattern.
- Cooldown logic (`evaluation.rs:526-528`), bootstrap-vs-Elo gate (`530-534`), per-cycle counter reset — all match plan.
- `run_baseline.sh` candidate_elo extraction: awk defaults to 1500.0 when the field is absent; the surrounding `[ "$EVAL_CYCLES" -gt 0 ]` guard sets `LAST_CANDIDATE_ELO=1500.0` for zero-cycle runs. No stray -75 score penalty.
- WARN-skip path when pool nonempty but handles unset: logs, emits a well-formed `ladder_match` line, `continue`s. Never triggered in production.
- Challenger-perspective sign in the opponent-as-White loop (`evaluation.rs:468`): matches the pre-existing `test_win_rate_black_side_sign` test.
- Concurrency between `load_weights` and the opponent batcher: games are awaited sequentially within the eval task, so no in-flight racing across opponent swaps.
