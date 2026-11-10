# Code Review Log

Tracked state for the scheduled "claude code review" routine. The next run reads `LAST_REVIEWED_SHA` below, reviews commits from that SHA..HEAD, and prepends a new entry under `## Reviews`.

Stored under `.claude/` rather than `docs/wiki/` because the latter is gated behind `/approve-wiki`, which a scheduled routine can't obtain.

## State

- LAST_REVIEWED_SHA: bde4f9b
- LAST_REVIEWED_UTC: 2026-11-10T16:18Z

## Reviews

### 2026-11-10 — bde4f9b (range `5f30ea8..bde4f9b`)

Focus: elo-promotion / elo-ladder-eval feature cluster, plus log/wiki refresh.

**Bugs (ranked):**

1. **[med] `src/selfplay/evaluation.rs:540-546` — champion-swap uses live challenger batcher, bypassing the dedicated champion backend.** On promotion, the new champion is set to `self.challenger_evaluator.clone()`, which is a `ChannelEvaluator` that points at the MAIN inference batcher, continuously overwritten by `weight_rx` from training. The `champion_backend` handle wired via `with_champion_backend` (field at `src/selfplay/evaluation.rs:103`, setter at `:144-150`) is stored but never read, so no actual weight hot-swap into the separate champion batcher/server ever happens.
   _Failure scenario:_ If training has not persisted a checkpoint when the first eval cycle fires (`latest_checkpoint_path` still `None`), promotion runs with `ckpt_path=None`, no `best_v{NNN}.pt` is written, and on the next cycle `latest_archive_versions()` returns empty → the bootstrap path runs again, now with `champion_store.champion()` returning the challenger's own channel evaluator. All subsequent "champion vs challenger" games become challenger-vs-itself, `win_rate` hovers around 0.5, and promotions stall indefinitely under the default 0.55 threshold. The dead `champion_backend` handle was introduced pre-`5f30ea8` but commit `a93b077` preserved it through the elo-ladder refactor, so the bootstrap-only exposure is now the primary risk.

2. **[low] `src/selfplay/evaluation.rs:526-528` — cooldown check has no bootstrap carve-out.** `cooldown_ok = total_games_since_last_promotion >= promotion_cooldown_games || promotion_cooldown_games == 0`.
   _Failure scenario:_ If a user sets `HYZERO_PROMOTION_COOLDOWN > 2 * pool_size * games_per_side`, the very first (bootstrap, no champion yet) promotion is blocked even when `candidate_elo` clears the delta, silently stalling the ladder. Not triggered by default (`cooldown=0`); no log line indicates cooldown blocked a promotion.

3. **[low] `src/bin/selfplay.rs:305-310` — `.expect(...)` on champion `InferenceServer.load_weights` panics the whole binary on an incompatible resume checkpoint.** The sibling `Err` arm at `:340-355` falls back to `RandomEvaluator` on _file-read_ failure, so this inconsistency is a plausible footgun.
   _Failure scenario:_ Any stale/incompatible `checkpoints/best.pt` in the working directory (differently-shaped model, torch format mismatch) crashes selfplay at startup.

**Clean areas (checked, nothing to flag):**

- `src/selfplay/elo.rs` — no division-by-zero, probabilities clamped, well-tested.
- `src/selfplay/pool.rs` — no panics on missing dirs or malformed filenames.
- Env-var wiring in `src/bin/selfplay.rs` — all uses `unwrap_or(default)`.
- Baseline `candidate_elo` extraction in `scripts/run_baseline.sh:210-219` — correct; `:-1500.0` fallback only matters in the `EVAL_CYCLES==0` branch (set on `:231`).
- Opponent inference-server concurrency — `load_weights` only fires between pool-member cycles after `play_game_dual().await` returns; no in-flight batch races weight swap.

**Next run should start from:** `bde4f9b`.
