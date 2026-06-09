# Elo Ladder Evaluation

The evaluation ladder measures the latest trained model (the **challenger**)
against the current champion and a pool of archived past champions, and decides
whether to promote. It is implemented and live (this replaces an earlier
"not-yet-implemented" design note). Code: `src/selfplay/elo.rs` (rating math),
`src/selfplay/evaluation.rs` (`EvaluationTask::run`), `src/selfplay/pool.rs`
(archive enumeration).

## Elo Math (`elo.rs`)

Pure functions, no state:

```rust
const INITIAL_RATING: f32 = 1500.0;   // candidate + opponents start here each cycle
const K_FACTOR:       f32 = 32.0;     // standard chess K

expected_score(r_a, r_b) = 1 / (1 + 10^((r_b − r_a) / 400))
update_rating(rating, opp_rating, score, k) = rating + k·(score − expected_score(rating, opp_rating))
```

`score` is 1.0 win / 0.5 draw / 0.0 loss from the candidate's perspective. Each
game the candidate's rating is updated against the opponent's **fixed** rating
(opponents never move from `opponent_initial_elo`).

## Configuration (`EvaluationConfig`)

| Field | Default | Env (binary) |
|-------|---------|--------------|
| `games_per_side` | 4 | `HYZERO_GAMES_PER_SIDE` |
| `promotion_threshold` | 0.55 | `HYZERO_PROMOTION_THRESHOLD` |
| `promotion_cooldown_games` | 0 | `HYZERO_PROMOTION_COOLDOWN` |
| `num_simulations` | 50 | `HYZERO_EVAL_SIMS` |
| `temperature_moves` | 15 | — |
| `champion_score_weight` | 2.0 | `HYZERO_CHAMPION_SCORE_WEIGHT` |
| `elo_k_factor` | 32.0 | `HYZERO_ELO_K_FACTOR` |
| `pool_size` | 3 | `HYZERO_POOL_SIZE` |
| `promotion_elo_delta` | 20.0 | `HYZERO_PROMOTION_ELO_DELTA` |
| `opponent_initial_elo` | 1500.0 | `HYZERO_OPPONENT_INITIAL_ELO` |
| `checkpoints_dir` | `checkpoints` | — |

A cycle plays `2 · games_per_side` games per opponent (challenger as White, then
as Black).

## Per-cycle Flow (`EvaluationTask::run`)

The task watches `model_version`; each new version triggers a cycle:

1. Read the live `champion_store.version()`.
2. Enumerate up to `pool_size` archived champions via
   `pool::latest_archive_versions(checkpoints_dir, champion_version, pool_size)` —
   newest `best_v{NNN}.pt` files, excluding the champion's own version.
3. **If the pool is empty → bootstrap (win-rate) path.** Play `2·games_per_side`
   games against the live `champion_store.champion()`. Promotion gate:
   `win_rate ≥ promotion_threshold`. This is the **only** path that can fire the
   **first** promotion; once any `best_v{NNN}.pt` archive exists, all later
   cycles route through the Elo gate. (If `champion_version > 0` but the pool is
   empty — archives deleted — it logs a WARN and still runs the fallback.)
4. **If the pool is non-empty → Elo path.** For each archived opponent: read its
   checkpoint bytes, hot-swap them into the held opponent `InferenceServer` via
   `load_weights(bytes)`, then play `2·games_per_side` games against it. After
   each game, `candidate_elo = update_rating(candidate_elo, opponent_initial_elo,
   score, k)` (opponents pinned). Promotion gate:
   `candidate_elo > opponent_initial_elo + promotion_elo_delta`.
   - If the pool is non-empty but the opponent evaluator / server handle is unset
     (`with_opponent` not called), it logs a WARN, emits a `ladder_match` line
     with zero games, and skips the ladder.

The opponent side requires `EvaluationTask::with_opponent(evaluator,
server_handle)`; the binary wires a dedicated opponent `InferenceServer` +
batcher for this (`src/bin/selfplay.rs`). This dual-model setup lets the
challenger and each archived champion run on separate batchers concurrently.

## The `ladder_match` Log Line

One structured line per cycle (parsed by `scripts/run_baseline.sh`):

```
[eval] v{challenger} cycle={c} ladder_wins={w} ladder_draws={d} ladder_losses={l} \
       win_rate={r:.3} champion_version={cv} candidate_elo={elo:.1} \
       pool_size={ps} opponents={v.., or none} pool_score={ps:.3} ladder_match
```

- `win_rate` = `(wins + 0.5·draws) / total_games`; on the pool path `pool_score`
  equals `win_rate` (the legacy field name is preserved so extractors keep
  working).
- `candidate_elo` carries the running Elo (the field the baseline composite score
  reads). On the bootstrap path it stays at `opponent_initial_elo` (1500.0).

On promotion an additional line is emitted:

```
[eval] promoted champion_version={challenger} challenger_version={challenger} \
       win_rate={r:.3} candidate_elo={elo:.1}
```

`champion_store.promote(...)` swaps the champion evaluator, bumps the version, and
archives the checkpoint (see [Champion Pool & Promotion](champion-pool-promotion.md)).

## Cooldown

`promotion_cooldown_games` counts **games** (not cycles). With `pool_size = K`
and `games_per_side = g`, one cycle is `2·K·g` games. The default 0 is a no-op.
The binary prints a NOTE clarifying this when it is set.

## Gotchas

- **The first promotion only ever comes from the win-rate fallback** (empty pool).
  After it lands `best_v001.pt`, gating switches to Elo permanently.
- **`HYZERO_PROMOTION_THRESHOLD` only governs the empty-pool path** now — the
  binary prints a NOTE if you set it. Use `HYZERO_PROMOTION_ELO_DELTA` to tune the
  Elo gate.
- **Opponents are fixed-rating per cycle**; `candidate_elo` resets to 1500 at the
  start of each cycle (it is not persisted across cycles).
- **Single-cycle Elo is noisy** (`2·pool_size·games_per_side` games). Read the
  trajectory, not a single cycle.
- `compute_candidate_elo_from_results(initial, opp_initial, k, scores)` is a
  test-only pure helper; production `run()` inlines the per-game update so it can
  log `candidate_elo` between games.

## Related

- [Champion Pool & Promotion](champion-pool-promotion.md) — archive pool, `best.pt`, version recovery
- [Baseline Scoring](baseline-scoring.md) — how `candidate_elo` feeds the composite score
- [Self-Play Coordinator](selfplay-coordinator.md) — the eval task shares one game slot
- `src/selfplay/elo.rs`, `src/selfplay/evaluation.rs`, `src/selfplay/pool.rs`
