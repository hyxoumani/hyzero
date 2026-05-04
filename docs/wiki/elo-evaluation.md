# Relative-Chain Elo Evaluation

> ⚠️ **Status: Not yet implemented.** The code described below was prototyped on 2026-05-01 in worktree `feat/elo-relative-chain` but lost when the worktree was reaped before commit. This page is preserved as the design spec for a future rebuild.

The relative-chain Elo metric would track cumulative champion strength across a single training run by converting each ladder match's score into a logit-Elo delta and summing those deltas every time a challenger is promoted. It would be anchored at zero for the v0 champion and increase monotonically with each promotion, producing a self-consistent training-curve signal without requiring external opponents (Stockfish, frozen reference, etc.). The metric is **not** a real Elo: it would be meaningless across runs that do not share v0, and a non-promoted match would contribute nothing to the running total. The number would live in memory on a new `champion_elo` field of `EvaluationTask`, would be logged once per cycle as `[eval] elo_total=… elo_delta=… score=… promoted=…`, and would be parsed by `scripts/run_baseline.sh` into `logs/baseline_score.json` under `metrics.elo` and `metrics.elo_delta_sum`.

## Key decisions

- **Relative chain over external anchor.** Anchoring against Stockfish or a frozen reference checkpoint would yield numbers that are interpretable across runs, but every existing W/D/L sample in this repo is intra-chain (champion vs. challenger only). Using the chain itself gives a self-consistent training curve for free, with the explicit cost that the value drifts and is not externally meaningful.
- **Logit-Elo (`400·log10(s/(1-s))`) instead of incremental K-factor updates.** Closed-form, no K to tune, and exactly invertible from score to delta. The match score `s = (W + 0.5·D) / N` would map to a delta in one shot per cycle; no per-game online update is needed.
- **Score clamped to `[0.001, 0.999]`.** Without clamping, a single shutout (e.g. 8-0) would map to ±∞. The clamp would bound shutouts to roughly ±1200.34 Elo per cycle, preventing one anomalous match from blowing the running total.
- **`champion_elo` updates only on promotion.** Non-promoted challengers are discarded — no future eval cycle uses their weights — so their match results carry zero information about the next champion's strength relative to the current one. Folding non-promotion deltas in would attribute Elo to weights that are never played again.
- **In-memory only, no persistence yet.** Scope cap. The full chain would be reconstructible from log replay (the per-cycle anchor line would be stable), so persistence is deferred until there is a concrete consumer that needs it across restarts.

## Gotchas

- **The number would be relative, anchored at v0 = 0.** Comparing `metrics.elo` across separate training runs would not be meaningful unless both runs start from the same v0 weights. Two runs that each climb to "Elo 400" would not necessarily reach the same playing strength.
- **The log line is a parsed contract.** Format `[eval] elo_total=<f> elo_delta=<f> score=<f> promoted=<bool>` would be consumed by `scripts/run_baseline.sh` (last `elo_total` wins, `elo_delta` values summed). Changing the prefix, the field names, or the order would silently break the JSON pipeline.
- **Promotion threshold sets the floor on per-promotion Elo gain.** At the default 0.55 promotion threshold, every promotion would contribute at least `400·log10(0.55/0.45) ≈ 34.85` Elo (the delta at exactly 55% score). Lowering the threshold would make promotions cheaper and shrink both the per-step gain and its signal-to-noise ratio.
- **Single-cycle deltas would be noisy.** With `2 × games_per_side` games per cycle (default `games_per_side = 4`, so 8 games), a single cycle's score has high binomial variance. A future implementation should advise reading the trajectory, not individual cycle deltas.
- **`elo_delta_sum` and `elo_total` should match for any complete run.** They would be computed two different ways (last-line read vs. summing applied deltas). A divergence in `baseline_score.json` between them would indicate either a parser regression or a truncated log.

## Related (intended file locations)

- `src/selfplay/evaluation.rs` — `EvaluationTask` would gain a `champion_elo` field, an `elo_delta_from_match` helper (the closed-form score→Elo conversion), promotion-gated mutation, the per-cycle log anchor line, and unit tests for the delta math (zero games, even match, 75% match, 55% match, shutout clamp).
- `scripts/run_baseline.sh` — would gain an awk parser that extracts `elo_total` (last-wins) and `elo_delta_sum` (sum) from the eval log, then writes them to JSON.
- `logs/baseline_score.json` — would gain `metrics.elo` and `metrics.elo_delta_sum` as the persisted run-level summary.

## Not yet implemented

- **Persistence across restarts.** A future `champion_elo` would reset to 0.0 every time `EvaluationTask` is constructed. A continued training session would have no way to inherit the prior chain's accumulated Elo without log replay.
- **Decay or regression-to-mean for stale chain estimates.** As the chain grows, early-cycle deltas (when the network was barely trained) would carry the same weight as late-cycle deltas. Some downweighting of stale segments may be desirable.
- **BayesElo / confidence intervals.** A point estimate has no associated uncertainty. With only 8 games per cycle, a Bayesian posterior over the per-cycle delta would be more honest than the raw logit-Elo number.
- **External anchors.** A frozen reference checkpoint replayed every N cycles, or a fixed-depth Stockfish opponent, would give a value with cross-run meaning. Both are deferred until the relative-chain metric proves useful enough to extend.
