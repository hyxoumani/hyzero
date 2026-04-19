# Passivity Trap — Experimental Investigation

**Hypothesis (reframed 2026-04-18)**: Visit-level MCTS is NOT collapsed (median
top_p=0.10, avg n_visited=22 across 5k+ calls). The real failure mode is the
**value head collapsing to ~0** under drawn self-play, which removes the only
corrective force in PUCT, letting passive prior biases compound.

## Evidence from the fresh-run baseline (23 min, v1→v29)

| metric | first | last |
|---|---|---|
| policy_loss | 7.35 | 4.19 |
| value_loss | 0.08 | **0.02** |
| consistency_loss | 0.76 | 0.21 |

Value loss **drops to 0.02** — the variance of value targets is near zero
because drawn outcomes dominate, so the head learns nothing useful.

Self-play outcomes over 24 sampled games: 13 draws (54%), 10 black wins,
1 white win — **huge white-passive asymmetry**.

## Experiment grid (10 min each, default env besides listed)

- **e1_control**: β=0.3, defaults.
- **e2_beta07**: β=0.7 — reduce bootstrap, amplify outcome signal.
- **e3_entropy**: HYZERO_POLICY_ENTROPY_WEIGHT=0.02 — prevent policy narrowing.
- **e4_value_w3**: HYZERO_VALUE_LOSS_WEIGHT=3.0 — force value head to matter.
- **e5_combo**: β=0.7 + VALUE_LOSS_WEIGHT=3.0.
- **e6_gamma**: HYZERO_REWARD_OUTCOME_GAMMA=0.5 — also weight reward targets on outcome.
- **e7_extreme**: β=0.9, VALUE_LOSS_WEIGHT=5.0, ENTROPY_WEIGHT=0.02 combined.

## Success criteria

- value_loss stays ≥ 0.1 at end (non-trivial learning signal).
- decisive_ratio ≥ 30% (from [game_outcome] traces).
- policy_loss continues decreasing monotonically.
- no runtime errors or game-length blowup.

## Metrics extracted per experiment

From the new `[game_outcome] v=<v> len=<l> outcome=<o> is_draw=<d>` trace line:
- n_outcomes, decisive_ratio, white_wins, black_wins, avg_abs_outcome.

From `logs/mcts_summary.log`:
- mcts_top_p_mean, mcts_entropy_mean, mcts_nvisited_mean.

From training logs: first/last policy_loss, first/last value_loss, n_train_steps,
last_version, games_total, avg_game_length.
