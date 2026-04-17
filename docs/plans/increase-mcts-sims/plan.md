# Plan: Increase MCTS Simulations (50 → 100)

## Hypothesis

`decisive_ratio` is the dominant term in the training score metric (weight ×10). It
collapsed in the recency-replay experiment (0.30 → 0.10) when the value head lost
diverse outcome signal. The root cause is that value targets in training come solely from
`step.root_value` — the MCTS Q-estimate at that position — NOT from bootstrapped game
outcomes. With only 50 simulations, Q estimates in the early network are noisy and
near-zero, providing weak gradient signal to the value head. The policy consequently
learns to avoid losing but not to win, producing drawish games.

Doubling simulations (50 → 100) improves the Q-value estimates used both during
self-play (better move selection → more decisive games directly) and in training
(better `root_value` targets → stronger value head gradient signal). Unlike changes to
the training distribution (recency replay) or loss weights, this change is architecturally
safe: it increases search quality without touching the optimizer, the model, or the
replay buffer distribution.

**Expected delta**: +0.5 to +1.0 on the training score.
- decisive_ratio: 0.30 → ~0.45–0.50 (+0.15–0.20, contributes +1.5–2.0 to score)
- avg_game_length: 182 → ~190–210 (2x sims per move will make some games longer; -0.1–0.3)
- policy_loss: minimal change (~3.96; better value targets may improve slightly)
- Games in 900s: fewer (2x sims → ~half the games), but training quality per game is higher

**Net expected score**: 5.7646 + ~0.8 = ~6.5 (conservative); up to ~7.5 (optimistic)

## Tradeoffs

| Factor | Impact |
|--------|--------|
| Search quality (decisive games) | + direct improvement |
| Value targets for training | + better Q estimates per step |
| Games completed in 900s | - roughly halved |
| avg_game_length | - slightly longer games |
| Risk level | near-zero (single env-var change) |

The games-completed tradeoff is acceptable: fewer games of higher quality is better than
many games of random-policy quality, especially given `train_steps_per_game = 8` means
each game already generates 8 gradient steps regardless of game count.

## Fallback

If decisive_ratio does not improve by +0.05 vs baseline (5.7646), or if policy_loss
regresses above 4.2 (recency-replay level), mark as discard. The next candidate would be
value head scaling (increase value loss weight 10x on value_loss alone, not reward_loss,
to strengthen outcome signal without the distribution-collapse risk of recency weighting).

## Subtasks

### 1. Change default MCTS simulation count

- **Files**: `src/bin/selfplay.rs`
- **Changes**: Change `num_simulations: 50` to `num_simulations: 100` and
  `eval_num_simulations: 50` to `eval_num_simulations: 100` in `RunConfig::default()`.
  The env-var overrides (`HYZERO_SIMS`, `HYZERO_EVAL_SIMS`) remain unchanged so the
  change is reversible without a code edit.
- **Tests**: No new tests required — existing `test_play_game_completes` uses
  `num_simulations: 2` directly; `RunConfig` is not independently tested. Verify via
  baseline run.
- **Dependencies**: none

## Testing Strategy

Run `bash scripts/run_baseline.sh 900`. Success criteria:
1. `decisive_ratio` in last `[eval]` line >= 0.40 (vs 0.30 baseline)
2. `policy_loss` <= 4.1 (no regression)
3. `avg_game_length` <= 250 (not runaway)
4. No errors in log

Compare against baseline entry `46c3d0d` (score 5.7646). Record result in `results.tsv`
with label `keep` or `discard`.
