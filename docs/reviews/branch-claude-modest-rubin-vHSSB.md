# Branch Review: `claude/modest-rubin-vHSSB`

Tracking file for code review on this branch. Bump `Last reviewed commit` after
each pass so subsequent sessions can diff only the unreviewed range.

| Field                | Value                                     |
| -------------------- | ----------------------------------------- |
| Branch               | `claude/modest-rubin-vHSSB`               |
| Last reviewed commit | `ee132c4`                                 |
| Diff base            | `4382bde` (Merge PR #2 — last main merge) |
| Reviewer             | Claude Opus 4.7 (1M context)              |
| Date                 | 2026-05-24                                |
| Focus                | bugs                                      |

## Scope

Commits reviewed (`git log 4382bde..ee132c4`):

- `7243aec` mcts+selfplay: fix color asymmetry in self-play move selection
- `38033d7` docs+wiki: document color asymmetry bug-hunt and fix (docs only — not reviewed for bugs)
- `ee132c4` train: TB supervision infrastructure + canonical MuZero backup + diverse starts

## Findings

### Bugs

**B1. Entropy bonus silently disabled on K-step unrolls when TB is active**

- File: `python/hyzero/training/trainer.py`
- Lines: `_policy_loss_per_sample` at 905-928, callsite at ~600-619
- When `HYZERO_TABLEBASE_PATH` is set, `is_tb_tensor` becomes non-None and the
  K>=1 policy loss switches to `_policy_loss_per_sample`, which does NOT apply
  the entropy bonus. The non-TB branch uses `_policy_loss`, which DOES.
- Result: setting `HYZERO_POLICY_ENTROPY_WEIGHT > 0` together with TB silently
  drops the regularizer on every latent step (still applied at step 0). The
  helper's docstring claims "entropy bonus … applied at the scalar level by
  the caller" but no caller does.
- Fix: either fold the entropy term into `_policy_loss_per_sample` and apply
  per-row with the same mask, or compute it once in the caller after the
  per-sample reduction.

**B2. Diverse starts break paired-eval fairness in the champion/challenger ladder**

- Files: `src/selfplay/evaluation.rs:188,215`, `src/selfplay/game_task.rs:245`
- `play_game_dual` is shared between self-play and the eval ladder. Once
  `HYZERO_STARTS_FILE` is configured, EVERY eval game samples an independent
  random FEN — including the 40% middlegame positions with `|Δmat| ≥ 2` and
  30% Syzygy endgames (7–12 pieces).
- Effect: the `gps` games of `challenger=W` and the `gps` games of `champion=W`
  use disjoint random FEN sets, so the win-rate comparison is no longer
  balanced. With a 0.55 promotion threshold, sampling noise can easily swamp
  real strength differences in both directions.
- Fix options: (a) skip diverse starts entirely in `play_game_dual` (cleanest —
  add an `init_eval_board()` that always uses the standard start); (b) pair the
  FENs so game-i as `challenger=W` and game-i as `champion=W` start from the
  same FEN. (a) is safer; eval semantics shouldn't depend on a self-play env var.

**B3. Adam optimizer state stale after value-head reinit**

- File: `python/hyzero/training/trainer.py`, `_reinit_value_head` at 351-388,
  invoked from `load_checkpoint` at ~1057.
- `_reinit_value_head` randomizes weights and (optionally) sets a +bias offset,
  but does not touch the optimizer's first/second-moment estimates for those
  parameters. Adam will keep applying stale momentum derived from the OLD
  weights, which can wash the bias offset out within tens of steps —
  defeating the whole point of the reinit.
- Fix: reset the optimizer state for the reinitialized parameter tensors
  (`self.optimizer.state.pop(p)` for each `p` in the value head), or rebuild
  the optimizer.

**B4. Per-step debug probe writing to `/tmp/hyzero_diag_probe.txt`**

- File: `python/hyzero/training/trainer.py:652-657`
- Every `train_batch` call opens this file in append mode, writes one line,
  flushes, closes. Leftover probe instrumentation that grows unboundedly and
  adds per-step syscalls. Production training runs will leak file size and pay
  the I/O cost forever.
- Fix: delete or gate behind an env var alongside the other periodic probes.

### Minor / polish

**M1. Per-step `[val_stats]/[reward_stats]/[policy_stats]` logging**

- `trainer.py:660-701` runs softmax/log_softmax/MSE on every batch every step
  for log printing. Useful during the current debugging campaign but worth
  gating to `% 50` like the canonical-position probes once the value-head
  collapse fix lands.

**M2. Inconsistent `unsafe { set_var }` blocks in tests**

- `src/py/training.rs:1238-1244` wraps `std::env::set_var` in `unsafe`; the
  test at `:1551` does not. The crate is edition 2021 so neither is required;
  the inconsistency is cosmetic but worth normalizing.

**M3. New `test_backpropagate_includes_mating_reward` only covers D=2**

- `src/mcts/tree.rs:1003-1085` exercises the new canonical-MuZero recurrence
  at exactly one path length. A second case at D=3 would catch sign-flip
  regressions at odd depths that the current D=2 test misses (paired by parity).

### Correctness wins worth calling out

- `select_action` random tie-break + `legal_actions.sort_unstable()` is the
  right fix and is pinned by `test_legal_actions_ordering_is_color_symmetric_after_sort`
  — empirically takes B-bias from 70% to ~44% (commit `7243aec`).
- `encode_action_spatial_for_color` now satisfies the
  `encode(flip(a), !c) == flip_planes(encode(a, c))` invariant for all 576
  underpromotion actions; preserving the all-zeros plane pattern for illegal
  file-pair slots is the right call.
- The new MuZero backup recurrence (`G_{k-1} = r_k - G_k`) reduces to the
  previous behavior bit-for-bit when all edge rewards are zero, so the existing
  zero-reward tests still pin the old sign convention.
- `_maybe_mix_tb_samples` correctly distinguishes snapshot vs trajectory caches
  by `is_tablebase` flag, so trajectory rows get full K-step + consistency
  loss while legacy snapshot rows still mask steps 1..K. Comment block at the
  consistency-loss site is misleading (says "TB rows have zero obs" — true
  only for snapshots) but the logic is right.

## Next review

If this branch picks up more commits, base the next pass at `ee132c4..HEAD` and
update the "Last reviewed commit" line above.
