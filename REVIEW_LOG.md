# Review Log

Tracks which commits/files have been reviewed in this branch, what bugs were found,
and a per-file verdict. Append-only; oldest reviews at top.

## Conventions

- **Status**: ✅ reviewed-clean | ⚠ findings | ❌ blocking-bug | ⏭ skipped (logs/data)
- **Severity**: H (high — likely bug, change-required) / M (medium — suspicious or risky) / L (low — nit / style)

---

## v1 — 2026-05-23

**Scope**: HEAD = `ee132c4` (squash of 23 commits — TB supervision + canonical MuZero backup + diverse starts).
Reviewing against `ee132c4^` = `38033d7`.

### Files in scope (HEAD only)

| File                                        | LOC Δ    | Status  | Notes                                           |
| ------------------------------------------- | -------- | ------- | ----------------------------------------------- |
| src/mcts/tree.rs                            | +247     | pending | canonical MuZero backup change                  |
| src/data/encoding.rs                        | +152     | pending | encode_action_spatial_for_color flip invariant  |
| src/py/training.rs                          | +287     | pending | terminal-reward POV, value/reward sign fixes    |
| src/selfplay/game_task.rs                   | +102     | pending | legal_actions sort + tie-break + diverse starts |
| python/hyzero/data/tablebase.py             | +417 NEW | pending | TB cache format detection                       |
| python/hyzero/data/board_encoder.py         | +250 NEW | pending | TB sample → encoded tensor                      |
| python/hyzero/training/trainer.py           | +354     | pending | TB routing + reinit + conditional β             |
| python/tests/test_tablebase.py              | +501 NEW | pending | unit tests                                      |
| python/tests/test_training.py               | +59      | pending | additions                                       |
| scripts/build_tablebase_cache.py            | +475 NEW | pending | offline cache gen                               |
| scripts/build_tablebase_trajectory_cache.py | +455 NEW | pending | K-step trajectory cache                         |
| scripts/build_starting_positions.py         | +219 NEW | pending | FEN sampling pool                               |
| scripts/rebalance_tb_cache.py               | +107 NEW | pending | label balancing                                 |
| scripts/gen_pretrain_dynamics.py            | +110 NEW | pending | (s,a,s') tuple gen                              |
| scripts/pretrain_dynamics.py                | +265 NEW | pending | dynamics-only pretrain                          |
| logs/                                       | —        | ⏭      | data artifacts, skipped                         |
| docs/wiki/, agent-memory/                   | —        | ⏭      | docs, skipped                                   |

### Files (status after review)

| File                                        | Status                                    |
| ------------------------------------------- | ----------------------------------------- |
| src/mcts/tree.rs                            | ⚠ M1                                      |
| src/data/encoding.rs                        | ✅                                        |
| src/py/training.rs                          | ⚠ L1,L2                                   |
| src/selfplay/game_task.rs                   | ✅                                        |
| python/hyzero/data/tablebase.py             | ❌ H1                                     |
| python/hyzero/data/board_encoder.py         | ✅                                        |
| python/hyzero/training/trainer.py           | ⚠ L3,L4                                   |
| python/tests/test_tablebase.py              | ⚠ M2 (test gap)                           |
| python/tests/test_training.py               | ✅                                        |
| scripts/build_tablebase_cache.py            | ❌ H1                                     |
| scripts/build_tablebase_trajectory_cache.py | ❌ H1                                     |
| scripts/pretrain_dynamics.py                | ❌ H1                                     |
| scripts/gen_pretrain_dynamics.py            | ✅                                        |
| scripts/build_starting_positions.py         | ✅ (not re-read; trivial FEN-list writer) |
| scripts/rebalance_tb_cache.py               | ✅ (not re-read; trivial label balancer)  |

### Findings

#### H1 — Tablebase pipeline stores ABSOLUTE action indices, training pipeline expects POV-flipped (HIGH)

**Files**: `python/hyzero/data/tablebase.py`, `python/hyzero/data/board_encoder.py:action_from_move`,
`scripts/build_tablebase_cache.py`, `scripts/build_tablebase_trajectory_cache.py`,
`scripts/pretrain_dynamics.py`.

**Symptom**: Black-to-move TB samples teach the network with inconsistent action targets.

**Root cause**: `action_from_move(move, board)` (board_encoder.py:223) returns the base-action
index as `from_sq * 64 + to_sq` using **absolute** chess squares — it never inspects
`board.turn` to flip for Black. Self-play replay, on the other hand, flips actions to POV
space in `src/selfplay/game_task.rs:273-290, 459-467` before storing in `StepRecord`. So:

- Replay (Rust): `step.action` is in **POV space** for both colors.
- TB (Python): `traj.actions[k]`, `optimal_actions[k]`, `legal_actions[k]` are in
  **absolute space** (no flip for Black).

`build_tb_batch_trajectories` (tablebase.py:392-401) then encodes the action plane via
`encode_action_spatial(action_idx, white_to_move)` — but `encode_action_spatial` only
uses `white_to_move` for **underpromotion rank inference**; base actions are placed at
the raw absolute (rank, file) positions encoded in the index. Meanwhile,
`encode_board_python` POV-flips the observation for Black-to-move. Result:

| Black-to-move TB step   | observation has black king at                     | action plane FROM at                     | target policy slot   |
| ----------------------- | ------------------------------------------------- | ---------------------------------------- | -------------------- |
| e8e7 example            | POV rank 0, file 4 (correct, flipped from abs 60) | abs rank 7, file 4 (uses abs index 3892) | slot 3892 (absolute) |
| What replay would store | same POV obs                                      | POV rank 0, file 4 (uses POV index 268)  | slot 268 (POV)       |

Reproducer (no python-chess needed):

```
abs action: 3892 -> from_sq=60 (rank 7, file 4), to_sq=52 (rank 6, file 4)
POV action: 268  -> from rank 0 file 4, to rank 1 file 4
```

**Blast radius**: ~50% of TB samples (those with `board.turn == BLACK`) feed the network
contradictory supervision. The kqk probe (white-to-move) is unaffected because white POV
is identity; this is consistent with the experimental result that `kqk_value` responded
to TB supervision while broader value-head metrics didn't recover as cleanly as expected.
Trajectory mode is worse: within one trajectory, plies alternate STM, so half the steps
per trajectory are corrupted.

**Fix sketch**: Either (a) flip absolute action indices to POV in `action_from_move`
when `board.turn == BLACK` (mirror Rust's behavior), or (b) flip them inside
`build_tb_batch*` after probing, before storing into the batch. Option (a) is the
single-source-of-truth fix. The Rust `flip_action` logic is already in
`src/data/encoding.rs` — mirror it as a Python helper. `pretrain_dynamics.py:60-78` has
the same defect and would be fixed by the same change.

**Recommended regression test** (`python/tests/test_tablebase.py`): for a Black-to-move
KQK position, build a TB sample and assert that the FROM square in the encoded action
plane is at the same rank as the moving piece in the observation.

#### M1 — Backup at re-visited terminal nodes uses parent-POV value as leaf-POV value (MEDIUM, rare branch)

**File**: `src/mcts/tree.rs:413-417`.

When the descent in `run_simulations` re-enters an already-expanded child whose
`legal_actions` is empty (terminal child), the code sets:

```rust
let child = parent.children[leaf_action_idx].as_ref().unwrap();
child.q_value()  // value passed to backpropagate
```

`child.q_value()` is in the **parent's POV** (per the documented storage convention),
but `backpropagate` treats `value` as `g_values[d] = G_d` in the **leaf's own POV**.
The recurrence `G_{k-1} = r_k − G_k` then propagates with a sign error from depth d−1
upward.

**Why it rarely fires**: internal MuZero nodes always have `legal_actions.len() == 64`
(top-K from policy), so `legal_actions.is_empty()` only fires when the root itself is
terminal or when the policy's top-K returns zero candidates (degenerate). In normal
self-play with `top_k=64`, this branch is effectively dead. Still, the new MuZero
backup makes the latent semantics depend on this branch being correct, so it's worth
either negating the value here or removing the branch entirely.

**Fix**: pass `-child.q_value() + child.reward` (which equals `V(child)` in child's own
POV), or simpler: pass `0.0` for terminal-absorbing leaves and let `child.reward`
carry the entire terminal signal through the new backup.

#### M2 — Black-to-move TB sample not covered by tests (MEDIUM, missing coverage)

**File**: `python/tests/test_tablebase.py`.

`test_tablebase_value_target_sign` uses Black-STM in its target value check, but no
test exercises `build_tb_batch*` with a Black-to-move FEN and verifies that the action
plane / target_policy / legal_mask are aligned with the POV-flipped observation.
Adding such a test would have caught H1 directly. Tests construct samples with
abstract action indices (42, 99, 100, 200) that don't correspond to real moves on
the FENs, masking the inconsistency.

#### L1 — env::set_var / env::remove_var calls inconsistent on `unsafe` (LOW)

**File**: `src/py/training.rs:1242-1248` wraps `std::env::set_var` in `unsafe { ... }`
but the matching `std::env::remove_var` calls at 1278-1280 are not wrapped, and the
new mirror-trajectory test at 1471-1473 calls both without `unsafe`. Edition 2021
permits both forms, so this compiles, but the styles diverge. Pick one. (Will break
if/when the crate moves to edition 2024 where these become `unsafe`.)

#### L2 — `notify_trajectory` docstring says "BEFORE add()", Rust calls AFTER add() (LOW)

**File**: `python/hyzero/training/trainer.py` ≈ line 494 (`notify_trajectory` docstring).
The Rust caller in `src/py/training.rs:482-497` runs `replay_buffer.add(trajectory)`
before `notify_trajectory`, with an in-code comment explaining the order is intentional
(so `replay_buffer.len()` is current in the subsequent log line). Update the Python
docstring to match.

#### L3 — Redundant `if k_steps > 0:` inside `if consistency_weight > 0 and k_steps > 0:` (LOW)

**File**: `python/hyzero/training/trainer.py` (consistency loss block, end of branch).
The inner guard is unreachable-when-false. Remove the inner check.

#### L4 — Reward loss normalization inconsistent across heads when TB rows present (LOW, design check)

**File**: `python/hyzero/training/trainer.py` (k>=1 unroll block).
For `k == 1` with TB rows, the reward loss uses `per_sample_rwd.mean()` (denominator B),
while policy/value at k>=1 normalize by `non_tb_count`. This is deliberate per the
comment ("TB step-1 reward carries the real mating-action signal"), but the relative
weighting of TB step-1 reward vs non-TB step-1 reward is now `1/(2 - tb_frac)` instead
of `1`. With `tb_frac=0.45`, that's a ~28% downweight of replay reward at step 1 vs
the matched policy/value normalizers. Confirm this matches the intended loss balance.
