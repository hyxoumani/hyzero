# Review of commit `ee132c4` — TB supervision + canonical MuZero backup + diverse starts

**Reviewed**: 2026-04-30
**Reviewer**: Claude (Opus)
**Scope**: ~50 files, ~60k insertions (squash of 23 commits from `autoresearch/apr13`).
**Focus**: bugs (not style, not perf).

The review prioritized the highest-risk areas: MCTS backup math, POV/color
augmentation in target construction, the new tablebase pipeline, and the
diverse-starts loader. I did not exhaustively read the docs, logs, or per-game
trace dumps in the diff.

---

## CRITICAL

### C1. Python `action_from_move` does not POV-flip — TB action IDs are in absolute coords while replay/self-play use POV-relative IDs

**File**: `python/hyzero/data/board_encoder.py:223-250`

```python
def action_from_move(move: chess.Move, board: chess.Board) -> int:
    from_sq = move.from_square   # 0-63 (ABSOLUTE)
    to_sq   = move.to_square     # 0-63 (ABSOLUTE)
    ...
    return from_sq * 64 + to_sq  # ABSOLUTE base-action id
```

Compare with Rust self-play (`src/selfplay/game_task.rs:402-409`):

```rust
let mut legal_actions: Vec<ActionIndex> = if side_to_move == Color::Black {
    raw_legal.iter().map(|&a| flip_action(a as usize) as ActionIndex).collect()
} else { raw_legal };
```

Self-play stores POV-relative action IDs (`step.action`, `step.legal_moves`,
`visit_distribution`). The Python TB pipeline stores ABSOLUTE IDs.

**Concrete divergence**: black plays e7e5.

- Self-play replay row: action id = `flip_action(52*64+36) = 12*64+28 = 796`.
- TB trajectory row: action id = `52*64+36 = 3364`.

Both rows enter the same training batch via `_maybe_mix_tb_samples`. The
network is forced to satisfy contradictory targets:

- For black-to-move positions, the policy head sees mass at index 796 from
  replay rows AND at index 3364 from TB rows — same physical move, two ids.
- The action plane fed to `g(hidden, action)` shows squares (12,28) for replay
  and (52,36) for TB — same physical move, two encodings — so the dynamics
  network is trained on inconsistent inputs.
- `legal_mask` is also at the absolute index for TB rows, so cross-entropy is
  internally consistent within TB rows — the inconsistency is across rows.

Why it isn't catastrophic in the experiments:

- Underpromotion encoding is files-only (color-independent), so underpromo
  ids are unaffected.
- For TB **snapshot** rows (`is_tablebase=True`), the trainer masks losses at
  `k≥1`, so the wrong action plane never contributes a gradient. The bug
  narrows to step-0 policy/legal-mask only.
- For TB **trajectory** rows (`is_tablebase=False`, the new format) the bug
  hits every step's policy target, every action plane, and the root legal
  mask. This is the format the commit message highlights ("score 8.16/9.03
  with TB trajectory cache") — and the fact it scored _below_ the no-TB
  β=0.3 baseline of 14.51 is consistent with this bug actively degrading
  the trajectory-format runs.

**Fix sketch** (mirrors `move_to_action` semantics + Rust's POV flip):

```python
def action_from_move(move: chess.Move, board: chess.Board) -> int:
    from_sq, to_sq = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        from_sq = (7 - from_sq // 8) * 8 + (from_sq % 8)
        to_sq   = (7 - to_sq   // 8) * 8 + (to_sq   % 8)
    promo = move.promotion
    if promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        piece_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promo]
        return NUM_BASE_ACTIONS + piece_idx * 192 + (from_sq % 8) * 24 + (to_sq % 8)
    return from_sq * 64 + to_sq
```

**Test gap**: there is no test that exercises black-to-move TB rows alongside
white-to-move replay rows on a position with a known POV-relative action id.
A targeted regression would catch this immediately. Suggested:
construct a black-to-move TB sample where `optimal_actions = [chess.Move.from_uci("e7e5")]`
and assert `target_policies[0, 796] > 0` (not `target_policies[0, 3364]`).

---

## HIGH

### H1. `select_action(0.0)` is no longer deterministic, but the existing test still asserts equality

**File**: `src/mcts/tree.rs:585-608` (production), `:701-719` (test).

The fix to break ties uniformly at random is correct (and matches
`.claude/rules/mcts-pov-symmetry.md`). But the existing test
`test_select_action_deterministic` calls `select_action(0.0)` twice in a row
and asserts the returned actions are equal. With 50 simulations, uniform prior,
Dirichlet noise, and only 5 legal actions, ties on `max_visits` are common
enough that this test is now a flake.

**Fix**: rename the test, drop the equality assertion, and instead assert that
the chosen action is among the tied-max set on a hand-built tree. Or pin a
`StdRng` seed.

### H2. `notify_trajectory` docstring contradicts the call site

**File**: `python/hyzero/training/trainer.py:498-513` vs `src/py/training.rs:485-499`.

Docstring: "Called by the Rust training loop ... **BEFORE** it is added to the
replay buffer."

Code: Rust calls `notify_trajectory` **AFTER** `self.replay_buffer.add(trajectory)`.

Not a behavioral bug today (the function only mutates a counter), but the
contract drift will mislead anyone who later wants to use buffer state inside
`notify_trajectory`. Pick one and document accordingly.

---

## MEDIUM

### M1. `init_self_play_board` discards the FEN's fullmove counter

**File**: `src/selfplay/game_task.rs:172-205`.

```rust
match board_from_fen(fen, precomputed.clone()) {
    Ok((board, side_to_move, _fullmove)) => {
        if board.result() == GameResult::Ongoing {
            return (board, side_to_move);
        }
```

The fullmove count from the FEN is discarded. `play_game()` initializes
`turn_count: usize = 0` and uses it for temperature decay
(`turn_count < config.temperature_moves` ⇒ T=1, else T≈0). When starting
from a 30-ply middlegame FEN, the first 30 _new_ moves use exploration
temperature even though the game is already deep — different behavior than
self-play from the standard position.

`board.halfmove_clock` _is_ preserved by `board_from_fen` (used by the
50-move rule), so that side is fine. The issue is purely about the
temperature schedule.

**Fix**: thread `fullmove` (or a derived ply count) through `init_self_play_board`
and seed `turn_count` with it, so temperature decays as if the game had been
played to that point. Or document explicitly that diverse-start games run at
high temperature for `temperature_moves` extra plies.

### M2. `tb_path` env var gates loading but `_tb_frac` is read unconditionally

**File**: `python/hyzero/training/trainer.py:482-496`.

`HYZERO_TABLEBASE_PATH` only controls _whether_ to load the cache — the actual
fraction comes from `HYZERO_TABLEBASE_FRAC`. If a user sets `_FRAC` without
`_PATH`, mixing is silently disabled. This is a benign footgun but worth a
warning print at startup if `_FRAC>0` and `_tb_cache is None`.

### M3. `reward_gamma_env_lock()` is used to serialize tests that mutate unrelated env vars

**File**: `src/py/training.rs` (`test_conditional_beta_decisive_uses_pure_outcome` etc.).

The test acquires `reward_gamma_env_lock()` to serialize, but actually mutates
`HYZERO_CONDITIONAL_BETA`, `HYZERO_DISABLE_COLOR_AUG`, and
`HYZERO_VALUE_OUTCOME_BETA`. Other tests that read those vars without
acquiring the same lock can race in parallel `cargo test` runs. Rust 2024's
`unsafe std::env::set_var` makes this explicit but doesn't fix the race.

**Fix**: rename the lock to a generic `env_var_test_lock` and consistently
acquire it in _every_ test that mutates _any_ `HYZERO_*` env var.

---

## LOW

### L1. `consistency_loss` divisor counts all `k_steps` even when some are skipped

**File**: `python/hyzero/training/trainer.py:835-836`.

```python
if k_steps > 0:
    consistency_loss = consistency_loss / k_steps
```

When the entire batch is TB-snapshot rows, the inner `if cos_sim.numel() > 0`
guard skips every k-iteration, leaving `consistency_loss == 0` (correct).
When some k-iterations contribute and others don't (e.g. due to NaN guards
elsewhere), this denominator is wrong. Today, `is_tb_tensor` doesn't vary
across k, so it's not a live bug — just brittle.

### L2. `build_starting_positions.py` **main** shim restoration is incomplete

**File**: `scripts/build_starting_positions.py:130-144`.

Restores `__main__` only if `_prev is not None`. `tablebase.py:140-144` does
the right thing (also `del`s the shim if `_prev is None`). Cosmetic — the
script exits immediately after.

### L3. `build_start_positions(n)` materializes `n` copies of the same FEN

**File**: `scripts/build_starting_positions.py:71-73`.

For `N=100k`, that's 30k duplicates of the standard FEN written to disk and
loaded into a `Vec<String>` in Rust. It works because Rust samples uniformly,
but it's ~2 MB of redundant data. Replacing the file format with `<weight>\t<fen>`
would be cleaner.

### L4. `cos_sim` masking concern with `is_tablebase=False` trajectory rows

**File**: `python/hyzero/training/trainer.py:830-832`.

```python
if is_tb_tensor is not None:
    cos_sim = cos_sim[~is_tb_tensor]
```

For trajectory-format TB rows (`is_tablebase=False`), this leaves them in
the consistency loss — which is correct because their step-`k` observations
are real (or zero-absorbing past mate). However, absorbing-state rows have
`obs_k = zeros`, and `h(zero_obs)` produces some learned latent. Forcing
`g(...)` to match `h(zero_obs)` for absorbing steps is canonical MuZero, so
this is fine — but it means the consistency target depends on whatever
`h` learns to produce for the all-zeros observation, which is not separately
supervised. Watch for `h(zero_obs)` drifting.

---

## NITS

### N1. `select_action` deterministic branch has unreachable empty-tied case

`src/mcts/tree.rs:600-607` handles `tied.is_empty()` returning index 0, but
that branch is unreachable: if `visits` has any entry, `max_visits` equals
some entry, which passes the `(v - max_visits).abs() < f32::EPSILON` filter.

### N2. Float-tie comparison via EPSILON for integer-valued visit counts

`src/mcts/tree.rs:597`. `visits` are `c.visit_count as f32` — integer-valued
floats below 2^24 are exact. `v == max_visits` would be equivalent and
clearer than `(v - max_visits).abs() < f32::EPSILON`.

### N3. `decode_underpromo_action` accepts `to_file ≥ 8` slots

`src/data/encoding.rs:268-...` — these "padding" slots in the underpromo
range (3*192 = 576 entries vs 3*8\*8 = 192 legal combinations) are encoded
as all-zero from/to planes plus the promotion-flag plane. The flip invariant
test asserts both colors produce the same all-zero pattern, so flipping is
consistent. But the network sees a legal-looking promotion flag with no
from/to — training those slots is pure noise. Not new, but worth flagging.

---

## What I verified positively

- `MCTSTree::backpropagate` (canonical MuZero recurrence): the
  `G_{k-1} = r_k − G_k` index math is correct. `g_values[k]` corresponds to
  `r_{k+1}` in the path-collected `rewards[k]` (0-indexed). Storage at each
  depth maps to the right POV. Zero-reward paths reduce bit-for-bit to the
  prior alternating-sign behavior. New test
  `test_backpropagate_includes_mating_reward` exercises a non-trivial reward.
- `encode_action_spatial_for_color` flip invariant: the regression test
  `test_flip_action_planes_matches_flip_action_invariant` covers all 4672
  actions for both colors.
- `assemble_batch_arrays` action encoding under `apply_flip=true`: passing
  the un-flipped `step.action` with the _flipped_ `pov_white` flag is correct,
  because action IDs are POV-relative. For base actions, the function ignores
  `pov_white` (squares are explicit); for underpromos it picks the right
  promotion ranks. The fix from `flip_action_planes(encode_action_spatial(a))`
  to `encode_action_spatial_for_color(a, !white_to_move)` actually corrects a
  prior bug for base actions (rank-mirroring was wrong for POV-relative IDs).
- `_build_trajectory` (trajectory cache builder): mate reward fires at
  `target_rewards[k+1]` where `k` is the action index of the mating move,
  matching the "transition into the mated state" convention. Absorbing-step
  padding (FEN=None, action=-1, value/reward=0) is consistent.
- TB-routing in `train_batch` (snapshot vs trajectory format): the
  `is_tb_tensor` masking correctly degenerates to no-op when all rows have
  `is_tablebase=False`, so trajectory rows participate in full K-step loss
  and consistency loss as intended.
- `test_legal_actions_ordering_is_color_symmetric_after_sort`: directly
  pins the rule from `.claude/rules/mcts-pov-symmetry.md`.

---

## Recommended next actions

1. **Fix C1 first**, then re-run the TB-trajectory experiment. If the score
   jumps materially, that's confirmation. If it doesn't, the bug was masked
   by other dynamics and the trajectory-format approach needs further work.
2. Add a regression test for C1: black-to-move TB row, optimal action e7e5,
   assert policy mass at id 796.
3. Fix H1 (rename + tighten or seed the test).
4. Decide on H2's contract and update either docstring or call-site.
