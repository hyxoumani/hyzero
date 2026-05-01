# Review Findings — ee132c4 (2026-05-01)

Bug-focused review of the squashed commit "TB supervision infrastructure +
canonical MuZero backup + diverse starts" (51 files, +62k / -8k).

## Severity legend

- **HIGH**: data/training correctness; affects gradient signal.
- **MED**: silently wrong output for an edge case (rare promo, atypical script input).
- **LOW**: cosmetic, dead code, slightly misleading comment.

---

## HIGH — TB cache action encoding is misaligned with observation POV (Black-to-move)

**Files**: `python/hyzero/data/tablebase.py:257`, `:401`;
`python/hyzero/data/board_encoder.py:167-218`;
`scripts/build_tablebase_trajectory_cache.py:319,332,362`.

`encode_board_python` rank-mirrors the observation when `board.turn ==
chess.BLACK` (current-player POV, planes 0–11 = my pieces, rank-mirrored).
But `action_from_move(move, board)` returns the action in **absolute**
coordinates (`move.from_square * 64 + move.to_square` — chess.Move uses
white-absolute squares), and `encode_action_spatial(action, white_to_move=False)`
for **base actions** writes plane 0 / plane 1 directly from the un-flipped
`from_sq` / `to_sq`.

Result: for any TB row where it is Black to move, the observation has the
moving piece at the rank-mirrored square, but the action's "from-square" plane
points to the un-mirrored absolute square. The (action, observation) pair is
inconsistent with how Rust self-play feeds the dynamics network in
`src/selfplay/game_task.rs:464,474`, where `selected_action` is run through
`flip_action()` _before_ it lands in `step.action`.

Compare the Rust contract (training.rs:153–167):

> `step.action` is in current-player POV. `encode_action_spatial_for_color`
> uses `white_to_move` only for **underpromotion** rank choice; for base
> actions, the from/to are taken straight from the action index, which is
> already in POV.

The Python TB pipeline silently violates that contract for ~50% of rows.

**Affected code paths**:

- Snapshot format (`build_tb_batch`, `is_tablebase=True`): the wrong action is
  fed at step 0 only; reward/value losses at step 0 still train, but the
  dynamics step `g(h(obs_0), a_0) → (h_1, r_1)` is supervised with mismatched
  inputs. Mate-in-1 reward signal is therefore noisy.
- Trajectory format (`build_tb_batch_trajectories`, `is_tablebase=False`,
  full K-step loss + consistency): every Black-to-move step in every
  trajectory feeds the dynamics network a misaligned (action, obs) pair. This
  is the canonical-MuZero supervision path — the mismatch is exactly what TB
  supervision was supposed to fix.

**Fix sketch** (in `tablebase.py` / `board_encoder.py`):

```python
# in build_tb_batch / build_tb_batch_trajectories, before encode_action_spatial:
if not white_to_move:
    action_idx = flip_action(action_idx)   # mirror Rust's selfplay path
actions[i, k] = encode_action_spatial(action_idx, white_to_move)
```

This requires a Python `flip_action` (rank-mirror from_sq/to_sq for base, identity
for underpromo) — currently Python has no such helper; one needs to be added.
Alternatively, change `encode_action_spatial` to rank-mirror base-action
from/to squares when `white_to_move=False`. Either way add a regression test
that asserts: for a Black-to-move TB position, the obs's `my_pawn` plane and
the action's `from_sq` plane point at the same `(rank, file)`.

**Impact on reported numbers**: the 2026-04-21 TB experiment got a peak score
of 8.16 (vs β=0.3 baseline 14.51); this bug plausibly explains a chunk of the
gap, since half the supervision is internally inconsistent.

---

## MED — `inference_backend.rs:207` uses deprecated `encode_action_spatial`

**File**: `src/py/inference_backend.rs:207`.

```rust
let planes = crate::data::encode_action_spatial(action);
```

`encode_action_spatial` is the white-default wrapper kept as a deprecated
shim (encoding.rs:275–277). For base actions it makes no difference (the
encoding ignores the color flag); for **underpromotion** actions it always
chooses white's promotion ranks (6 → 7), regardless of MCTS depth.

Training, by contrast, alternates POV per step (training.rs:162). So during
inference the dynamics network sees underpromotion encodings that don't match
the per-depth POV it was trained with.

**Severity is MED, not HIGH**, because:

1. Underpromotion actions sit in the top-K=64 candidates only when the policy
   prior puts noticeable mass on them (rare).
2. The `white_to_move` flag in MCTS internal nodes is not currently tracked,
   so even fixing this requires plumbing the per-depth POV through
   `expand_leaf`. Today it is silently wrong rather than detectably wrong.

**Fix sketch**: thread a "depth parity" or "white-to-move-at-this-depth" flag
through `MCTSNode → expand_leaf_batch`, then call
`encode_action_spatial_for_color(action, pov_white)`.

---

## MED — `scripts/rebalance_tb_cache.py` only handles snapshot caches

**File**: `scripts/rebalance_tb_cache.py:60–66`.

```python
samples: list[TBSample] = [
    TBSample(
        fen=item.fen,
        target_value=float(item.target_value),    # AttributeError on TBTrajectory
        ...
```

If a user runs this on a trajectory-format cache (the new
`cache_trajectories.pkl` from `build_tablebase_trajectory_cache.py`), it will
explode with `AttributeError: 'TBTrajectory' object has no attribute 'fen'`
(or similar) because the dataclass differs. Tablebase.py's `TablebaseCache`
auto-detects format; this script doesn't.

**Fix**: detect by attribute (mirror tablebase.py's `hasattr(first, "fens")`)
and either branch on a per-step value or refuse with a clear error.

---

## LOW — Marsaglia–Tsang Dirichlet sampler has dead random draw

**File**: `src/mcts/tree.rs:189-196`.

```rust
let x: f32 = rng.random::<f32>() * 6.0 - 3.0;   // computed, never used
let u1: f32 = rng.random::<f32>();
let u2: f32 = rng.random::<f32>();
let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
...
if v <= 0.0 {
    let _ = x; // suppress unused warning
    continue;
}
```

`x` is sampled but never used; the explicit `let _ = x` is plastered to dodge
the warning. This is just a leftover. Drop the `let x = …` and the
`let _ = x` — Box-Muller via (u1, u2) → z is the actual normal sample.

Behaviour is correct, just wasteful (3 RNG calls per sample → 2).

---

## LOW — Halfmove-clock plane can exceed 1.0

**File**: `src/data/encoding.rs:113-117`,
`python/hyzero/data/board_encoder.py:124-126`.

```rust
let clock_val = board.halfmove_clock as f32 / 100.0;
```

In normal play the 50-move rule triggers a draw at 100 plies, so the value
caps at 1.0. But position-loading code paths (eg. diverse-start FENs from
`HYZERO_STARTS_FILE`) can present halfmove_clock > 100 for FENs that arose
mid-game, in which case this plane fills with > 1.0. The network has never
seen those values during normal training, so out-of-distribution.

Fix: clamp with `.min(1.0)` (both Rust and Python).

---

## LOW — `select_action` random tie-break uses `rand::rng()` per call

**File**: `src/mcts/tree.rs:600-607`.

```rust
let best_idx = if tied.len() == 1 {
    tied[0]
} else if tied.is_empty() {
    0
} else {
    use rand::Rng;
    tied[rand::rng().random_range(0..tied.len())]
};
```

`tied.is_empty()` cannot occur when `visits.len() > 0` because the max-fold
returns `f32::NEG_INFINITY` only when `visits` itself is empty (already
handled at line 574). The `else if tied.is_empty() { 0 }` branch is dead;
remove it or `unreachable!()` it.

Not a bug; just confusing dead code.

---

## LOW — Trajectory-format consistency loss trains `g` to land on `h(zero_obs)` after mate

**File**: `python/hyzero/training/trainer.py:820-836`,
`python/hyzero/data/tablebase.py:339,367-371`.

For trajectory-format TB rows, absorbing steps past the mate have
`fens[k] = None` → observation is all-zeros (zero-init, not overwritten).
The trainer's consistency loss then forces
`g(h(obs_{k-1}), a_{k-1}) ≈ h(zero_obs).detach()`. Since
`is_tablebase=False` for trajectories (intentional, per the docstring),
those rows are NOT excluded by the `cos_sim = cos_sim[~is_tb_tensor]` mask
at line 832.

This is most likely fine — the dynamics learns a stable "absorbing latent" —
but worth flagging because the comment on line 818-819 ("TB rows have zero
obs at steps 1..K, which would force the consistency target toward a
zero-latent and poison the dynamics network") suggests the author _intended_
to exclude this case but only implemented it for snapshot rows.

If absorbing-state consistency is empirically a problem, mask trajectory
rows past `mate_step` similarly (would need a separate per-row mask, since
`is_tb_tensor` already says False for trajectories).

---

## Notes / non-issues considered

- **Color-augmentation sign math** in `src/py/training.rs:172-237`: re-derived
  by hand; the `flip_sign · root_value · root_side_sign · ply_flip` decomposition
  is consistent with the obs-flip semantics. Pinned by
  `test_value_target_sign_under_flip_matches_observation_pov`.
- **Canonical MuZero backup** in `src/mcts/tree.rs:505-542`: `G_{k-1} = r_k - G_k`
  with γ=1, root POV at depth-0, parent's POV at depth-k. Test
  `test_backpropagate_includes_mating_reward` covers the mating edge case;
  zero-reward case degenerates to old behavior bit-for-bit.
- **`legal_actions.sort_unstable()` after POV flip** (game_task.rs:290,419):
  matches the rule in `.claude/rules/mcts-pov-symmetry.md`. Regression test
  `test_legal_actions_ordering_is_color_symmetric_after_sort` verifies it.
- **`flip_action_planes` invariant test** (encoding.rs:921-973): exercises
  all 4672 actions for both colors. Solid.
- **Off-by-one regression test** for dynamics action indexing
  (training.rs:699+): well-targeted; the comment explains the historical bug.

---

## Recommendation order

1. Fix the TB-cache POV mismatch (HIGH) — most likely real explanation for
   why TB supervision underperformed expectations in the 2026-04-21 / 04-22
   experiments.
2. Plumb POV through `expand_leaf` or document the inference-encoder
   mismatch (MED).
3. Auto-detect cache format in `rebalance_tb_cache.py` (MED).
4. The LOW items can be batched into a cleanup commit.
