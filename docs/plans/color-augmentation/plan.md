# Plan: Color Augmentation for Self-Play Training

## Approach

For each `TrainingSample` drawn from the replay buffer, flip a fair coin. On heads,
produce a color-swapped version by: (1) rank-mirroring and color-swapping all 103
observation planes, (2) rank-mirroring the encoded action planes, (3) remapping the
4672-element policy vectors to mirrored action indices, (4) negating `game_outcome`.
All of this happens inside `assemble_batch_arrays` in `src/py/training.rs` — no Rust
struct changes, no Python changes, no replay buffer changes.

---

## 103-Plane Map

```
Planes 0-11:   Current position pieces
  0-5:  White pieces — Pawn(0), Knight(1), Bishop(2), Rook(3), Queen(4), King(5)
  6-11: Black pieces — Pawn(6), Knight(7), Bishop(8), Rook(9), Queen(10), King(11)

Planes 12-23:  History position 1 (oldest) — same 0-5 White, 6-11 Black layout
Planes 24-35:  History position 2
...
Planes 84-95:  History position 7

Planes 96-99:  Castling rights (constant planes, all squares same value)
  96: White kingside     97: White queenside
  98: Black kingside     99: Black queenside

Plane 100:  En passant target (one-hot single square)
Plane 101:  Side to move (all-1.0 = White, all-0.0 = Black)
Plane 102:  Halfmove clock (all squares = clock/100.0)
```

Square indexing: A1=0, B1=1, …, H1=7, A2=8, …, H8=63.
`sq = rank * 8 + file` where rank 0 = rank-1 (White home) and rank 7 = rank-8 (Black home).
Planes are stored flat: `planes[plane_idx * 64 + sq]`.

---

## Flip Transformation — Observations

### Rank mirror

For each positional plane (a plane that is a bitboard, not a constant-fill plane), every
square `sq = rank * 8 + file` maps to mirrored square `sq_flip = (7 - rank) * 8 + file`.
Equivalently: `sq_flip = (63 - 7*(sq/8) + (sq%8)) = (56 - (sq & !7)) + (sq & 7)`.

In practice, for each 64-element block: read it as 8 rows of 8, reverse the row order,
write back. Or: `planes_flipped[sq] = planes[(7 - sq/8) * 8 + sq%8]` for sq in 0..64.

### Per-plane treatment

| Planes      | Contains           | Action under color flip                              |
|-------------|--------------------|------------------------------------------------------|
| 0-5 (W pcs) | White piece bboards | → become planes 6-11; rank-mirror each              |
| 6-11 (B pcs)| Black piece bboards | → become planes 0-5; rank-mirror each               |
| 12-95 (hist)| W/B piece bboards  | Same swap: within each 12-plane group, swap 0-5↔6-11; rank-mirror each 64-block |
| 96 (W K-side)| constant fill     | → becomes plane 98 (B kingside); no rank mirror needed (constant-fill plane) |
| 97 (W Q-side)| constant fill     | → becomes plane 99 (B queenside)                    |
| 98 (B K-side)| constant fill     | → becomes plane 96 (W kingside)                     |
| 99 (B Q-side)| constant fill     | → becomes plane 97 (W queenside)                    |
| 100 (EP sq) | one-hot bitboard   | rank-mirror the one-hot (ep square flips rank too)   |
| 101 (side)  | constant fill      | flip: if was all-1.0, becomes all-0.0 (and vice versa) |
| 102 (clock) | constant fill      | unchanged (halfmove clock is position-invariant)     |

### Summary of plane index remapping

```
For i in 0..8 history groups (group 0 = current, groups 1-7 = past):
  base = i * 12
  new_obs[base + pt]     = rank_mirror(old_obs[base + 6 + pt])  for pt in 0..6  (Black → White)
  new_obs[base + 6 + pt] = rank_mirror(old_obs[base + pt])      for pt in 0..6  (White → Black)

new_obs[96] = old_obs[98]   (B kingside → W kingside)
new_obs[97] = old_obs[99]   (B queenside → W queenside)
new_obs[98] = old_obs[96]   (W kingside → B kingside)
new_obs[99] = old_obs[97]   (W queenside → B queenside)
new_obs[100] = rank_mirror(old_obs[100])  (EP square flips rank)
new_obs[101] = 1.0 - old_obs[101]        (side-to-move flip; plane is constant-fill)
new_obs[102] = old_obs[102]              (halfmove clock unchanged)
```

---

## Flip Transformation — Base Actions (0..4095)

Base action encoding: `action = from_sq * 64 + to_sq`.
Under rank flip: `from_sq_flip = (7 - from_sq/8)*8 + (from_sq%8)` and similarly for to_sq.
Flipped action: `flipped = from_sq_flip * 64 + to_sq_flip`.

Helper (works for sq in 0..63):
```
fn flip_sq(sq: usize) -> usize { (7 - sq / 8) * 8 + (sq % 8) }
fn flip_base_action(a: usize) -> usize { flip_sq(a / 64) * 64 + flip_sq(a % 64) }
```

---

## Flip Transformation — Underpromotion Actions (4096..4671)

Encoding: `action = 4096 + piece_idx * 192 + from_file * 24 + to_file`
where `piece_idx` ∈ {0=Knight, 1=Bishop, 2=Rook}, `from_file` ∈ 0..7, `to_file` ∈ 0..23.

Under rank flip:
- `piece_idx` is unchanged (piece type doesn't swap)
- `from_file` is unchanged (file is unchanged under rank mirror)
- `to_file` (which encodes destination file directly as 0..7) is unchanged

However, the color flip changes which player is promoting. Under color flip, a White
underpromotion (rank 6→7) becomes a Black underpromotion (rank 1→0). The piece_idx
and file encoding are identical, so the action index is the **same number**. No
transformation is needed for underpromotion actions beyond keeping the index as-is.

Wait — this is correct because the underpromotion encoding does not embed the rank
explicitly; it only stores `from_file` and `to_file`. The `action_to_move` function
infers the rank from the `color` argument. So `flip_underpromo_action(a) = a` — the
underpromotion action index is invariant under color flip.

---

## Flip Transformation — Policy Vectors (4672 elements)

The policy vector is a distribution over 4672 action indices. Under color flip, each
probability `p[a]` must move to `p_flipped[flip_action(a)]`.

```
fn flip_action(a: usize) -> usize {
    if a < 4096 { flip_base_action(a) }
    else { a }  // underpromotion indices are invariant
}

// Apply to policy vector:
let mut flipped_policy = vec![0.0f32; 4672];
for a in 0..4672 {
    flipped_policy[flip_action(a)] = policy[a];
}
```

---

## Action Plane Flip (3 × 8 × 8 spatial encoding)

`encode_action_spatial` produces 3 planes: source one-hot, dest one-hot, promo flag.
Under color flip, source and dest squares are rank-mirrored. Promotion flag is unchanged.

```
fn flip_action_planes(planes: &[f32; 192]) -> [f32; 192] {
    let mut out = [0.0f32; 192];
    for sq in 0..64 {
        let fsq = flip_sq(sq);
        out[fsq] = planes[sq];          // plane 0: source
        out[64 + fsq] = planes[64 + sq]; // plane 1: dest
        out[128 + sq] = planes[128 + sq]; // plane 2: promo flag (constant, no flip)
    }
    out
}
```

---

## Augmentation Site — training.rs

Inside `assemble_batch_arrays` in `src/py/training.rs`, immediately after the
`for (bi, sample) in samples.iter().enumerate()` loop opens and before any array
writes, sample a boolean `apply_flip: bool` with 50% probability using `rand::random::<bool>()`.

If `apply_flip`:
1. Negate `game_outcome` (used in `outcome_in_step_perspective` computation)
2. For each `steps[k].observation`: apply the 103-plane flip in-place (or to a cloned buffer)
3. For each action written (steps[k+1].action): pass through `flip_action()` before calling `encode_action_spatial`
4. For each policy step: remap the visit_distribution entries to flipped action indices

The `root_white_to_move` sentinel is derived from `steps[0].observation.planes[101 * 64]`
after flipping — it will correctly read as Black-to-move, which is the right perspective.

---

## Simpler Alternative Assessment

**Option A: Negate outcome only (no board flip)**
Wrong. Tells the network that position X (encoded from White's perspective) has a
Black-wins outcome. Teaches side-bias, not position quality. Actively harmful.

**Option B: Flip at trajectory level in game_task.rs before replay buffer insertion**
Viable in principle, but requires changing `GameTrajectory` / `StepRecord` types or
duplicating trajectories at insertion time. Increases replay buffer memory by 2x.
Does not simplify the actual transformation — same math is needed.
Conclusion: No benefit over batch-assembly-time flip; prefer training.rs.

**Option C: Perspective normalization (encode always from side-to-move's view)**
Would require changing `encode_board` to always rotate the board based on `side_to_move`.
This is a much larger change (affects all 8 history planes, action decoding, the
`action_to_move` color argument, inference server). Also breaks backward compatibility
with existing checkpoints. High risk, high complexity.
Conclusion: Reject for this task. The batch-time augmentation is cleaner.

**Option D: Implement only base-action flip, not underpromotion**
Since underpromotion action indices are invariant under color flip (files don't change
under rank mirror), this is actually complete — underpromotion naturally doesn't need
a separate transformation. The implementation naturally handles both.

**Recommended approach**: Batch-assembly-time augmentation in `assemble_batch_arrays`.
This is the lowest-risk, smallest-diff approach.

---

## Subtasks

### 1. Add flip_sq and flip_action helpers

- **Files**: `src/data/encoding.rs`
- **Changes**: Add two pub(crate) functions:
  - `pub fn flip_sq(sq: usize) -> usize { (7 - sq / 8) * 8 + (sq % 8) }`
  - `pub fn flip_action(action: usize) -> usize` (dispatches base vs underpromo)
  - `pub fn flip_obs_planes(obs: &[f32]) -> Vec<f32>` (103*64 in, 103*64 out)
  - `pub fn flip_action_planes(planes: &[f32; 192]) -> [f32; 192]`
- **Tests**: Unit tests in `mod tests`:
  - `flip_sq(0) == 56` (A1 → A8), `flip_sq(63) == 7` (H8 → H1)
  - `flip_action(12*64+28) == flip_sq(12)*64 + flip_sq(28)` (e2e4 ↔ e7e5)
  - `flip_action(4096) == 4096` (underpromo invariant)
  - `flip_obs_planes` round-trip: `flip_obs_planes(flip_obs_planes(x)) == x`
  - Piece plane swap: White pawn at A2 (plane 0, sq 8) maps to Black pawn at A7 (plane 6, sq 48)
- **Dependencies**: none

### 2. Integrate augmentation in assemble_batch_arrays

- **Files**: `src/py/training.rs`
- **Changes**:
  - Add `use rand::Rng;` and `use crate::data::{flip_action, flip_obs_planes, flip_action_planes};`
  - At loop top: `let apply_flip = rand::rng().random::<bool>();`
  - If `apply_flip`:
    - Override `game_outcome` with `-sample.game_outcome`
    - For step 0 observation copy: use `flip_obs_planes(&steps[0].observation.planes)`
    - For each action k: apply `flip_action` before `encode_action_spatial`
    - For policy remapping: remap `visit_distribution` entries to `flip_action(legal_moves[slot])` instead of `legal_moves[slot]` directly
    - For legal_masks: remap mask bits through `flip_action`
  - The `root_white_to_move` read from plane 101 must happen AFTER the flip (so it reads the flipped plane, which is correct — after flip, the side-to-move plane reflects the flipped perspective)
- **Tests**: Integration-level assertion test in `mod tests`:
  - Build a known `TrainingSample` with a single White-wins game (outcome=1.0)
  - Force flip (extract the flip logic into a helper for testability, or use a seeded rng)
  - Assert flipped outcome = -1.0
  - Assert observation plane 101 is all-0.0 after flip (was all-1.0 before)
  - Assert flipped policy mass maps to flipped action indices
- **Dependencies**: Subtask 1

---

## Testing Strategy

1. **Unit tests** (Subtask 1): Run `cargo test data::encoding` — all flip_sq / flip_action /
   flip_obs_planes tests. Focus on round-trip identity and known-square spot checks.

2. **Integration test** (Subtask 2): `cargo test py::training` — verify the augmentation
   path produces correct negated outcomes and remapped policies.

3. **Smoke test**: Run `bash scripts/smoke_dual_eval.sh` — confirm training runs to
   completion without panic (no assertion failures, no NaN in loss).

4. **Baseline run**: `bash scripts/run_baseline.sh 1800` — compare score vs 14.51 baseline.
   Expected: score should be >= baseline (augmentation reduces bias; may see modest gain
   in policy learning from more balanced value targets). The score should not regress
   meaningfully (>3 pts) if the flip is correct.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| History planes flipped incorrectly (swap within each 12-group but forget to rank-mirror) | High — silently wrong position encoding for 7 of 8 planes | Unit test: pick a specific history bitboard, verify sq mapping |
| EP plane not rank-mirrored | Low — EP is rare; silently wrong for a few positions | Unit test: set EP plane bit, verify it moves to correct rank-flipped square |
| Underpromotion policy remapping wrong | Medium — promotion bias if wrong | Unit test: policy mass at underpromo index unchanged after flip |
| `root_white_to_move` read before flip applied | High — perspective sign inverted for all value targets | Code: read plane 101 from already-flipped buffer |
| Policy remapping builds a new vec but legal_masks still use old indices | Medium — illegal move bits at wrong positions | Code review: ensure both policy and legal_mask go through flip_action |
| Castling rights plane swap correctness | Low — 4 plane indices swapped in pairs | Unit test: WK↔BK and WQ↔BQ |
| Stale wiki | Low — neural-networks.md says 19 planes but code has 103 | Flag to context-keeper: wiki plane count is stale (19 vs 103) |

---

## Stale Wiki Note

`docs/wiki/neural-networks.md` line 11 states "Observation planes (19): ..." and all
network shape tables show `[B, 19, 8, 8]`. The actual code uses 103 planes (confirmed
by `NUM_OBS_PLANES = 103` in `src/data/types.rs` and the `encode_board` function).
The Python model `input_planes=103` in config would also contradict the wiki.
Context-keeper should update this page.
