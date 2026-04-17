# Phase 3: Training Pipeline Cleanup — Plan

Prerequisites: Phase 1 fixes landed (commits ee4aeaf, f37f719, 6cbe610, a09b09e).
Trigger: Phase 2 validation complete — i.e. the 10-hour run finished and its analysis
(via `scripts/analyze_run.py`) confirms the Phase 1 fixes broke the 85/15 adjudication
skew and unblocked promotions.

---

## 3a. Self-play PGN logging (SAFE, ~30 LOC)

**Why**: Phase 1 analysis showed eval games are near-deterministic (no Dirichlet in eval
means a fixed network produces the same moves every time). Opening diversity can only be
evaluated from **self-play games** (which have Dirichlet noise). Currently only eval games
are PGN-logged. Adding a small sampled logger for self-play gives us diversity visibility.

**Risk**: Very low. Pure logging, no training-dynamics changes. Throttle to 1% of games
to avoid disk bloat over long runs.

**Files**: `src/selfplay/game_task.rs`

Sampling logic: reuse `rand::random::<f32>() < 0.01` check at end of `play_game`, write
to `logs/selfplay_sample.pgn`. Match existing PGN format from `evaluation.rs:write_pgn_game`.

Add a `moves: Vec<String>` accumulator to `play_game` (currently only `play_game_dual`
has this). Store each move's notation at the same point action_to_notation is called.

### Subtask 3a.1: Refactor PGN writer to a shared helper

Move `write_pgn_game` from `evaluation.rs` to a new file `src/selfplay/pgn.rs` or
inline into `game_task.rs`. Signature:
```rust
pub fn write_pgn_game(
    path: &str,
    event: &str,
    white_label: &str,
    black_label: &str,
    result: &str,
    moves: &[String],
)
```

### Subtask 3a.2: Capture moves in `play_game`

Add `let mut moves: Vec<String> = Vec::new();` at top of loop. After `action_to_notation`,
push the result. Currently the notation is only built for `process_move` via the
`move_str` variable; reuse it.

### Subtask 3a.3: Throttled write at game end

At end of `play_game`, if `rand::random::<f32>() < 0.01`:
```rust
write_pgn_game(
    "logs/selfplay_sample.pgn",
    &format!("Selfplay v{model_version}"),
    "selfplay",
    "selfplay",
    match game_outcome { 1.0 => "1-0", -1.0 => "0-1", _ => "1/2-1/2" },
    &moves,
);
```

### Testing
- `cargo test` — no new tests needed; manual smoke: 60s run, check `logs/selfplay_sample.pgn` exists and has valid PGN.

---

## 3b. Plane 101 cleanup — side-to-move out-of-band (MEDIUM complexity)

**Why**: Plane 101 (absolute side-to-move) is the only asymmetric channel in the
current-player-perspective observation. Removing it would make the observation truly
colour-invariant, so the network cannot learn colour-specific policies at all.

**Risk**: Medium. Changes observation encoding (all plane indices shift), changes
`StepRecord` layout (add `white_to_move: bool`), changes `training.rs` (read it from
the new field instead of from plane 101). Requires retraining from scratch (checkpoints
with old shape will be incompatible).

**Prerequisite**: Phase 1 must be validated as fundamentally working before breaking
backward compat. If Phase 1 alone recovers 10+ score, this fix becomes lower priority.

### Files

- `src/data/types.rs` — add `white_to_move: bool` to `StepRecord`, decrement `NUM_OBS_PLANES` to 102
- `src/data/encoding.rs` — delete plane 101 emission in `encode_board`, shift plane 102 down to index 101 (halfmove clock). Update `flip_obs_planes` to match (delete the `1.0 - obs[101 * 64 + i]` block).
- `src/selfplay/game_task.rs` — in `play_game`, when pushing `StepRecord`, set `white_to_move: side_to_move == Color::White`
- `src/py/training.rs` — replace `observations[obs_base + 101 * 64]` with `steps[0].white_to_move` (reading from StepRecord via sample). Apply color-aug flip of `white_to_move` in the flip branch.
- `python/hyzero/config.py` — `input_planes: 102`
- `python/hyzero/models/representation.py` — no changes needed (reads from config)
- `scripts/run_baseline.sh` — no changes
- Tests — update shape assertions in trainer tests

### Implementation notes

Color augmentation: flipping the sample requires negating `game_outcome` AND flipping
`white_to_move`. The training math (`root_side_sign = white_to_move ? +1 : -1`) is
unchanged, just the source of the bool is different.

History planes don't need to carry "side at history k" — they're just position snapshots
encoded from the CURRENT player's perspective anyway.

### Testing

- Existing tests for encode_board and flip_obs_planes need shape updates.
- Add new test: `test_encode_board_has_no_side_plane` — verify plane 101 doesn't exist (or is clock).
- Existing test `test_value_target_outcome_blend_white_root` — update to set `white_to_move` directly instead of plane 101.

### Validation

After implementing, run a fresh 1800s baseline and compare to Phase 1 baseline. Expect:
- Identical score ± noise (if plane 101 was vestigial)
- OR slight improvement (if network was mis-using plane 101 to learn colour-specific policies)

Both outcomes are informative. A regression would indicate plane 101 carried useful
tempo signal and should be kept.

---

## 3c. Delete dead `src/selfplay/training.rs` stub (TRIVIAL, SAFE)

The stub `TrainingThread` was replaced by `PyTrainingThread` in `src/py/training.rs`.
The stub has tests but is not used by any production code.

### Files
- Delete `src/selfplay/training.rs`
- `src/selfplay/mod.rs` — remove `pub mod training;` and `pub use training::...`

### Testing
- `cargo build` — must succeed (no broken imports)
- `cargo test` — tests of `TrainingThread` go away; everything else unchanged

---

## 3d. Fix watch-channel initialization race (TRIVIAL)

The `(version_tx, version_rx) = watch::channel(1u64)` initial value is broadcast before
`load_checkpoint` restores the actual version. Observers see version=1 briefly even
when resuming from v500.

### Files
- `src/bin/selfplay.rs` — initialize channel to 0u64 OR defer creation until after load
- `src/py/training.rs` — already sends correct version after load; just need consistency

### Testing
- `cargo test` — existing resume test still passes

---

## 3e. Logarithmic value loss (EXPERIMENTAL, LOW PRIORITY)

**Why**: MSE loss scales quadratically; small errors give small gradients. Value loss
is ~30× smaller than policy loss at defaults, and the value head trains slowly. A
log-cosh loss amplifies gradients on small errors without the instability of raising
loss weight above 1.0.

**Risk**: Experimental. The value_weight=5.0 experiment regressed score from 11.63 to
4.84. This is a different mechanism (loss shape, not magnitude) but still risky.

**Only pursue if**: Phase 3a/3b analysis shows the value head is still not learning
meaningfully after Phase 1+3b fixes.

### Files
- `python/hyzero/training/trainer.py` — replace `F.mse_loss(value, target)` with `torch.log(torch.cosh(value - target)).mean()` as an env-var-gated option.

### Testing
- Ablation 1800s run with the new loss; compare to same-config MSE baseline.

---

## Priority order

1. **3a** (self-play PGN) — do first, gives us visibility for all subsequent work.
2. **Analyze Phase 2 data** (via `scripts/analyze_run.py`) — identify whether 3b is needed.
3. **3b** (plane 101) — only if Phase 1 results suggest colour-specific bias remains.
4. **3c** (dead code) — cleanup, do whenever.
5. **3d** (version race) — cleanup, do whenever.
6. **3e** (log-cosh value loss) — only if value head is visibly dead after 3a+3b.

## Commit strategy

One commit per subtask. Keep 3a separate (safe, can ship immediately). Bundle 3c and
3d into one "cleanup" commit. Keep 3b and 3e as separate standalone commits with
clear revert paths.
