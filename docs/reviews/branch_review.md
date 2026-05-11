# Branch Review Log

Tracks code review findings on the `claude/modest-rubin-BYqpI` branch
(currently aligned with `origin/main` at commit `ee132c4`).

Each entry records the latest reviewed commit, reviewer, and bugs found.
On the next review, start from the commit after `last_reviewed`.

---

## Review 1 — 2026-05-11

- **Reviewer**: Claude (Opus 4.7)
- **Scope**: focused bug review of `ee132c4` (squash of `autoresearch/apr13`)
- **last_reviewed**: `ee132c4ec4dc32f61d688af6f776a0424e56e2d3`

### Findings (severity ordered)

#### B1 — TB action coordinate system mismatch (HIGH)

Locations:

- `python/hyzero/data/tablebase.py:257` (`build_tb_batch`)
- `python/hyzero/data/tablebase.py:401` (`build_tb_batch_trajectories`)
- `python/hyzero/data/tablebase.py:264-273, 282-285, 379-390, 404-407`
  (policy / legal-mask population)
- `scripts/build_tablebase_trajectory_cache.py:319, 321, 332, 362, 364`
  (cache source: actions stored in absolute coords)

Bug. Rust self-play stores `step.action`, `legal_moves`, and
`visit_distribution` indices in **POV-flipped** (current-player) coordinates
(`src/selfplay/game_task.rs:474`, see also the dedicated rule
`.claude/rules/mcts-pov-symmetry.md`). The Python tablebase pipeline stores
actions in **absolute** coordinates (output of `action_from_move`, which uses
python-chess `move.from_square` directly with no rank-mirror).

When TB samples are mixed into a batch and the trainer's POV-encoded
observation is paired with absolute-coord action / policy / legal-mask
indices, the action plane's from-square does not align with the
observation's "my piece" plane for **any black-to-move TB sample**.
Concretely, a black knight on b8 (sq 57) yields `action_from_move ≈ 57*64+…`,
which the action encoder places at rank 7 file 1; but `encode_board_python`
flips ranks for black, so the same knight appears at rank 0 file 1 in the
observation. The action plane and the piece plane disagree.

Same disagreement applies to `target_policies` and `legal_masks`: indexed
by absolute action ID for TB rows vs POV-flipped action ID for replay rows.
The network sees inconsistent supervision on roughly half of all TB
samples (the black-to-move half).

Fix sketch: in both `build_tb_batch*` builders, derive `white_to_move` per
sample, and when False, apply `flip_action(action_idx)` (rank-mirror via
`flip_base_action` for base actions; identity for underpromo) to
`actions[k]`, every entry of `legal_actions[k]`, and every entry of
`optimal_actions[k]` before writing into the planes / masks. Add a
black-to-move regression test (the existing `test_tablebase_*` set has no
black-to-move action-coord coverage).

#### B2 — MCTS terminal revisit double-counts edge rewards (MEDIUM)

Location: `src/mcts/tree.rs:413-416`

```rust
if parent.children[leaf_action_idx].is_some() {
    let child = parent.children[leaf_action_idx].as_ref().unwrap();
    child.q_value()      // ← bug: q_value already encodes path rewards
} else { … expand … }
```

After the canonical-MuZero backup change (`backpropagate` now adds
`r_k` along the path), revisiting an already-expanded terminal and
passing `child.q_value()` as the leaf value to `backpropagate(path, value)`
double-applies the edge rewards. First visit: `value=0`, `G\_{d-1} = r_d

- 0 = +1`(correct mate signal). Revisit:`value = q*value ≈ +1`,
`G*{d-1} = r_d - 1 = 0`, so the second visit contributes zero to
  ancestors. After N visits, root's accumulated mate evidence is roughly
  1/N of what it should be — directly weakens the very signal the new
  backup is intended to propagate.

Fix: pass the original leaf value (which is `0` for absorbing terminals)
or store `leaf_value` on the node. Simplest correct change:
`backpropagate(&path, 0.0)` in the terminal-revisit branch (assuming
absorbing-state convention `v_leaf = 0`). Add a test that calls
`run_simulations` twice on a one-mate tree and asserts root's
`total_value` grows by exactly the mate contribution each time.

#### B3 — Mirror-trajectory regression test is gated behind `--ignored` (LOW)

Location: `src/py/training.rs` — `test_mirror_trajectory_targets_are_symmetric`
marked `#[ignore = "mirror-trajectory symmetry regression — expensive; run with --ignored"]`.

The whole point of regression tests is to catch reintroductions of the
specific bug they target. Because this one is `#[ignore]`'d, the default
`cargo test` invocation won't catch a future POV-symmetry regression.
The test body looks fast (constructs 5-step trajectories with mock
observations); the "expensive" justification seems weak. Recommend
removing the `#[ignore]` unless there is a real cost reason.

Same observation for `test_mcts_visit_distribution_ordering_invariance`
in `src/mcts/tree.rs` (200 MCTS runs — that one is plausibly slow; OK
to keep `--ignored` but consider adding a tiny sibling that runs ~5
trials for the default suite).

#### B4 — Diverse-start games may unexpectedly start with Black to move (LOW)

Location: `src/selfplay/game_task.rs` — `init_self_play_board` returns
`(board, side_to_move)` and both `play_game` / `play_game_dual` correctly
consume the returned `side_to_move`. The path is correct as wired.

Concern is downstream: `turn_count` always starts at 0 even when the FEN
represents (say) ply 40. `MAX_GAME_LENGTH = 300` therefore allows 300
_additional_ plies, not 300 total — fine, but worth documenting. More
substantively, the history `VecDeque` is empty even for mid-game FENs,
which silently breaks any threefold-repetition reasoning across the FEN
boundary. For the new default 100k-FEN starts file that includes
middlegame and endgame positions, this means repetition can only ever
be detected from positions that occur _after_ the FEN start. Probably
acceptable for training diversity, but flag for awareness.

#### B5 — Unused `unsafe` block in env-var test (NIT)

Location: `src/py/training.rs:1240-1245` in
`test_conditional_beta_decisive_uses_pure_outcome`.

In Rust 2021 (this project's edition), `std::env::set_var` and
`remove_var` are not `unsafe`. The test wraps `set_var` calls in an
`unsafe` block but the matching `remove_var` cleanups at the end are
not wrapped. The `unused_unsafe` lint is warn-by-default, so this
likely produces a warning but compiles. Either drop the `unsafe`
wrapper or wrap both halves consistently.

### Notes (non-bugs)

- The new `backpropagate` recurrence `G_{k-1} = r_k − G_k` (γ=1) is
  algebraically correct and reduces to the old behavior bit-for-bit
  when all edge rewards are zero (confirmed by reading the diff and
  the new `test_backpropagate_includes_mating_reward`).
- `encode_action_spatial_for_color` correctly fixes the
  under-promotion POV-flip invariant; the regression test exercises
  both colors over all 576 underpromo indices.
- `flip_obs_planes` / `flip_action_planes` / `flip_action` semantics
  match between Rust and Python (verified visually).
- The conditional-β code path is gated by env var (default off) and
  default semantics are preserved.
- Dirichlet sampler: Marsaglia-Tsang with `d = (α+1) - 1/3` is valid
  for any `α > 0`. The stray `let _ = x;` on line 196 is dead but
  harmless.

### Verification commands

```bash
# Existing regression tests that pin the fixes in this commit:
cargo test --release test_backpropagate -- --nocapture
cargo test --release test_flip_action_planes_matches_flip_action_invariant
cargo test --release test_legal_actions_ordering -- --nocapture
cd python && pytest tests/test_tablebase.py -v

# Ignored regression tests that should be unlocked for routine CI:
cargo test --release -- --ignored test_mirror_trajectory_targets_are_symmetric
cargo test --release -- --ignored test_mcts_visit_distribution_ordering_invariance
```
