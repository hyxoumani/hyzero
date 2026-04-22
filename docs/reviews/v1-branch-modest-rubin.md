# Review v1 — branch `claude/modest-rubin-4U3Op`

**Review date:** 2026-04-22
**Range reviewed:** `origin/main..HEAD` (35 commits, ~65k line diff)
**Reviewer scope:** bugs only. Not reviewed: logs, PGN dumps, agent-memory/wiki prose, generated caches.

## Commits covered (all from `origin/main..HEAD`)

- ee132c4 train: TB supervision infrastructure + canonical MuZero backup + diverse starts ← **primary focus**
- 38033d7 docs+wiki: color asymmetry bug-hunt
- 7243aec mcts+selfplay: fix color asymmetry in self-play move selection
- 72d0c8e training: diagnostic instrumentation
- 773cb90 selfplay+training: POV isolation infra + terminal-reward POV fix
- b012944 training: fix value/reward target sign under color augmentation
- 2edb194 mcts: Dirichlet ε/α env-controllable
- 6aff3d4 encoding: initial-position color-symmetry regression test
- 5e201cc training: fix off-by-one in dynamics action indexing
- 65b54e7 encoding: remove plane 101 (side-to-move)
- (plus 25 smaller commits — mostly docs, tests, env-gated instrumentation)

## Files audited for bugs

- `src/mcts/tree.rs` — new canonical-MuZero backup + tie-break RNG
- `src/py/training.rs` — POV-flip sign under color augmentation
- `src/data/encoding.rs` — `encode_action_spatial_for_color`
- `src/selfplay/game_task.rs` — `legal_actions.sort_unstable()`
- `python/hyzero/training/trainer.py` — TB routing, biased reinit, SimSiam consistency
- `python/hyzero/data/tablebase.py` — TBSample / TBTrajectory / TablebaseCache
- `scripts/build_tablebase_trajectory_cache.py` — absorbing-state convention

---

## Findings

### [Medium] `rand::rng()` thread-local RNG shared across concurrent self-play tasks

**`src/mcts/tree.rs:606`** — `tied[rand::rng().random_range(0..tied.len())]`.

`rand::rng()` is `thread_rng()`. Multiple tokio tasks multiplexed onto the same
worker thread share one RNG state, so their tie-breaks are interleaved from a
common stream. Not a correctness bug (all samples are still uniform on their own),
but means reproducibility (seeding) won't work without per-task RNGs, and per-game
randomness is weakly correlated across coresident tasks. Low-impact in practice.

### [Medium] `γ = 1` hardcoded in backup with no validation of intermediate-reward contract

**`src/mcts/tree.rs:521-526`** — The new backup computes `G_{k-1} = r_k − G_k`
(γ implicit 1). This relies on the invariant _"only the terminal edge carries a
non-zero reward"_ — i.e., `child.reward == 0.0` for every non-terminal child.
Nothing enforces or asserts this. If a future change (material shaping, draw
penalty in the tree, etc.) ever writes a non-zero `reward` on a non-terminal
edge, the new formula will silently double-count signals up every ancestor.

Suggest adding a debug_assert!(child.reward == 0.0 || child_is_terminal) at the
reward-write site, or a similar guard at the top of `backpropagate`.

### [Medium] `is_trajectory_format` detection via `hasattr(first, "fens")` is fragile

**`python/hyzero/data/tablebase.py:151`** — Format detection inspects a single
attribute name on the first element. If `TBSample` is ever extended with a
`fens` field (rename or unification), every old snapshot cache silently
reclassifies as trajectory and flows through the wrong decoder. Prefer an
explicit `isinstance(first, TBTrajectory)` check.

### [Low] `TablebaseCache.sample()` silently switches to replacement for `n >= len(pool)`

**`python/hyzero/data/tablebase.py:189-191`** — `random.choices(pool, k=n)` is
used when the pool is smaller than `n`; `random.sample(pool, n)` otherwise. If
the TB cache is a few hundred positions and the trainer requests, say, 128 TB
samples per step, every batch will draw duplicates without warning. Log a
one-time warning when `n >= len(pool)` or an env-controlled hard-error.

### [Low] Biased value-head reinit assumes `tanh` activation without guard

**`python/hyzero/training/trainer.py:352-389`** — Docstring states the final
bias produces `tanh(0.3) ≈ 0.29`. The code does not verify the prediction-head
activation is `tanh`. If the head is ever swapped (e.g. to `sigmoid` or
`identity`), the documented calibration silently misaligns. Low-impact because
this is an opt-in env-gated recovery path, but worth a single `assert` on the
activation type or a comment pinning the assumption to the `PredictionNetwork`
definition.

### [Low] Reward-loss mixing at `k=1` dilutes non-TB rows when TB is present

**`python/hyzero/training/trainer.py:633-637`** — At `k==1`, snapshot-format TB
rows legitimately carry a real `+1` mating target, and non-TB rows carry their
replay target (usually 0). The code averages both together unmasked:
`total_reward_loss += per_sample_rwd.mean()`. This is intentional per the
comment, and it's the right call for the TB rows — but it means for replay
rows at k=1, the MSE is blended into a loss that is dominated by the TB mating
signal's larger magnitude. With a mostly-TB batch, the replay rows' k=1 reward
signal is numerically diluted. Not incorrect, but the per-source loss weight is
not controllable — consider weighting the TB vs non-TB contributions explicitly.

### [Low] `ply_flip` interaction with `original_root_side_sign` is correct but under-tested

**`src/py/training.rs:183-214`** — The composite
`flip_sign * game_outcome * original_root_side_sign * ply_flip` is correct
(outcome is White-absolute → root-POV via `original_root_side_sign` → step-k-POV
via `ply_flip`, then globally negated under flip). But this is four multiplications
deep and the regression test named in the comment
(`test_value_target_sign_under_flip_matches_observation_pov`) is the only thing
pinning it. Recommend expanding the test to cover: decisive outcome at odd-ply
root, flipped + non-flipped, and check each k ∈ 0..K independently.

### [Low] `flip_action_planes` stale docstring referencing plane 101

**`src/data/encoding.rs`** — `#[allow(dead_code)] fn flip_action_planes(...)`
references "plane 101 (side-to-move)" which was removed in commit 65bfd7a.
Function is unused. Either delete it or drop the stale plane-101 reference.

### [Nit] Dead-code / stale comment hygiene

Several modules retain commentary referring to the removed side-to-move plane
and to `max_by` in `select_action`. Fine to leave for now, but clean up on the
next pass through each file.

---

## Things explicitly verified _not_ to be bugs

These came up during review (including from subagent probes) and were confirmed
sound:

- **SimSiam asymmetric stop-gradient** (`trainer.py:820-834`). The online branch
  is `project → predict` and the target branch is `project → detach`. This is
  canonical SimSiam (Chen & He 2021): the predictor asymmetry _is_ how SimSiam
  prevents collapse. No bug.
- **Reward masking at k=1 for TB rows**. The code intentionally does NOT mask
  the TB reward at k=1 because that's where the mating signal lives in the
  snapshot format. The comment is correct.
- **Absorbing-state value targets**. The trajectory builder
  (`build_tablebase_trajectory_cache.py:340-343`) fires `reward=+1` at the
  mate-transition step and leaves all later `target_values` at `0.0`. No +1
  value-target drift past terminal.
- **Trajectory-format consistency-loss masking**. `is_tablebase=False` on
  trajectory rows is intentional: these rows carry real observations at every
  step, so consistency loss applies. Snapshot rows are correctly excluded.
- **`sort_unstable()` on `legal_actions`** is deterministic per run because
  action indices are distinct `u16`s.

---

## Not yet reviewed (deferred to v2)

- `src/data/replay_buffer.rs` changes (+164/-lines) — priority sampling + decisive fraction.
- `src/mcts/puct.rs` changes — PUCT-specific edits.
- `src/data/types.rs` — new fields / changes to replay sample struct.
- `scripts/pretrain_dynamics.py` and `scripts/gen_pretrain_dynamics.py` — pretraining pipeline.
- `python/hyzero/data/board_encoder.py` — Python port of board encoder (parity vs Rust).
- Test-file additions (`python/tests/test_tablebase.py`, new Rust tests) — assumed to pass by CI; coverage not audited.

If v2 is requested, start by spot-checking board_encoder.py parity against
`src/data/encoding.rs` (Python port bugs are a classic source of silent training
corruption) and the new replay_buffer priority-sampling invariants.
