# Plan: MCTS Tree Reuse

## Hypothesis

Discarding the MCTS tree after every move wastes prior search work. After selecting
action A, the subtree rooted at child(A) already contains visit counts and Q-values
from the previous search. Carrying that subtree forward makes PUCT scores better
calibrated from simulation 1 instead of simulation 50+. This produces stronger move
selection, more decisive games (higher `decisive_ratio`), and richer training signal
(policy targets reflect deeper searches). Expected score improvement: +0.3 to +0.8
from `decisive_ratio` increase and marginal policy loss improvement.

---

## Approach

`MCTSNode` stores children as `Vec<Option<Box<MCTSNode>>>`. After selecting action at
index `i`, `children[i]` is a `Some(Box<MCTSNode>)`. A single `Option::take()` call
extracts it without unsafe code. The extracted node becomes the root for the next move.
After the opponent plays action `j`, repeat: take `children[j]` from the current root.
If either lookup misses (None — unexpanded branch), fall back to a fresh `root_setup`
call as today.

The change is confined to `MCTSTree` (add `reuse_subtree`) and `game_task.rs` (pass
tree across loop iterations). No changes to inference, training, Python, or data types.

---

## Subtasks

### 1. Add `MCTSTree::reuse_subtree(action: ActionIndex)` to `src/mcts/tree.rs`

- **Files**: `src/mcts/tree.rs`
- **Changes**:
  - Add method `pub fn reuse_subtree(self, action: ActionIndex) -> Option<MCTSTree>`.
  - Iterate `self.root.legal_actions` to find the index `i` where
    `legal_actions[i] == action`.
  - If not found or `children[i]` is `None`, return `None`.
  - Otherwise, take the child via `children[i].take()`, unwrap the `Box`, and
    construct a new `MCTSTree { root: *child_box, config: self.config }`.
  - The new root already has `visit_count`, `total_value`, `priors`, `children` from
    the previous search. Do NOT re-add Dirichlet noise here — noise is added at
    initial `MCTSTree::new` only. The reused root needs fresh noise mixed in to keep
    exploration diversity; add a `mix_root_noise()` private helper that applies
    Dirichlet to the existing priors (same formula as in `new()`).
- **Tests**:
  - `test_reuse_subtree_returns_some_for_expanded_child`: run 10 sims, reuse a child
    that was expanded, verify `root.visit_count > 0`.
  - `test_reuse_subtree_returns_none_for_unexpanded_child`: construct a tree where one
    child index was never expanded (0 visits), verify `reuse_subtree` returns `None`.
  - `test_reuse_subtree_returns_none_for_unknown_action`: pass an `ActionIndex` not in
    `legal_actions`, verify `None`.
- **Dependencies**: none

### 2. Refactor `play_game()` in `src/selfplay/game_task.rs` to carry tree across moves

- **Files**: `src/selfplay/game_task.rs`
- **Changes**:
  - Introduce `let mut maybe_tree: Option<MCTSTree> = None` before the game loop.
  - At the start of each turn, after computing `legal_actions` and calling
    `evaluator.root_setup()`, check `maybe_tree`:
    - If `Some(prev_tree)`, try `prev_tree.reuse_subtree(last_opponent_action)`.
      If it returns `Some(reused)`, use it as the tree (skip `MCTSTree::new`).
      If it returns `None`, fall back to `MCTSTree::new(hidden_state, &policy, ...)`.
    - If `None` (first move), always call `MCTSTree::new(...)`.
  - After the tree is obtained (reused or fresh), call `tree.run_simulations()` as
    today.
  - After `select_action`, store the selected action as `last_own_action`.
  - After `board.process_move()`, call `maybe_tree = Some(tree.reuse_subtree(last_own_action)
    .unwrap_or_else(|| MCTSTree::new(...)))`. This advances the tree past our own move.
  - Track `last_opponent_action: Option<ActionIndex>` — set to the action the opponent
    just played (i.e., the action stored in the `StepRecord` from the previous
    half-turn). Since `play_game` controls both sides, `last_opponent_action` is just
    the action selected two half-turns ago by the alternating player.
  - Keep `root_setup` call unconditional — we always need the hidden state for the
    `MCTSTree::new` fallback path. When reusing, the hidden state is already stored in
    the reused root (`root.hidden_state`) but `root_setup` is still needed because
    `hidden_state` in the reused node was computed via `g(s,a)` (dynamics), not
    `h(obs)` (representation). In MuZero these are different latent spaces. The reused
    node's hidden state is valid as a starting point for further simulation, but the
    representation network `h(obs)` should still be called to ground the root in the
    real board observation. Therefore, always call `root_setup`, and when we have a
    reused subtree, replace its `root.hidden_state` with the fresh `h(obs)` output
    before running simulations.
  - Specifically: `reused.root.hidden_state = fresh_hidden_state;` — one field
    assignment after the reuse.
- **Tests**:
  - `test_play_game_with_tree_reuse_completes`: run a full game with `num_simulations=5`,
    verify trajectory is non-empty and outcome is valid (same as existing
    `test_play_game_completes`, just validates the refactored function still works).
  - The existing `test_play_game_completes` test continues to serve as a regression
    check; no change needed if it still passes.
- **Dependencies**: Subtask 1 must complete first (uses `reuse_subtree`).

### 3. Update `GameConfig` and env-var parsing in `src/bin/selfplay.rs`

- **Files**: `src/bin/selfplay.rs`
- **Changes**:
  - Add `tree_reuse: bool` field to `GameConfig` in `src/selfplay/game_task.rs`
    (default `true`).
  - Add corresponding `HYZERO_TREE_REUSE` env-var parse in `src/bin/selfplay.rs`
    (parse "0"/"1", default 1 = enabled).
  - Thread `config.tree_reuse` through `SelfPlayConfig` → `GameConfig` → `play_game`.
  - Guard the reuse path with `if config.tree_reuse { ... } else { None }` so the
    feature can be disabled for ablation without a code change.
- **Tests**: No new tests needed — the existing `test_coordinator_produces_trajectories`
  covers the default path (tree_reuse=true by default).
- **Dependencies**: Subtask 2 must complete first.

---

## Testing Strategy

1. `cargo test` — all 82 existing tests must still pass. New unit tests in `tree.rs`
   and `game_task.rs` must pass.
2. `cargo test --release -- --ignored` — slow perft and Python-dependent tests.
3. `bash scripts/e2e_test.sh` — end-to-end: 5 games, 13 train steps, loss decrease.
4. `bash scripts/run_baseline.sh 900` — 15-min training run, extract
   `training_score` from `logs/baseline_score.json` and compare to baseline of 5.7646.

Ablation: set `HYZERO_TREE_REUSE=0` and rerun to confirm the baseline is stable.
Any regression means the hidden-state replacement (step 2) is the culprit — in that
case, remove the `root.hidden_state = fresh_hidden_state` assignment and instead always
build a fresh tree with the same `MCTSConfig` (pure fallback mode).

---

## Expected Score Delta

| Component | Direction | Reasoning |
|-----------|-----------|-----------|
| `decisive_ratio` | +0.03 to +0.10 | Better calibrated PUCT → stronger, more decisive games |
| `avg_game_length` | -5 to -15 moves | Cleaner wins terminate sooner |
| `policy_loss` | -0.05 to -0.15 | Richer visit distributions in training targets |
| **Net score delta** | **+0.35 to +1.25** | Formula: decisive_ratio×10 dominates |

Conservative estimate: **+0.4** (decisive_ratio goes from 0.30 to 0.34).
Optimistic estimate: **+1.0** (decisive_ratio goes from 0.30 to 0.40).

---

## Fallback

If score regresses:
1. First try removing the `root.hidden_state = fresh_hidden_state` override — maybe
   the dynamics-space hidden state is better as a root than the fresh representation.
2. If still regressing, disable reuse entirely (`HYZERO_TREE_REUSE=0`) and confirm
   baseline is restored. The feature flag means rollback is a one-line env change.
3. The `maybe_tree` pattern leaves the existing code path fully intact as the fallback.

---

## Context Summary

### Project
- Stack: Rust (cargo) + Python (PyTorch via PyO3)
- Build: `cargo build`
- Test: `cargo test` (82 pass, 7 ignored), `cd python && pytest`
- Baseline run: `bash scripts/run_baseline.sh 900` (15-min budget)
- Metric: `(8.55 - policy_loss) + (decisive_ratio * 10) - (avg_length / 100)`

### Relevant Code
- `src/mcts/tree.rs`: `MCTSTree` — owns `root: MCTSNode`, `config: MCTSConfig`. Tree is
  discarded after every move (line 88 comment "transient"). `MCTSNode` children are
  `Vec<Option<Box<MCTSNode>>>` — safe ownership extraction via `Option::take()`.
- `src/mcts/node.rs`: `MCTSNode` — has `hidden_state: HiddenState`, `visit_count`,
  `total_value`, `priors: Vec<f32>`, `children: Vec<Option<Box<MCTSNode>>>`,
  `legal_actions: Vec<ActionIndex>`. `legal_actions` is empty for un-expanded leaves
  (important: reused subtree children also have empty legal_actions until expanded by
  simulator — this is fine, the simulator expands them).
- `src/selfplay/game_task.rs`: `play_game()` — creates fresh `MCTSTree::new()` every
  turn (line 85-91). `GameConfig` holds `num_simulations`, `exploration_constant`,
  `temperature_moves`. `mcts_config` is cloned per-turn (line 52-55).
- `src/bin/selfplay.rs`: Env-var overrides for `RunConfig`. `HYZERO_SIMS=50` default.
  `from_default_config` hardcodes `train_steps_per_game=8`.

### Patterns
- Env-var config: `RunConfig` parsed from env at binary startup (selfplay.rs:54-87).
  Pattern: `env::var("HYZERO_X").ok().and_then(|v| v.parse().ok()).unwrap_or(default)`.
- Feature flags: no existing precedent; `tree_reuse: bool` on `GameConfig` establishes
  the pattern.
- Dirichlet noise: applied in `MCTSTree::new()` (tree.rs:110-117) using
  `NOISE_EPSILON=0.25` and `NOISE_ALPHA=0.03`. Must re-apply to reused root to maintain
  exploration diversity.

### Constraints
- `Cargo.lock` — do not modify
- `docs/wiki/` — do not modify (context-keeper owned)
- `logs/` — do not modify
- Checkpoints in `checkpoints/` are compatible — no shape changes in Batch 2
- `expand_leaf` does NOT take a real board state, only `hidden_state`. Leaves in the
  reused tree have `legal_actions: Vec::new()` — this is correct and unchanged.

### Risks
- **Hidden-state grounding**: The reused root's `hidden_state` was produced by
  dynamics `g(s,a)` whereas `root_setup` uses `h(obs)`. Replacing the hidden state
  restores MuZero's invariant that the root is always in representation space.
  If this causes instability, the ablation (keep dynamics-space hidden state) is the
  fallback.
- **Dirichlet noise on reused root**: Must re-apply noise after reuse or the root
  will retain the previous game's noise mixture, reducing exploration diversity.
- **Memory**: at 50 sims/move with ~40 legal moves, the tree is small (~50 nodes).
  Carrying one tree between turns has negligible memory impact.
- **`legal_actions` on reused root**: The reused node has correct `legal_actions`
  (from the previous `MCTSTree::new` call where it was first created as a child and
  then expanded). No change needed to legal action bookkeeping.
