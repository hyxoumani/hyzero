# Review 001 — claude/modest-rubin-7VwAn

| Field         | Value                                      |
| ------------- | ------------------------------------------ |
| Branch        | `claude/modest-rubin-7VwAn`                |
| Reviewed HEAD | `4382bde` (merge PR #2 autoresearch/apr13) |
| Base          | `origin/main` @ `30419020`                 |
| Range         | `30419020..4382bde` (30 commits, 29 files) |
| Date          | 2026-04-20                                 |
| Focus         | Bugs / correctness only                    |

## Scope

8,231 insertions / 661 deletions across Rust core (training, MCTS, encoding,
replay, selfplay) and Python trainer / representation. Key commits reviewed:

- `b012944` value/reward target sign under color aug
- `5e201cc` off-by-one in dynamics action indexing
- `773cb90` POV isolation + terminal-reward POV fix
- `65b54e7` remove plane 101 (side-to-move) for color symmetry
- `7fca9ea` EfficientZero consistency loss
- `0882a7b` / `08911f1` draw penalty → prioritized replay
- `bdc8301` policy entropy bonus
- `a09b09e` / `2edb194` opt-in / env-controllable Dirichlet noise
- `ee4aeaf` randomized tie-break in `select_child`

## Findings

### CRITICAL

**C1. `flip_action` ignores underpromotion actions**

- File: `src/data/encoding.rs:318-325`
- For indices `>= NUM_BASE_ACTIONS` the function returns the raw index
  unchanged. Underpromotion actions encode `from_file` / `to_file`; both must
  be mirrored when the board is flipped for color augmentation.
- Impact: training samples containing underpromotions have a ~50% probability
  (augmentation rate) of receiving a policy/action target from the wrong POV,
  injecting noise into the dynamics loss in critical endgame positions.
- Fix: decode underpromotion action → mirror files → re-encode.

### MEDIUM

**M1. Underpromotion spatial encoding is always white-perspective**

- File: `src/data/encoding.rs:274-279` (`encode_action_spatial`)
- Underpromotions encode rank 6→7 regardless of the side to move, while the
  piece planes are already player-relative. `flip_action_planes` rank-mirrors
  the spatial encoding afterward, which may compensate — but the code has no
  explanatory comment and the symmetry is not tested. Related to C1; fixing
  C1 will require reasoning about this too.

**M2. Division by zero when `k_steps == 0`**

- File: `python/hyzero/training/trainer.py:585`
- `avg_reward_loss = total_reward_loss / k_steps` raises `ZeroDivisionError`
  if the unroll length is 0. Unlikely in production (`k_steps >= 1` from
  self-play), but a correctness issue at the boundary.
- Fix: guard with `if k_steps > 0` or use `max(k_steps, 1)`.

**M3. Stale docstring: `[B, 103, 8, 8]` after plane 101 removal**

- File: `python/hyzero/inference/server.py:60, 71`
- Documentation drift; runtime reads from `DEFAULT_CONFIG` so inference is
  correct. Fix by updating the docstrings to `[B, 102, 8, 8]`.

## Verified correct (investigated, no bug)

Rust

- `src/py/training.rs:143` — dynamics uses `steps[k].action`, not `[k+1]` (5e201cc fix holds).
- `src/py/training.rs:200,196-207` — `flip_sign` applied to both `root_value_target` and `outcome_in_step_perspective` (b012944 fix holds).
- `src/selfplay/game_task.rs:434-435` — terminal `last.reward = outcome * last_side_sign` (POV correct).
- `src/py/training.rs:130` — `(bi*kp1 + k)*obs_stride` flattening bounds-safe up to the last element.
- `src/selfplay/evaluation.rs:167` vs self-play `add_root_noise: true` (line 287) — Dirichlet correctly gated.
- `src/mcts/puct.rs:64-82` — tied-PUCT tie-break uses uniform random over ties (ee4aeaf).
- `src/data/replay_buffer.rs:76-132` — decisive-sample weighted sampling; falls back to all-pool when empty.
- `src/py/training.rs:104,129` — all K+1 observations stored, shape matches consistency-loss expectation.
- Plane 101 removal — castling (player-relative) and en passant (rank-mirrored) remain unambiguous.

Python

- `python/hyzero/training/trainer.py:595-607` — consistency loss: `p2 = self.h.project(target_latent).detach()` correctly blocks gradient through the target branch (PyTorch detach severs the graph regardless of upstream `requires_grad`).
- Policy entropy sign: `ce_loss + β · (-H)` ⇒ minimizing pushes toward higher entropy. Correct.
- Loss-weight env vars clamped to `[0.0, 100.0]`; zero-weight contributes zero loss (no NaN path).
- Diagnostic instrumentation wrapped in `torch.no_grad()` and `try/except` — no training-path interference.
- `_flip_obs_planes`: piece-plane swap + rank mirror consistent with Rust `flip_action_planes`.

## Recommended next steps

1. Fix **C1** (underpromotion flip) before the next training run — this one silently corrupts training targets.
2. Fix **M2** (divide-by-zero) as a cheap boundary guard.
3. Address **M1** / **M3** in a follow-up — either document the invariant or add a test that enforces it.

## Review ledger

| Commit / Tag | Reviewed | Notes             |
| ------------ | -------- | ----------------- |
| `4382bde`    | ✅       | This review (001) |
