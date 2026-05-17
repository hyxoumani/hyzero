# Code Review Log

Tracks which commits have been reviewed by Claude, with a summary of bugs found
per review. Append-only: each review session adds a new entry at the bottom.

## Format

Each entry:

- **Reviewed-at**: ISO date of the review
- **Commit range**: `<base>..<head>` (SHAs) — what was reviewed
- **Branch**: branch the review targeted
- **Status**: pending fixes / no bugs / blocked-on-author
- **Findings**: bug list with severity (HIGH / MED / LOW)

## Reviews

### 2026-05-17 — ee132c4 (squash of autoresearch/apr13)

- **Reviewed-at**: 2026-05-17
- **Commit range**: `4382bde..ee132c4` (single squash commit, 51 files, +62308/−7989)
- **Branch**: `claude/modest-rubin-hTf15` (= main HEAD)
- **Focus**: bugs only
- **Status**: see findings

Headline of the squash:

- MuZero canonical MCTS backup (`src/mcts/tree.rs::backpropagate` now propagates edge rewards via `G_{k-1} = r_k − G_k`).
- Color-asymmetry self-play fix (`src/selfplay/game_task.rs`: `legal_actions.sort_unstable()` after POV flip; random argmax tie-break in `MCTSTree::select_action`).
- POV-aware action encoding (`src/data/encoding.rs::encode_action_spatial_for_color`).
- PyO3 training targets under color augmentation (`src/py/training.rs`): action encoding routes through the color-aware encoder; conditional β gated by `HYZERO_CONDITIONAL_BETA`.
- Diverse self-play starts via `HYZERO_STARTS_FILE`.
- Syzygy TB supervision infrastructure (Python: `tablebase.py`, `board_encoder.py`, snapshot + K-step trajectory formats; trainer routing).

Findings recorded below.

#### Findings

Test gate: `cargo test --lib --release` 133 passed / 0 failed / 10 ignored.
`cd python && pytest tests/` 46 passed. No regressions from this commit.

##### HIGH — TB-trajectory action indices not POV-flipped for Black-to-move positions

- **Where**: `python/hyzero/data/board_encoder.py:223` (`action_from_move`) — and every call site in `scripts/build_tablebase_trajectory_cache.py` (`_build_trajectory` lines 319, 321, 332, 362, 364).
- **What**: `action_from_move(move, board)` returns the absolute-square base-action index `from_sq * 64 + to_sq`. It never applies `flip_action` for Black-to-move boards. The Rust self-play pipeline DOES POV-flip the action before storing it in `StepRecord.action` (`src/selfplay/game_task.rs:273-280`, again at 402-409). Consequence: for the same physical move (e.g. Black's a7→a6), self-play stores action index `528` (POV-flipped) and TB stores `3112` (absolute). The trainer feeds both to the same policy head.
- **Concrete reproduction** (verified): for FEN `4k3/p7/8/8/8/8/8/4K3 b - - 0 1` and move a7a6:
  - `action_from_move(...)` returns `3112` (POV-flipped index would be `528`).
  - `encode_action_spatial(3112, white_to_move=False)` places the FROM plane at `(rank=6, file=0)` — absolute coords.
  - `encode_board_python(board)` places the moving pawn at `(rank=1, file=0)` — POV coords.
  - Observation and action are in different coordinate frames inside the same training sample.
- **Impact**: ~50% of TB-trajectory positions (those where Black is to move) train the policy head against action indices that disagree with the self-play data distribution, and the dynamics network `g` receives action planes whose `from`/`to` squares don't align with the observation it just consumed. Once trajectory-format TB supervision is mixed in (`HYZERO_TABLEBASE_FRAC > 0` with a trajectory-format cache), the policy signal is incoherent for half the TB data, and the dynamics network is taught a misaligned mapping.
- **Why the existing tests miss it**: `test_trajectory_batch_shapes_and_mate_signal` uses a white-to-move FEN. `test_trajectory_value_targets_alternate_pov_sign` checks only the value-target signs, not action indices. The `action_from_move` docstring claims it "mirrors `src/data/encoding.rs::move_to_action`" — which is true: Rust's `move_to_action` also returns absolute. Both are intended for callers who then POV-flip; the Rust caller does, the Python TB caller doesn't.
- **Fix**: in `_build_trajectory` (and `build_tablebase_cache.py` snapshot path), wrap every `action_from_move(m, board)` with a POV flip when `board.turn == chess.BLACK`. A `_flip_action(idx)` helper that mirrors `src/data/encoding.rs::flip_action` (rank-mirrors base actions, identity on underpromotion indices) should go into `board_encoder.py`. Add a regression test that builds a trajectory from a Black-to-move FEN and asserts the encoded FROM plane lands at the same square as the OBS piece plane for the moving piece.

##### HIGH — Syzygy "optimal move" picks min |DTZ| for both sides (kills losing-side defense)

- **Where**: `scripts/build_tablebase_trajectory_cache.py::_find_optimal_moves` lines 271-284. Same issue in `scripts/build_tablebase_cache.py` (mating-in-1 branch is fine; the DTZ fallback isn't).
- **What**: after `board.push(m)`, the code calls `tb.probe_dtz(board)` and stores `abs(dtz)`, then picks moves with `min |DTZ|`. python-chess returns DTZ from the **new** side-to-move's POV, so for a winning side this correctly picks "force-mate fastest." For a losing side it picks "let the opponent win fastest" — the opposite of optimal defense.
- **Impact**: roughly half the TB starting positions are losing-side STM. Their `optimal_actions` become "fastest surrender" moves, and the policy head is supervised to play the most resistanceless capitulating move. This directly contradicts the goal of TB supervision (teach correct endgame play) and will actively make the value/policy head worse on losing endgames the more `HYZERO_TABLEBASE_FRAC` is increased.
- **Fix**: branch on root WDL sign. Pseudo-code:

  ```python
  wdl = tb.probe_wdl(board)  # STM POV at root
  if wdl > 0:
      # Winning: minimize |DTZ| to mate fastest.
      best = min(...)
  elif wdl < 0:
      # Losing: maximize |DTZ| to delay loss.
      best = max(...)
  else:
      # Drawn: keep any move that preserves a draw (|DTZ| at the resulting
      # position should be 0 from the new STM's POV under correct play).
      best = [m for d, m in dtz_scores if d == 0] or dtz_scores
  ```

  Snapshot builder (`build_tablebase_cache.py`) needs the same branching for its `optimal_actions` field.

##### MED — Policy entropy bonus silently dropped when TB mixing is active

- **Where**: `python/hyzero/training/trainer.py` k≥1 TB-masking branch (`_policy_loss_per_sample` usage).
- **What**: at k≥1 with `is_tb_tensor is not None`, the policy loss goes through `_policy_loss_per_sample`, which does the cross-entropy but omits the `policy_entropy_weight * (−H(π))` term applied inside `_policy_loss`. The k=0 branch and the no-TB k≥1 branch both go through `_policy_loss` which DOES apply the entropy bonus. Net effect: when `HYZERO_POLICY_ENTROPY_WEIGHT > 0` AND TB mixing is on, the entropy regularization at k≥1 quietly evaporates for that fraction of the loss.
- **Impact**: dormant at the default `HYZERO_POLICY_ENTROPY_WEIGHT=0.0`. Becomes a silent inconsistency the moment someone turns entropy regularization on while TB is active — the loss the network optimizes is no longer the loss the env vars describe.
- **Fix**: either re-add the entropy term at the caller in the k≥1 TB branch (per-sample, then mask + average like the cross-entropy), or call `_policy_loss` and apply the row mask outside it.

##### MED — Trace-writer thread-local is unsound under tokio task migration

- **Where**: `src/mcts/tree.rs:57-64, 290-297`.
- **What**: `try_claim_writer()` is a global CAS, but the "did I win the CAS" answer is stored in a `thread_local!`. Tokio tasks can move OS threads across `.await` points (e.g. `evaluator.expand_leaf(...).await` inside the simulation loop). If task A wins the CAS on thread X and later resumes on thread Y, `WRITER_LOCAL[Y]` is false → `is_writer()` returns the wrong answer. Inversely, if task B then runs on thread X, `WRITER_LOCAL[X]` is still true → task B is treated as the writer despite never winning the CAS.
- **Impact**: tracing is opt-in via `HYZERO_MCTS_TRACE`. In production (env var unset) all of this is dead. When tracing is on, the trace file may interleave or silently drop entries — but the file mutex protects the file handle, so the data is corrupt-resistant. Treat as latent bug; flag before anyone relies on the trace log for correctness analysis.
- **Fix**: replace the thread-local with an async-aware mechanism — e.g. let `try_claim_writer` return an owned token (e.g. an `Arc<TraceWriter>`) that the task threads through `run_simulations` so the writer identity moves with the task.

##### LOW — Dead-or-misleading code in TB snapshot cache builder

- **Where**: `scripts/build_tablebase_cache.py`, the dual mating-action loops (per Python agent).
- **What**: the first loop computes `mating_actions` AFTER `board.push(move)`, so `action_from_move(move, board)` sees the post-move board. The second loop overwrites `mating_actions` so the corrupted first-loop result is unused. The dead code is currently harmless but is one delete away from silently breaking the snapshot cache.
- **Fix**: delete the first loop.

##### LOW — `__main__` shim leak on pickle failure in `build_starting_positions.py`

- **Where**: `scripts/build_starting_positions.py` (per Python agent — `__main__` shim cleanup).
- **What**: the script installs a `__main__` shim to unpickle TB caches but only restores `sys.modules["__main__"]` when `_prev is not None`. If `_prev is None` and `pickle.load` raises, the shim remains installed for the rest of the script's lifetime. `TablebaseCache.__init__` handles this correctly with an `else: del sys.modules["__main__"]`; the script doesn't.

##### LOW — Unreachable `tied.is_empty()` arm in `select_action`

- **Where**: `src/mcts/tree.rs:602-604`.
- **What**: `max_visits = visits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)` over a non-empty vector necessarily equals an actual element of `visits`, so the `(v - max_visits).abs() < f32::EPSILON` filter is guaranteed non-empty. The `tied.is_empty()` branch is dead. If a future change flips the epsilon comparison to strict `<`, this dead arm silently returns index 0 — reinstating the first-max bias the surrounding diff was meant to fix.
- **Fix**: replace with `unreachable!()` to fail loud if the invariant breaks, or drop the empty case entirely.

##### NOTE (not new in this commit) — `select_action` sampling branch overflows at temperature ≤ 0.01

- **Where**: `src/mcts/tree.rs:609-628`.
- **What**: with `temperature = 0.01` (the value used in `play_game` after the exploration window and in `play_game_dual` throughout), `inv_temp = 100`. `v.powf(100.0)` overflows f32 (`f32::MAX ≈ 3.4e38`, so any `v > ~2.4` overflows). All max-tied visit counts > ~2 become `f32::INFINITY`, the cumulative-sample loop returns the FIRST `inf`-weight action, and visit-distribution proportionality is lost. The random tie-break introduced for the `temperature ≤ ε` branch never executes in production (no caller passes 0). The previously-flagged "color asymmetry" cure via `legal_actions.sort_unstable()` works in practice because both colors now have the same `legal_actions[0]` — but selection is still effectively deterministic-first-max, just color-symmetric. Pre-existing; not introduced by this commit, but the commit message implies the tie-break fix runs in production when it does not.
- **Fix (if pursued)**: do the powf in `f64` and renormalize, or sample directly from a softmax-of-log-visits.

---

#### Summary

- 2 HIGH-severity bugs in the new Syzygy TB trajectory pipeline that are both silent and that compound when `HYZERO_TABLEBASE_FRAC > 0` with a trajectory cache. Either bug alone would degrade TB supervision; together they make the trajectory cache actively harmful for ~50% of positions.
- 1 MED gotcha that activates the moment policy-entropy regularization is combined with TB mixing.
- 1 MED latent unsoundness in the MCTS trace infrastructure (off by default).
- 3 LOW issues: dead code, error-path cleanup, unreachable arm.
- 1 NOTE on a pre-existing f32 overflow in `select_action` that the commit's narrative inadvertently misrepresents.

Recommend blocking trajectory-format TB rollout until the action-POV-flip and DTZ-direction bugs are fixed. The two LOW issues in the TB scripts can be cleaned up in the same patch.

