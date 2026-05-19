# Review Log

Tracks bug-focused code reviews performed by Claude on this branch.

## Schema

Each review entry records:

- `version`: monotonically increasing integer
- `date`: ISO date of the review
- `head_sha`: full SHA of HEAD at review time
- `range`: git range reviewed (e.g., `4382bde..ee132c4`)
- `files`: files actually opened/inspected
- `findings`: numbered list with severity (bug | suspicious | nit) and location

A finding is **not** a TODO list — it is the snapshot of what the review saw.
If a finding is later fixed, leave the entry alone and add a follow-up entry
referencing it. The log is append-only.

---

## Review v1 — 2026-05-19

- **head_sha**: `ee132c4ec4dc32f61d688af6f776a0424e56e2d3`
- **range**: `4382bde..ee132c4` (single squash commit "TB supervision + canonical MuZero backup + diverse starts")
- **scope**: bug-focused review, parallelized across four sub-agents + direct reads of `src/mcts/tree.rs` and `src/mcts/puct.rs`
- **files inspected**:
  - `src/mcts/tree.rs`, `src/mcts/puct.rs`
  - `src/py/training.rs`
  - `src/selfplay/game_task.rs`
  - `src/data/encoding.rs`
  - `python/hyzero/data/tablebase.py`
  - `python/hyzero/data/board_encoder.py`
  - `python/hyzero/training/trainer.py`
  - `scripts/build_tablebase_cache.py`
  - `scripts/build_tablebase_trajectory_cache.py`
  - `scripts/build_starting_positions.py`
  - `scripts/pretrain_dynamics.py`
  - `scripts/gen_pretrain_dynamics.py`
  - `scripts/rebalance_tb_cache.py`

### Findings (severity ordered)

#### F1 — **bug** — TB trajectory cache picks the _fastest-losing_ move for losing/drawn positions

- **Location**: `scripts/build_tablebase_trajectory_cache.py:271-284` (`_find_optimal_moves`) and the same shape in `scripts/build_tablebase_cache.py:395-414`.
- **What**: After `board.push(m)`, `tb.probe_dtz(board)` returns DTZ from the **opponent's** POV. The code ranks by `min(|DTZ|)`. Correct Syzygy "optimal play" is rank by `(−wdl_after, sign·|DTZ|)`: among wins, min |DTZ|; among losses, max |DTZ|. The current rule picks the fastest mate for winning STM (correct) but picks the move that _loses fastest_ for losing STM and can pick a blunder for drawn STM.
- **Why it matters**: `optimal_actions` is the supervised policy target. Roughly half of TB positions (losses + draws — KvK class is mostly draws) get poisoned targets. Reward target sign for the trajectory rollout also loses meaning (F2). This silently undermines the entire TB-supervision recovery the commit was built for; the 0.45-TB-frac 2h run scored 8.16 vs. the absolute β=0.3 baseline of 14.51, and this bug is consistent with the gap.

#### F2 — **bug** — Trajectory rollouts in losing/drawn positions fire `target_rewards[1]=+1` with no signed POV

- **Location**: `scripts/build_tablebase_trajectory_cache.py:330-344` interacting with F1.
- **What**: After `_find_optimal_moves` returns the worst-defensive move in a losing position, the rollout often resolves to checkmate one ply later. `target_rewards[k+1] = 1.0` is set unconditionally — there is no sign convention attached to _who_ delivered mate. In a losing-STM root, STM was just mated, but the reward target says `+1`.
- **Why it matters**: Pollutes the reward head with sign-incorrect terminal signals on the same fraction of positions as F1.

#### F3 — **bug** — Consistency loss includes absorbing-state zero observations from TB trajectory rows

- **Location**: `python/hyzero/training/trainer.py:818-834` interacting with `python/hyzero/data/tablebase.py:339,367-371,345`.
- **What**: TB trajectory rows are emitted with `is_tablebase=False` (line 345) so the trainer applies the full K-step + consistency loss. But trajectory rows have `fens[k]=None` past the mate step and the encoder leaves the corresponding observation at all-zeros (line 339 init + `continue` line 371). The consistency-loss exclusion at line 832 (`cos_sim[~is_tb_tensor]`) only protects snapshot rows; trajectory absorbing steps are _not_ excluded. As a result `self.h(zeros)` becomes a SimSiam target for `g(real_latent, action)`, pulling the dynamics network to collapse latents toward `h(0)` whenever a trajectory reached terminal early.
- **Why it matters**: Silent dynamics-net corruption proportional to the fraction of TB trajectories that terminate before K plies. The trainer comment at line 818-819 explicitly intends this to be protected but the protection only catches the legacy snapshot format.

#### F4 — **bug** — Loss normalization differs between step 0 and steps 1..K when TB rows are present

- **Location**: `python/hyzero/training/trainer.py:594-598` vs `618-634`.
- **What**: Step 0 uses `F.mse_loss(...)` and `self._policy_loss(...)` — full-batch mean over all B rows. Steps 1..K use `(per_sample * non_tb).sum() / non_tb_count` — mean over the non-TB subset. With `tb_frac=0.45`, a non-TB row contributes ≈ `1/B` at step 0 but ≈ `1/(0.55·B)` at steps 1..K. After `avg = total/(K+1)`, the gradient is biased.
- **Why it matters**: Not catastrophic but biases the learning rate per step in a way the K-step weighting was not designed for.

#### F5 — **bug** — `_reinit_value_head` uses `nonlinearity='linear'` for trunk Linear layers

- **Location**: `python/hyzero/training/trainer.py:377` (Kaiming init loop).
- **What**: `kaiming_normal_(weight, nonlinearity='linear')` gives `gain=1`, but trunk Linear layers feed a ReLU (gain should be `sqrt(2)`). Only the final layer (before tanh) is correctly `linear`. The loop applies the same mode to every linear layer in `value_head`.
- **Why it matters**: After reinit-on-load (`HYZERO_REINIT_VALUE_HEAD=1`), trunk activations are under-scaled and the value head may saturate near tanh(+0.3)≈0.29 regardless of input until SGD recovers. Combined with `+0.3` bias this matches the observed "kqk_value peaked +0.85 for ~3k steps" behavior in the commit message.

#### F6 — **bug** — PGN sampling at 1% races between concurrent self-play tasks

- **Location**: `src/selfplay/game_task.rs:559-572` calling `write_pgn_game` (`src/selfplay/pgn.rs:9-59`).
- **What**: `OpenOptions::new().create(true).append(true).open(path)` is taken per call, with multiple `writeln!` calls per game (one per PGN tag + one per move chunk). Only a single `write(2)` ≤ PIPE_BUF on `O_APPEND` is atomic; concurrent tokio tasks can interleave header tags and move lines.
- **Why it matters**: The sample is debug/analysis only (not training), but PGN parsers will reject interleaved games and silently skip them. Fix: a process-global `Mutex<BufWriter<File>>` as already used by `summary_writer()` elsewhere.

#### F7 — **suspicious** — Sampled-start FEN handling does not retry on terminal/invalid positions

- **Location**: `src/selfplay/game_task.rs:185` (`init_self_play_board`).
- **What**: If the sampled FEN is already terminal or fails to construct, the code logs once and falls back to the default starting position for that game. There is no bounded retry. A FEN file with many terminal positions silently degrades to all-standard-start games.
- **Why it matters**: Loss of training diversity is invisible — the only signal is the log line, which is easy to miss across hundreds of concurrent tasks.

#### F8 — **suspicious** — `is_trajectory_format` is inferred from `hasattr(first_entry, "fens")`

- **Location**: `python/hyzero/data/tablebase.py:151`.
- **What**: Only the first cache entry is inspected. A mixed cache routes every entry through the wrong builder; partially-corrupt pickles can yield surprising failures or silent skips. The classes use disjoint attribute names (`fen` vs `fens`), so it's robust for clean caches, but there's no schema validation.
- **Why it matters**: A buggy cache script could produce a heterogeneous file and the trainer would NaN-crash or skip TB supervision silently.

#### F9 — **suspicious** — Biased value-head reinit fires every checkpoint load with `HYZERO_REINIT_VALUE_HEAD=1`

- **Location**: `python/hyzero/training/trainer.py:1058-1059`.
- **What**: No idempotency guard — leaving the env var set in the run script wipes the value head on every process restart, including after the value head has recovered.
- **Why it matters**: Long training runs that auto-restart will repeatedly destroy progress on the value head. Needs a one-shot checkpoint marker.

#### F10 — **suspicious** — `target_values` POV / `original_root_side_sign` parity invariant is implicit

- **Location**: `src/py/training.rs:184,212-214`.
- **What**: `original_root_side_sign` reads `steps[0].white_to_move`; `ply_flip = (-1)^k` then assumes strict alternation per ply. Today this holds, but a future change (sampled-start games that splice non-consecutive plies, or null-move padding) would silently corrupt every value target. A `debug_assert!(step.white_to_move == (steps[0].white_to_move ^ (k % 2 == 1)))` would pin the invariant.
- **Why it matters**: The invariant is enforced 1000 lines away in `selfplay/game_task.rs`; cross-file invariants without an assertion are exactly the kind of bug that bit this codebase before (color-asymmetry, commit 7243aec).

#### F11 — **suspicious** — Endgame piece-count filter contradicts docstring

- **Location**: `scripts/build_starting_positions.py:163` vs module docstring at lines 9-11 and CLAUDE.md "Metric" note.
- **What**: Filter is `p_count < 3 or p_count > 12` (accepts 3-12). Docstring and CLAUDE.md claim "7-12 pieces" for the endgame bucket. The actual bucket contains 3-piece KvK-class positions, which are trivial / immediate-draw.
- **Why it matters**: Documented training-data distribution is false. Either the comment or the filter is wrong; fix one to match the other.

#### F12 — **suspicious** — `rebalance_tb_cache.py` does not seed RNG and does not balance the draw bucket

- **Location**: `scripts/rebalance_tb_cache.py:81-84`.
- **What**: `random.sample` runs without `random.seed`, so balanced output is non-reproducible. The draw bucket (`list_zero`) is kept in full, so a typical Syzygy distribution (~50% draws) remains ~50% draws after "balancing" — only the win/loss bucket is balanced.
- **Why it matters**: Ablations that re-balance from the same source produce different caches each time; the "balanced" claim is misleading.

#### F13 — **suspicious** — Dead duplicate mate-detection loop in `_probe_position`

- **Location**: `scripts/build_tablebase_cache.py:374-392`.
- **What**: Lines 374-381 build `mating_actions` from `action_from_move(move, board)` _after_ `board.push(move)` (wrong board state). Lines 384-392 reassign and recompute correctly. The first loop is dead code.
- **Why it matters**: Cosmetic, but signals an incomplete refactor — easy to mistakenly trust the first loop in a later edit.

#### F14 — **nit** — `target_rewards[0]` is computed but never read

- **Location**: `src/py/training.rs:235-236`; consumed by `python/hyzero/training/trainer.py:589,632`.
- **What**: Rust populates slot 0 with `(1-γ)*step[0].reward + γ*outcome_term`. The trainer inserts a zero placeholder at step 0 (line 589) and only averages reward over k=1..K (line 807, `/ k_steps`). The Rust slot is therefore dead but documented as live by the shape doc.

#### F15 — **nit** — `dirichlet_noise` samples and discards `x` per loop iteration

- **Location**: `src/mcts/tree.rs:189-198`.
- **What**: `let x: f32 = rng.random::<f32>() * 6.0 - 3.0;` is sampled before the proper Box-Muller `z`, then suppressed with `let _ = x;` to silence the unused-warning. Wastes one RNG draw per Marsaglia-Tsang iteration.

#### F16 — **nit** — Unused `math` import in `build_tablebase_cache.py:28`.

#### F17 — **nit** — GIL held during pure-Rust batch assembly

- **Location**: `src/py/training.rs:521-523` wraps `assemble_batch_arrays` (~6 MB f32 allocation per batch) inside `Python::attach`. Not a correctness issue; perf-only.

### Verified-correct (intentionally listed so a future reviewer can skip)

- **MCTS canonical backup** (`src/mcts/tree.rs:505-541`): `G_{k-1} = r_k − G_k`, root stores `G_0` own-POV, depth-k stores `G_{k-1}` in parent-POV. Manual trace through `test_backpropagate_alternates_signs` (D=1/2/3) and `test_backpropagate_includes_mating_reward` matches.
- **PUCT sign convention** (`src/mcts/puct.rs:41-83`): reads `child.q_value` directly; tied scores broken uniformly at random (`test_select_child_breaks_ties_uniformly`).
- **`select_action` tie-break** (`src/mcts/tree.rs:585-607`): max-visit ties broken uniformly at random — satisfies `.claude/rules/mcts-pov-symmetry.md`.
- **Action-list sort + flip order** in both `play_game` and `play_game_dual` matches the rule (`sort_unstable()` after `flip_action()`).
- **Underpromotion flip invariant** (`encode_action_spatial_for_color` ↔ `flip_action_planes`): manually traced for white `e7→e8=knight`; both colors produce planes that are exact `flip_sq` of each other. `test_flip_action_planes_matches_flip_action_invariant` covers all 4672 × 2.
- **Plane-101 removal**: every consumer (`src/data/encoding.rs`, `python/hyzero/data/board_encoder.py`, `python/hyzero/training/trainer.py`) treats plane 101 as halfmove clock; no stale side-to-move references.
- **Terminal reward POV** (`src/selfplay/game_task.rs:539-542` → `src/py/training.rs:234`): step-POV reward consumed with one global `flip_sign` flip; pinned by `test_terminal_reward_in_step_pov_not_white_absolute`.
- **`pretrain_dynamics.py` freeze**: `f.requires_grad=False`, `f` never forwarded — `f.eval()` not strictly required because BN running stats only update on forward.

### Suggested fix order (highest-impact first)

1. F1 + F2 together — DTZ direction fix in `_find_optimal_moves` + signed reward target in the trajectory builder. This is the single biggest correctness fix and would directly affect the next TB-supervision run.
2. F3 — exclude absorbing-step rows from consistency loss (per-step mask, not per-row).
3. F5 — Kaiming `nonlinearity='relu'` for trunk layers in `_reinit_value_head`.
4. F4 — unify step-0 vs steps-1..K loss normalization when TB rows are mixed.
5. The rest can be batched as cleanup.

---
