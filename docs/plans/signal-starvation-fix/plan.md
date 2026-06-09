# Plan: signal-starvation-fix

## Approach
Restore training signal to the value head by computing n-step TD value targets at the only
site with the full trajectory (`replay_buffer.rs::sample_batch`), carried on the **non-serde**
`TrainingSample` so no on-disk format changes; enabling decisiveness in self-play (guarded
resignation) plus eval-side adjudication-at-cap; and normalizing MCTS Q with MinMaxStats +
FPU. Every behavioral change is `HYZERO_*` env-gated with a sensible new default and is
additive to serde/PyO3/log contracts so existing artifacts and sign-convention regression
tests keep passing. The value-symmetry probe is surfaced as a structured log metric (plus an
optional flag-gated antisymmetry regularizer at weight 0) so progress becomes detectable.

REVIEW FIXES folded in: (1) td_target is NOT added to `StepRecord` — bincode is positional
and `#[serde(default)]` does not make it additive; the test at `replay_buffer.rs:332` would
break. It is a derived sample-time quantity carried on `TrainingSample`. (2) GameConfig gains
two fields, so EVERY exhaustive struct literal must be touched; those sites are enumerated and
owned. (3) Subtask file ownership re-cut so parallel subtasks are truly disjoint;
`coordinator.rs` and `selfplay.rs` ownership is explicit. (4) FPU fallback while MinMaxStats
is degenerate (max==min) is specified. (5) TD/β composition is pinned: on the TD path β→0.

## Subtasks

### 1. N-step TD value targets computed at sample time (no serde change)
- Files: `src/data/replay_buffer.rs`, `src/py/training.rs`
- Changes:
  - `replay_buffer.rs`: do NOT touch `types.rs` / `StepRecord`. Extend the **non-serde**
    `TrainingSample` (`replay_buffer.rs:11-16`) with `pub td_targets: Vec<Option<f32>>`
    (one entry per step in the K+1 slice; `None` ⇒ no TD target for that step). This struct
    is never bincode-serialized, so `test_checkpoint_roundtrip` (`replay_buffer.rs:332`) and
    all on-disk `ReplayBuffer.bin` / `.pt` artifacts are unchanged.
  - `sample_batch` (after slicing `steps[start..start+K+1]`, line 137): when TD is enabled,
    for each step at absolute trajectory index `t = start + k` compute the n-step TD return in
    that step's POV from the **full** `traj` (which IS available here — the K+1 slice is not):
    `G_t = Σ_{i=0}^{m-1} (-1)^i · γ^i · r_{t+i} + (-1)^m · γ^m · root_value[t+m]`
    where `n = HYZERO_TD_NSTEP` (default 5), `γ = HYZERO_TD_GAMMA` (default 0.997),
    `m = min(n, last - t)` and `last = traj.steps.len()-1`. The `(-1)^i` / `(-1)^m` factors
    convert each future reward and the bootstrap value into step-t's POV — this matches the
    canonical backup recurrence `G_{k-1}=r_k−G_k` (γ=1) in `tree.rs:705,733`, generalized to
    arbitrary γ. If `t+m == last` (window runs to the trajectory end) the bootstrap term uses
    the terminal signal instead of `root_value`: replace `root_value[t+m]` with
    `traj.game_outcome` converted to step-t POV via `(-1)^(last-t) · outcome_sign(traj, t)`
    (same sign chain training.rs uses to put White-absolute outcome into step POV). Per-step
    `root_value` and terminal `reward` are already stored (`game_task.rs:528-536,600-603`).
    Push the computed `G_t` into `TrainingSample.td_targets[k]`; when TD is disabled push
    `None`. Add cached env helpers `td_enabled()` (`HYZERO_TD`, default ON),
    `td_nstep()`/`td_gamma()` mirroring the existing helper at `replay_buffer.rs:70-74`.
  - `training.rs::assemble_batch_arrays` (value-target line 218-232): the `for k in 0..kp1`
    loop already iterates each step; thread the parallel `sample.td_targets[k]` in. When it
    is `Some(g)`, set `root_value_target = flip_sign * g` (apply the same `flip_sign` for
    color-aug) AND force `effective_beta = 0.0` for that step (see TESTING / composition below).
    When `None` (TD disabled or legacy path), behavior is byte-for-byte unchanged
    (`(1-β)·flip_sign·root_value + β·outcome`). Reconcile DOC DRIFT: fix the stale "default
    0.3" comments at `training.rs:221,224` and the test docstring (1238) to the actual default
    0.1; do not change the runtime default.
- TD/β composition (explicit, resolves review #5): the n-step return `G_t` already bootstraps
  toward the eventual game outcome through its discounted tail (the terminal-bootstrap branch
  literally substitutes `game_outcome`). Re-adding `β·outcome` on top would double-count the
  outcome through the TD tail. Therefore **when a step has a TD target, β collapses to 0** and
  the final value target is exactly `target = flip_sign · G_t` (with `G_t` already POV-correct
  and outcome-aware via its tail). β/`HYZERO_VALUE_OUTCOME_BETA` and conditional-β retain their
  current meaning ONLY on the legacy (`td_targets[k]==None`) path.
- Tests:
  - `replay_buffer.rs`: `td_target_equals_signed_discounted_reward_plus_bootstrap` (hand-built
    3-step trajectory with known rewards/root_values; assert exact `G_t` per step under known
    γ/n including the `(-1)^i` sign chain; serialize via the `Mutex` env lock pattern at
    `replay_buffer.rs:264`). FAILS without the new computation.
  - `replay_buffer.rs`: `td_target_uses_outcome_bootstrap_at_trajectory_end`.
  - `replay_buffer.rs`: `td_disabled_yields_all_none_td_targets`.
  - `training.rs`: `td_target_overrides_root_value_and_zeroes_beta` (asserts the value target
    equals `flip_sign·G` regardless of β; FAILS without the override + β→0 branch) and
    `legacy_none_td_target_preserves_old_value_target` (β-blend identical to today).
- Dependencies: none.

### 2. GameConfig adjudication fields + ALL construction sites (foundation)
- Files: `src/selfplay/game_task.rs`, `src/selfplay/coordinator.rs`
- Rationale for ownership: adding fields to `GameConfig` (defined in `game_task.rs:230-239`)
  breaks every exhaustive struct literal. The literals live in: `coordinator.rs:122`,
  `selfplay.rs:470`, `evaluation.rs:253`, and `game_task.rs` tests `1004/1104/1162/1295`.
  This subtask owns `game_task.rs` (definition + its 4 test literals + self-play behavior)
  AND `coordinator.rs` (its one literal). `selfplay.rs:470` and `evaluation.rs:253` are owned
  by subtask 3. No other subtask touches these two files, so file lists stay disjoint.
- Changes:
  - `game_task.rs`: add to `GameConfig` (`:230-239`, additive — GameConfig is NOT
    serde-persisted): `pub adjudicate_at_cap: bool` and `pub adjudication_material_margin: i32`.
    `GameConfig::default()` (`:241-250`) sets `adjudicate_at_cap=false`,
    `adjudication_material_margin=5` so all existing callers/self-play are unchanged. Update
    the 4 exhaustive test literals (`1004/1104/1162/1295`) by appending the two fields
    (or switching them to `..GameConfig::default()` where the test doesn't assert them).
  - `coordinator.rs:122`: append the two fields to that `GameConfig` literal (use defaults:
    `adjudicate_at_cap:false, adjudication_material_margin:5`). `coordinator.rs:21` already
    uses `GameConfig::default()` and needs no change.
  - Self-play behavior in `play_game` (self-play, NOT dual):
    - Resignation: add cached helpers `resign_enabled()` (`HYZERO_RESIGN`, default ON),
      `resign_threshold()` (`HYZERO_RESIGN_THRESHOLD`, default `-0.90`, clamp `[-1.0,-0.5]`),
      `resign_plies()` (`HYZERO_RESIGN_CONSECUTIVE`, default 4), `resign_min_ply()`
      (`HYZERO_RESIGN_MIN_PLY`, default 30 — never resign during the high-temperature
      exploration window). Track consecutive plies where the side-to-move's `root_value <=
      threshold`; on reaching `resign_plies` end the game and set `game_outcome` to a WIN for
      the opponent (±1, `is_draw=false`). PASSIVITY-ATTRACTOR GUARD (research
      `game_task.rs:569-574`): resignation is value-based, not material-adjudication, so it
      cannot be gamed by shuffling to preserve material — passive play that avoids checkmate
      still drives root_value negative and triggers resignation. Material shaping stays OFF.
    - Annealed temperature: replace the hard step at `game_task.rs:498-502` with a linear
      anneal 1.0→0.01 over `HYZERO_TEMP_ANNEAL_PLIES` (default 60) once past
      `temperature_moves`; gate with `HYZERO_TEMP_ANNEAL` (default ON). Existing step behavior
      preserved when OFF.
  - Eval cap-adjudication in `play_game_dual`: at the cap branch (`game_task.rs:382-386`),
    when `config.adjudicate_at_cap` and `result()` is non-checkmate, award ±1 to the side
    ahead by ≥ `config.adjudication_material_margin` material (reuse `compute_material_diff`,
    `game_task.rs:903-913`), else 0. Default config keeps this OFF, so self-play and existing
    callers see no change.
- Tests (all in `game_task.rs`):
  - `resigns_after_consecutive_losing_plies_below_threshold` and
    `does_not_resign_before_min_ply` (FAIL without the counter logic).
  - `temperature_anneals_linearly_within_window`.
  - `dual_game_adjudicates_material_lead_at_cap` and
    `dual_game_draws_when_material_within_margin` (FAIL without adjudication branch).
  - Existing sign test `game_task.rs:1281` must still pass.
- Dependencies: none.

### 3. Eval ladder: enable adjudication, more games, opening diversity
- Files: `src/selfplay/evaluation.rs`, `src/bin/selfplay.rs`
- Ownership: owns `evaluation.rs` (incl. its `GameConfig` literal at `:253`) and `selfplay.rs`
  (incl. its `GameConfig` literal at `:470`). Disjoint from subtask 2 (`game_task.rs` +
  `coordinator.rs`).
- Changes:
  - `evaluation.rs:253`: append the two new GameConfig fields, sourcing them from env:
    `adjudicate_at_cap = HYZERO_EVAL_ADJUDICATE` (default ON — eval outcomes never enter
    training targets, so adjudication here is safe and the antisymmetry/passivity risk does
    not apply), `adjudication_material_margin = HYZERO_EVAL_ADJ_MARGIN` (default 5). Add the
    two cached env helpers in `evaluation.rs`.
  - `selfplay.rs:470`: append the two fields to that self-play `GameConfig` literal using
    defaults (`adjudicate_at_cap:false, adjudication_material_margin:5`) — self-play must NOT
    adjudicate.
  - Raise discrimination: `games_per_side` default 4→8 in `EvaluationConfig::default()`
    (`evaluation.rs:73`); it is already env-plumbed through `config.games_per_side`
    (`selfplay.rs:497`). Update the default-asserting test (`evaluation.rs` default test).
  - Opening diversity (no new subsystem): `init_self_play_board` already honors
    `HYZERO_STARTS_FILE` (`game_task.rs:144-187`) and is called inside `play_game_dual` — just
    document that setting it diversifies eval openings. No code change for this bullet.
  - The `[eval] ... ladder_match` println (`evaluation.rs:506-521`) is UNCHANGED — adding
    games/adjudication only changes numeric values of existing fields, not the format, so the
    `run_baseline.sh` greps for `win_rate=`/`candidate_elo=` keep working.
- Tests (`evaluation.rs`):
  - `default_games_per_side_is_eight` (update existing default test).
  - `eval_game_config_enables_adjudication_when_env_set` (set `HYZERO_EVAL_ADJUDICATE`,
    construct the config path, assert `GameConfig.adjudicate_at_cap==true`; FAILS without
    wiring; uses the module `Mutex` env-lock pattern).
- Dependencies: subtask 2 (consumes the `GameConfig.adjudicate_at_cap` /
  `adjudication_material_margin` fields it adds). Build-orders after 2.

### 4. MCTS MinMaxStats Q-normalization + FPU
- Files: `src/mcts/puct.rs`, `src/mcts/node.rs`, `src/mcts/tree.rs`
- Changes:
  - `node.rs`: add `MinMaxStats { min: f32, max: f32 }` with `new()` (min=+INF, max=-INF),
    `update(q)`, and `normalize(q)`. FPU/normalization degenerate-window fallback (resolves
    review #4): `normalize` returns `(q-min)/(max-min)` only when `max > min + EPS`
    (EPS=1e-8); otherwise (no two distinct Q's seen yet, the common early-search case) it
    returns the **pass-through** `q`. Add an explicit `is_degenerate()` helper.
  - `puct.rs`: add `select_child_normalized(node, c, stats, fpu)` and keep the existing
    `select_child`/`puct_score`/`puct_score_detail` byte-for-byte so the raw-Q tests
    (`puct.rs:89-192`) pass. In the new fn:
    - VISITED child: use `stats.normalize(child_q)`.
    - UNVISITED child: FPU = `parent_q_norm − HYZERO_FPU_REDUCTION` (default 0.25,
      `HYZERO_FPU`, clamp `[0.0,1.0]`), where `parent_q_norm = stats.normalize(parent.q())`.
      DEGENERATE-WINDOW FALLBACK: when `stats.is_degenerate()` (max==min — true on the first
      visits each search, since MinMaxStats has seen ≤1 distinct Q), the subtraction
      `parent_q_norm − 0.25` would push every unvisited child to a uniform negative value and
      re-create the "exploration dominates uniformly" failure. In that case fall back to
      FPU = `parent_q_norm` (no reduction) — i.e. unvisited children inherit the parent's
      (pass-through) value with no pessimism until the window has scale. Once `max>min`,
      apply the full reduction.
    - Gate the whole path with `HYZERO_MCTS_QNORM` (default ON); when OFF, `tree.rs` calls the
      legacy `select_child`.
  - `tree.rs`: add `min_max: MinMaxStats` to `MCTSTree` (`tree.rs:266-269`), reset per
    `run_simulations`; update it in `backpropagate` (`tree.rs:738-749`) with each node's
    running Q AFTER the visit/total update. Thread `&min_max` + fpu into both `select_child`
    call sites (`tree.rs:417,651`) when QNORM is on. Normalization affects ONLY selection,
    never `total_value`/`reward` stored in backprop, so `tree.rs:1140` (alternating signs) and
    `tree.rs:1291` (mating reward) are unaffected.
- Tests:
  - `node.rs` (new `#[cfg(test)] mod tests`): `minmax_normalizes_to_unit_range`,
    `minmax_passes_through_when_degenerate`.
  - `puct.rs`: `fpu_pessimism_deprioritizes_unvisited_when_parent_low` (non-degenerate stats),
    `fpu_falls_back_to_parent_value_when_window_degenerate`,
    `normalized_selection_prefers_high_q_after_minmax` (FAIL without the new fn).
  - `tree.rs`: `qnorm_does_not_change_stored_total_value` (run sims with QNORM on/off; assert
    backprop totals identical) — guards the sign regressions.
- Dependencies: none (file-disjoint).

### 5. Wire value-antisymmetry into baseline-visible output (+ optional regularizer)
- Files: `python/hyzero/training/trainer.py`, `scripts/run_baseline.sh`
- Changes:
  - Promote `[sym_probe_batch]` (`trainer.py:748-753`) from a periodic 50-step diagnostic to a
    per-`train_batch` STRUCTURED metric: emit a stable anchor line
    `[antisym] step=<v> mean_sum=<f> corr=<f>` via `_diag_print` every call (move the ≤10-
    sample batch probe out of the `model_version % 50` gate but cap N at
    `HYZERO_ANTISYM_PROBE_N` default 8 to bound cost). The 5-key return dict
    (`trainer.py:861-869`) and `train_batch` signature are UNCHANGED (bridge-safe).
  - `run_baseline.sh`: add an awk extractor for `[antisym]` lines mirroring the existing
    `ladder_match` extractors (`run_baseline.sh:188-225`); write additive field
    `last_antisym_mean_sum` (latest) into `logs/baseline_score.json` (`run_baseline.sh:291-327`).
  - OPTIONAL flag-gated regularizer: in the loss block (`trainer.py:847-852`), when
    `_parse_loss_weight_env("HYZERO_ANTISYM_LOSS_WEIGHT", default=0.0) > 0`, add
    `w · mean((f(h(obs))+f(h(flip(obs))))^2)` using `_flip_obs_planes` (already imported,
    `trainer.py:50`). Default 0.0 ⇒ zero extra forward passes and identical loss to today.
- Tests:
  - `python/hyzero/training/`: pytest `test_antisym_loss_zero_when_weight_unset` (asserts
    `total_loss` unchanged vs baseline with weight=0) and
    `test_antisym_loss_penalizes_nonantisymmetric_value` (stub where `v(flip)=v`, assert
    positive penalty when weight>0; FAILS without the loss term).
  - `run_baseline.sh`: a shellcheck-clean awk parse unit asserting `last_antisym_mean_sum` is
    extracted from a sample `[antisym]` fixture line.
- Dependencies: none (file-disjoint; pure Python + script).

## File-ownership summary (disjoint check)
- Subtask 1: `replay_buffer.rs`, `training.rs`
- Subtask 2: `game_task.rs`, `coordinator.rs`
- Subtask 3: `evaluation.rs`, `selfplay.rs` (orders after 2 for the shared field contract)
- Subtask 4: `puct.rs`, `node.rs`, `tree.rs`
- Subtask 5: `trainer.py`, `run_baseline.sh`
No file appears in two subtasks. `types.rs` is NOT touched by anyone (review #1). Every
GameConfig literal (`coordinator.rs:122`, `selfplay.rs:470`, `evaluation.rs:253`,
`game_task.rs:1004/1104/1162/1295`) has exactly one owner.

## Testing strategy
1. Unit: `cargo test` (subtasks 1-4 — Rust inline `#[cfg(test)]`, env tests serialized via
   module `Mutex`) and `pytest python/` (subtask 5). Each NEW test must FAIL on the pre-fix
   tree (per testing rule) and pass after.
2. Serde-compat guard: `test_checkpoint_roundtrip` (`replay_buffer.rs:332`) must still pass —
   it does, because no serde type changed.
3. Regression guard: sign tests (`training.rs:1442,1349`, `game_task.rs:1281`,
   `tree.rs:1140,1291`, `puct.rs:89-192`) still pass with all defaults ON.
4. Smoke integration: short `bash scripts/run_baseline.sh 1800` with new defaults. Verify in
   `logs/baseline_score.json` + selfplay log: nonzero decisive fraction (resignations in
   `[cm_count]`/PGN), `[antisym] mean_sum` trending toward 0, `last_win_rate`/
   `last_candidate_elo` diverging from 0.5/1500, `[value_spread]` (`trainer.py:790-799`)
   widening.
5. A/B confidence: re-run smoke with `HYZERO_TD=0 HYZERO_RESIGN=0 HYZERO_MCTS_QNORM=0
   HYZERO_EVAL_ADJUDICATE=0` (near-legacy) to confirm deltas come from the fixes, not noise.

## Rollback
Every behavioral change has an env kill switch: `HYZERO_TD=0` (value target reverts to
`(1-β)·root_value + β·outcome` exactly), `HYZERO_RESIGN=0`, `HYZERO_TEMP_ANNEAL=0`,
`HYZERO_EVAL_ADJUDICATE=0`, `HYZERO_MCTS_QNORM=0`, `HYZERO_ANTISYM_LOSS_WEIGHT=0`. Set all
off to recover pre-fix runtime behavior without code revert. No on-disk format changed
(`TrainingSample` is non-serde; `GameConfig` is non-serde), so old `ReplayBuffer.bin` and
`.pt` checkpoints load unchanged and new buffers remain readable by old code. Do work on
branch `feat/signal-starvation-fix`, one logical commit per subtask (`{scope}: {description}`);
revert a subtask by reverting its commit — file lists are disjoint except subtask 3 builds on
the GameConfig fields from 2, so revert 3 before 2 if unwinding both.
