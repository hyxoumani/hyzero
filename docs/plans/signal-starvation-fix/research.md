# Signal-Starvation Fix — Research (Phase 1)

Diagnosis: training makes no progress because the supervised value/reward signal is
near-zero almost everywhere, self-play almost never produces decisive games, the eval
ladder cannot discriminate, MCTS Q is un-normalized, and a measured value-antisymmetry
violation is logged but never gated. All five problems verified below against current code.

## What exists today

- Value target = `(1-β)·rootQ + β·outcome` per step (training.rs:226-232). `rootQ` is the
  same-ply MCTS root Q (`StepRecord.root_value`) — no n-step TD, no bootstrap window.
- Reward target = `(1-γ)·step.reward + γ·outcome` (training.rs:235-237); `step.reward` is
  non-zero only on the trajectory's last step (game_task.rs:533, 600-603). γ default 0.0.
- Non-checkmate terminals → `outcome=0.0, is_draw=true` (game_task.rs:579-595). Material
  shaping (`tanh(Δmaterial/scale)`) exists but is OFF by default.
- Self-play: `MAX_GAME_LENGTH=300` (game_task.rs:413), temperature 1.0 for first 30 plies
  then 0.01 (game_task.rs:498-502; `temperature_moves` default 30, game_task.rs:246).
  No resignation, no adjudication. Dirichlet noise on (ε=0.25, α=0.3; tree.rs:136-140).
- Eval: `play_game_dual` — T=0.01 fixed (game_task.rs:351), `add_root_noise=false`
  (game_task.rs:286), 300-ply cap, checkmate-only outcome (game_task.rs:382-386).
  `games_per_side=4`, `num_simulations=50`, `temperature_moves=15` (evaluation.rs:71-86).
- Replay buffer: prioritized by decisive fraction `HYZERO_DECISIVE_SAMPLE_FRAC` default
  0.25 (replay_buffer.rs:62-147); sampling window = `traj.steps[start..start+K+1]`,
  `start ∈ 0..(len-K)` (replay_buffer.rs:134-137). Full per-step `root_value` IS stored,
  so n-step/TD-λ CAN be computed at sample time here.
- MCTS Q-init for unvisited child = 0.0 (node.rs:56-62, puct.rs:48-59). No MinMaxStats /
  Q-normalization anywhere. Backup is canonical MuZero `G_{k-1}=r_k−G_k` (tree.rs:713-749).
- Loss = weighted policy+value+reward+consistency (trainer.py:847-852); per-loss env
  weights default 1.0 (trainer.py:469-472), consistency default 0.5 (trainer.py:823).

## Patterns & conventions

- Env-var config is the dominant knob pattern: each feature reads `HYZERO_*` via a small
  cached helper (e.g. training.rs:58-88, tree.rs:143-164, game_task.rs:868-898). Defaults
  declared inline in those helpers (Rust) and via `_parse_loss_weight_env` (trainer.py:322).
- Error handling: `PyResult`/`io::Error` propagation in lib code; self-play swallows+logs
  (`eprintln!`) so a bad game never aborts the run (game_task.rs:213-220, 641).
- Tests: inline `#[cfg(test)] mod tests` in every Rust module (training.rs:633, puct.rs:85,
  tree.rs:862, node has none, replay_buffer.rs:174, game_task.rs:915, evaluation.rs:562).
  Env-mutating tests serialize via module-local `Mutex` locks (training.rs:1122,
  replay_buffer.rs:264). Testing rule: regression tests must FAIL without the fix.
- Python diagnostics use `_diag_print` (multi-fallback to fd 1/2; trainer.py:12-42), NOT
  gated — they run every `train_batch`.

## What can't change

- PyO3 bridge signatures used by `python/`: `train_batch(batch_dict)` returns the 5-key
  loss dict (training.rs:322-331; trainer.py:861-869); `notify_trajectory(outcome, is_draw)`;
  `get_weights`/`load_weights(bytes)`/`save_checkpoint(path, metrics)`/`load_checkpoint`.
  batch_dict keys/shapes fixed (training.rs:294-319).
- `ReplayBuffer` is `serde::Serialize` (replay_buffer.rs:19) and checkpointed via bincode
  (replay_buffer.rs:162-171). `StepRecord`/`GameTrajectory`/`BoardObservation` are all
  serde+bincode persisted (types.rs:41-105). Reordering/removing fields breaks on-disk
  replay/buffer artifacts and `.pt` checkpoints across runs — additive only, or version.
- DEFAULT_CONFIG keys consumed by model ctors (config.py:3-11).

## What could break

- Dual-model eval consumers: `DualGameOutcome{game_outcome,num_moves,moves}` (game_task.rs
  :252-261); champion-perspective negation convention (sign tests evaluation.rs:997-1014,
  game_task.rs:1186-1224).
- `scripts/run_baseline.sh` greps `[eval] ... ladder_match` for `win_rate=` and
  `candidate_elo=` (run_baseline.sh:188-225) and writes `logs/baseline_score.json` fields
  incl. `last_win_rate`, `last_candidate_elo`, `promotions` (run_baseline.sh:291-327).
  The log line format (evaluation.rs:506-521) is a load-bearing contract.
- Sign-convention regression tests: training.rs:1442 (value sign under flip),
  training.rs:1349 (terminal reward POV), game_task.rs:1281 (terminal reward POV).
- MCTS backup-sign tests: tree.rs:1140 (alternating signs), tree.rs:1291 (mating reward).
  Any Q-normalization must preserve these. PUCT tests puct.rs:89-192 assume raw Q.

## Per-problem detail

### 1. Value-target zero-attractor (training.rs)
- Formula at training.rs:231-232: `target = (1-effective_beta)*flip_sign*root_value
  + effective_beta*outcome_in_step_perspective`. `outcome_in_step_perspective` =
  `flip_sign*game_outcome*original_root_side_sign*ply_flip` (training.rs:212-214).
- β from `HYZERO_VALUE_OUTCOME_BETA` default 0.1 (training.rs:58-64). `effective_beta=1.0`
  for decisive games iff `HYZERO_CONDITIONAL_BETA` truthy (training.rs:69-77, 226-230);
  default OFF. DOC DRIFT: comment at training.rs:222-225 and test docstring (1238) say
  "default 0.3"; the actual default is 0.1. Planner should reconcile.
- Reward γ from `HYZERO_REWARD_OUTCOME_GAMMA` default 0.0 (training.rs:82-88). With shaping
  off, both rootQ→0 and outcome→0 for the ~all-draw stream → target ≈ 0 everywhere.
- Signal enters via `StepRecord{root_value, reward}` (game_task.rs:528-536) and terminal
  reward POV-set (game_task.rs:600-603); outcome from `GameTrajectory.game_outcome`.
- n-step TD would compute, at sample time, `G = Σ γ^i r_{t+i} + γ^n V(s_{t+n})` from the
  per-step `root_value` already stored. Best site: replay_buffer.rs sample_batch (full
  trajectory + window known there, replay_buffer.rs:134-137) OR a new TrainingSample field
  carrying the bootstrap value; assembling in training.rs only has the K+1-step slice, so
  the n-step tail beyond K is unavailable there — compute in replay_buffer.rs.

### 2. No decisive games (game_task.rs)
- Caps: `MAX_GAME_LENGTH=300` self-play (:413) and eval (:281). Non-checkmate → outcome 0,
  is_draw true (:576-596). `material_shaping_enabled` (:890-898) gates
  `tanh(Δmaterial/scale)` outcome; `compute_material_diff` P/N/B/R/Q=1/3/3/5/9 (:903-913);
  scale from `HYZERO_MATERIAL_SHAPING_SCALE` default 5, clamp [0.5,100] (:868-874).
- KNOBS THAT EXIST: material shaping (off), temperature_moves, decisive-sample frac,
  Gumbel. NEEDS BUILDING: resignation (compare rootQ/value to threshold over N plies),
  adjudication-at-cap (e.g. award decisive on large material/Q at 300), longer/annealed
  temperature schedule (currently a hard 30-ply step to 0.01).

### 3. Blind eval ladder (evaluation.rs + game_task.rs)
- Flow: `EvaluationTask.run` (evaluation.rs:227-559) waits for new model version, builds a
  pool of `best_v{NNN}.pt` (pool_size 3), plays `2*games_per_side` per opponent via
  `play_game_dual`, updates `candidate_elo` per game with `elo::update_rating` (vs fixed
  `opponent_initial_elo=1500`, K=32; evaluation.rs:440-486). Promotion gate:
  pool-path `candidate_elo > 1500 + promotion_elo_delta(20)`; bootstrap path
  `win_rate >= promotion_threshold(0.55)` (evaluation.rs:530-534).
- candidate_elo is computed inline in run(); pure helper `compute_candidate_elo_from_results`
  (evaluation.rs:198-209). Game settings come from GameConfig built at evaluation.rs:253-258
  (T=0.01 via play_game_dual:351, no noise). Eval games are checkmate-only outcomes — with
  no decisive games, every eval is a draw, win_rate→0.5, Elo→1500, never promotes.
- Where to add knobs: games_per_side/num_simulations in EvaluationConfig (evaluation.rs
  :36-87); openings via existing `HYZERO_STARTS_FILE` path (game_task.rs:144-187, already
  wired into play_game_dual via init_self_play_board); adjudication shares game_task.rs
  :382-386 with self-play.

### 4. MCTS Q-normalization (puct.rs/node.rs/tree.rs)
- PUCT: `select_child` (puct.rs:41-83) computes `q + c*P*sqrt(Nparent)/(1+N)`; unvisited
  child contributes `(q=0, N=0)` (puct.rs:57-59), so exploration term alone ranks it.
  No min/max tracking. With value head dead at ≈0, Q has no scale → exploration dominates
  uniformly (degenerate search).
- Canonical AlphaZero/MuZero: maintain `MinMaxStats{min,max}` per search; normalize
  `Q_norm=(Q-min)/(max-min)` before adding the exploration term, and init unvisited child
  Q to parent's value (or normalized 0). MinMaxStats would live in tree.rs (per-MCTSTree
  field, threaded into `select_child`); updated in `backpropagate` (tree.rs:738-749) on
  each node's running Q. `puct_score`/`select_child` signatures would gain a stats arg.
- Tests that must keep passing: puct.rs:89-192 (raw-Q math — will need updating or a
  normalization-bypass for the unit cases), tree.rs:893-1006 (sim/visit), tree.rs:1140 &
  1291 (backup sign + mating reward — normalization must not change stored total_value).

### 5. Ungated value-symmetry probe (trainer.py:712-753)
- `[sym_probe]`: for a real batch obs, computes `v1=f(h(obs))`, `v2=f(h(flip(obs)))`,
  `sum=v1+v2` and `ratio=sum/|v1|` (trainer.py:715-727). POV-antisymmetry requires
  `v2≈-v1` → sum≈0; nonzero sum = degeneration. `[sym_probe_batch]` adds Pearson corr of
  `v` vs `-v_flip` and mean(v+v_flip) over ≤10 samples (trainer.py:729-753).
- Runs only when `self.model_version % 50 == 0` (trainer.py:713). NOTE: model_version is
  incremented at END of train_batch (trainer.py:859), so the probe sees the pre-increment
  value — effectively every 50th step. Output via `_diag_print` to stdout/stderr → captured
  in the selfplay log; NOT written to any structured artifact and NOT consumed anywhere.
- Hooking it in: (a) as a logged gate — promote/continue only if |mean_sum| below threshold;
  would touch evaluation.rs or run_baseline.sh parsing (new grep anchor) and require the
  Python value to surface (currently log-only). (b) As an auxiliary antisymmetry
  regularizer loss — compute `(f(h(obs))+f(h(flip(obs))))^2` inside the loss block
  (trainer.py:577-652) under a new `HYZERO_*_WEIGHT`, add to `total_loss` (trainer.py
  :847-852); needs the flip helper `_flip_obs_planes` (already imported, trainer.py:50) and
  an extra h/f forward pass per batch (cost). Both are additive; neither changes the bridge.

## Open questions
- Conditional-β default: code 0.1 vs comments 0.3 — which is the intended baseline?
- Resignation vs adjudication-at-cap: prior note (game_task.rs:569-574) warns adjudication
  was the cause of a passivity attractor — does the plan re-introduce a guarded form?
