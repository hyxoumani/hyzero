# MCTS & Self-Play Infrastructure

## Closed-Loop Training Paradox: Faster Training ≠ Better Model

**CRITICAL FINDING (2026-04-15)**: MCTS self-play is a closed loop. The model generates games (via MCTS with value-guided search), which train the model. If training becomes faster (lower loss, more steps, different loss weighting) WITHOUT a corresponding improvement in MCTS quality, the result is poorer play, not better. This appears as a contradiction: policy loss decreases while promotions drop to zero.

**Evidence from autoresearch sweep** (11 experiments, 30-min each):

| Config | policy_loss | avg_game_len | promotions | score |
|--------|-------------|---|---|---|
| **β=0.3 (winner)** | 3.40 | 151.6 | 4 | **11.63** |
| value_weight=5.0 | 2.70 ✓ | 113.2 | 0 ✗ | 4.84 |
| games_per_side=6 | 2.41 ✓ | 106.9 | 0 ✗ | 5.48 |
| β=0.4 | 2.63 ✓ | 105.4 | 1 ✗ | 6.80 |
| β=0.5 | 2.45 ✓ | 107.2 | 2 ✗ | 8.07 |

**The paradox**: Every configuration that achieved lower policy loss (2.4–2.7 vs 3.4) regressed in promotions and score. The challenger **lost to Random** at eval cycles 1–4 despite training metrics looking excellent.

**Root cause**: Self-play games are the training data source. If MCTS quality doesn't improve alongside training speed, the model trains on garbage targets. Policy loss appears good because the network is learning the visit-distribution targets faithfully — but those targets reflect whatever MCTS produced (which may be poor if value estimates are unstable early). The metrics are decoupled:
- **Policy loss** = how well the network memorizes MCTS visit distributions (local signal)
- **Promotions** = whether the learned model actually plays better (global signal)

When the two diverge, promotions are the ground truth. Lower loss alone is a false positive.

**Intuition**: β=0.3 had longer games (151.6 moves vs ~106) and higher loss (3.40), meaning more exploration and slower convergence. Slower convergence → MCTS had more time to refine value estimates → better training data → promotions happened. Faster training (β>0.3) meant the model converged before MCTS built good value estimates → it learned on noisy targets → play regressed despite loss curves looking great.

**Key Decision**: Always validate experiments by promotions (real play), not training loss. In this codebase, measuring by loss is nearly useless — measure by wins against evaluation opponents.

## MCTS Tree Search (Per Move)

**Per-move flow**:
1. Encode board → `BoardObservation` (103 planes: current position + 7 historical positions × 12 piece planes + 7 auxiliary channels)
2. Call `evaluator.root_setup(observation)` → (hidden_state, policy, value)
3. Run N simulations (50 for CPU dev, 200+ for GPU production)
4. Extract visit count distribution, select action by visits (temperature scheduling)
5. Apply move, record StepRecord, **discard entire tree** (transient)

## Single MCTS Simulation

1. **SELECT**: Walk down tree using PUCT = Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a))
2. **EXPAND**: Call g(s, a) → (s_new, reward) in latent space
3. **EVALUATE**: Call f(s_new) → (policy, value) to initialize children
4. **BACKUP**: Propagate value to root with negation per ply, increment visit counts (two-player zero-sum)

## Inference Batching

Game threads send `InferenceRequest` (RootSetup or ExpandLeaf) to batcher. Batcher collects up to `batch_size` requests or times out (T_timeout), makes single PyO3 call to Python, distributes results via oneshot channels. Reduces GIL acquisitions from ~800/move to ~1/move.

## Replay Buffer

Ring buffer (`VecDeque<GameTrajectory>`) with K-step sampling: pick random trajectory (weighted by length), pick random index t where t+K ≤ len, return steps t..=t+K. Each trajectory tagged with model_version. Serialized with bincode for checkpoints.

## Self-Play Coordinator

Spawns N persistent long-lived tokio game loop tasks (no semaphore gating). Each task loops indefinitely: read current `model_version` from watch channel → play one game → send GameTrajectory → repeat. Each game: root setup → MCTS → move selection → step record, repeating until terminal (or 300-move limit). GameTrajectory tagged with model_version and outcome. Awaits all JoinHandles on shutdown.

**Design**: No semaphore — tasks are persistent, not spawned per game. This reduces overhead and scales cleanly to many concurrent games. Watch channel updates when trainer publishes new weights, ensuring game tasks always use current model.

### Root Noise for Exploration

Dirichlet noise added to root policy before move selection. Implemented via Marsaglia-Tsang Gamma sampling (MuZero paper).
- **NOISE_ALPHA = 0.3** (AlphaZero chess value — spreads exploration across many moves)
- **NOISE_EPSILON = 0.25** (fraction of noise mixed into prior)

**Game-specific constants**: AlphaZero paper specifies α={0.3, 0.15, 0.03} for {chess, shogi, Go}. Using the wrong value (e.g., 0.03 for chess) over-concentrates noise on 1-2 random moves, starving exploration of the state space (see 2026-04-14 mistakes log: Dirichlet alpha bug).

**WARNING**: Slow in debug mode; use `--release` builds for end-to-end testing.

## Checkpoint Management

`PyTrainingThread` manages rolling window of saved checkpoints with zero-padded filenames (`model_v000050.pt`). Every `checkpoint_interval_steps` training steps (default 50), saves weights to disk. Maintains list of saved files in `VecDeque<PathBuf>`, prunes oldest when window size exceeds `checkpoint_keep_last` (default 5).

### Resume from Checkpoint

`from_default_config(resume_checkpoint: Option<&str>)` loads a prior checkpoint. PyO3 call loads weights into trainer, reads `model_version` from trainer object attribute, returns both. Weights pushed to InferenceServer, `model_version` published via watch channel. All inference and game tasks resume from the loaded model state.

## Evaluation Task

Separate async task watches `model_version` via watch channel. When version advances by `eval_interval_steps` (default 200), spawns `eval_games` (default 10) self-play games using the current model playing against itself (ChannelEvaluator, not random). After all games complete, logs statistics:
- `white_wins`, `black_wins`, `draws`
- `white_win_rate` (white_wins / total)
- `decisive_ratio` ((white_wins + black_wins) / total)
- `avg_length` (mean game length)

Provides continuous signal of model quality during training. Runs in parallel with main self-play loop.

## Action Encoding

Action space: 4672 (AlphaZero 8×8×73 encoding: 64 origin squares × 64 destination squares + 64 underpromotion slots). Encoding: `action = from_sq * 64 + to_sq` for regular moves; underpromotion offsets at 4096-4671. Network learns to suppress illegal moves via legal-move masking (before softmax, with `nan_to_num` handling).

## Replay Buffer Distribution Dynamics

**Equal-weight sampling intuition**: Early random games (weak model) contribute the same gradient updates as recent on-policy games (strong model). Intuitively wasteful — why train on old noise?

**Recency weighting tradeoff**: Exponential decay (weight ∝ exp(-decay × version_age)) prioritizes recent batches. However, this **starves the value head of outcome diversity**. Early random games have wide outcome distributions (draws, decisive wins, losses); recent games from a converging policy narrow toward repetitive patterns. The value head needs signal from diverse positions and outcomes to learn stable value estimates.

**Catastrophic forgetting signature**: Commit `003eaf9` ran with `HYZERO_REPLAY_DECAY=0.1` and observed:
- v20 eval: decisive_ratio = 0.50 (good)
- v57 eval: decisive_ratio = 0.10 (collapsed)
- Policy loss kept improving (3.96→3.02) but tactical play regressed

This divergence signals value-head collapse: policy learned to avoid loss but value estimates became unreliable, leading to drawish play. Second run with decay=0.05 showed same pattern (v20 decisive=0.20, v61 decisive=0.10).

**Future options**: (a) Separate value/policy samplers with different decay schedules (policy favors recent; value preserves old diversity); (b) Add diversity floor (minimum 10-20% uniform sampling to force value head to see wide outcomes); (c) Prioritize by outcome (high-variance games preferred over recent repetitive ones).

## Dead Value & Reward Heads

**Value head dead** (training logs: `value=0.0000`): Self-referential bootstrap. Target is MCTS root_value (untrained), initialized from f-network output and backed up from leaf. Loop: `f(s) ≈ 0` → `root_value ≈ 0` → target ≈ 0` → loss ≈ 0 → `f` stays 0 (refs: `src/selfplay/game_task.rs:96`, `src/py/training.rs:98`, `src/data/replay_buffer.rs:93`). Canonical MuZero uses outcome targets (Schrittwieser 2020, Appendix F); our approach loses this signal. With Q ≈ 0, MCTS search reduces to prior sampling — policy self-imitates with no improvement.

**Reward head dead** (training logs: `reward=0.0006`): Class imbalance — only terminal steps have non-zero targets. For 100-move games, terminal appears in ~1% of K+1-step slices. MSE-optimal solution is 0 everywhere. MuZero needs reward head for latent-space terminal detection; dead reward breaks backup signal and MCTS may expand past terminal states.

See neural-networks.md sections "Canonical MuZero Value Target" and "MCTS as Policy Improvement" for architecture discussion and fix proposals.

## Dual-Model Ladder Stall — Symmetry Collapse

After a promotion to v1 (new challenger = champion snapshot), both models have identical architecture + weights at t=0. They drift together due to simultaneous training on the same replay buffer. Eval often returns **win_rate ≈ 0.50** for many cycles, delaying the next promotion. This is **expected behavior** (symmetric play → draws) but slows improvement.

**Mitigations observed** (2026-04-15):
- **β=0.1** (soft outcome blend): 1 promotion in 30 min
- **β=0.2**: 2 promotions in 30 min

**Note**: This remains an unsolved problem for longer runs (>1 hour). Future work needed on breaking symmetry inside a single training run (e.g., adversarial ensemble, asymmetric eval). See `docs/plans/next-steps/resume.md` for proposed solutions.

## Key Gotchas

1. **Value negation**: Negated per ply during backprop (two-player zero-sum). See `MCTSTree::backpropagate()`: `child.total_value += -value`.
2. **Dirichlet noise overhead**: Marsaglia-Tsang Gamma sampling for Dir(0.03) is very slow in debug mode. For e2e testing, always use `--release` builds or run the binary directly (not via `cargo run`).
3. **Game length with Dirichlet**: Games run ~200 moves now (vs ~60 before); correct behavior for better play, but impacts iteration speed.
4. **Transient tree**: Discarded after each move. Fresh tree per move, no caching.
5. **Batch timeout tuning**: 10ms timeout is empirical. Few concurrent games → small batches → lower GPU utilization.
6. **Stale model data**: Old trajectories in buffer. Loss initially high (bootstrapping).
7. **Action space mismatch**: 4672 logits vs ~40 legal moves. Legal-move masking applied before softmax + nan_to_num.
8. **Visit distribution sparsity**: replay buffer stores StepRecord with visit counts for each move. Array is sparse (length = num_visits). PyTrainingThread zero-pads to 4672 before passing to trainer.
9. **Stdout buffering in scripts**: `cargo run` buffers output. For log capture in shell scripts, run the binary directly: `target/release/selfplay` instead of `cargo run --bin selfplay`.
10. **action_to_move signature**: `action_to_move(action, board, color)` requires board state and active color to correctly reconstruct castling and en passant moves. The selfplay game_task.rs already handles this correctly; only direct callers of action_to_move need updating.
11. **Legal-move masking NaN**: `log_softmax(-inf)` produces NaN in log-probs. Use `nan_to_num(neginf=0.0)` after softmax to replace illegal-move NaNs with 0.
12. **Game outcome perspective**: `game_outcome` is absolute White-perspective (+1 for White win, -1 for Black win), but observation planes encode absolute piece positions. When using outcome as a value target, account for whose turn it is (plane 101).

## Related Files

- `src/mcts/{mod,node,tree,puct,evaluator}.rs` — tree operations, PUCT selection
- `src/selfplay/{mod,game_task,coordinator,inference,training}.rs` — orchestration
- `src/data/{types,encoding,replay_buffer}.rs` — board observation, action space, buffer
- `docs/TASKS_MCTS_SELFPLAY.md` — detailed task specs (Tasks 17-23)
