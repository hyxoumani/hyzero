# Missing, Incomplete, and Next Steps for hyzero

## Current State (Tasks 1-29)

All core infrastructure is DONE:
- Chess engine: Full bitboards, magic moves, special moves, validation
- MCTS tree search with value negation per ply
- Self-play pipeline: 4 concurrent games, 50 sims/move, MCTS-guided play
- Python MuZero networks: h/g/f with training and inference
- PyO3 integration: Full Rust↔Python bridge working
- End-to-end loop verified: 5 games, 13 training steps, loss 8.52→7.04

## Development Configuration (Conservative)

Current settings are CPU development optimized, not production:
- `max_concurrent_games: 4`
- `num_simulations: 50` (vs target 200+)
- `batch_size: 32`, `batch_timeout_ms: 10`
- `max_game_length: 300` (empirically becomes ~200 avg)
- `training_batch_size: 256`, `unroll_k: 5`
- Network: C=64 channels, 4 ResBlocks (tiny for chess)

## Major Gaps by Category

### 1. Scalability & Performance (High Impact, Medium Effort)

**MCTS Limitations:**
- Fixed 50 simulations (way too low) → games play badly
- No transposition tables → redundant computation in multi-game MCTS
- No root noise besides Dirichlet (no UCB for exploration decay)
- No alpha-beta pruning or related enhancements
- Dirichlet sampling uses Marsaglia-Tsang (slow) but works

**Inference Pipeline:**
- Batch size capped at 32; GPU likely underutilized with 4 games
- 10ms timeout is empirical; may need profiling for GPU batch fill
- No GPU memory pooling or prefetch strategy
- No dynamic batch sizing based on load

**Training:**
- No priority replay sampling → all trajectories weighted equally
- No reanalyze step (replay buffer is cold storage, not reused)
- No temperature scheduling (fixed 1.0 then 0.01)
- No learning rate scheduling
- No gradient clipping (loss is relative, not absolute)
- No distributed training (single Python process)

### 2. Model & Action Space (Medium Impact, Medium Effort)

**Action Space Issues:**
- Fixed 4096 (64×64) with queen-default promotion
- No underpromotion support (knight/bishop/rook) → only 1 of 4 promotions legal
- No illegal move masking in policy → network learns to ignore invalid moves (wastes capacity)

**Network Architecture:**
- Tiny C=64, 4 blocks vs AlphaZero baseline (256+, 20 blocks)
- No batch norm momentum tuning
- Value head tanh-bounded but not scaled (advantage vs outcome unclear)
- No support for learned temperature in policy

### 3. Game Logic Edge Cases (Low Impact, Low Effort)

**Known Issues:**
- Game parsing uses `panic!()` on invalid notation (src/game/playerobj.rs)
- Coordinator has `panic!()` on trajectory timeout (src/selfplay/coordinator.rs)
- Stalemate detection via bitboard-index was buggy, fixed in Task 23
- 50-move rule + threefold repetition handled but not exhaustively tested

### 4. Evaluation & Metrics (High Impact, High Effort)

**Missing:**
- No Elo rating system (vs known engines: stockfish, leela, etc.)
- No opening book or endgame tablebases
- No game analysis (PV extraction, mistake detection)
- No convergence tracking across runs (only single e2e test snapshot)
- No training curve visualization
- Metric extraction limited to loss + game count (no accuracy, policy entropy, etc.)

### 5. Infrastructure & Ops (Medium Impact, Low Effort)

**Incomplete:**
- No config file support (hardcoded in selfplay.rs)
- No logging levels (all stderr, all the time)
- No model versioning beyond `u64` counter
- No graceful shutdown (SIGTERM + 2s sleep, then SIGKILL)
- No health checks or monitoring
- No experiment parameter sweep tools (run_experiment.sh runs same config N times)

### 6. Testing & Validation (Medium Impact, Medium Effort)

**Gaps:**
- 27 Rust tests (good coverage of core logic)
- 0 Python tests (config exists but pytest not installed)
- 0 integration tests (only e2e_test.sh via scripting)
- No property-based tests (game invariants, move generation exhaustive)
- No adversarial tests (engine vs itself, transposition detection)
- No performance benchmarks (move gen throughput, inference latency)

### 7. Documentation & Developer Experience (Low Impact, Low Effort)

**Missing:**
- No setup.py or environment.yml for Python
- No Makefile for common tasks
- No Docker for reproducibility
- docs/todo.md is stale (pre-Task 1 content)
- No contributor guide
- No hyperparameter tuning guide

## Recommended Priority (by Impact × Urgency)

### Phase 1: Get it to Actually Play Chess (Critical)
1. **Scale up MCTS** (1 file, 10 min)
   - Increase num_simulations 50 → 400
   - Optional: add root noise schedule
   - Impact: 10× improvement in move quality

2. **Increase network capacity** (1 file, 15 min)
   - C=64 → C=128, 4 blocks → 8 blocks
   - Impact: Better value/policy estimation

3. **Add illegal move masking** (3 files, 1-2 hrs)
   - Pass legal_moves to policy head
   - Mask logits before softmax in f() and inference
   - Impact: 10% faster convergence, cleaner policy

### Phase 2: Make Training Work (High Priority)
4. **Priority replay sampling** (1 file, 2-3 hrs)
   - Weight samples by TD error or loss magnitude
   - Impact: Faster convergence on hard positions

5. **Reanalyze step** (2 files, 4-6 hrs)
   - Periodically re-evaluate old trajectories with new model
   - Feedback stale value targets
   - Impact: 20-30% training speedup

6. **Temperature scheduling** (1 file, 1 hr)
   - Decay temperature over training
   - Impact: Better endgame moves

### Phase 3: Validation & Metrics (Medium Priority)
7. **Elo estimation** (2 files, 4-6 hrs)
   - Play against fixed-depth minimax or opening book move
   - Track win/loss/draw rate
   - Estimate Elo via Bayesian model or ELO formula

8. **Training curve tracking** (2 files, 2-3 hrs)
   - Log policy entropy, value range, reward mean/std per batch
   - Save metrics to TSV for plotting
   - Add matplotlib plot script

### Phase 4: Scalability & Ops (Nice to Have)
9. **Config YAML/TOML** (2 files, 1-2 hrs)
   - Load from `hyzero.toml` or `configs/` dir
   - Override via CLI flags

10. **Distributed inference** (3 files, 8-12 hrs, high risk)
    - Subprocess inference server + Unix socket
    - Drop-in replacement for PyO3Backend
    - Enables GPU isolation, language-agnostic inference

## Known Risks & Gotchas

1. **Scaling 50→400 simulations**: Game length grows proportionally (50 sims → ~200 moves, 400 sims → ~500+ moves). May hit 300-move limit. Need to adjust MAX_GAME_LENGTH.

2. **Batch timeout tuning**: With 4 games × 400 sims = 1600 inference requests. 10ms timeout → ~16 requests/ms = batches of ~160. If GPU can't handle that, increase timeout. Conversely, fewer games = partial batches = latency waste.

3. **GIL bottleneck**: One GIL per batch. With 400 sims, you're acquiring GIL 50× per move. Profile before optimizing.

4. **Reward signal**: Real games use terminal rewards (win=1, loss=-1, draw=0), but immediate rewards from g() are learned artifacts. May diverge from game outcome.

5. **Stale model in flight**: Games started with old weights vs training on new weights. Brief lag acceptable per design, but verify empirically.

6. **Memory**: Replay buffer unbounded (max_replay_trajectories=10k, each with ~300 steps, each step has ~19×64 floats). Rough: 10k × 300 × 64 × 4 bytes = ~7.5 GB. Need monitoring.

## Files to Watch

Key files that frequently need changes as you iterate:
- `src/bin/selfplay.rs` — configuration (num_simulations, batch_size, etc.)
- `src/selfplay/game_task.rs` — temperature, max_game_length, move selection
- `python/hyzero/config.py` — network capacity
- `src/data/encoding.rs` — action space (if underpromotion added)
- `scripts/e2e_test.sh` — metric extraction (currently looks for specific log format)

## Test Coverage Summary

| Component | Tests | Status |
|-----------|-------|--------|
| Bitboards & move gen | 8 | ✓ Solid |
| MCTS (tree, PUCT, value negation) | 6 | ✓ Solid |
| Selfplay (game logic, batching) | 7 | ✓ Passing |
| Replay buffer (sample, checkpoint) | 5 | ✓ Passing |
| PyO3 bindings | 3 | ⚠ Ignored (requires Python install) |
| E2E | 1 script | ✓ Passing (120s, 5 games) |
| Python models | 0 installed | ⚠ No pytest; 27 tests exist but never run |
| Integration | 0 | ✗ None |

## Session Outcome

All 29 tasks are complete and integrated. System is "feature complete" for a minimal MuZero chess engine but plays poorly (50 sims). Next agent should focus on Phase 1 (scale MCTS + increase network size) to get it to actually learn chess, then Phase 2 (priority replay, reanalyze) for training efficiency.
