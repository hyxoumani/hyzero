# Autoresearch Program — hyzero MuZero Chess Engine

## Instructions

Optimize the `training_loss` metric (lower is better) for the hyzero MuZero chess engine.
The metric is the final training loss from a 120-second self-play + training run.

### What to search for

**Training efficiency improvements:**

- Learning rate scheduling (warmup, cosine decay, reduce-on-plateau)
- Loss component weighting (policy vs value vs reward balance)
- Gradient clipping thresholds
- Optimizer changes (Adam → AdamW, SGD+momentum, weight decay tuning)
- K-step unroll depth (currently 5)
- Training batch size vs steps-per-game ratio
- Gradient scaling factor at dynamics boundary (currently 0.5)

**Network architecture improvements:**

- Hidden channel count (currently 64 — try 96, 128)
- Number of residual blocks (currently 4 — try 6, 8)
- Squeeze-and-excitation blocks in residual stack
- Different activation functions (GELU, SiLU/Swish instead of ReLU)
- Normalization (LayerNorm vs BatchNorm, pre-norm vs post-norm)
- Policy/value head architecture (deeper heads, different pooling)

**Self-play quality improvements:**

- MCTS simulation count (currently 50 — try 100, 200)
- PUCT exploration constant (currently 1.5)
- Dirichlet noise alpha (currently 0.03) and epsilon (currently 0.25)
- Temperature schedule (currently 15 moves at T=1.0, then T=0.01)
- Max game length (currently 300)
- Concurrent games count (currently 4)

**Data pipeline improvements:**

- Replay buffer capacity and minimum samples threshold
- Batch sampling strategy (uniform → prioritized by TD error)
- Board encoding (add move history planes, attack maps, piece mobility)
- Action encoding improvements

**Illegal move masking:**

- Mask policy logits to legal moves before softmax (currently not done)
- Could significantly speed up convergence

## Constraints

**Must NOT change:**

- The Rust game logic in `src/game/` (move generation, validation, rules)
- The MCTS tree structure in `src/mcts/tree.rs` (node/tree types)
- The PyO3 bridge protocol in `src/py/inference_backend.rs` (method signatures, array shapes)
- The e2e test script `scripts/e2e_test.sh` (metric extraction must keep working)
- The log format: `[py_training] step N: total=X.XXXX policy=... value=... reward=...`
- Test contracts: all `cargo test` must still pass
- Board encoding dimensions must stay consistent across Rust encoding and Python networks
  (if you change input_planes, update BOTH `src/data/encoding.rs` AND `python/hyzero/config.py`)

**Read-only files:**

- `Cargo.lock`
- `target/`
- `scripts/e2e_test.sh`
- `src/game/`

**In-scope files (safe to modify):**

- `python/hyzero/config.py` — hyperparameters
- `python/hyzero/models/*.py` — network architectures
- `python/hyzero/training/trainer.py` — training loop, loss, optimizer
- `python/hyzero/inference/server.py` — inference pipeline
- `src/bin/selfplay.rs` — runtime config (sims, concurrency, batch size)
- `src/selfplay/game_task.rs` — game config, temperature, max length
- `src/selfplay/coordinator.rs` — coordinator config
- `src/selfplay/inference.rs` — batcher config
- `src/data/encoding.rs` — board/action encoding
- `src/data/types.rs` — data types
- `src/data/replay_buffer.rs` — replay buffer sampling
- `src/mcts/puct.rs` — PUCT selection formula
- `src/py/training.rs` — batch assembly, training bridge

## Stopping criteria

- Stop after 50 experiments, OR
- Stop when the metric plateaus for 10 consecutive runs (no improvement), OR
- Stop if loss drops below 3.0 (strong convergence signal)
