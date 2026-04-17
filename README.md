# hyzero

A chess engine emulating MuZero — a Rust core for game logic and search, bridged
to a Python neural network layer via PyO3.

## Overview

hyzero implements the MuZero algorithm applied to chess. The engine uses a bitboard
representation for fast move generation, Monte Carlo Tree Search (MCTS) with PUCT
selection for planning, and three PyTorch networks (representation, dynamics,
prediction) for evaluation. Rust and Python communicate through a PyO3 bridge, letting
the search run at native speed while keeping model code in idiomatic Python.

## Features

- Bitboard-based move generation for all piece types, including castling, en passant,
  and pawn promotion (queen and underpromotion)
- Full check, checkmate, and stalemate detection; perft-validated against python-chess
  (53M nodes across 6 standard positions)
- MCTS with PUCT selection, Dirichlet root noise, legal-move masking, and neural-network-
  backed value/policy estimates
- MuZero-style representation, dynamics, and prediction networks in PyTorch
- Self-play training loop with replay buffer and rolling-window checkpoints
- Dual-model champion/challenger evaluation ladder with atomic promotion
- Soft value-outcome blend (β) to break the dead-value-head bootstrap loop
- PGN-formatted eval-game logging for introspection
- Async client/server architecture via Tokio
- End-to-end validation script

## Architecture

```
src/
  pieces/     — move generation (bishop, king, knight, pawn, queen, rook)
  game/       — board state, game history, player abstraction, perft
  mcts/       — MCTS node/tree, PUCT selection, evaluator interface
  selfplay/   — self-play coordinator, inference backend, training pipeline,
                  dual-model eval task
  data/       — board encoding (103 planes), replay buffer, shared data types
  session/    — game session management
  py/         — PyO3 bridge to Python inference and training
  bin/        — binary entry points (server, client, selfplay, perft)

python/
  hyzero/     — MuZero model definitions (h/g/f), trainer, inference server
```

## Build and Run

```bash
# Build
cargo build --release

# Run binaries
cargo run --bin server       # game server
cargo run --bin client       # game client
cargo run --bin selfplay     # self-play training loop

# Perft (move-generation validation)
cargo run --release --bin perft -- <FEN> <depth>
```

## Test

```bash
cargo test                              # Rust unit tests (82 pass, 7 ignored)
cargo test --release -- --ignored       # slow perft + Python-dependent tests
python3 scripts/cross_validate.py --all # move-gen validation vs python-chess
cd python && pytest                     # Python model/trainer tests
bash scripts/e2e_test.sh                # end-to-end pipeline validation
```

## Training

```bash
# Controlled 30-minute baseline run (deletes checkpoints, computes score)
bash scripts/run_baseline.sh 1800

# Long-form keep-training mode (preserves checkpoints)
bash scripts/run_training.sh            # 7200s default
```

### Training Metric

```
training_score = (8.55 − final_policy_loss)
               + (promotions × HYZERO_CHAMPION_SCORE_WEIGHT)
               − (avg_game_length / 100)
```

- `promotions` counts successful champion replacements on the eval ladder.
- `policy_loss` is the final cross-entropy on MCTS visit-distribution targets.
- `avg_game_length` is mean plies across completed self-play games.

**Baseline (commit 294e63e)**: `14.51` at `HYZERO_VALUE_OUTCOME_BETA=0.3` with all other
defaults. Run-to-run variance is roughly ±3 points; expect 11–14.5 on repeats.

### Key environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HYZERO_VALUE_OUTCOME_BETA` | `0.1` (recommended `0.3`) | Value-target blend: `β·outcome + (1-β)·MCTS_Q` |
| `HYZERO_REWARD_OUTCOME_GAMMA` | `0.0` | Analogous soft-blend for reward head |
| `HYZERO_CHAMPION_SCORE_WEIGHT` | `2.0` | Score multiplier for each promotion |
| `HYZERO_PROMOTION_THRESHOLD` | `0.55` | Win-rate needed to promote challenger |
| `HYZERO_GAMES_PER_SIDE` | `4` | Eval games per colour per cycle |
| `HYZERO_ADJ_THRESHOLD` / `HYZERO_ADJ_PLIES` | `6` / `10` | Material-adjudication gate |
| `HYZERO_VALUE_LOSS_WEIGHT` | `1.0` | **Keep at 1.0** — amplification destabilises the closed loop |
| `HYZERO_LR_SCHEDULE` | `none` | Set to `cosine` to enable annealing |

## Lint and Format

```bash
cargo clippy
cargo fmt
```

## Documentation

- `docs/wiki/` — synthesised knowledge base (maintained by `context-keeper` agent)
  - [`index.md`](docs/wiki/index.md) — entry point
  - [`project-roadmap.md`](docs/wiki/project-roadmap.md) — current state, baselines, next batches
  - [`mcts-selfplay.md`](docs/wiki/mcts-selfplay.md) — search, coordinator, closed-loop paradox
  - [`neural-networks.md`](docs/wiki/neural-networks.md) — h/g/f networks, value-head bootstrap
  - [`board-encoding.md`](docs/wiki/board-encoding.md) — observation tensor, current-player perspective
  - [`mistakes.md`](docs/wiki/mistakes.md) — agent-failure log with root-cause analysis
- `docs/plans/` — per-experiment plans (one directory per attempted change)
- `CLAUDE.md` — project conventions for AI agents

## Current Status

- Peak baseline: **14.51** (β=0.3, commit 294e63e, 2026-04-15)
- Known open issue: **adjudication passivity trap** (2026-04-17) — model converges to
  passive shuffle patterns (e.g. Na3 + rook shuffle a1↔b1) because material-based
  adjudication rewards "don't lose material" without punishing "don't move." See
  `mcts-selfplay.md` for the full analysis and the proposed fix (remove adjudication,
  keep material-at-cap only).

## Tech Stack

| Layer       | Technology                        |
|-------------|-----------------------------------|
| Game logic  | Rust (bitboards)                  |
| Async I/O   | Tokio                             |
| Search      | MCTS / PUCT (Rust)                |
| Neural nets | PyTorch (Python)                  |
| FFI bridge  | PyO3 + numpy                      |
| Serde       | serde + bincode                   |
| Randomness  | rand                              |
