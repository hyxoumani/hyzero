# hyzero

My attempt at a chess engine emulating MuZero — a Rust core for game logic and search,
bridged to a Python neural network layer via PyO3.

## Overview

hyzero implements the MuZero algorithm applied to chess. The engine uses a bitboard
representation for fast move generation, Monte Carlo Tree Search (MCTS) with PUCT
selection for planning, and a trio of PyTorch networks (representation, dynamics,
prediction) for evaluation. Rust and Python communicate through a PyO3 bridge, letting
the search run at native speed while keeping model code in idiomatic Python.

## Features

- Bitboard-based move generation for all piece types, including castling, en passant,
  and pawn promotion
- Full check, checkmate, and stalemate detection
- MCTS with PUCT selection backed by neural network value and policy estimates
- MuZero-style representation, dynamics, and prediction networks in PyTorch
- Self-play training loop with replay buffer
- Async client/server architecture via Tokio
- End-to-end validation script

## Architecture

```
src/
  pieces/     — move generation (bishop, king, knight, pawn, queen, rook)
  game/       — board state, game history, player abstraction
  mcts/       — MCTS node/tree, PUCT selection, evaluator interface
  selfplay/   — self-play coordinator, inference backend, training pipeline
  data/       — board encoding, replay buffer, shared data types
  session/    — game session management
  py/         — PyO3 bridge to Python inference and training
  bin/        — binary entry points (server, client, selfplay)

python/
  hyzero/     — MuZero model definitions and training config
```

## Build and Run

```bash
# Build
cargo build

# Run binaries
cargo run                    # main entry point
cargo run --bin server       # game server
cargo run --bin client       # game client
cargo run --bin selfplay     # self-play training loop
```

## Test

```bash
cargo test                   # Rust unit tests
cd python && pytest          # Python model tests
bash scripts/e2e_test.sh     # end-to-end validation
```

## Lint and Format

```bash
cargo clippy
cargo fmt
```

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
