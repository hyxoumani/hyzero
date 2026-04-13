# Testing Procedures

How to validate the hyzero engine at every level — from quick smoke tests to exhaustive cross-validation.

## Quick Reference

```bash
cargo test                                          # 80 Rust tests (~20s debug)
cargo test --release -- --ignored                    # + 3 slow perft tests (~1s release)
python3 scripts/cross_validate.py --all              # full oracle comparison (~3 min)
bash scripts/e2e_test.sh                             # end-to-end self-play loop
```

## 1. Rust Unit Tests

**Command**: `cargo test`
**Time**: ~20s (debug), ~1s (release)
**Result**: 80 pass, 6 ignored

### By module

| Module | Tests | What they cover |
|--------|-------|-----------------|
| `game::board` | 30 | Move gen (all pieces), check/mate/stalemate, castling, EP, promotion, pins, zobrist, game_status, threefold repetition, 50-move rule, insufficient material |
| `game::perft` | 13 (3 ignored) | Perft node counts for 6 standard positions × depths 1–6 |
| `game::fen` | 5 | FEN parsing: startpos, midgame, castling rights, EP target, black-to-move |
| `data::encoding` | 6 | action↔move round-trip: normal, castling, EP, promotion |
| `data::replay_buffer` | 7 | Buffer add/evict, sampling, checkpointing, step tracking |
| `mcts::puct` | 4 | PUCT scoring, child selection |
| `mcts::tree` | 3 | Simulation runs, visit distribution, action selection |
| `selfplay::*` | 8 | Coordinator, game task, inference batching, training pipeline |
| `py::*` | 4 (3 ignored) | PyO3 batch assembly/shapes; inference+training need Python env |

### Ignored tests

| Test | Why ignored | How to run |
|------|-------------|------------|
| `perft_startpos_d4/d5` | ~15s debug mode | `cargo test --release -- --ignored perft_startpos` |
| `perft_kiwipete_d3` | ~12s debug mode | `cargo test --release -- --ignored perft_kiwipete` |
| `py::*` (3 tests) | Require `hyzero` Python package installed | `cd python && pip install -e . && cd .. && cargo test -- py::` |

## 2. Cross-Validation (python-chess oracle)

**Script**: `scripts/cross_validate.py`
**Dependency**: `python-chess` (pip: `chess==1.11.2`)
**Prerequisite**: `cargo build --release --bin perft`

The script calls our `perft` binary and compares results against python-chess — a battle-tested reference implementation.

### Modes

| Flag | What it tests | Positions | Time |
|------|---------------|-----------|------|
| `--perft` | Node counts at depth for 6 standard positions | 28 depth/position combos (~53M nodes) | ~30s |
| `--moves` | Legal move lists for edge-case FENs | 10 positions | ~2s |
| `--termination` | Game status (checkmate/stalemate/draw) | 13 positions | ~2s |
| `--fuzz N` | Random games, compare moves + status every position | N×~250 positions | ~60s/game |
| `--all` | Runs perft + moves + termination + fuzz 5 | ~1,500 positions | ~3 min |

### Perft suite detail

All reference counts from [chessprogramming.org/Perft_Results](https://www.chessprogramming.org/Perft_Results):

| Position | FEN | Depths tested | Max nodes |
|----------|-----|---------------|-----------|
| Startpos | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` | d1–d5 | 4,865,609 |
| Kiwipete | `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1` | d1–d4 | 4,085,603 |
| Position 3 | `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1` | d1–d6 | 11,030,083 |
| Position 4 | `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1` | d1–d5 | 15,833,292 |
| Position 5 | `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8` | d1–d4 | 2,103,487 |
| Position 6 | `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 10` | d1–d4 | 3,288,373 |

### Edge-case move comparison

Each FEN targets a specific chess engine pitfall:

| Edge case | FEN | Validates |
|-----------|-----|-----------|
| EP discovered check (legal) | `8/8/8/k2pP2R/8/8/8/4K3 w - d6 0 1` | EP that gives discovered check is legal for moving side |
| EP discovered check (illegal) | `8/8/8/8/R2pPk2/8/8/4K3 b - e3 0 1` | EP that exposes own king is correctly rejected |
| Castling through check | `r3k2r/8/8/8/3q4/8/8/R3K2R w KQkq - 0 1` | Queenside blocked by queen on d-file |
| Promotion + capture | `r7/P7/8/8/8/8/8/4K1k1 w - - 0 1` | All 4 promo types × push and capture |
| Stalemate (0 moves) | `k7/2Q5/1K6/8/8/8/8/8 b - - 0 1` | Legal move list is empty |
| Checkmate (0 moves) | `k7/8/8/8/8/8/QQ6/4K3 b - - 0 1` | Legal move list is empty, king in check |

### Termination status comparison

| Category | Positions tested | What's checked |
|----------|-----------------|----------------|
| Checkmate | 3 (fool's mate, scholar's mate, Qg7#) | `board.game_status() == "checkmate"` matches `chess.Board.is_checkmate()` |
| Check | 1 (queen check, king can capture) | Correctly reports `check` not `checkmate` |
| Stalemate | 2 (basic, pawn block) | `stalemate` matches `chess.Board.is_stalemate()` |
| Insufficient material | 3 (K vs K, K+B vs K, K+N vs K) | `insufficient_material` matches `chess.Board.has_insufficient_material()` |
| Not insufficient | 2 (K+R vs K, K+P vs K) | Correctly reports `ongoing` |
| Ongoing | 2 (startpos, midgame) | Normal positions report `ongoing` |

## 3. Perft CLI

**Binary**: `src/bin/perft.rs`
**Build**: `cargo build --release --bin perft`

```bash
# Node count
cargo run --release --bin perft -- "<FEN>" <depth>

# Per-move breakdown (like Stockfish "go perft")
cargo run --release --bin perft -- --divide "<FEN>" <depth>

# Legal moves in UCI format (sorted, one per line)
cargo run --release --bin perft -- --moves "<FEN>"

# Game termination status
cargo run --release --bin perft -- --status "<FEN>"
# outputs: checkmate | stalemate | insufficient_material | check | ongoing
```

Useful for debugging mismatches: run `--divide` on both our engine and Stockfish/python-chess to find which move's subtree diverges.

## 4. End-to-End Self-Play

**Script**: `bash scripts/e2e_test.sh`
**What it tests**: Full MuZero loop — self-play games → training → loss decrease
**Prerequisite**: Python env with `hyzero` package, `--release` build
**Time**: ~2 min

## 5. Adding New Tests

### For move generation bugs
1. Find the failing FEN (via fuzz or manual play)
2. Run `--divide` at depth 1 to compare move list against python-chess
3. Add the FEN to `EDGE_CASE_FENS` in `cross_validate.py`
4. If it's a standard position, add a perft test in `src/game/perft.rs`

### For game termination bugs
1. Construct the FEN, verify expected status with python-chess
2. Add to `TERMINATION_FENS` in `cross_validate.py`
3. Add a Rust unit test in `board.rs` `mod tests`

### For draw rule bugs (threefold, 50-move)
These require move sequences, not static FENs. Add Rust tests in `board.rs` that call `compute_turn_items()` in a loop and assert `game_result` at each step.

## Related

- [Chess Engine](chess-engine.md) — board representation, move generation, gotchas
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline architecture
- [Mistakes Log](mistakes.md) — past bugs with root cause analysis
