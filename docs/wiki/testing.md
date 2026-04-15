# Testing Procedures

How to validate the hyzero engine at every level — from quick smoke tests to exhaustive cross-validation.

## Quick Reference

```bash
cargo test                                          # 89 Rust tests (82 pass, 7 ignored; ~60s debug)
cargo test --release -- --ignored                    # + 7 slow/Python tests (~2s release)
python3 scripts/cross_validate.py --all              # full oracle comparison (~3 min)
bash scripts/e2e_test.sh                             # end-to-end self-play loop
```

## 1. Rust Unit Tests

**Command**: `cargo test`
**Time**: ~60s (debug), ~2s (release)
**Result**: 82 pass, 7 ignored

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
| `selfplay::*` | 11 | Coordinator, game task, inference batching, training pipeline, checkpoint window, evaluation task, resume |
| `py::*` | 4 (3 ignored) | PyO3 batch assembly/shapes; inference+training need Python env |

### Ignored tests

| Test | Why ignored |
|------|-------------|
| `perft_startpos_d4/d5`, `perft_kiwipete_d3` | ~15s debug mode; run `--release -- --ignored perft_` |
| `py::*` + selfplay evaluation/resume | Require `hyzero` Python package; `cd python && pip install -e . && cargo test -- py::` |

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

Six reference positions from [chessprogramming.org/Perft_Results](https://www.chessprogramming.org/Perft_Results) tested at depths 1–6 (up to 53M nodes total). Includes Startpos (d1–d5: 4.8M), Kiwipete (d1–d4: 4.0M), and edge positions with promotion/castling interactions.

10 edge-case FENs: EP discovered check (legal/illegal), castling through check, promotion+capture, stalemate, checkmate. 13 termination status tests: checkmate/check/stalemate/insufficient-material/draw detection vs python-chess.

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

Find failing FEN, verify vs python-chess, add to appropriate test list (`EDGE_CASE_FENS`, `TERMINATION_FENS`) and Rust test in `board.rs`.

## 6. Baseline & Validation

Baseline: **6.78** (commit d407281, 2026-04-14 — Dirichlet alpha fix: 0.03 → 0.3 for chess).

The metric has ±1 point noise floor. **Rule**: Changes <1.5 points need 2–3 reruns; median reported.

## 7. Metric Definition Precision

Score multipliers must count discrete events (e.g., promotions), not version tags or checkpoint indices. A single promotion can update a version tag by arbitrary amounts depending on producer/consumer rate ratios. Always verify metric extraction against ground truth:
```bash
# Example: validate promotion count
expected_promotions=2
actual=$(grep -c "\[eval\] promoted" run.log)
if [ "$actual" -ne "$expected_promotions" ]; then
  echo "ERROR: Metric extraction mismatch"
fi
```

See 2026-04-15 mistakes.md entry "Metric Inflation from Training-Version Tag vs Promotion Count" for the inflation bug and fix.

## Related

- [Chess Engine](chess-engine.md) — move generation, gotchas
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline
- [Mistakes Log](mistakes.md) — past bugs
