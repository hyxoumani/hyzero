# Testing Procedures

How to validate the hyzero engine — from quick unit tests to exhaustive
cross-validation against a python-chess oracle.

## Quick Reference

```bash
cargo test                                   # Rust unit tests (debug, ~minute)
cargo test --release -- --ignored            # slow perft + Python-backed tests
cargo clippy                                 # lint
cargo fmt                                     # format
python3 scripts/cross_validate.py --all      # full python-chess oracle comparison
bash scripts/e2e_test.sh                      # end-to-end self-play loop
bash scripts/run_baseline.sh 1800             # 30-min controlled baseline (see baseline-scoring.md)
```

## 1. Rust Unit Tests

`cargo test` runs the in-crate `#[test]` / `#[tokio::test]` suites. Some tests are
`#[ignore]`d for speed or because they need the Python env.

| Area | Module(s) | Coverage |
|------|-----------|----------|
| Move gen + rules | `game::board`, `game::fen` | all pieces, check/mate/stalemate, castling, EP, promotion, pins, zobrist, repetition, 50-move, insufficient material |
| Perft | `game::perft` | node counts for standard positions (deep ones `#[ignore]`d) |
| Encoding | `data::encoding` | action↔move round-trip (normal, castling, EP, promotion) |
| Replay buffer | `data::replay_buffer` | add/evict, prioritized sampling, checkpoint round-trip |
| Replay capture | `selfplay::replay_writer` | serialize round-trip, filename format |
| MCTS | `mcts::puct`, `mcts::tree` | PUCT scoring, tie-break uniformity, visit distribution, action selection, color-symmetry after sort |
| Elo ladder | `selfplay::elo`, `selfplay::evaluation`, `selfplay::pool`, `selfplay::champion` | rating math, candidate-Elo folding, bootstrap vs Elo gate, archive enumeration, promotion/version |
| PyO3 + selfplay | `py::*`, `selfplay::coordinator` | batch assembly/shapes, coordinator trajectories (some `#[ignore]`d — need the Python package) |
| Binary config | `bin::selfplay` | env-var → `RunConfig` parsing (serialized under a lock) |

**Ignored tests** are the deep perft positions (`cargo test --release -- --ignored
perft_`) and the Python-backed `py::*` / selfplay-eval tests (need
`cd python && pip install -e .`). Run them in `--release` for speed.

## 2. Cross-Validation (python-chess oracle)

`scripts/cross_validate.py` calls the `perft` binary and compares against
python-chess. Requires `python-chess` and `cargo build --release --bin perft`.

| Flag | Tests | Notes |
|------|-------|-------|
| `--perft` | perft node counts for standard positions | the heavy mode |
| `--moves` | legal-move lists for `EDGE_CASE_FENS` | EP discovered check, castling through check, promotion+capture, stale/checkmate |
| `--termination` | game status for `TERMINATION_FENS` | checkmate/stalemate/check/insufficient material/draw |
| `--fuzz N` | N random games, compare moves + status at every position | |
| `--all` | perft + moves + fuzz 5 + termination | default when no flag is given |

Use `--divide` on a divergent FEN against both engines to find which move's
subtree differs (see Perft CLI).

## 3. Perft CLI (`src/bin/perft.rs`)

```bash
cargo run --release --bin perft -- "<FEN>" <depth>          # node count
cargo run --release --bin perft -- --divide "<FEN>" <depth> # per-move breakdown
cargo run --release --bin perft -- --moves "<FEN>"          # legal moves (UCI, sorted)
cargo run --release --bin perft -- --status "<FEN>"         # termination status
```

## 4. End-to-End Self-Play

`bash scripts/e2e_test.sh` exercises the full MuZero loop (self-play → training →
loss decrease). Needs the `hyzero` Python package and a `--release` build.

## 5. Conventions (`.claude/rules/testing.md`)

- Test names describe behavior, not implementation
  (`rejects_invalid_move`, not `test_func_3`).
- One assertion per test where practical.
- New code paths require tests before merging.
- **Flaky policy**: run 3×. Pass 2/3 = flaky (flag, continue); 0/3 = real failure.
- **Regression tests** must fail without the fix and pass with it.
- The `test-gate` hook runs the test command on session end and blocks completion
  on failure (see [Dev Workflow](dev-workflow.md)).

## 6. Adding New Tests

Find a failing FEN, confirm the correct answer with python-chess, add it to the
appropriate list (`EDGE_CASE_FENS`, `TERMINATION_FENS`) in `cross_validate.py`
and a Rust unit test in `board.rs`.

## 7. Baseline & Score Validation

The headline benchmark is `scripts/run_baseline.sh` → `logs/baseline_score.json`.
Score has a binomial noise floor on small game samples; compare medians of reruns
for small deltas, and validate metric extraction (e.g. promotion count) against
the log before trusting a delta. See [Baseline Scoring](baseline-scoring.md).

## Related

- [Chess Engine](chess-engine.md) — move generation, gotchas
- [Baseline Scoring](baseline-scoring.md) — composite score, `baseline_score.json`
- [Dev Workflow & Framework](dev-workflow.md) — hooks, the test-gate
- `scripts/cross_validate.py`, `scripts/e2e_test.sh`, `src/bin/perft.rs`
