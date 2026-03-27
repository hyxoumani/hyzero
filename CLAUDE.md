# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System: Local LLM Review Hook

A PostToolUse hook sends every Edit/Write diff to **Qwen2.5-Coder:32b** (Ollama, localhost:11434) for automatic bugs-and-correctness review. See `docs/system.md` for full details.

- **Hook script**: `~/.claude/hooks/ollama-review.sh`
- **Hook config**: `~/.claude/settings.local.json`
- **Behavior**: blocks with review text if issues found, silent on LGTM or non-source files
- **Failure mode**: silent (Ollama down = no disruption)
- **Disable**: remove/rename `~/.claude/settings.local.json`

---

## Project Overview

`hyzero` is a Rust chess engine attempting to emulate MuZero. It uses bitboard representation and magic bitboards for sliding piece move generation, with an async TCP server for networked play.

## Commands

```bash
# Build
cargo build

# Run main binary (currently a scratch/test entry point)
cargo run

# Run the server binary
cargo run --bin server

# Run the client binary
cargo run --bin client

# Check for errors without building artifacts
cargo check

# Run tests
cargo test

# Run a single test
cargo test <test_name>
```

## Architecture

### Core Types (`src/lib.rs`)
All fundamental types live here and are re-exported for use across the crate:
- `Bitboard = u64` — squares 0–63, A1=0, H8=63, rank-major order (sq = rank*8 + file)
- `Square` enum (A1..H8) with `u8`/`usize` conversions via `From`
- `Color`, `PieceType`, `Piece`, `CastleOption`
- `PrecomputedItems` — computed once at startup via `begin_precomputing()`, shared via `Arc`. Contains knight/king/pawn lookup tables, magic bitboard tables for rooks and bishops, and precomputed ray/line masks used for pin detection.

**Note:** The `Move` struct is defined in `src/game/mod.rs` (fields: `from`, `to`, `promotion_piece_type`, `castle_option`, `en_passant`). There is only one Move struct in the codebase.

### Game Layer (`src/game/`)
- `mod.rs` — `GameState` wraps `GameBoard`; also defines the `game::Move` struct used by players
- `board.rs` — `GameBoard` is the core engine. It holds per-player `Player` structs, bitboards for white/black/combined occupancy, castling rights, pin masks, and the last move. Key methods:
  - `get_move_mask()` — dispatch to piece-specific move generation
  - `get_sliding_moves()` — magic bitboard lookup for rooks/bishops
  - `get_attackers()` — reverses move generation to find all pieces attacking a square
  - `validate_move()` — pseudo-legal check + simulated board clone to verify king safety
  - `calculate_pins()` — uses precomputed ray masks between king and enemy sliders
  - `calculate_checkmate()` / `calculate_stalemate()` — called after each turn
- `playerobj.rs` — `Player` holds per-piece-type bitboards (`pieces_bb: [u64; 6]`, indexed by `PieceType as usize`), combined occupancy `pieces`, and an `own_board: [Option<Piece>; 64]` mailbox. Move input is parsed from coordinate notation (e.g. `"e2e4"`).
- `externplayer.rs` — stub for TCP-connected external players

### Piece Move Generation (`src/pieces/`)
- `mod.rs` — `Piece` trait with `get_piece_type()` and `get_color()`
- `mod_rook.rs`, `bishop.rs` — Magic bitboard implementation: each square has a `RookEntry`/`BishopEntry` with a `mask`, randomly-found `magic_num`, `sig_bits`, and a precomputed `magic_table`. Lookup: `table[(mask & occupancy).wrapping_mul(magic) >> (64 - sig_bits)]`
- Other piece files contain structs implementing the `Piece` trait but move generation is handled centrally in `GameBoard::get_move_mask()`

### Game History (`src/game/history.rs`)
- `GameHistory` stores move history (`Vec<String>` with W/B prefixes) and board snapshots (`Vec<[Option<Piece>; 64]>`) after each validated move

### Session & Server (`src/session/`, `src/bin/`)
- `session/mod.rs` — `SessionObj` holds `Arc<PrecomputedItems>`; intended as the top-level container for session state (MCTS info, etc.)
- `src/bin/server.rs` — async Unix domain socket server on `/tmp/hyzero.sock` using Tokio. Accepts 2 clients (White, Black), coordinates turns via `mpsc` channels, validates moves via `GameBoard::process_move()`, records move history with W/B prefixes, stores board snapshots, and sends bitboard representation after every move.
- `src/bin/client.rs` — async Unix domain socket client. Connects to server, receives color assignment, handles protocol messages (`YOUR_TURN`, `OK`, `INVALID`, `OPPONENT_MOVED`, `BOARD`, `GAME_OVER`), prompts for move input via stdin.

### Architecture
See `docs/ARCHITECTURE.md` for the full MuZero system design including MCTS, neural network components, self-play threading model, training loop, and data storage.

### Remaining Work (`docs/todo.md`)
Key incomplete areas: en passant edge cases, full legal move filtering, and the MuZero/MCTS search layer are not yet implemented.

---

## Refactoring Plan — Chess Game Logic

### Bug Summary (18 issues found)

| # | File | Bug | Severity |
|---|------|-----|----------|
| 1 | `Cargo.toml:4` | Edition "2024" invalid | Blocks build |
| 2 | `server.rs` | `crate::` in binary, missing arg, syntax error | Blocks build |
| 3 | `board.rs:320` | `if let Some()` on bool field | Blocks build |
| 4 | `board.rs:58,67,74` | count not mut, `return` instead of `break` | Game loop broken |
| 5 | `board.rs:91-95` | Colors inverted in `compute_turn_items` | Wrong side checked |
| 6 | `board.rs:241-242` | Wrong pins in `calculate_checkmate` | Checkmate wrong |
| 7 | `board.rs:249` | Wrong color to `get_attackers` | Checkmate wrong |
| 8 | `board.rs:251` | `& player.pieces` restricts king moves to own pieces | King escapes broken |
| 9 | `board.rs:260` | Falls through when not in check (no early return) | False checkmates |
| 10 | `board.rs:514-515` | Double-push mask uses wrong ranks | Pawn moves broken |
| 11 | `board.rs:306-316` | Queenside castling unvalidated | Illegal castles |
| 12 | `lib.rs:127-134` | Castle squares include king sq (always occupied) | Castling always fails |
| 13 | `board.rs:334` | Uses old occupancy after simulated move | King safety wrong |
| 14 | `board.rs:410-415` | EP target calculated but never stored | EP impossible |
| 15 | `board.rs:105-128` | `update_castling` never called | Rights never update |
| 16 | `board.rs:96` | Pins recalculated for wrong/single color | Pin data stale |
| 17 | `board.rs:417` | `board_arr` update after capture overwrites correctly set value | Minor |
| 18 | `board.rs:28` | `is_en_passant` field unused | Dead code |

### Tasks

#### Phase 1: Sequential (must run in order)

**Task 1: Fix Compilation Errors**
- `Cargo.toml:4` — change `edition = "2024"` to `edition = "2021"`
- `server.rs` — fix `crate::` imports to `hyzero::`, add missing arg to `start_session()`, fix `num_waiting = i32` to `num_waiting: i32`
- `board.rs:320` — remove the dead `if let Some(is_en_passant)` block
- Verify: `cargo check` passes

**Task 2: Fix Game Loop (`start_game`)**
- `board.rs:57-82` — make `count` mutable, change `return` to `break`, restructure `piece_moved` assignment
- Verify: `cargo build`; game accepts alternating moves

#### Phase 2: Parallel (these are independent, spawn subagents)

After Tasks 1-2 pass, the following can run **in parallel as subagents**:

**Task 3: Fix `compute_turn_items` Color Logic** `[PARALLEL]`
- `board.rs:84-103` — swap colors (when white just moved, check Black next)
- Recalculate pins for BOTH sides
- Call `update_castling(piece_moved)` (currently never called)

**Task 4: Fix `calculate_checkmate`** `[PARALLEL]`
- `board.rs:240-282` — change signature to take `color: Color` instead of `count`
- Fix pin selection, `get_attackers` color, remove `& player.pieces` on line 251
- Add early `return false` when `attackers == 0`

**Task 5: Fix Pawn Double-Push Mask** `[PARALLEL]`
- `board.rs:514-515` — white mask to `0x0000_0000_FF00_0000` (rank 4), black to `0x0000_00FF_0000_0000` (rank 5)

**Task 6: Fix Castling Validation** `[PARALLEL]`
- `lib.rs:125-136` — split `castle_squares` into `castle_empty_squares` and `castle_path_squares`
  - Empty: f1,g1 / b1,c1,d1 / f8,g8 / b8,c8,d8
  - Path: e1,f1,g1 / e1,d1,c1 / e8,f8,g8 / e8,d8,c8
- `board.rs:306-316` — remove kingside-only guard, check empty for occupancy, path for attacks

**Task 7: Fix `validate_move` King Safety** `[PARALLEL]`
- `board.rs:327-336` — after `temp_state.update_board()`, recalculate occupancy from temp player bitboards and recalculate king square; use these for `get_attackers`

#### Phase 3: Sequential (depends on Phase 2)

**Task 8: Implement En Passant**
- Add `en_passant_target: Option<usize>` to `GameBoard`, remove unused `is_en_passant`
- `update_board`: store `en_passant_target` on double pawn push, clear on other moves
- `get_pawn_moves`: include EP target square in valid attacks
- `update_board`: detect EP capture and remove captured pawn from square behind
- Clone-and-check in `validate_move` handles pin edge cases automatically

**Task 9: Add Draw Rules**
- **9a: 50-Move Rule** — add `halfmove_clock: u32`, reset on pawn move/capture, game over at 100
- **9b: Threefold Repetition** — add `position_history: HashMap<u64, u8>`, hash position after each move, game over at count 3
- **9c: Insufficient Material** — detect K vs K, K+N vs K, K+B vs K, K+B vs K+B (same color)

**Task 10: Add Game Result Reporting**
- Add `GameResult` enum: `Ongoing`, `Checkmate(Color)`, `Stalemate`, `FiftyMoveRule`, `ThreefoldRepetition`, `InsufficientMaterial`
- Replace `is_game_over: bool` with `game_result: GameResult`
- Print result when game ends

### Execution Strategy

Every task runs as a **subagent**. Sequential tasks wait for the previous to complete before spawning. Parallel tasks are spawned simultaneously.

```
Task 1 (subagent) -> Task 2 (subagent) -> [Task 3+4 | Task 5 | Task 6 | Task 7] (parallel subagents) -> Task 8 (subagent) -> Task 9 (subagent) -> Task 10 (subagent)
```

Tasks 3+4 are combined into one subagent since they both modify `compute_turn_items` / `calculate_checkmate`.

After each subagent completes, update CLAUDE.md with status (`DONE` / `FAILED`) and note any changes made.

### Task Status

| Task | Status | Notes |
|------|--------|-------|
| 1. Fix Compilation | DONE | Cargo.toml edition, server.rs imports/syntax, board.rs type mismatch |
| 2. Fix Game Loop | DONE | count mut, return->break, piece_moved restructure |
| 3+4. Fix Turn Colors + Checkmate | DONE | Swapped colors, both-side pins, update_castling call, checkmate signature/logic |
| 5. Fix Pawn Double-Push | DONE | Corrected rank masks (rank 4/5 instead of 7-8/1-2) |
| 6. Fix Castling Validation | DONE | Split castle_squares into empty/path, both sides validated |
| 7. Fix validate_move Safety | DONE | Use temp state occupancy and recalculate king sq |
| 8. Implement En Passant | DONE | en_passant_target field, EP capture in update_board, EP in get_pawn_moves |
| 9. Add Draw Rules | DONE | 50-move clock, threefold repetition hash, insufficient material |
| 10. Game Result Reporting | DONE | GameResult enum replaces is_game_over bool, prints result |
| 11. Create GameHistory | DONE | history.rs with move_history and board_snapshots |
| 12. Add process_move() | DONE | process_move, board_snapshot, result, bitboard_string methods; parse_move made pub |
| 13. Rewrite server.rs | DONE | Unix domain socket server with turn coordination, history, bitboard output |
| 14. Rewrite client.rs | DONE | Unix domain socket client with protocol handling and stdin input |
| 15. Update CLAUDE.md | DONE | Updated architecture section, task status table |
| 16. Write ARCHITECTURE.md | DONE | Full MuZero architecture doc |

### MCTS & Self-Play Infrastructure (Tasks 17-23)

Detailed task document: `docs/TASKS_MCTS_SELFPLAY.md`

| Task | Status | Notes |
|------|--------|-------|
| 17. Module Structure + Data Types | DONE | data/types.rs, data/encoding.rs, mcts/mod.rs, selfplay/mod.rs |
| 18. MCTS Tree + PUCT | DONE | MCTSNode, MCTSTree, Evaluator trait, PUCT selection, simulation loop |
| 19. Inference Channel + Batching | DONE | InferenceBatcher, ChannelEvaluator, RandomBackend stub |
| 20. Replay Buffer | DONE | VecDeque ring buffer, weighted sampling, bincode checkpoints |
| 21. Self-Play Game Task | DONE | play_game() async fn, MCTS per move, trajectory building |
| 22. Coordinator + Training Thread | DONE | SelfPlayCoordinator with semaphore, TrainingThread stub with replay buffer, selfplay binary |
| 23. Integration + Cleanup | DONE | Clippy fixes in new modules, removed DEBUG print, verified selfplay+tests |

**Execution order:**
```
Task 17 → Task 18 → [Task 19 | Task 20] (parallel) → Task 21 → Task 22 → Task 23
```

### Python Neural Network Layer (Tasks 24-26)

Detailed task document: `docs/TASKS_PYTHON.md`

| Task | Status | Notes |
|------|--------|-------|
| 24. Python Project Setup + Models | TODO | config, ResidualBlock, h/g/f networks, shape tests |
| 25. Training Loop + Checkpointing | TODO | Trainer class, MuZero loss, K-step unroll, weight save/load |
| 26. Inference Server | TODO | InferenceServer class, batch methods, weight loading |

**Execution order:**
```
Task 24 → [Task 25 | Task 26] (parallel)
```

Python tasks can run in parallel with Rust Tasks 17-20 (shared interface spec only).

Every task runs as a **subagent** with edit and bash permissions.
Update CLAUDE.md and docs/ status after each task. When any task changes architecture, adds modules, modifies APIs, or alters behavior, update all relevant docs in `docs/` (ARCHITECTURE.md, TASKS_MCTS_SELFPLAY.md, TASKS_PYTHON.md, todo.md, etc.) to reflect those changes.

### Verification
After each task: `cargo check` / `cargo build`
After all tasks: `cargo test` with unit tests for:
- Pawn moves (single push, double push, blocked, en passant)
- Castling (kingside/queenside, blocked, through check)
- Checkmate (back-rank mate, not-in-check returns false)
- Stalemate (known position)
- Pin handling (pinned piece restricted to pin line)
- Draw rules (50-move, repetition, insufficient material)

### Correction to Architecture Docs
The CLAUDE.md previously stated there are two Move structs (one in lib.rs, one in game/mod.rs). There is only ONE Move struct, defined in `src/game/mod.rs` with fields: `from`, `to`, `promotion_piece_type`, `castle_option`, `en_passant`.
