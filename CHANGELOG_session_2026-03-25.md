# Session Summary — 2026-03-25

## Overview

This session added networked two-player chess via Unix domain sockets, created a comprehensive MuZero architecture document, and added game history tracking with bitboard output.

---

## Changes Made

### 1. Game History Tracking (`src/game/history.rs` — NEW)

Added a `GameHistory` struct to record validated moves and board state:

- `move_history: Vec<String>` — stores every validated move with a color prefix (e.g., `"W: e2e4"`, `"B: e7e5"`)
- `board_snapshots: Vec<[Option<Piece>; 64]>` — stores a full copy of the board array after each validated move
- `record_move(prefix, notation, board_state)` — appends a move and snapshot
- Registered as `pub mod history` in `src/game/mod.rs`

### 2. Step-Based Game Engine API (`src/game/board.rs`)

The game loop previously ran as a blocking stdin loop in `start_game()`. Added methods to drive the engine one move at a time for networked play:

- **`process_move(&mut self, move_str: &str, color: Color, turn_count: usize) -> Result<(Move, GameResult), String>`**
  - Parses the move string via `Player::parse_move()`
  - Validates via `validate_move()`
  - Applies via `compute_turn_items()`
  - Returns the move and updated game result, or an error string
- **`board_snapshot(&self) -> [Option<Piece>; 64]`** — returns a copy of the current board array
- **`result(&self) -> GameResult`** — returns the current game result
- **`bitboard_string(&self) -> String`** — returns all 12 piece bitboards (6 per side) as labeled hex values:
  ```
  wp=000000000000ef00 wn=0000000000000042 wb=0000000000000004 wr=... wq=... wk=... bp=... bn=... bb=... br=... bq=... bk=...
  ```

### 3. Made `parse_move` Public (`src/game/playerobj.rs`)

Changed `Player::parse_move()` from `fn` to `pub fn` so the server can parse move strings without going through the stdin-based `make_move()` method.

### 4. Unix Domain Socket Server (`src/bin/server.rs` — REWRITE)

Completely rewrote the server from a WIP TCP stub to a working Unix domain socket game server:

- **Socket**: Listens on `/tmp/hyzero.sock` (cleans up stale socket on startup and shutdown)
- **Connection flow**: Accepts exactly 2 connections — first is White, second is Black. Sends `COLOR white/black` on connect.
- **Shared state**: `Arc<Mutex<SharedState>>` holding `GameBoard`, `GameHistory`, and `turn_count`
- **Turn coordination**: Per-client `tokio::sync::mpsc` channels. After a move is validated, the handler sends `OPPONENT_MOVED` + `BOARD` + `YOUR_TURN` to the opponent's channel.
- **Protocol** (newline-delimited text):
  - Server -> Client: `COLOR`, `YOUR_TURN`, `OK`, `INVALID`, `OPPONENT_MOVED`, `BOARD`, `GAME_OVER`
  - Client -> Server: `MOVE <notation>`
- **Bitboard output**: Sends `BOARD <bitboards>` to both players after every validated move
- **Server logging**: Prints each move with prefix and bitboard state to stdout (capturable to log file)
- **Move history**: Printed to stdout when the game ends

### 5. Unix Domain Socket Client (`src/bin/client.rs` — REWRITE)

Completely rewrote the client from a stub to a working interactive client:

- Connects to `/tmp/hyzero.sock`
- Reads color assignment on connect
- Main loop reads server messages line by line, handles multi-line notifications
- Message handlers:
  - `YOUR_TURN` — prompts user for move via stdin
  - `OK` — confirms move accepted
  - `INVALID` — shows error and re-prompts
  - `OPPONENT_MOVED` — displays opponent's move
  - `BOARD` — displays bitboard state
  - `GAME_OVER` — displays result and exits
  - `WAIT` — displays waiting message
- Uses `tokio::io::stdin()` for async stdin reading

### 6. Architecture Document (`ARCHITECTURE.md` — NEW)

Created a comprehensive architecture document for the full MuZero chess engine design. Covers:

- **System overview**: Rust game engine + Python neural networks
- **Component diagram**: Game state, MCTS, inference coordinator (Rust) connected via PyO3/FFI to policy/value/dynamics networks + replay buffer (Python)
- **Architectural decisions table**:
  - MCTS ownership: Rust (avoids GIL)
  - Rust-Python IPC: PyO3/FFI (simpler than shared memory)
  - Self-play: Multi-game parallel from the start
  - Data storage: In-memory replay buffer + disk checkpoints
  - MCTS persistence: Visit counts + root value only (tree discarded per move)
- **Threading model**: N game threads (pure Rust) -> shared inference queue -> 1 inference thread (acquires GIL, batches to PyTorch) -> results back via per-thread channels
- **MCTS per-move flow**: Build tree, run N simulations, extract visit distribution, discard tree, apply move
- **Training data schema**: Per-step (observation, action, visit distribution, root value, reward, legal moves) and per-game metadata (outcome, model version, temperature)
- **Training loop**: Sample position, unroll K steps, compute policy/value/reward loss, backpropagate
- **Replay buffer design**: In-memory ring buffer with random access, periodic disk checkpoints
- **Neural network descriptions**: Representation (h), dynamics (g), prediction (f) networks
- **Game server protocol**: Unix socket protocol for human/external play
- **Planned Python directory structure**

### 7. Test Scripts

- **`test_server.sh`** — starts the server and pipes output to `server_output.log`
- **`test_clients.sh`** — launches two clients that play Scholar's Mate (White checkmates in 4 moves: `e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7#`), captures both client outputs

### 8. CLAUDE.md Updates

- Updated Session & Server architecture section to reflect Unix sockets, protocol, and GameHistory
- Added reference to `ARCHITECTURE.md`
- Added tasks 11-16 to the task status table
- Added GameHistory section to architecture docs

---

## Files Changed

| File | Action | Lines Changed |
|------|--------|---------------|
| `src/game/history.rs` | Created | New file (21 lines) |
| `src/game/mod.rs` | Modified | Added `pub mod history;` |
| `src/game/board.rs` | Modified | Added `process_move`, `board_snapshot`, `result`, `bitboard_string` |
| `src/game/playerobj.rs` | Modified | `parse_move`: `fn` -> `pub fn` |
| `src/bin/server.rs` | Rewritten | 57 lines -> 173 lines |
| `src/bin/client.rs` | Rewritten | 3 lines -> 84 lines |
| `ARCHITECTURE.md` | Created | New file (~180 lines) |
| `CLAUDE.md` | Modified | Updated architecture section + task status |
| `test_server.sh` | Created | Server test launcher |
| `test_clients.sh` | Created | Automated Scholar's Mate test |

---

## Architectural Decisions Discussed

These decisions were made through discussion and are documented in `ARCHITECTURE.md`:

1. **MCTS in Rust** — tree operations are performance-sensitive, co-located with game engine avoids cross-language calls per node
2. **PyO3/FFI over shared memory** — simpler to implement; shared memory is a future optimization if GIL contention becomes a bottleneck during multi-threaded self-play
3. **Multi-game parallel self-play** — designed from the start to avoid future refactor; N game threads batch leaf nodes to one inference thread
4. **Replay buffer + disk** — in-memory ring buffer for fast training sampling, periodic disk checkpoints for crash recovery
5. **Visit counts only** — MCTS tree is transient working memory; only visit distribution + root value persisted per move (all training needs)
6. **Training data format** — per step: observation, action, visit distribution, root value, reward, legal move mask; supports K-step unrolling for MuZero loss
