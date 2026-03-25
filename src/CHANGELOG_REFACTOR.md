# Chess Game Logic Refactor — Session Summary

## Starting State

The hyzero chess engine had solid foundations (magic bitboards, pin detection, precomputed tables) but **could not run a single game** due to critical bugs throughout the codebase. The code did not compile, the game loop was broken, and several chess rules were missing or incorrect.

### 18 Bugs Found

| # | File | Bug | Severity |
|---|------|-----|----------|
| 1 | `Cargo.toml:4` | Edition "2024" invalid (doesn't exist) | Blocks build |
| 2 | `server.rs` | `crate::` in binary, missing arg to `start_session()`, syntax error `= i32` | Blocks build |
| 3 | `board.rs:320` | `if let Some()` on a `bool` field | Blocks build |
| 4 | `board.rs:58,67,74` | `count` not mutable, `return` instead of `break` in game loop | Game loop broken |
| 5 | `board.rs:91-95` | Colors inverted in `compute_turn_items` — checked wrong side for checkmate/stalemate | Wrong side checked |
| 6 | `board.rs:241-242` | Wrong pins used in `calculate_checkmate` (white_pins when checking black) | Checkmate wrong |
| 7 | `board.rs:249` | Wrong color passed to `get_attackers` in checkmate detection | Checkmate wrong |
| 8 | `board.rs:251` | `& player.pieces` restricted king escape moves to own-piece squares | King escapes broken |
| 9 | `board.rs:260` | No early return when `attackers == 0` — false checkmates when not in check | False checkmates |
| 10 | `board.rs:514-515` | Double-push mask used wrong ranks (7-8/1-2 instead of 4/5) | Pawn moves broken |
| 11 | `board.rs:306-316` | Only kingside castling validated, queenside skipped | Illegal castles |
| 12 | `lib.rs:127-134` | Castle squares included king's own square (always occupied) | Castling always fails |
| 13 | `board.rs:334` | `validate_move` used old occupancy after simulated move | King safety wrong |
| 14 | `board.rs:410-415` | En passant target calculated but never stored | EP impossible |
| 15 | `board.rs:105-128` | `update_castling()` defined but never called | Rights never update |
| 16 | `board.rs:96` | Pins recalculated for wrong/single color | Pin data stale |
| 17 | `board.rs:28` | `is_en_passant` field unused (dead code) | Dead code |
| 18 | `board.rs:282` | Operator precedence: `1u64 << sq & pins` missing parentheses | Stalemate wrong |

### Missing Features
- En passant (completely unimplemented)
- 50-move rule
- Threefold repetition
- Insufficient material detection
- Game result reporting (only a boolean `is_game_over`)

---

## Changes Made

### Task 1: Fix Compilation Errors
**Files:** `Cargo.toml`, `src/bin/server.rs`, `src/game/board.rs`, `src/main.rs`

- Changed `edition = "2024"` to `edition = "2021"`
- Fixed `crate::` imports to `hyzero::` in server binary
- Added missing `Arc<PrecomputedItems>` arg to `start_session()`
- Fixed `num_waiting = i32` syntax to `num_waiting: i32`
- Removed dead `if let Some(is_en_passant)` block on bool field

### Task 2: Fix Game Loop (`start_game`)
**File:** `src/game/board.rs`

- Made `count` mutable: `let mut count: usize = 0`
- Changed `return` to `break` so valid moves exit the inner loop, not the function
- Restructured so `compute_turn_items()` and `count += 1` actually execute
- Added "Invalid move, try again." feedback

### Task 3+4: Fix Turn Colors + Checkmate Logic
**File:** `src/game/board.rs`

**compute_turn_items:**
- Swapped color logic: when `count % 2 == 0` (white just moved), now correctly sets `color_to_move = Black`
- Added `self.update_castling(piece_moved)` call (was defined but never called)
- Recalculates pins for BOTH sides instead of just one

**calculate_checkmate:**
- Changed signature from `count: usize` to `color: Color`
- Fixed pin selection: uses `black_pins` when checking black, `white_pins` for white
- Fixed `get_attackers` to use opponent color correctly
- Removed `& player.pieces` that was restricting king moves to own-piece squares
- Added early `return false` when `attackers == 0` (not in check = not checkmate)
- Added `continue` to skip king square when iterating blocking pieces

### Task 5: Fix Pawn Double-Push Mask
**File:** `src/game/board.rs`

- Changed rank-wide masks (`0xFFFF_0000_0000_0000` / `0x0000_0000_0000_FFFF`) to correct ranks: `0x0000_0000_FF00_0000` (rank 4) and `0x0000_00FF_0000_0000` (rank 5)
- Later refined to per-pawn blocking: computes the specific double-push square for the blocked pawn instead of masking the entire rank

### Task 6: Fix Castling Validation
**Files:** `src/lib.rs`, `src/game/board.rs`

- Split `castle_squares` into two arrays in `PrecomputedItems`:
  - `castle_empty_squares` — squares between king and rook (must be unoccupied)
  - `castle_path_squares` — squares king passes through (must not be under attack, includes king's starting square)
- Removed the `if castle_option == CastleOption::Kingside` guard so both sides are validated
- Updated `validate_move` to check empty squares for occupancy and path squares for attacks separately

### Task 7: Fix validate_move King Safety
**File:** `src/game/board.rs`

- After `temp_state.update_board(piece_moved)`, now recalculates occupancy from temp player bitboards (`temp_state.player1.pieces`, `temp_state.player2.pieces`)
- Recalculates king square from temp state (king may have moved)
- Uses these updated values for the `get_attackers` call

### Task 8: Implement En Passant
**File:** `src/game/board.rs`

- Added `en_passant_target: Option<usize>` field to `GameBoard`, replacing unused `is_en_passant: bool`
- In `update_board`: stores EP target square on double pawn push, clears on any other move
- In `get_pawn_moves`: adds EP target to valid attacks when a pawn can reach it diagonally
- In `update_board`: detects EP capture (pawn moves to `en_passant_target`) and removes the captured pawn from the square behind
- King safety for EP handled automatically by the clone-and-check in `validate_move`

### Task 9: Add Draw Rules
**File:** `src/game/board.rs`

- **50-move rule**: Added `halfmove_clock: u32`. Resets on pawn moves or captures, increments otherwise. Game drawn at 100 halfmoves.
- **Threefold repetition**: Added `position_history: HashMap<u64, u8>`. Hashes position after each move using piece bitboards + side to move + castling rights + EP target. Draw at 3 occurrences.
- **Insufficient material**: Added `is_insufficient_material()` detecting K vs K, K+minor vs K, and K+B vs K+B with same-color bishops.

### Task 10: Add Game Result Reporting
**Files:** `src/game/board.rs`, `src/game/mod.rs`

- Added `GameResult` enum: `Ongoing`, `Checkmate(Color)`, `Stalemate`, `FiftyMoveRule`, `ThreefoldRepetition`, `InsufficientMaterial`
- Replaced `is_game_over: bool` with `game_result: GameResult` throughout
- `start_game` prints the specific result when the game ends
- Removed unused `game_over: bool` from `GameState` in `mod.rs`

### Bug Fix: Stalemate Operator Precedence
**File:** `src/game/board.rs`

- Fixed `1u64 << sq & pins` to `(1u64 << sq) & pins` — `<<` has higher precedence than `&`

### Bug Fix: Per-Pawn Double-Push Blocking
**File:** `src/game/board.rs`

- Replaced rank-wide mask with single-square mask targeting only the specific pawn's double-push destination

---

## Current State

### Working Chess Rules
- Move generation (magic bitboards for sliding pieces, precomputed tables for others)
- Castling (both sides, rights tracking, validation for empty/safe squares)
- En passant (target storage, capture generation, capture execution)
- Promotion (detection, defaults to queen, bitboard updates)
- Checkmate detection (king escape, single/double check, blocking, pin-aware)
- Stalemate detection (all pieces checked for legal moves, pin-aware)
- 50-move rule
- Threefold repetition
- Insufficient material
- Game result reporting with specific variants

### Remaining Work
- **Unit tests**: No tests exist — all rules are unverified beyond compilation
- **Board display**: No visual representation during play
- **Move notation**: No move history or algebraic notation output
- **Input handling**: `parse_move` panics on bad input
- **Server/client**: `handle_connection` is empty, no protocol
- **MCTS/MuZero**: Not started
- **26 compiler warnings**: Unused imports, mutable vars, dead fields (cosmetic)
