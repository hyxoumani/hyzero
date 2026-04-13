# Plan: Engine Foundation

## Approach

Add a FEN parser, full unit test suite for move generation and game rules, perft
validation, fix the `action_to_move` castling/en-passant reconstruction bug, and
replace the homebrew position hash with Zobrist. All work is organized so subtasks
have disjoint file sets and can be implemented in parallel once their dependencies
are met.

---

## Subtasks

### 1. Zobrist hashing (replaces `position_hash`)

- **Files**: `src/game/board.rs`, `src/lib.rs`
- **Changes**:
  - Add a `ZobristTable` struct to `src/lib.rs` (or inline in `board.rs`) with:
    - `piece_sq: [[[u64; 64]; 6]; 2]` — indexed `[color][piece_type][square]`
    - `side_to_move: u64` — XOR in when it is Black's turn
    - `castling: [u64; 4]` — WK, WQ, BK, BQ availability bits
    - `en_passant_file: [u64; 8]` — one per file (XOR in when EP target exists)
  - Initialize with deterministic pseudorandom values at startup (seed with a
    fixed `u64`; use splitmix64 or a simple LCG to fill the table so it is
    reproducible without an external crate).
  - Add `ZobristTable` as a field of `PrecomputedItems` so it is computed once.
  - Replace `position_hash()` in `board.rs` with an incremental Zobrist hash:
    - `pub(crate) zobrist_hash: u64` field on `GameBoard`, initialized in
      `init_game_board` by hashing the starting position.
    - `update_zobrist()` called from `update_board()` to XOR out the old piece
      and XOR in the new piece for every change (moving piece, captured piece,
      rook in castling, captured en-passant pawn, promoted piece).
    - XOR the side-to-move token each half-move.
    - XOR castling/EP delta bits on any change.
  - Replace the `position_hash(color_to_move)` call in `compute_turn_items` with
    `self.zobrist_hash` (already maintained incrementally).
  - Remove the old `position_hash()` method.
- **Tests**: Add to the new `tests` module in `board.rs`:
  - `test_zobrist_starting_position` — hash of the initial board is non-zero and
    deterministic across two independent `GameBoard` instances.
  - `test_zobrist_roundtrip` — make a move and then the reverse move; hash must
    equal the original.
  - `test_zobrist_castling_rights_differ` — position before WK castle vs after
    must differ.
  - `test_zobrist_ep_differs` — position with EP target vs without must differ.
- **Dependencies**: None (self-contained). Complete before subtask 4 (threefold
  repetition correctness is needed before perft).

---

### 2. FEN parser

- **Files**: `src/game/fen.rs` (new), `src/game/mod.rs`, `src/game/board.rs`
- **Changes**:
  - Create `src/game/fen.rs` with a public function:
    ```rust
    pub fn board_from_fen(
        fen: &str,
        precomputed: Arc<PrecomputedItems>,
    ) -> Result<GameBoard, String>
    ```
  - Parse the six FEN fields:
    1. Piece placement — iterate ranks 8→1, files a→h. Build `board_arr`,
       `player1.pieces_bb`, `player2.pieces_bb`, `player1.pieces`,
       `player2.pieces`, `player1.own_board`, `player2.own_board`.
    2. Active color — set `side_to_move` (return this too; add it as a second
       return value or embed a wrapper struct).
    3. Castling availability — set `white_kingside`, `white_queenside`,
       `black_kingside`, `black_queenside`.
    4. En-passant target square — parse algebraic (e.g. "e3") to `en_passant_target: Option<usize>`.
    5. Halfmove clock — parse integer → `halfmove_clock`.
    6. Fullmove number — parse but only use to derive `turn_count` context (not
       stored; caller may use it).
  - Return `(GameBoard, Color, u32)` where Color is the side to move and u32 is the
    fullmove number, so callers can reconstruct `turn_count`.
  - Add `pub mod fen;` to `src/game/mod.rs`.
  - Add a public re-export: `pub use fen::board_from_fen;` in `src/game/mod.rs`.
  - After constructing the `GameBoard`, call `calculate_pins` for both colors and
    compute the initial `zobrist_hash` (depends on subtask 1 being done, or use
    the old `position_hash` temporarily and switch after subtask 1 lands).
  - The standard starting position FEN:
    `"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`
    must round-trip to a board identical to `GameBoard::init_game_board`.
- **Tests**: Add `#[cfg(test)]` block in `src/game/fen.rs`:
  - `test_fen_starting_position` — parse the starting FEN; verify all bitboards
    match `Player::new_white()` / `Player::new_black()`, castling all true,
    EP None, halfmove 0.
  - `test_fen_midgame` — parse
    `"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"`.
    Spot-check: white bishop on c4 (sq 26), knight on f3 (sq 21), black knight
    on c6 (sq 42), halfmove 4.
  - `test_fen_castling_partial` — parse
    `"r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"` and check only
    `white_kingside=true`, `white_queenside=false`, `black_kingside=false`,
    `black_queenside=true`.
  - `test_fen_en_passant` — parse
    `"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"`;
    check `en_passant_target == Some(43)` (d6 = rank5*8+file3 = 43).
  - `test_fen_roundtrip` — starting position FEN produces same bitboards as
    `GameBoard::init_game_board`. (Depends on subtask 1 only if the Zobrist
    field is checked; otherwise no dependency.)
- **Dependencies**: Subtask 1 (for `zobrist_hash` initialization in parsed boards);
  can be stubbed with old `position_hash` if needed.

---

### 3. Engine unit tests — move generation and special moves

- **Files**: `src/game/board.rs` (add `#[cfg(test)] mod tests` at bottom),
  `src/game/fen.rs` (additional tests that need FEN)
- **Changes**: Add a `#[cfg(test)] mod tests` block to `src/game/board.rs`
  containing the following tests. All tests construct positions via FEN (depends
  on subtask 2) or via `init_game_board`.

  **Move generation correctness**:
  - `test_initial_white_pawn_moves` — white pawns on rank 2 all have 2 pushes
    each; iterate `get_move_mask` for each pawn square.
  - `test_pawn_blocked_single_push` — place a piece on e3; white pawn on e2
    has 0 moves.
  - `test_pawn_double_push_blocked` — place a piece on e4; white pawn on e2
    has 1 move (e3 only), not 2.
  - `test_pawn_captures_diagonal` — place black piece on d5 and f5; white pawn
    on e4 has captures to d5 and f5 in its mask.
  - `test_knight_moves_center` — knight on e4 (sq 28) has exactly 8 moves.
  - `test_knight_moves_corner` — knight on a1 (sq 0) has exactly 2 moves.
  - `test_bishop_moves_empty_board` — bishop on d4 (sq 27) with empty board
    has 13 squares in move mask.
  - `test_rook_moves_blocked` — rook on e1, friendly piece on e4; rook cannot
    reach e5 or beyond.
  - `test_queen_moves_center` — queen on d1 starting position; verify move
    mask is 0 (blocked by own pieces).

  **Special move validation**:
  - `test_castling_kingside_white` — FEN
    `"r1bqk2r/pppp1ppp/2n2b2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"`;
    construct a castle move (`from=E1, to=G1, castle_option=Some(Kingside)`);
    `validate_move` returns true; after `compute_turn_items`, king is on G1 and
    rook is on F1.
  - `test_castling_blocked_by_piece` — kingside castle impossible if F1 or G1
    occupied.
  - `test_castling_blocked_by_check` — king cannot castle through an attacked
    square.
  - `test_castling_rights_revoked_after_king_move` — after king moves, both
    castling rights are false.
  - `test_castling_rights_revoked_after_rook_move` — after a1 rook moves,
    `white_queenside=false`.
  - `test_en_passant_capture` — FEN after 1.e4 d5 2.e5 f5 (en passant target
    f6): white pawn on e5 can capture to f6; after applying the move, black
    pawn on f5 is gone.
  - `test_en_passant_clears_after_non_ep_move` — EP target is None after any
    non-double-pawn move.
  - `test_promotion_queen` — white pawn on a7 can advance to a8 with promotion;
    after applying, `player1.pieces_bb[Queen]` has bit 56 set.
  - `test_promotion_knight` — same with knight promotion; knight bit set on a8.

  **Check and checkmate detection**:
  - `test_in_check_detection` — FEN `"4k3/8/8/8/8/8/8/4K2R w K - 0 1"` after
    white plays Rh8+; `get_attackers(black_king_sq, White, ...)` returns non-zero.
  - `test_simple_checkmate` — FEN for "fool's mate" result position
    `"rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"`;
    `calculate_checkmate(Color::White)` returns true.
  - `test_not_checkmate_can_block` — position where check can be blocked;
    `calculate_checkmate` returns false.
  - `test_not_checkmate_king_can_move` — position where king has escape square;
    `calculate_checkmate` returns false.
  - `test_stalemate` — FEN for a known stalemate position (e.g.
    `"k7/8/1Q6/8/8/8/8/7K b - - 0 1"` — black to move, stalemate);
    `calculate_stalemate(Black, ...)` returns true.

  **Draw rules**:
  - `test_fifty_move_rule` — call `compute_turn_items` 100 times without a pawn
    move or capture; `game_result == FiftyMoveRule`.
  - `test_fifty_move_rule_reset_on_capture` — reset on capture keeps clock at 0.
  - `test_threefold_repetition` — reach the same position three times;
    `game_result == ThreefoldRepetition`. (Requires subtask 1 so Zobrist hash
    is collision-resistant enough to distinguish positions.)
  - `test_insufficient_material_k_vs_k` — only kings remain; result is
    `InsufficientMaterial`.
  - `test_insufficient_material_kn_vs_k` — king + knight vs king;
    `InsufficientMaterial`.
  - `test_sufficient_material_two_bishops` — two bishops present; NOT
    insufficient material.

- **Dependencies**: Subtask 2 (FEN) for position setup. Most tests can also be
  written with manual `init_game_board` and forced board mutations; FEN
  makes them cleaner. Mark any test requiring FEN with `// needs subtask 2`.

---

### 4. Perft tests

- **Files**: `src/game/perft.rs` (new), `src/game/mod.rs`
- **Changes**:
  - Create `src/game/perft.rs` with:
    ```rust
    /// Count leaf nodes at `depth` from the given position.
    pub fn perft(board: &GameBoard, color: Color, depth: u32, precomputed: &Arc<PrecomputedItems>) -> u64
    ```
    Implementation: enumerate all legal moves for `color`, for each apply on a
    clone, call `perft` recursively with the opposite color and `depth - 1`.
    At `depth == 0` return 1. At `depth == 1` return `legal_move_count`
    (leaf-count optimization).
  - `get_legal_moves_for_perft(board, color) -> Vec<Move>` — same logic as
    `game_task::get_legal_moves` but returns `Vec<Move>` instead of
    `Vec<ActionIndex>`. Castling and EP must be enumerated. Underpromotions
    must also be enumerated (Queen, Rook, Bishop, Knight) for accurate perft
    counts at positions with promotions.
  - Add `pub mod perft;` to `src/game/mod.rs`.
  - Tests (`#[cfg(test)]` block in `perft.rs`):
    - `test_perft_startpos_d1` — starting position, depth 1: 20 nodes.
    - `test_perft_startpos_d2` — depth 2: 400 nodes.
    - `test_perft_startpos_d3` — depth 3: 8902 nodes.
    - `test_perft_startpos_d4` — depth 4: 197281 nodes. (Mark `#[ignore]`
      so it doesn't run in CI; run explicitly with `cargo test -- --ignored`.)
    - `test_perft_kiwipete_d1` — Kiwipete position
      `"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"`,
      depth 1: 48 nodes.
    - `test_perft_kiwipete_d2` — depth 2: 2039 nodes.
    - `test_perft_kiwipete_d3` — depth 3: 97862 nodes. (Mark `#[ignore]`.)
    - `test_perft_pos3_d1` — position 3
      `"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"`, depth 1: 14 nodes.
    - `test_perft_pos3_d2` — depth 2: 191 nodes.
    - `test_perft_pos3_d3` — depth 3: 2812 nodes.
    - `test_perft_pos5_d1` — position 5
      `"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"`,
      depth 1: 44 nodes.
    - `test_perft_pos5_d2` — depth 2: 1486 nodes.
  - **Note**: Perft correctness directly validates the entire move generation
    pipeline including edge cases. Any discrepancy pinpoints the bug (can add
    split-perft to identify which move class is off).
- **Dependencies**: Subtask 2 (FEN), subtask 3 (ensure engine logic is fixed
  before perft validates it).

---

### 5. Fix `action_to_move` for castling and en-passant reconstruction

- **Files**: `src/data/encoding.rs`
- **Changes**:

  **Current bug**: `action_to_move` reconstructs a `Move` from an `ActionIndex`
  (which is just `from_sq * 64 + to_sq`). It always sets `castle_option: None`
  and `en_passant: false`. This function is used only in `src/data/mod.rs` as a
  re-export; it is NOT called in the hot game-play path (which uses
  `action_to_notation` → `parse_move` instead). However it IS in the public API
  and must be correct for external callers and future use.

  **Analysis of what information is recoverable from `ActionIndex` alone**:
  - Castling: The move `e1g1` (sq4 → sq6, action=262) always means white
    kingside castle when made by the king. Without knowing the board state
    (piece type on from_sq), the decoder cannot distinguish king-moving-to-g1
    from a regular move. The action encoding must carry board context.
  - En passant: Similarly cannot be determined without board state.

  **Fix approach — pass board context**:
  Change the signature to:
  ```rust
  pub fn action_to_move(action: ActionIndex, board: &GameBoard, color: Color) -> Move
  ```
  Inside, look up `board.board_arr[from_sq]` to determine piece type, then:
  - If piece is King and `|to_file - from_file| == 2`: set `castle_option`.
  - If piece is Pawn and `Some(to_sq as usize) == board.en_passant_target`
    and files differ: set `en_passant: true`.
  - Promotion detection: as currently (pawn reaching rank 0 or 7).

  **Callers to update**: Search for all calls to `action_to_move` and update
  signatures. Currently only re-exported via `src/data/mod.rs`; confirm no
  callers in Python bridge or MCTS that use the old signature.

  **Alternative if signature change is too disruptive**: Document that
  `action_to_move` is intentionally incomplete and that the game-play pipeline
  (which uses `action_to_notation` → `parse_move`) is the canonical path and
  works correctly. Leave the function with a corrected docstring. This is the
  minimum fix if callers cannot be updated.

  **Preferred fix**: Update the signature. Any direct caller currently passes no
  board context so there are few call sites — check with `grep action_to_move`
  before starting.

- **Tests** (add to `src/data/encoding.rs`):
  - `test_action_to_move_normal` — action 796 (e2e4) with starting board and
    white; returns Move{from=E2, to=E4, castle=None, ep=false}.
  - `test_action_to_move_castling` — construct a board where white king is on
    E1 and action is 4→6 (e1g1); returns `castle_option: Some(Kingside)`.
  - `test_action_to_move_en_passant` — construct board with EP target on f6
    (sq 45), white pawn on e5 (sq 36), action = 36*64+45 = 2349; returns
    `en_passant: true`.
  - `test_action_to_move_promotion` — action with pawn reaching rank 8; returns
    `promotion_piece_type: Some(Queen)`.
- **Dependencies**: Subtask 2 (FEN makes constructing test boards easy).

---

## Testing Strategy

Run tests in this order:

1. `cargo test --lib` — all unit tests (with `#[ignore]` perft tests skipped).
2. `cargo test -- --include-ignored` — full suite including slow perft tests.
3. `cargo test selfplay::game_task::tests` — verify the existing self-play game
   loop tests still pass after any changes to `action_to_move`.
4. `bash scripts/e2e_test.sh` — full end-to-end validation.

**Perft acceptance criteria**: depths 1-3 of startpos and Kiwipete must match
reference counts exactly. A discrepancy of even 1 node at depth 3 indicates a
move generation or rule bug.

**Regression check**: `cargo test` must show all 24 existing Rust tests still
passing after each subtask.

---

## Implementation Order

```
Subtask 1 (Zobrist)   ──────────────────────────────► Subtask 3 (draw rule tests)
                      \
                       ──► Subtask 2 (FEN) ──────────► Subtask 3 (special move tests)
                                           \──────────► Subtask 4 (perft)
                                           \──────────► Subtask 5 (action_to_move)
```

Subtasks 1 and 5 have no dependencies between them and can start simultaneously.
Subtask 3 can begin with manual board setup for non-FEN tests before subtask 2
is done.

---

## Key Implementation Notes

### Zobrist table seeding

Use splitmix64 with seed `0x517cc1b727220a95` (already used in the old hash).
Generate 2 * 6 * 64 + 1 + 4 + 8 = 781 values.

### FEN square mapping

FEN rank 8 = board rank index 7, rank 1 = board rank index 0. File a = file
index 0. Square index: `rank * 8 + file`. FEN iterates from rank 8 down to
rank 1, left to right within each rank.

### Perft underpromotion

The current `get_legal_moves` only generates queen promotions (one action per
promotion square). Perft requires all four promotion types. The perft-specific
`get_legal_moves_for_perft` must generate four `Move` variants per promotion
square. The existing MCTS pipeline can continue using queen-only (this is a
separate concern).

### `action_to_move` — grep result summary

Before starting subtask 5, verify the exact callers:
```
grep -rn "action_to_move" src/
```
Expected: only `src/data/mod.rs` (re-export) and `src/data/encoding.rs` (definition).
The game-play pipeline does NOT call `action_to_move`; it uses the
`action_to_notation` → `parse_move` path (which correctly detects castling via
king file-diff and EP via board state lookup).

### Move.en_passant field status

`Move.en_passant` is set in `get_legal_moves` (game_task.rs line 193-195) but
`update_board` does NOT use it — it checks `Some(to_idx) == self.en_passant_target`
directly (board.rs line 474). The field is therefore cosmetically set but not
load-bearing in the current pipeline. Keep it for completeness but do not rely
on it being the trigger.

### position_hash collision risk

The existing hash uses `wrapping_mul` with a constant multiplier applied to the
bitboard value. Identical piece configurations at different positions could
collide if two bitboards happen to map to the same hash after multiplication.
Zobrist with independent random values per (piece, square) pair eliminates this
class of collision.
