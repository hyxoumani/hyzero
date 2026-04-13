# Plan: Engine Validation — Move Generation Confidence to 100%

## Approach

Extend the perft suite to cover all five standard positions at meaningful depths,
then add targeted unit tests for edge cases not covered by perft alone (stalemate
with castling as only escape, promotion + capture, double check, EP discovered check,
castling right loss on rook capture). Also fix the known stalemate detection bug
(castling not checked in `calculate_stalemate`).

---

## Subtasks

### 1. Unignore and expand existing perft tests (low-hanging fruit)

- **Files**: `src/game/perft.rs`
- **Changes**:
  - Remove `#[ignore]` from `test_perft_startpos_d4` (0.23s release) and `test_perft_startpos_d5` (0.58s release)
  - Remove `#[ignore]` from `test_perft_kiwipete_d3` (0.24s release)
  - Add `test_perft_kiwipete_d4` (expected 4,085,603; ~0.49s release) — add as `#[ignore]`
  - Add `test_perft_pos3_d4` (expected 43,238; ~0.01s) — no `#[ignore]` needed
  - Add `test_perft_pos3_d5` (expected 674,624; ~0.08s) — no `#[ignore]` needed
  - Add `test_perft_pos3_d6` (expected 11,030,083; ~1.31s) — add as `#[ignore]`
  - Add `test_perft_pos5_d3` (expected 62,379; ~0.01s) — no `#[ignore]` needed
  - Add `test_perft_pos5_d4` (expected 2,103,487; ~0.25s) — no `#[ignore]` needed
  - Add `test_perft_pos5_d5` (expected 89,941,194; ~10.72s) — add as `#[ignore]`
- **Tests**: These are themselves the tests
- **Dependencies**: none

### 2. Add Position 4 perft tests

Position 4 FEN: `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1`

This position is the heaviest test for promotions (white has a-pawn on a7 with
promotion captures), black castling (kq rights), and captures with en passant
absent. Reference counts from chessprogramming.org:
- D1: 6 D2: 264 D3: 9,467 D4: 422,333 D5: 15,833,292 D6: 706,045,033

- **Files**: `src/game/perft.rs`
- **Changes**:
  - Add `test_perft_pos4_d1` (6) — no `#[ignore]`
  - Add `test_perft_pos4_d2` (264) — no `#[ignore]`
  - Add `test_perft_pos4_d3` (9,467) — no `#[ignore]`
  - Add `test_perft_pos4_d4` (422,333; ~0.05s release) — no `#[ignore]`
  - Add `test_perft_pos4_d5` (15,833,292; ~1.89s release) — add as `#[ignore]`
  - Add `test_perft_pos4_d6` (706,045,033; ~84s release) — add as `#[ignore]`
- **Tests**: These are themselves the tests
- **Dependencies**: none

### 3. Add Position 6 perft tests

Position 6 FEN: `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 10`

The engine is confirmed correct for D1=42, D2=1892, D3=76031, D4=3288373 (verified
against python-chess). This position has no EP, no castling (both kings have already
moved), heavy piece activity with bishops on long diagonals and knights.
Reference counts from chessprogramming.org:
- D1: 42 D2: 1,892 D3: 76,031 D4: 3,288,373 D5: 133,248,296

- **Files**: `src/game/perft.rs`
- **Changes**:
  - Add `test_perft_pos6_d1` (42) — no `#[ignore]`
  - Add `test_perft_pos6_d2` (1,892) — no `#[ignore]`
  - Add `test_perft_pos6_d3` (76,031) — no `#[ignore]`
  - Add `test_perft_pos6_d4` (3,288,373; ~0.39s release) — add as `#[ignore]`
  - Add `test_perft_pos6_d5` (133,248,296; ~15.88s release) — add as `#[ignore]`
- **Tests**: These are themselves the tests
- **Dependencies**: none

### 4. Fix stalemate detection: castling not checked

`calculate_stalemate` in `src/game/board.rs` (lines 360-415) iterates over all
friendly pieces and checks if any has a legal move. For the king, it only calls
`get_move_mask` (1-square moves). Castling is never attempted. If the only
non-stalemate escape is a castle move, this function returns `true` (stalemate)
incorrectly.

The fix must attempt both castle options for the king in addition to 1-square moves,
using the same `validate_move` castling check that perft and the game loop use.

This bug does NOT affect perft correctness (perft uses `get_legal_moves_for_perft`
which correctly includes castling, and ignores `game_result`). It affects game-play
correctness — a position where the king has no 1-square escapes but a valid castle
would be declared stalemate.

- **Files**: `src/game/board.rs` (function `calculate_stalemate`)
- **Changes**: After the king's 1-square move loop (lines 393-411), add a loop that
  tries `CastleOption::Kingside` and `CastleOption::Queenside` using
  `validate_move(castle_mv, color, combined_bits, friendly_bits, opponent_bits)`.
  If any castle is valid, return `false` (not stalemate).
- **Tests**: Add `test_stalemate_only_escape_is_castling` — construct a position where
  all 1-square king moves are attacked but one castle is valid. Example approach:
  place the king on e1 with rook on h1 and kingside rights; block and attack all
  adjacent squares except via castling to g1. Verify `calculate_stalemate` returns
  `false` (not stalemate) and that a game-play call on this position does NOT end in
  stalemate.
- **Dependencies**: none (perft unaffected, but fix before adding deep perft tests)

### 5. Add targeted edge-case unit tests (no code changes, tests only)

These tests document and verify behavior of specific chess edge cases. Each should
be in `src/game/board.rs` (existing `mod tests` block) or a new `mod edge_cases`
submodule in the same file.

#### 5a. En passant discovered check (EP capture exposes king)

FEN: `8/8/8/k2pP2R/8/8/8/4K3 w - d6 0 1`
- White pawn on e5, black pawn on d5 (just double-pushed), EP target d6
- White rook on h5 on the same rank as both pawns and the black king on a5
- White plays e5xd6 e.p. → removes both the e5 pawn AND the d5 pawn from rank 5
- After the capture, the rook on h5 has a clear line to the black king on a5
- This is en passant discovered check — white exposes a rook check by moving its pawn

Expected: `validate_move(e5xd6ep, White, ...)` returns `true` (the move IS legal —
it's white moving, not black; white gives check to black).

Add the reverse (black captures EP and exposes own king):
FEN: `4k3/8/8/8/3pP3/8/8/r3K3 w - e3 0 1` (adjust so black is to move variant)
Actually construct: White pawn on e4, black pawn on d4, EP target e3, white rook on a4,
black king on h4. FEN (black to move): `4K3/8/8/8/r2pP2k/8/8/4k3 b - e3 0 1` —
needs careful construction.

Simpler canonical position: `8/8/8/8/k2pP2R/8/8/3K4 b - e3 0 1`
- Black pawn on d4, white pawn on e4 just double-pushed (EP target e3)
- White rook on h4, black king on a4
- Black tries d4xe3 e.p. → removes d4 pawn and e4 pawn from rank 4
- Rook on h4 would have clear line to black king on a4
- This is the illegal EN passant discovered check: `validate_move` must return `false`

Verify: `validate_move(d4xe3ep, Black, ...)` returns `false` (illegal — exposes black king).

#### 5b. Pinned piece en passant

FEN: `8/8/8/8/k2pP2r/8/8/3K4 b - e3 0 1` — same structure but with black ROOK on h4
pinning the d4 pawn relative to... wait, the pawn would be pinned to the king. Let me
use: `4k3/8/8/8/rp1PP3/8/8/4K3 b - d3 0 1` — black pawn on b4, white pawn on c4
just double-pushed (EP target c3), black rook on a4, black king on e4. The b4 pawn is
pinned along the 4th rank. Playing b4xc3ep would expose the black king on e4 to the
rook on a4... wait, the rook is black, so it's friendly.

Correct setup: `4k3/8/8/8/rp1PP3/8/8/4K3 b - d3 0 1`
- Hmm — need the pinning piece to be WHITE. White rook on a4, black pawn on b4 (pinned
  to black king somewhere on rank 4 beyond a4). This requires careful arrangement.

Canonical for this test: `8/8/4k3/8/r1pPP3/8/8/4K3 b - d3 0 1`
- Black pawn c4, white pawn d4 just pushed (EP target d3), white pawn e4
- White rook on a4 — but rook is white, not the pinner here.

Actually the canonical position for pinned EP: white rook on a5, black king on e5,
black pawn on b5, white pawn on c5 (just double-pushed, EP target c6).
FEN: `8/8/8/k1pP3R/8/8/8/4K3 b - d6 0 1`
- Black pawn b5, white pawn d5 just double-pushed (target d6), white rook on h5,
  black king on a5. Black tries b5xd6 — wait, b5 can only capture c6, not d6.

Corrected: `8/8/8/kp1P3R/8/8/8/4K3 b - d6 0 1` (but d5 != c5+1)
Use: `8/8/8/k1pP3R/8/8/8/4K3 b - c6 0 1` — wait, white pawn pushed to d5 (EP on d6)?
No: white pawn double-push means it was on d2 → d4 (EP target d3). The position needs
a pawn that JUST pushed.

Simplest canonical: use the FEN from chessprogramming.org pinned EP test:
FEN for pinned EP illegal: `8/8/8/2k5/3Pp3/8/8/3KR3 b - d3 0 1`
- Black king c5, black pawn e4, white pawn d4 (EP target d3), white rook on e1.
- Black tries e4xd3 ep → removes black e4 pawn and white d4 pawn from e/d files
- After capture, white rook on e1 has open e-file to... no, black king is on c5.
Hmm, e-file check on e-file doesn't pin along d-file.

The correct arrangement: white rook on a4 or h4, black king on one end, black pawn
in between, white pawn next to it with EP option. The two pawns are both on rank 4.
After EP capture, both are gone → horizontal pin revealed.

FEN: `8/8/8/8/R2pPk2/8/8/4K3 b - e3 0 1` (corrected attempt)
- White rook a4, black pawn d4, white pawn e4 (EP target e3), black king f4
- Black tries d4xe3 ep → removes d4 and e4 pawns
- Rook a4 has clear line to black king f4 → discovered check/exposed → illegal
- `validate_move(d4xe3ep, Black, ...)` must return `false`

This is the correct pinned-EP-illegal test position.

#### 5c. Castling through check (not out of check, but through an attacked square)

FEN: `r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1` — both sides can castle
White tries queenside castle (e1→c1, passes through d1). If d1 is attacked:
FEN: `r3k3/8/5b2/8/8/8/8/R3K2R w KQ - 0 1`
- White bishop... no, we need an enemy piece attacking d1.
FEN: `r3k2r/3q4/8/8/8/8/8/R3K2R w KQkq - 0 1`
- Black queen on d7 attacks d1 through the open d-file.
- White tries O-O-O (e1→c1), king passes through d1 which is attacked.
- `validate_move(queenside_castle, White, ...)` must return `false`.
- White can still castle kingside (e1→g1, passing through f1 which is safe).

Also test: castling while king is in check (king attacked on e1):
FEN: `r3k2r/8/8/8/8/8/8/R1q1K2R w KQkq - 0 1`
- Black queen on c1 gives check to white king on e1.
- White cannot castle either side (king passes through/starts on attacked square).

#### 5d. Castling rights lost after opponent captures rook (not just rook move)

FEN: `r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1`
- White takes black's a8 rook with Ra1xa8.
- After the move, black no longer has queenside castling rights.
- Verify: after `compute_turn_items(0, Ra1xa8)`, `board.black_queenside == false`

Check `update_castling` in board.rs (lines 241-285):
```rust
if from_idx == 56 || to_idx == 56 { self.black_queenside = false; }
if from_idx == 63 || to_idx == 63 { self.black_kingside = false; }
```
The `to_idx` check correctly revokes castling rights when the rook's square is
captured by the opponent. So this IS already correctly handled.

Test: confirm the Zobrist hash also differs after losing castling rights this way
(the `update_castling` function XORs the appropriate castling key).

#### 5e. Promotion with capture (not just straight push)

FEN: `r7/P7/8/8/8/8/8/4K1k1 w - - 0 1`
- White pawn on a7 can capture to b8 (black rook) and promote to any piece.
- 4 promotion types via capture = 4 legal moves from a7xb8.
- 4 promotion types via push = 4 legal moves from a7xa8.
- Total: 8 legal moves for the pawn, plus any king moves.
- Verify `get_legal_moves_for_perft` returns all 8 pawn moves.

#### 5f. Double check — only king can move

FEN: `r1b1k3/ppp2p1p/2n3pN/3Q4/4P3/2BB4/PPP3PP/R4RK1 b - - 0 1`
This is a composed position. Use a cleaner one:
FEN: `4k3/8/8/8/8/4N3/3Q4/4K3 b - - 0 1`
- White knight on e3, white queen on d2 — arrange so after a discovered check move,
  both pieces attack black king simultaneously.
Simpler: use a known double-check position from a real game.

A reliable test: after the double-check move is made (by white), verify:
- `calculate_checkmate(Black)` returns `false` (king can still move somewhere)
- A black non-king piece that could block a single check cannot block in double check
- The only legal moves for black are king moves

Since this is complex to set up as a pre-move FEN (we need the position AFTER the
double-checking move), use:
FEN: `8/8/8/4k3/8/2N5/1Q6/4K3 b - - 0 1` — knight on c3 and queen on b2 both
attacking e5 king: `c3xe5?` no, knight attacks d1,d5,b1,b5,a2,a4,e4,e2 from c3.
Queen on b2 attacks along diagonal b2-e5 if no blockers? b2=sq10, e5=sq36, rank diff=3,
file diff=3, yes diagonal. Knight from c3 attacks d5 and e4. Not a double check on e5.

Use a verified double-check FEN: the position where both a sliding piece and a
discovered piece attack the king after a single move has already been made:
`4k3/8/8/3pR3/4K3/8/8/8 b - - 0 1` is a position where black has no moves except
king and is in check from the rook. Not double check.

For the double-check test, the simplest reliable approach is to make the move that
causes double check and verify via `get_legal_moves_for_perft` that only king moves
are available afterward. The specific FEN will be determined by the implementer
from a real double-check scenario.

**Key invariant to test**: in any double-check position, `get_legal_moves_for_perft`
returns only king moves (no blocking or interposing moves).

---

## Perft Reference Counts

All expected node counts from chessprogramming.org/Perft_Results:

| Position | D1 | D2 | D3 | D4 | D5 | D6 |
|---|---|---|---|---|---|---|
| Startpos | 20 | 400 | 8,902 | 197,281 | 4,865,609 | 119,060,324 |
| Kiwipete | 48 | 2,039 | 97,862 | 4,085,603 | 193,690,690 | — |
| Pos3 | 14 | 191 | 2,812 | 43,238 | 674,624 | 11,030,083 |
| Pos4 | 6 | 264 | 9,467 | 422,333 | 15,833,292 | 706,045,033 |
| Pos5 | 44 | 1,486 | 62,379 | 2,103,487 | 89,941,194 | — |
| Pos6 | 42 | 1,892 | 76,031 | 3,288,373 | 133,248,296 | — |

FENs:
- Startpos: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
- Kiwipete: `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`
- Pos3: `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1`
- Pos4: `r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1`
- Pos5: `rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8`
- Pos6: `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 10`

---

## Timing Budget

All estimates assume release mode (measured ~8.4M nodes/sec on this machine).

**Fast (no `#[ignore]` needed, < 2s):**
- All D1-D3 for new positions
- Pos4 D4, Pos5 D3-D4, Pos6 D3-D4
- Startpos D4, Kiwipete D3

**Slow (add `#[ignore]`, feasible in CI with explicit `--ignored`):**
- Startpos D5 (~0.6s), D6 (~14s)
- Kiwipete D4 (~0.5s), D5 (~23s)
- Pos3 D6 (~1.3s)
- Pos4 D5 (~1.9s), D6 (~84s — borderline, maybe skip)
- Pos5 D5 (~11s)
- Pos6 D5 (~16s)

**Too slow to include (> 90s):**
- Pos4 D6 (~84s) — include as `#[ignore]` but flag that it's very slow
- Startpos D6 (~14s) — acceptable as `#[ignore]`

---

## Testing Strategy

End-to-end verification:
1. Run `cargo test --release -- game::perft` — all non-ignored tests pass.
2. Run `cargo test --release -- --ignored game::perft` — ignored (slow) perft tests pass.
3. Run `cargo test` (debug) — all unit tests pass including new edge-case tests.
4. For each new perft position, verify the D1 count manually against the position's
   legal move list before trusting deeper counts.

Regression guarantee: the existing 10 passing perft tests must continue to pass after
every change. Any change that breaks an existing perft test is a regression.

---

## Known Issues / Risks

### Bug: `calculate_stalemate` misses castling (Subtask 4)

- **Location**: `src/game/board.rs` lines 360-415
- **Impact on perft**: None (perft uses `get_legal_moves_for_perft`, ignores `game_result`)
- **Impact on gameplay**: A position where the only non-stalemate escape is castling
  would be declared stalemate. This is a correctness bug in game termination, not
  in move generation.
- **Fix risk**: Low — the fix adds a `validate_move` call for each castle option,
  same as the existing perft and game_task logic.

### Suspicious pattern: `calculate_checkmate` does not try castling

- **Location**: `src/game/board.rs` lines 417-500
- **Assessment**: NOT a bug. Chess rules forbid castling when in check. The `validate_move`
  castling check includes the king's starting square in `castle_path_squares`, so
  it correctly rejects castling from check. No fix needed.

### Suspicious pattern: `update_board` ordering — board_arr update after promotion

- **Location**: `src/game/board.rs` line 766 (`self.board_arr[to_idx] = self.board_arr[from_idx].take()`)
  This line runs AFTER the promotion piece is set (line 785). If this line runs after
  the promotion block, it would overwrite the promoted piece with the pawn. Let's verify:
  
  Line 766: `self.board_arr[to_idx] = self.board_arr[from_idx].take();` — copies from→to
  Line 775-788: checks `promotion_piece_type` and if set, overwrites `self.board_arr[to_idx]`
  with the promoted piece.
  
  The order is: (1) move pawn to to_sq in board_arr, (2) if promotion, replace with promoted piece.
  This is correct — the promotion code at line 785 correctly overwrites the pawn.

### Reference counts for Position 4

Position 4 FEN contains a black queen on a3 (`q4N2` = rank 3 from bottom, so
rank 2 in 0-indexed = squares 16-23; a-file = sq 16 = a3; queen on a3). The `q` 
at the start means the black queen is on a3 (sq 16). The position is complex but
well-specified. The D1=6 count is low (white has very limited moves while black queen
threatens many white pieces). This low D1 count makes it a good regression detector —
any off-by-one in promotion or castling at this depth would be immediately visible.

### Stale wiki note

`docs/wiki/chess-engine.md` (Test Coverage section) says "10 pass, 3 ignored". After
subtask 1 the counts will change. Flag for context-keeper to update.
