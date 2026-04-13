# Plan: Fix perft D5 Overcount (+8 nodes)

## Approach

Remove the `nodes += 1` terminal branch from the `perft()` function in `src/game/perft.rs`.
Standard perft counts positions reachable at depth D: a terminal position at depth D-1
contributes 0 (no onward positions exist). The current code adds 1 for every checkmate or
stalemate encountered mid-tree, which overcounts by exactly 8 at depth 5 (the 8 Fool's Mate
positions reachable at ply 4). The `depth == 1` fast path already returns 0 for checkmates
via `get_legal_moves_for_perft`, so removing the terminal branch makes the two code paths
consistent with the standard perft definition.

## Subtasks

### 1. Fix `perft()` in perft.rs

- **Files**: `src/game/perft.rs`
- **Changes**:
  1. Remove the `if board.result() != GameResult::Ongoing { break; }` dead-code guard inside
     the loop (the board passed to perft is always Ongoing; this check never fires and
     creates misleading code).
  2. Replace the terminal-position branch:
     ```rust
     // REMOVE:
     if new_board.result() == GameResult::Ongoing {
         nodes += perft(&new_board, next_color, depth - 1, precomputed);
     } else {
         nodes += 1;  // wrong
     }

     // REPLACE WITH:
     nodes += perft(&new_board, next_color, depth - 1, precomputed);
     ```
  3. Remove the now-unused `use crate::game::board::GameResult;` import if it becomes dead.
- **Tests**:
  - The existing `test_perft_startpos_d3` (8902), `test_perft_startpos_d4` (197281),
    `test_perft_kiwipete_d2` (2039), `test_perft_pos3_d3` (2812), `test_perft_pos5_d2`
    (1486) must all still pass — the fix must not break any depth ≤ 4 tests.
  - Add `test_perft_startpos_d5` asserting 4,865,609 (marked `#[ignore]` for CI speed).
- **Dependencies**: none

### 2. Optionally fix `perft_divide()` convention mismatch

- **Files**: `src/game/perft.rs`
- **Changes**: In `perft_divide`, when `depth == 1`, the count for a move that leads to
  a terminal position should be 1 (not the result of `perft(depth-1)`). Change the count
  assignment to:
  ```rust
  let count = if depth > 1 {
      perft(&new_board, next_color, depth - 1, precomputed)
  } else {
      1
  };
  ```
  This is already the current code for `depth == 1`. The divide mismatch is at `depth > 1`
  where a terminal position gives `perft(depth-1) = 0` instead of 1. To fix: check
  `new_board.result()` and use 1 if non-Ongoing.
  ```rust
  let count = if depth > 1 {
      if new_board.result() == GameResult::Ongoing {
          perft(&new_board, next_color, depth - 1, precomputed)
      } else {
          1
      }
  } else {
      1
  };
  ```
  Note: once perft() is fixed (subtask 1), the divide total will match `perft()`. This
  subtask makes the divide's per-move annotations correct, but is lower priority.
- **Tests**: Manual inspection — after 1.g4 e5 2.f3, `divide(d=2)` should show `d8h4: 1`
  (not 0) and total should match `perft(d=2)`.
- **Dependencies**: subtask 1 must complete first.

## Testing Strategy

After applying subtask 1:
1. `cargo test` — all existing perft tests pass.
2. `cargo test --release test_perft_startpos_d5 -- --include-ignored` — asserts 4,865,609.
3. Verify the kiwipete d3 value (97,862) still holds with `-- --include-ignored`.
4. The `compare_fast_vs_slow_no_terminal_d5` diagnostic test (if re-added temporarily)
   should show `fast == slow_no_terminal == 4,865,609`.
