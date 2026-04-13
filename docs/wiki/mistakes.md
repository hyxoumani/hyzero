# Mistakes Log

Record of agent failures with root cause analysis and error classification.

## 2026-04-13: Zobrist Implementation — Incomplete Cleanup

**Date**: 2026-04-13
**Agent**: Implementation agent (Task 30.1)
**Domain**: Chess engine (zobrist hashing)
**Error Type**: Quality — incomplete cleanup of old code

**What happened**: Zobrist table implemented correctly with incrementally-maintained hash, but old `position_hash()` function left in codebase and no tests added initially.

**Root Cause**: First implementation didn't follow the "test-driven" pattern. Function was added but old code wasn't removed, causing confusion about which hash to use. Lack of explicit test coverage for zobrist incremental updates meant the bug wasn't caught at commit.

**Fix**: Follow-up agent removed `position_hash()`, added 4 tests for zobrist hash consistency (initial, after move, after capture, after castling/EP). Tests now verify that `zobrist_hash` field matches recalculated hash.

**Escalation Tier**: Gotcha → added to wiki (chess-engine.md, item 1: Zobrist maintains incrementally, replaces old position_hash)

---

## 2026-04-13: Auto-format Hook — CWD-Relative Paths

**Date**: 2026-04-13
**Agent**: Implementation agent (Task 30.3)
**Domain**: Development infrastructure (formatting hooks)
**Error Type**: Context — acted on incomplete environment state

**What happened**: `auto-format.sh` hook ran after edit, but failed silently when cwd was `/home/devs/workspace/hyzero/python/` (agent had `cd python` to run pytest). Hook script used relative paths (e.g., `./target/release/rustfmt`) that don't exist from `python/` directory.

**Root Cause**: Hook script didn't check its own directory context or use absolute paths. When an agent `cd`'s into a subdirectory (reasonable for running tests), the hook becomes misconfigured.

**Fix**: Update `auto-format.sh` to use absolute paths or explicitly cd to repo root before running formatters. Ensure hooks always work from any cwd.

**Escalation Tier**: Rule — add to `.claude/rules/hooks.md` or CLAUDE.md: "Hooks must use absolute paths or explicitly cd to repo root before executing."

---

## 2026-04-13: Perft Agent — Missing Castling Constraint

**Date**: 2026-04-13
**Agent**: Validation agent (Task 30.5 perft)
**Domain**: Chess engine (move generation)
**Error Type**: Context — incomplete understanding of move generation API

**What happened**: Perft agent wrote test that assumed all legal king moves (including castling) would be in `get_move_mask(king_sq, color)`. Test failed because castling moves are generated separately via `get_castling_moves()`, not included in the basic move mask. This caused discrepancy in node counts.

**Root Cause**: Move generation API asymmetry: king's move mask only includes 1-square moves (precomputed table). Castling is special-cased. The interface doesn't make this obvious. Selfplay code (game_task.rs) already handles this correctly by calling both functions, but the constraint wasn't documented.

**Fix**: (1) Perft test updated to call both `get_move_mask()` and `get_castling_moves()` for king. (2) Wiki updated with gotcha #6: "Castling not in king move mask" and explicit note that move generation code must call both functions.

**Escalation Tier**: Gotcha → encoded in chess-engine.md and code comments. Added to perft.rs as inline doc comment on how to correctly enumerate king moves.

---

## 2026-04-13: Perft Terminal Counting — Non-Standard Convention

**Date**: 2026-04-13
**Agent**: Validation agent (Task 30.5 perft)
**Domain**: Chess engine (perft counting)
**Error Type**: Quality — convention violation

**What happened**: `perft(startpos, d=5)` returned 4,865,617 instead of 4,865,609 — exactly 8 extra nodes. Test comparison against python-chess reference showed mismatch. Root cause found: perft implementation was counting checkmate positions as +1 leaf instead of recursing to depth 0.

**Root Cause**: In `src/game/perft.rs`, the perft function had a terminal-position check: `if new_board.result() == GameResult::Ongoing { recurse } else { nodes += 1 }`. This implements a non-standard perft convention. Standard perft counts reachable positions at depth D, not terminal nodes. The +1 was counting Fool's Mate checkmates (8 of them reachable at ply 4 with depth=5 remaining).

**Fix**: Removed the terminal check and always recurse. The `depth == 1` shortcut already returns `legal_moves.len()`, which is 0 for checkmate. Verified with diagnostic `slow_perft_no_terminal` variant.

**Escalation Tier**: Gotcha → documented in chess-engine.md test coverage section ("terminal counting fixed"). Code comment added to perft.rs explaining standard convention.

---

## 2026-04-13: Researcher Session Timeout — Missing Memory Persist

**Date**: 2026-04-13
**Agent**: Researcher agent (perft d5 bug investigation)
**Domain**: Development infrastructure (agent workflow)
**Error Type**: Context — incomplete session close procedure

**What happened**: Researcher agent investigating perft d=5 overcount timed out at 2.7M ms (45 min) without writing findings to `/home/devs/workspace/hyzero/.claude/agent-memory/researcher/perft_d5_bug.md`. The analysis was correct but lost because the file wasn't written before timeout.

**Root Cause**: Agent got caught in a deep investigation loop (hunt_overcount, hunt_wrong_terminal, hunt_duplicates, hunt_missed_terminal) without periodically flushing findings. Timeout occurred before the summary could be written.

**Fix**: (1) Researcher completed the perft_d5_bug.md file in a follow-up session. (2) Added note to agent-memory/README.md: "Persist findings every 10 min of investigation to avoid timeout loss."

**Escalation Tier**: Rule (candidate) — could add to CLAUDE.md under agent workflow: "Write findings to agent-memory every 10-15 minutes during long investigations to avoid timeout loss."

---

## 2026-04-13: calculate_pins() Queen Omission

**Date**: 2026-04-13
**Agent**: Researcher agent (perft position 6)
**Domain**: Chess engine (pin detection)
**Error Type**: Quality — incomplete logic

**What happened**: `calculate_pins()` in `src/game/board.rs` was missing Queen from the `enemy_sliders` bitboard. This caused false negatives in `calculate_checkmate()` and `calculate_stalemate()` when the pinning piece was a queen. Regular move validation was unaffected.

**Root Cause**: Pin calculation built `enemy_sliders` from Rook and Bishop only:
```rust
player2.pieces_bb[Rook] | player2.pieces_bb[Bishop]  // Missing Queen
```
Copy-paste error from earlier code that may have predated Queen sliding piece support.

**Fix**: Added Queen to `enemy_sliders` in both Color branches. Tests already pass because the pin-detection bug only affects the internal bitmask; `validate_move()` uses clone+apply+check instead.

**Escalation Tier**: Gotcha → added to chess-engine.md gotcha #8 as a reminder to include all sliders in pin detection.

---

## 2026-04-13: Stalemate Castling Escape — Missing Check

**Date**: 2026-04-13
**Agent**: Validation agent (Task 31 engine validation)
**Domain**: Chess engine (game termination — stalemate)
**Error Type**: Quality — incomplete logic

**What happened**: `calculate_stalemate()` checked if the king had legal moves by iterating 1-square king moves and checking pins. Castling was never checked as an escape option. This is extremely rare but legal: in some positions, castling is the only legal move and prevents stalemate.

**Root Cause**: Stalemate logic assumed castling was not a valid escape. The function checked `get_move_mask(king_sq, color)` which returns only 1-square moves; castling generated separately via `get_castling_moves()`. Code path didn't call the second function.

**Fix**: After the king 1-square loop, added explicit calls to `validate_move()` for both kingside and queenside castling options. If either is legal, return `false` (not stalemate). Added 2 tests covering this edge case.

**Escalation Tier**: Gotcha → added to chess-engine.md gotcha #10 (stalemate must check castling escape).

---

## 2026-04-13: Stalemate Parameter Ordering — Bits Swapped for Black

**Date**: 2026-04-13
**Agent**: Validation agent (Task 31 engine validation)
**Domain**: Chess engine (game termination — stalemate)
**Error Type**: Quality — parameter swapping

**What happened**: `calculate_stalemate()` called `get_move_mask(sq, color)` by passing `(friendly_bits, opponent_bits)` as the occupancy parameter. However, `get_move_mask()` expects `(white_pieces, black_pieces)` — canonical color order, not relative to the moving player. For Black-to-move, the bits were swapped, causing incorrect move masks and missed legal escapes.

**Root Cause**: Parameter naming confusion. The function tried to optimize by passing relative (friendly/opponent) bits, but the magic bitboard lookup tables precompute moves for absolute color positions (white/black). The mismatch meant Black-to-move got swapped occupancy and returned wrong moves.

**Fix**: Derive canonical `white_bits` and `black_bits` from `color` at the function entry, then pass these to `get_move_mask()` instead of friendly/opponent. Added 7 tests covering Black stalemate scenarios.

**Escalation Tier**: Gotcha → added to chess-engine.md gotcha #9 (stalemate must pass canonical white/black bits to get_move_mask).

---

## 2026-04-13: Threefold Repetition Off-by-One — Initial Position Not Counted

**Date**: 2026-04-13
**Agent**: Validation agent (Task 31 engine validation)
**Domain**: Chess engine (draw rules — threefold repetition)
**Error Type**: Quality — initialization bug

**What happened**: `threefold_repetition()` always returned `false` in positions where the same position appeared 3 times total, requiring 4 repetitions to trigger the draw. Root cause: `position_history` map started empty. The initial board position was never inserted, so the first occurrence wasn't counted. A position seen after move 5 would count as occurrence #1 (not #2), requiring move 15 and move 25 to reach 3 total.

**Root Cause**: Both `init_game_board()` and `board_from_fen()` created an empty `position_history`, then called `update_board()` for the first time. The update increments the hash count, but never initializes it as occurrence #1. Standard threefold repetition counts the initial position as the first occurrence.

**Fix**: After board construction, explicitly insert `board.position_history.insert(board.zobrist_hash, 1)` to register the initial position as occurrence #1. Now the 2nd and 3rd occurrences properly trigger the draw rule at counts 2 and 3. Added 2 tests.

**Escalation Tier**: Gotcha → documented in wiki but considered a critical draw-rule fix. Not escalated to Rule because it only affects initialization (single fix point).

---

## 2026-04-13: PyO3 Test Mock Data — Production Code Mismatch

**Date**: 2026-04-13
**Agent**: Implementation agent (Task 32.3 — checkpoint resume)
**Domain**: Rust-Python Integration (PyO3)
**Error Type**: Quality — test setup masking production bug

**What happened**: `test_resume_checkpoint_restores_model_version` passed, but production `load_checkpoint()` was reading `model_version` from the wrong source. Test constructed a mock Python return dict and manually inserted `"model_version"` into it. Production code attempted to read from this dict, but the Python trainer object never populated it. The bug would have caused a panic at runtime with real checkpoints.

**Root Cause**: Test setup did not match production Python behavior. Production `load_checkpoint()` in Python trainer saves model state but does not return model_version in a dict. Instead, it's available as a trainer object attribute (`trainer.model_version`). The test mocked a return dict, and production code was written to read from it. The mock made the test pass, but the code would fail at runtime.

```rust
// WRONG (what was written):
let version_dict: Py<PyDict> = ...;  // mock dict with "model_version" key
let version: u64 = version_dict.getattr(py, "model_version")?.extract(py)?;

// RIGHT (what was fixed):
let version: u64 = self.trainer.getattr(py, "model_version")?.extract(py)?;
```

**Fix**: Changed `load_checkpoint()` to read `model_version` directly from the trainer object attribute via `trainer.getattr("model_version")` instead of expecting it in a return dict. Updated test to match.

**Escalation Tier**: Rule — add to `.claude/rules/testing.md` or create new `.claude/rules/pyo3.md`: "When testing PyO3 return values, verify the test setup matches what production Python code actually returns. Don't manually insert values that production code never provides. Mock data should mirror real Python behavior or use a fixture that does."

---

## Escalation Tiers

Mistakes escalate from manual avoidance to automation:
1. **Gotcha** (wiki page section) — documented, read-once, agent uses judgment
2. **Rule** (CLAUDE.md / `.claude/rules/`) — loaded into every session context
3. **Hook** (pre-commit/pre-edit) — blocked automatically by tooling

Error types: **context** (wrong/stale info), **breakage** (reintroduced bug), **security** (secrets/injection), **quality** (incomplete logic/validation).
