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

## Escalation Reference

### Error Types
- **context** — acted on wrong, missing, or stale information
- **breakage** — broke existing behavior or reintroduced a known bug
- **security** — hardcoded secrets, injection, unsafe patterns
- **quality** — missing validation, convention violations

### Tiers
1. **Gotcha** (wiki `## Gotchas`) → agent reads and avoids manually
2. **Rule** (CLAUDE.md / `.claude/rules/`) → loaded into context automatically
3. **Hook** (pre-commit/pre-edit) → blocked programmatically
