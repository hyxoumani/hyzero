# Review State

Tracks which commits/changes have been reviewed by Claude on this branch.

## Version 1 — 2026-05-25

**Scope**: Squash commit `ee132c4` ("TB supervision infrastructure + canonical
MuZero backup + diverse starts"), HEAD of `claude/modest-rubin-nrOde`. Branch is
even with `origin/main`.

**Files read in full** (focus: bug-hunting, not style):

- `src/mcts/tree.rs` (1087 lines) — backpropagate change, Dirichlet noise, trace
- `src/data/encoding.rs` (1007 lines) — color flip, underpromo, action encoding
- `src/selfplay/game_task.rs` (1227 lines) — POV flip, legal action sort, terminal reward
- `src/py/training.rs` (1693 lines) — batch assembly, color aug POV, β/γ blending
- `python/hyzero/data/tablebase.py` (417 lines) — TB snapshot/trajectory builders
- `python/hyzero/data/board_encoder.py` (250 lines) — Python encoder mirror
- `python/hyzero/training/trainer.py` (1061 lines) — Trainer.train_batch + TB routing
- `scripts/build_tablebase_trajectory_cache.py` (455 lines) — DTZ trajectory builder
- `scripts/build_starting_positions.py` (219 lines) — diverse-start FEN generator
- `scripts/pretrain_dynamics.py` (265 lines) — h+g SimSiam pretrain

**Files NOT reviewed** (deferred): logs, docs/wiki, scripts/{build_tablebase_cache,
gen_pretrain_dynamics,rebalance_tb_cache}.py, python/tests/test_tablebase.py,
test_training.py, agent-memory entries.

**Findings**: see review reply for 2026-05-25.
