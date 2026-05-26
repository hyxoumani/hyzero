# Review Log

Tracking which commits have been reviewed for bugs. New sessions should pick up
from the latest entry's `head_sha` and review only commits not already covered.

---

## v1 — 2026-05-26 — head_sha ee132c4

**Scope**: HEAD commit `ee132c4` ("train: TB supervision infrastructure + canonical
MuZero backup + diverse starts"). Squash of `autoresearch/apr13` (23 commits).

**Files reviewed** (substantive code only; logs/docs/agent-memory skipped):

- `src/data/encoding.rs` — action/obs encoding, color flip
- `src/mcts/tree.rs` — canonical MuZero backup, Dirichlet noise, tie-break
- `src/py/training.rs` — color augmentation, batch assembly
- `src/selfplay/game_task.rs` — POV flip + sort, diverse starts, terminal reward
- `python/hyzero/training/trainer.py` — K-step unroll, TB masking, diag probes
- `python/hyzero/data/tablebase.py` — TBSample / TBTrajectory builders
- `python/hyzero/data/board_encoder.py` — Python port of Rust encode_board
- `scripts/build_tablebase_trajectory_cache.py`
- `scripts/build_starting_positions.py`
- `scripts/pretrain_dynamics.py` (partial)

**Findings**: see `findings_v1.md` in this directory.
