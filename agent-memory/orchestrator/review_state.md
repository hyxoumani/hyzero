# Review State

Tracks the most recent commit reviewed by an interactive "Review changes & give
feedback" pass. Append a new entry on each review; keep history so we can see
what was covered when.

## Current head reviewed

- **Branch**: `claude/modest-rubin-v4101`
- **HEAD reviewed**: `ee132c4` — `train: TB supervision infrastructure + canonical MuZero backup + diverse starts`
- **Reviewed at**: 2026-05-01
- **Scope**: full squash (23 commits), files: src/mcts/tree.rs, src/data/encoding.rs,
  src/py/training.rs, src/py/inference_backend.rs, src/selfplay/game_task.rs,
  python/hyzero/data/tablebase.py, python/hyzero/data/board_encoder.py,
  python/hyzero/training/trainer.py, scripts/build_tablebase_trajectory_cache.py,
  scripts/rebalance_tb_cache.py.

Findings written to: `agent-memory/orchestrator/review_findings_ee132c4.md`.

## History

| Date       | SHA     | Reviewer | Notes                   |
| ---------- | ------- | -------- | ----------------------- |
| 2026-05-01 | ee132c4 | Claude   | initial review baseline |

## How to use this file

On each new review pass:

1. Read `Current head reviewed` → the SHA below is the last point covered.
2. `git log <SHA>..HEAD --oneline` to enumerate new commits.
3. Review only the diff `git diff <SHA>..HEAD` plus any commits in that range.
4. Append a row to `History`, update `Current head reviewed`, write findings to
   `agent-memory/orchestrator/review_findings_<new_SHA>.md`.
