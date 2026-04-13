# Development Workflow & Framework

Guide to orchestration, agents, baseline scoring, and persistent memory system.

## Orchestrator-Coordinator Pattern

**Task assessment**: Simple (1-2 files) → direct. Medium (3-5 files) → researcher → implementer → verifier. Complex (6+ files) → parallel worktrees.

**Pipeline**: Research → Plan → Implement → Verify → Decide → Log

## Baseline Scoring

Every architecture change is validated by a 30-minute training run.

**Run**: `bash scripts/run_baseline.sh 1800`
**Score formula**: `(8.55 - final_policy_loss) + (decisive_ratio * 10) - (avg_game_length / 100)`
**Direction**: Higher is better
**Stored at**: `logs/baseline_score.json`
**Current baseline**: 4.78 (commit c1e5cdc)

The script runs selfplay for 30 minutes, extracts metrics (loss, eval win rate, game length), computes the composite score, and compares against the previous baseline.

**Env var overrides**: `HYZERO_EVAL_INTERVAL`, `HYZERO_GAMES`, `HYZERO_SIMS`, etc.

## Autoresearch Loop

1. Read current score + wiki
2. Researcher proposes a single change to improve score
3. Implementer builds it (worktree)
4. Run 30-min baseline
5. Score improved? → keep and update baseline. Regressed? → revert and log why.
6. Loop

## Worktree Workflow Gotchas

1. Branches not fully merged — use `git branch -D` not `-d` after merge
2. Uncommitted worktree = submodule — always `git worktree remove` before staging
3. CLAUDE.md conflicts — only one agent touches per cycle

## Agent Memory System

Persistent memory in `.claude/agent-memory/{role}/{topic}.md`:
- **researcher** — architecture decisions, patterns, constraints
- **orchestrator** — integration decisions, debugging traces
- **context-keeper** — synthesizes into wiki, CLAUDE.md, rules

Memory discipline: researcher writes before returning; context-keeper encodes into wiki; verify freshness before acting.

## Related

- [CLAUDE.md](../../CLAUDE.md) — project config, metric definition
- [Project Roadmap](project-roadmap.md) — current state, baseline
- [Development Roadmap](../plans/next-steps/roadmap.md) — detailed next steps
