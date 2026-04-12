# Development Workflow & Framework

Guide to orchestration, agents, and persistent memory system.

## Orchestrator-Coordinator Pattern

**Entry point**: `claude --agent orchestrator` coordinates all development work.

**Task assessment**: Simple (1-2 files) → direct advice. Medium (3-5 files) → researcher → planner → implementer → tester → reviewer. Complex (6+ files) → parallel worktrees.

**Pipeline**: Research → Plan → Decompose → Implement → Test → Review → Merge → Document

**Agent isolation**: Each agent receives focused brief, relevant code only. Outputs disposable; plans compound across retries. Subagents never see full conversation.

**Autoresearch**: User says "go autonomous" → confirm metric in CLAUDE.md → spawn fresh agents in loop (researcher proposes → implementer in worktree → tester measures → context-keeper logs).

## Worktree Workflow Gotchas

1. **Branches not fully merged** — use `git branch -D` not `-d` after merge
2. **Uncommitted worktree = submodule** — always `git worktree remove` before staging
3. **CLAUDE.md conflicts** — only one agent touches per cycle
4. **Squashing** — use `git reset --soft {base}` then re-commit by logical group

## Subagent Permissions Issue

**Problem**: Agents use absolute paths but permission patterns use relative. Result: silent write denials.

**Fix**: Add BOTH patterns to `settings.json`:
```json
"allow": [
  "Write(file_path:docs/wiki/*.md)",
  "Write(file_path:**/docs/wiki/*.md)",
  "Edit(file_path:docs/wiki/*.md)",
  "Edit(file_path:**/docs/wiki/*.md)"
]
```

Applies to wiki, rules, CLAUDE.md, agent-memory.

## Context-Keeper Best Practices

- Works for small, focused writes (one wiki page, one rule)
- Use implementer for multi-file edits or conflict resolution
- Scope to ≤3 source files (avoid timeout)
- If it fails, fall back to implementer; don't retry same approach

## Agent Memory System

Persistent memory in `.claude/agent-memory/{role}/{topic}.md`:
- **user** — project lead role, preferences
- **feedback** — how to approach work (do/avoid)
- **project** — ongoing work, goals, deadlines
- **reference** — external systems (Linear, Grafana)

Memory discipline: researcher writes before returning; context-keeper encodes findings into CLAUDE.md; verify freshness before acting; do NOT save code patterns, git history, or debugging recipes.

## Related

- [CLAUDE.md](../../CLAUDE.md) — project config
- [.claude/PRINCIPLES.md](../../.claude/PRINCIPLES.md) — engineering rules
- [Project Roadmap](project-roadmap.md) — current state
