---
name: Persist Agent Findings
description: Agents must persist findings to agent-memory before returning results
type: workflow
---

## MANDATORY: Persist Agent Findings

### Ownership

- **Researcher** writes to `agent-memory/researcher/{topic}.md` (snake_case).
  Content: architecture decisions, patterns, constraints, gotchas — focus on *why*.
  One file per topic. Update existing files if the topic was previously researched.

- **Orchestrator** writes to `agent-memory/orchestrator/{topic}.md` at session end.
  Content: integration decisions, debugging traces, cross-cutting concerns discovered
  during coordination. The context-keeper compacts these into wiki pages.

- **Context-keeper** writes CLAUDE.md, `.claude/rules/`, `docs/wiki/`, and `results.tsv`.
  It does NOT write agent-memory files. It synthesizes all agent memory into the wiki.

### Trigger

After EVERY researcher agent completes, it MUST update its own memory before returning
findings to the orchestrator. This is NOT optional. Skipping means future sessions
re-discover the same things from scratch.

The orchestrator SHOULD persist key decisions and findings to its own memory before
session end, especially: integration issues, cross-subsystem constraints, and
debugging conclusions that aren't captured in any single agent's memory.
