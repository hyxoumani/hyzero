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
  Content: integration decisions, debugging traces, cross-cutting concerns.

- **Context-keeper** writes CLAUDE.md, `.claude/rules/`, and `docs/wiki/`.
  It does NOT write agent-memory files. It synthesizes all agent memory into the wiki.

### Trigger

After EVERY researcher agent completes, it MUST update its own memory before returning
findings to the orchestrator. Skipping means future sessions re-discover from scratch.
