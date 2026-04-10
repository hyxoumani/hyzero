---
name: Persist Research Findings
description: Orchestrator must spawn context-keeper to persist researcher findings after each phase
type: workflow
---

## MANDATORY: Persist Research Findings

### Ownership

- **Researcher** writes its own memory to `agent-memory/researcher/{topic}.md` (snake_case).
  Content: architecture decisions, patterns, constraints, gotchas — focus on *why*, not implementation details.
  One file per topic. Update existing files if the topic was previously researched.

- **Context-keeper** writes CLAUDE.md, `.claude/rules/`, `results.tsv`, and `docs/sessions/` summaries.
  It does NOT write agent-memory files. It encodes reviewer findings into permanent rules.

### Trigger

After EVERY researcher agent completes, it MUST update its own memory before returning
findings to the orchestrator. This is NOT optional. Skipping means future sessions
re-discover the same things from scratch.
