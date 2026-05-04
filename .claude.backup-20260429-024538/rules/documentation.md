---
paths:
  - "**/*.md"
  - "**/CLAUDE.md"
  - "**/docs/**"
---

# Documentation Conventions

- Update docs in the same commit as the code change
- ASCII diagrams for data flow, state machines, threading models
- Stale docs are worse than no docs. If you touch code near a diagram, verify it.
- CLAUDE.md is the source of truth for project config (commands, architecture, read-only files)
- Session summaries go in `docs/sessions/YYYY-MM-DD-HH.md`
- Plans go in `docs/plans/{feature}/`
