# Path-Scoped Rules

Rules in this directory load ONLY when Claude touches matching files. This keeps
CLAUDE.md lean while still enforcing domain-specific conventions.

## Structure

Each rule file has YAML frontmatter with a `paths` glob:

```yaml
---
paths:
  - "**/*.rs"
---
# Rust conventions for this project

- Use `thiserror` for error types, not manual `impl Display`.
- All public functions must have doc comments.
- Prefer `&str` over `String` in function parameters.
```

## How it works

Claude Code loads rules from `.claude/rules/` automatically. Rules with `paths`
frontmatter only activate when Claude reads or edits files matching those globs.
Rules without `paths` load every session (same as CLAUDE.md).

## When to create a rule

- The context-keeper discovers a language-specific convention during a task.
- A reviewer flags a pattern violation that should be permanent.
- A convention is too specific for CLAUDE.md but important for consistency.

## File naming

Use the domain as the filename:
- `rust.md`, `python.md`, `typescript.md` — language conventions
- `testing.md` — test patterns
- `api.md` — API design conventions
- `database.md` — schema and query conventions

## Maintenance

These files are maintained by the context-keeper agent. They should be:
- Concise (under 50 lines each)
- Actionable (specific rules, not general advice)
- Verified (each rule should address a real mistake that occurred)
