# Path-Scoped Rules

Files in this directory encode project conventions. Each rule loads automatically when matching files are touched, so cost stays low when irrelevant.

## Frontmatter

Add `paths:` to scope a rule. Without it, the rule loads every session.

```yaml
---
paths:
  - "src/**/*.rs"
  - "crates/*/src/**"
---
```

## Conventions

- Keep each rule under 30 lines. Long rules get ignored.
- Write rules as constraints ("don't do X", "always do Y"), not narratives.
- Update rules when patterns change. Stale rules are worse than none.
- Cross-reference `.claude/PRINCIPLES.md` for global behavior — rules are project-specific.

## Files in this directory

- `git.md` — commit format, branching, force-push policy
- `testing.md` — test naming, flake handling, regression policy
- Add language/domain rules per project as needed (`rust.md`, `python.md`, etc.)
