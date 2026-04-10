---
paths:
  - "**/*"
---

# Git Conventions

- Commit format: `{scope}: {description}` (lowercase, imperative mood)
- One logical change per commit. Rename separate from behavior change.
- Never commit secrets, credentials, .env files, or large binaries
- Experimental work on branches, not main
- Prefer new commits over amending. Amend only when explicitly asked.
- Never force-push to main
