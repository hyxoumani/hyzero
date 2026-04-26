# Code Review State

Tracks what has been reviewed so future reviews can resume from where the last left off.

## Latest review

- **Reviewer**: Claude Code
- **Date**: 2026-04-26
- **HEAD reviewed**: `ee132c4` (`train: TB supervision infrastructure + canonical MuZero backup + diverse starts`)
- **Branch**: `claude/modest-rubin-tjT3a`
- **Findings**: see `docs/reviews/2026-04-26-ee132c4.md`

## How to resume

```
LAST_REVIEWED=$(grep '^- \*\*HEAD reviewed\*\*' docs/reviews/REVIEW_STATE.md | head -1 | grep -oE '`[^`]+`' | head -1 | tr -d '`')
git log --oneline ${LAST_REVIEWED}..HEAD
```

The diff between `${LAST_REVIEWED}` and `HEAD` is the new surface to review.
