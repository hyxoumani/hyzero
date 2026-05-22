# Review Log

Tracking which commits/branches have been reviewed by Claude. New review entries
are appended at the bottom.

## Schema

- **commit**: Reviewed commit SHA (HEAD at time of review)
- **branch**: Branch name
- **reviewer**: Agent / model
- **date**: ISO date
- **scope**: What was reviewed
- **findings**: Path to detailed findings file (relative to repo root)

## Entries

| commit  | branch                    | reviewer            | date       | scope                                                                                    | findings                           |
| ------- | ------------------------- | ------------------- | ---------- | ---------------------------------------------------------------------------------------- | ---------------------------------- |
| ee132c4 | claude/modest-rubin-KgdzN | claude-opus-4-7[1m] | 2026-05-22 | TB supervision + canonical MuZero backup + diverse starts (squash of autoresearch/apr13) | docs/reviews/2026-05-22-ee132c4.md |
