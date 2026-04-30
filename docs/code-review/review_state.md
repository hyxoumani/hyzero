# Code Review State

Tracks which commits have been reviewed and what was found. Append-only log.

## Reviewed

| Date       | Commit    | Reviewer      | Notes                                                                              |
| ---------- | --------- | ------------- | ---------------------------------------------------------------------------------- |
| 2026-04-30 | `ee132c4` | Claude (Opus) | Squash: TB supervision + canonical backup. See [review-ee132c4](review-ee132c4.md) |

## Conventions

- Each commit's findings live in a sibling file `review-<short_sha>.md`.
- Severity tags: **CRITICAL** / **HIGH** / **MEDIUM** / **LOW** / **NIT**.
- Findings reference `file:line` so they can be jumped to from any editor.
