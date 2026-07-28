# scheduled-review-cursor.json — Scheduled Bug-Review Cursor

This state file tracks the cursor for the scheduled bug-review routine so
subsequent runs can pick up where the previous one left off. It lives at
`agent-memory/orchestrator/scheduled-review-cursor.json` (tracked in git so
a fresh clone can pick it up).

## Update protocol

- On next run, review only the range `<last_reviewed_head>..HEAD`.
- If that range is empty, notify "nothing new to review" and skip.
- After each review, append a new entry to `history` and update
  `last_reviewed_head` and `last_reviewed_at` to the new tip.
- Preserve `schema_version`; bump it only on breaking format changes.
