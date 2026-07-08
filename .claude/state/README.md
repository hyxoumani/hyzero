# Orchestrator State

This directory holds orchestrator state files used across sessions.

## Files

- `last-reviewed-sha` — SHA of the most recent commit whose diff has been
  reviewed for bugs. Line 1 is the SHA; line 2 is the ISO date of the review.

## Review workflow

Future review passes should:

1. Read the SHA from `last-reviewed-sha`.
2. Diff `{sha}..HEAD` to find unreviewed changes.
3. Update `last-reviewed-sha` after each pass with the new HEAD and date.
