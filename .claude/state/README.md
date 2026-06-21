# .claude/state

This directory holds state files used by scheduled Claude routines.

- `last-reviewed-sha` — records the last commit SHA reviewed by the recurring
  "review changes" routine. The next run should review the range
  `<sha>..HEAD`.

The marker is updated by the routine itself after each successful review;
do not edit it by hand unless you need to reset the review window.
