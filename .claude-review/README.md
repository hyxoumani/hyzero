# .claude-review

State for the scheduled bug-review routine.

Each scheduled run should:

1. Read `state.json` and take `last_reviewed_sha`.
2. Diff from `last_reviewed_sha` to current `HEAD` on the recorded branch, and
   review every new commit for correctness bugs.
3. Skip commits whose only changes are under `.claude-review/` — those are
   routine self-references, not review scope.
4. Update `state.json` with the new `last_reviewed_sha`, `last_reviewed_at`,
   `scope`, `status`, and `findings_count`.
5. On a non-clean run (bugs found), append (do not overwrite) a dated entry to
   `log.md` next to this file so the review history is preserved across runs.
