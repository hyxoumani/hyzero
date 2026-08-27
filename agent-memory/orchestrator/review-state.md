# Review State

Tracker for the recurring bug-review scheduled task. Each run reviews commits
in `<last_reviewed>..HEAD` on main, focuses on bugs, and updates the SHA below.

- last_reviewed_sha: bde4f9be00d1c59b648a4f3c8e59d63c9121d99c
- last_reviewed_short: bde4f9b
- last_reviewed_at: 2026-08-27
- notes: |
  First run — established baseline at the tip of main after the elo-promotion
  feature landed. Future runs will review commits after this SHA.
