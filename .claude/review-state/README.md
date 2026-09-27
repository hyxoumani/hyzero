# Review State

State files for the scheduled bug-focused review routine.

- The scheduled review routine reads `last-reviewed.json` on each run to know
  which commit was last reviewed.
- On each run it reviews commits in the range `<reviewed_head>..HEAD` for bugs,
  notifies the user of findings via push notification, and then rewrites
  `last-reviewed.json` with the new HEAD.
- Do not edit these files by hand unless resetting the baseline; the routine
  manages them.
