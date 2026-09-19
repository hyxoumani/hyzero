# Scheduled review cursor

Tracks the latest commit that the scheduled "review changes & give feedback"
routine has already inspected on this branch. Next run should review
`<cursor>..HEAD` and skip anything already covered.

Update this file at the end of every review run.

## Current cursor

- branch: claude/modest-rubin-pipx6m
- reviewed_through: bde4f9be00d1c59b648a4f3c8e59d63c9121d99c
- reviewed_on: 2026-09-19
- range_covered_this_run: 7b5dd87^..bde4f9b (elo-ladder / champion-pool promotion)

## Findings surfaced this run

- LOW scripts/run_baseline.sh:255 — Elo bonus term hard-codes 1500.0; ignores HYZERO_OPPONENT_INITIAL_ELO.
- LOW scripts/run_baseline.sh:216-218 — dead `${LAST_CANDIDATE_ELO:-1500.0}` fallback; awk always prints, so :- never fires.
- LOW src/selfplay/evaluation.rs:526-528 — cooldown counts from 0 at startup; blocks the FIRST promotion when configured non-zero.
- DOC docs/wiki/elo-ladder-eval.md:1-5 — wording implies challenger plays current champion; code excludes it, only past archived champions play.

All findings are latent under default env-vars (cooldown=0, initial_elo=1500). No HIGH/MED bugs found in this range.

## History

- 2026-09-19 first run; established cursor at HEAD after reviewing 7b5dd87^..bde4f9b.
