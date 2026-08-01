# Scheduled Review Tracker

Tracks the SHA of the most recent commit that the scheduled bug-review job has
covered on `origin/main`. Each run of the schedule reviews `LAST_REVIEWED..origin/main`
and updates the SHA below on success.

Do not edit by hand — the scheduled job maintains this file.

## Last reviewed

- **sha**: `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`
- **subject**: `logs: refresh baseline, eval, and self-play artifacts`
- **date**: 2026-08-01
- **scope**: initial run — covered last 10 commits on `origin/main` (elo-promotion feature and adjacent).
- **outcome**: clean (no bugs found)

## Review log

| Date       | Reviewed range              | Files                                                                                                               | Outcome |
| ---------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------- |
| 2026-08-01 | `<init>..bde4f9b` (last 10) | src/selfplay/elo.rs, src/selfplay/pool.rs, src/selfplay/evaluation.rs, src/bin/selfplay.rs, scripts/run_baseline.sh | clean   |
