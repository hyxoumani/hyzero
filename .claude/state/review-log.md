# Review Log

Tracks the scheduled review-changes routine. The first line under "Cursor" is the SHA of the most recently reviewed commit on `claude/modest-rubin-dd1h78`. Subsequent runs should `git log <cursor>..HEAD` to find new work.

## Cursor

bde4f9be00d1c59b648a4f3c8e59d63c9121d99c

## Entries

### 2026-06-26 — bootstrap (range: initial..bde4f9b)

Reviewed the elo-promotion feature added on this branch (commits `7b53e5d` through `bde4f9b`). Focus: bugs. Scope: `src/selfplay/elo.rs`, `src/selfplay/evaluation.rs`, `src/selfplay/pool.rs`, `src/bin/selfplay.rs`, `scripts/run_baseline.sh`.

**Verdict:** No bugs found.

Notes (not bugs, but worth knowing on a future read):

- Bootstrap branch always reports `candidate_elo=1500.0` in the log line — by design; pre-promotion cycles contribute 0 to the ELO score component in `scripts/run_baseline.sh`.
- `selfplay.rs:21` hardcodes `"checkpoints"` while `evaluation.rs:247` uses `config.checkpoints_dir`. They cannot diverge today (no env override exists), but a future `HYZERO_CHECKPOINTS_DIR` would need both updated.
- Skipped: pure docs/log commits (`06e6129`, `bde4f9b`) and formatting/doc-comment commits (`924f6be`, `df794b3`).
