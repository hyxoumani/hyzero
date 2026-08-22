# Scheduled Bug-Review Log

This file tracks what the scheduled bug-review task has already looked at, so
each run can review only what's new since the last one. Focus is bugs
(correctness), not style.

## Cursor

- Last reviewed HEAD: `bde4f9b` (branch `claude/modest-rubin-4itdpf`, == `main`)
- Last reviewed at: 2026-08-22

## Review 1 — 2026-08-22 — initial baseline (Elo-ladder feature series)

Scope: the recently-landed "Elo ladder eval + champion-pool promotion" feature.
Commits reviewed:

- `7b5dd87` selfplay: add elo math module
- `9ddee3a` selfplay: add archive pool enumeration helper
- `2a38e77` selfplay: plumb opponent inference server for elo ladder
- `a93b077` selfplay: refactor eval task to per-opponent elo ladder
- `0c35f8f` selfplay: wire elo ladder env vars + startup notices
- `9450e38` baseline: extract candidate_elo from ladder_match and add to score
- `924f6be` selfplay: apply cargo fmt to new elo-promotion code
- `df794b3` selfplay: fix doc-list indentation in eval task run() doc

Adjacent tuning commits skimmed (no findings): `aff97fb` cosine LR,
`7b30e29` policy entropy, `5f30ea8` Gumbel-Top-K.

### Confirmed bugs (adversarially verified)

**1. Champion "backend" swap is never called — champion tracks the challenger.**
`src/selfplay/evaluation.rs:143-150` records a `champion_backend` handle via
`with_champion_backend(...)`, and `promote()` at `evaluation.rs:543-546`
stores `self.challenger_evaluator.clone()` as the new champion Arc. That Arc
is a `ChannelEvaluator` routed to `inference_tx` (`src/bin/selfplay.rs:495`),
which is the CHALLENGER inference server — whose weights are overwritten on
every training bump at `src/bin/selfplay.rs:388-403`. Nothing anywhere calls
`.swap()` on the SwappableBackend (grep for `swap` returns only
type/comment mentions at `inference.rs:62-86` and `evaluation.rs:143-150`).
The frozen champion inference server booted at `src/bin/selfplay.rs:313-331`
is orphaned the moment `promote()` first runs.
Impact: from the second promotion onward, "champion vs challenger" eval
games are played by whatever weights the challenger server currently
holds — so both sides use the same weights and the gate degenerates to
~0.5 win-rate noise. Promotion decisions after the first are structurally
meaningless. Severity: HIGH.

**2. Elo gate does not activate after the first promotion, contrary to docs.**
`src/selfplay/pool.rs:33` filters `if v == exclude_version { continue }`,
and `src/selfplay/evaluation.rs:247-251` passes `champion_version` as
`exclude_version`. After `promote(1)` writes `best_v001.pt` (only archive
on disk), the next cycle excludes v=1 and returns an empty pool, so the
code falls through to the bootstrap win-rate branch (which even emits its
own `champion_version > 0` WARN at `evaluation.rs:277-281`). The Elo gate
first has a chance to fire only after the SECOND promotion produces
`best_v002.pt`. `docs/wiki/champion-pool-promotion.md:71-72` and the
docstring at `evaluation.rs:223-224` both claim transition happens once
`best_v001.pt` lands — code and docs disagree. Severity: medium.

**3. PGN "Game N" numbers collide across pool opponents within one cycle.**
`src/selfplay/evaluation.rs:422-428, 460-466` writes PGN entries with
`game_idx + 1` (and `gps + game_idx + 1`) from inside the per-opponent
`pool_loop`. With `pool_size=3, gps=4` you get three PGNs all tagged
"Eval Cycle N Game 1", three "Game 2", etc. The distinguishing
`pool v{opponent_version}` label lives inside the PGN body, not the
event/round tag, so anything grouping by (cycle, round) silently merges
distinct games. Debug/analytics only. Severity: low.

### No-bugs-found notes (so we don't re-check next run)

- `latest_archive_versions` filename parser (`pool.rs`) — `parse::<u64>`
  correctly rejects malformed names; safe.
- Cooldown counter, per-cycle Elo reset, `scored_games` accounting when an
  opponent inference server fails to load — all check out.
- `run_baseline.sh` candidate_elo extraction defaults to 1500.0 on missing
  field and to the last observed value on new logs.
