# Review Log

Scheduled code-review routine. Each entry records what was reviewed, from
what commit range, and what bugs were found. The next scheduled run picks
up from the latest `head_sha` here.

Entries are newest-first.

---

## 2026-10-05 — elo-promotion feature

- **Scope:** commits `7b53e5d..9450e38` (elo math module, archive pool,
  opponent inference plumbing, per-opponent elo ladder eval task, env
  var wiring, baseline score integration)
- **Head reviewed:** `bde4f9bde4f9be00d1c59b648a4f3c8e59d63c9121d99c` (branch `main`)
- **Reviewer:** analyst subagent, orchestrated review

### Findings (bugs only)

1. **MAJOR — Baseline script hardcodes 1500 Elo reference, decoupled from
   Rust config.**
   `scripts/run_baseline.sh:251` computes
   `(last_candidate_elo - 1500.0) * elo_score_weight`, but Rust reads the
   starting rating from `HYZERO_OPPONENT_INITIAL_ELO`. Setting the env var
   above 1500 produces a bogus "progress" bonus every cycle, even with zero
   games played.

2. **MAJOR — Elo-path soft lockout when pool is nonempty but opponent
   handle is unset.**
   `src/selfplay/evaluation.rs:375` warns and `continue`s the outer loop
   with no fallback to the win-rate bootstrap path. Any driver that drops
   `with_opponent(...)` with `best_v*.pt` on disk → promotion never fires
   however strong the challenger.

3. **MAJOR — Pool excludes current champion, so Elo gate never tests
   "beat the actual champion".**
   `src/selfplay/evaluation.rs:247-251` (via
   `latest_archive_versions(..., exclude_version=champion_version, ...)`,
   `src/selfplay/pool.rs:33`). With opponents pinned at flat 1500 and the
   champion excluded, a challenger that dominates older archives but loses
   to the current champion can still cross the promotion gate — the
   champion can drift downward across cycles.

4. **MINOR — PGN game numbering collides across pool opponents within one
   cycle.**
   `src/selfplay/evaluation.rs:423-424, 462-463` — `game_idx + 1` resets
   per opponent in the pool loop. Only the opponent label distinguishes
   PGN entries; parsers keyed on `(cycle, game_num)` collapse records.

5. **MINOR — Unprotected `.unwrap()` on `opp_handle` mutex inside GIL
   section.**
   `src/selfplay/evaluation.rs:398` — a future helper panicking while
   holding this lock would poison it, then the next eval cycle panics
   inside `Python::attach`.

### Not bugs (verified)

- Elo math (`src/selfplay/elo.rs`) — expected-score formula, update sign,
  draw=0.5 handling.
- Archive enumeration — empty pools, single entries, exclusion.
- Sign convention for Black-side games.
- Per-cycle reset of `candidate_elo` to `opponent_initial_elo` is explicit
  design.
- Env var parse-failure fallback matches the rest of the codebase.

### Next starting point

Next scheduled run should review new commits landed on `origin/main` after
`bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`. If none, extend coverage to the
MCTS Gumbel-Top-K / sequential-halving root selection (commit
`5f30ea8`) — not yet reviewed by this routine.
