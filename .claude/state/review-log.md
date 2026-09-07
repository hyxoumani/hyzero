# Scheduled review log

Bug-focused reviews performed by the scheduled routine. Newest first.
Each entry records the range reviewed and any bug findings.

## 2026-09-07 — first run

**Range reviewed:** `7b53e5d~1..bde4f9b` (10 commits, 29 files, elo-promotion series)
**HEAD at review:** `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`
**Previous marker:** none (first run)

### Findings (bug-focused)

1. **[MEDIUM] `src/selfplay/evaluation.rs:836-886`** — `test_evaluation_task_promotes_when_threshold_zero` is cwd-dependent. Uses default `checkpoints_dir = PathBuf::from("checkpoints")` (relative) and does not call `.with_opponent(...)`. If cwd contains stale `checkpoints/best_v*.pt` from a prior `cargo run --bin selfplay`, `pool` is non-empty, `run()` enters the Elo branch, hits the "opponent unset" WARN at lines 341-377, `continue`s, no promotion fires, and `assert_eq!(store_ref.version(), 5)` fails. `bootstrap_path_*` tests already point `checkpoints_dir` at `/nonexistent/...`; this one and `test_evaluation_task_completes_one_cycle` (line 793) were not updated during the refactor.
2. **[LOW] `src/selfplay/evaluation.rs:277-281`** — misleading WARN fires on every normal cycle after the first bootstrap promotion. After promotion #1, champion_version==1 and only `best_v001.pt` exists; `latest_archive_versions` excludes v==champion_version (`pool.rs:33`), so pool is empty and the code prints `"WARN: pool empty despite champion_version=1 > 0; using win-rate fallback"` each cycle until a second promotion lands. Comment claims the state is unexpected ("archives were deleted") but it's the normal steady state between promotions.
3. **[LOW] `src/selfplay/evaluation.rs:222-224` doc vs `src/selfplay/pool.rs:33` behavior** — Elo gate does NOT activate on the first archive. Doc says "transitions to the Elo gate once best_v001.pt lands" and the comment at line 525 says "once any archive lands, all subsequent cycles route through the Elo gate". Because `exclude_version` skips the current champion's own version, at least TWO archived versions are required before the Elo branch ever runs. Not a crash; contradicts the plan/docs.
4. **[LOW] `src/selfplay/evaluation.rs:341-377`** — Elo branch is an infinite no-op loop when `with_opponent()` was omitted. If pool is non-empty but `opponent_evaluator/opponent_server_handle` are `None`, the WARN branch `continue`s without playing games, without incrementing `total_games_since_last_promotion`, without hitting the promotion gate — task spins forever. Production wiring at `bin/selfplay.rs:530` always sets the opponent, so latent; but any future construction site (or the test in finding 1) hits it.

Verified NOT bugs in this range: `opp_handle` mutex + `Python::attach` GIL ordering (evaluation.rs:396-404), `champion_backend` dead-field (predates range), `run_baseline.sh:121` archive wipe (predates range).
