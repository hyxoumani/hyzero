# Review Log

Running log of bug-focused reviews of the working branch. Each entry pins the HEAD SHA reviewed so the next pass can scope to commits since the marker.

## 2026-06-23 — `bde4f9b` (branch `claude/modest-rubin-hduaj0`, 19 commits 5f30ea8..bde4f9b)

Scope: Elo-promotion / Elo-ladder eval feature + supporting plumbing. Branch is 0 commits ahead of `origin/main`; reviewed the feature window the brief was about.

### Bugs

- **HIGH** `src/selfplay/evaluation.rs:198-209,440-487` — Unit tests cover only the `compute_candidate_elo_from_results` helper; production `run()` re-implements the per-game Elo update inline. A sign/argument-swap regression in the inline path would not be caught by any current test. Fix: either call the helper from `run()` or add a test against the production path.
- **HIGH** `src/selfplay/evaluation.rs:341-377` — "Pool nonempty but opponent handle unset" fallback emits a log and `continue`s after consuming `last_evaluated_version`, so the documented "falls back to single-opponent eval" behavior (see doc lines ~110-112, 222-224) is a lie — that branch actually plays zero games. Today `with_opponent` is wired unconditionally in the binary so the branch is unreachable in prod, but the comment and the branch disagree; one of them should change.
- **MED** `src/bin/selfplay.rs:204` — Initial `watch::channel(1u64)` versus `champion_store_version=0` immediately triggers an eval cycle on startup with the random-init challenger versus the (RandomEvaluator) bootstrap champion. With default `promotion_threshold=0.55` on the bootstrap path this can fire a spurious first promotion before any training has happened.
- **MED** `src/selfplay/evaluation.rs:103,134,144-150` — `champion_backend` swappable handle is stored on `EvaluationTask` and wired in `main` (`src/bin/selfplay.rs:529`) but never referenced inside `run()`. Promotion only swaps the `Arc<dyn Evaluator>` inside `ChampionStore`; the champion _batcher's_ underlying weights are never hot-swapped. Either drop the field or actually use it.
- **MED** `src/selfplay/evaluation.rs:413,422,451,461` — PGN `game_num` restarts at 1 inside each pool-member loop, so within one cycle multiple games get the same "Game N" tag in PGN output, only distinguishable by the player labels. Logging-only bug.
- **LOW** `src/selfplay/pool.rs:30-37` — Filenames like `best_v01.pt` and `best_v001.pt` both parse to `v=1` and both end up in the returned pool with the same version but different paths, causing duplicate ladder opponents. Not currently exploitable but a footgun.
- **LOW** `src/selfplay/evaluation.rs:374` — `let _ = total_games;` is a dead store in the "opponent unset" branch; the shadowed local goes out of scope immediately.

### Couldn't assess

PyO3 GIL re-entrancy: `Python::attach` inside the async `run()` loop while concurrent self-play tasks call `root_setup_batch` — potential GIL contention/deadlock under a single-thread tokio runtime. Would need a stress test, not a static read.
