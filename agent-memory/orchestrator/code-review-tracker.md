# Code Review Tracker

Tracks commits reviewed by the scheduled bugs-focused code-review routine.

## Last reviewed HEAD

`bde4f9b` (full SHA: `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`)

## History

### 2026-10-08

- **Range**: `7b53e5d..bde4f9b`
- **Scope**: elo-promotion feature (new elo math module, archive pool helper, opponent inference server plumbing, per-opponent ladder eval, env-var wiring, candidate_elo extraction in baseline)
- **Findings**:
  1. **CONFIRMED — Test regression**: `src/selfplay/evaluation.rs:848-857` — `test_evaluation_task_promotes_when_threshold_zero` builds `EvaluationConfig` without overriding `checkpoints_dir`, so it defaults to `PathBuf::from("checkpoints")`. After any `cargo run --bin selfplay` run leaves `best_v{NNN}.pt` on disk, `cargo test` picks up a nonempty pool, the test never calls `.with_opponent(...)`, the Elo path hits the `(None, None)` arm at `src/selfplay/evaluation.rs:346-376`, logs WARN, `continue`s without playing games, loop ends, `store_ref.version()` stays 0, and the `assert_eq!(..., 5)` at line 885 fails. The sibling bootstrap tests at lines 660 and 711 defensively set `checkpoints_dir` to `/nonexistent/test/dir/*`; this one did not. Same latent flakiness in `test_evaluation_task_completes_one_cycle` (line 783) but its assertions don't check promotion so it still passes.
  2. **PLAUSIBLE — Startup notice false after first promotion**: `src/bin/selfplay.rs:174-177` claims gating switches to Elo once any `best_v{NNN}.pt` exists, but `src/selfplay/evaluation.rs:247-251` calls `latest_archive_versions(..., champion_version, pool_size)` which filters out the current champion's archive. After the first promotion, the only on-disk archive is the new champion's → pool is empty → bootstrap branch fires. In that branch the live "champion" is `self.challenger_evaluator.clone()` (`src/selfplay/champion.rs:81`), so challenger vs. champion inferences share the same backend and win_rate ≈ 0.5, below the 0.55 default threshold. A cold-start with `HYZERO_RESUME_FROM=checkpoints/best.pt` (no best_v files) therefore stalls at exactly one promotion until some unrelated best_v archive appears. Pre-refactor code was stuck here too; the new Elo path doesn't rescue it because the exclude filter suppresses it.
