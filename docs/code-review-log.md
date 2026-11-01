# Code Review Log

Running log of bug-focused code reviews of hyzero. Each entry records the commit range reviewed and the real bugs found (correctness defects only — not style, naming, or coverage). Later review runs should read the latest "Reviewed through" sha and only cover commits newer than it.

## 2026-11-01 — Reviewed through `bde4f9b`

**Scope:** commits `68b29ef..bde4f9b` (elo-ladder feature series + iter-2/iter-5 tweaks). Earlier history is treated as pre-existing baseline, not reviewed in this pass.

**Reviewer:** Claude Code (scheduled routine, bug-focus only).

### Findings

1. **MEDIUM — `src/selfplay/evaluation.rs:397` — opp_handle mutex poisoning kills the eval ladder forever.** `let guard = opp_handle.lock().unwrap();` inside `Python::attach`. If any prior `load_weights` call panics while holding the `std::sync::Mutex<Py<PyAny>>`, the mutex is poisoned and every future cycle panics on the unwrap. Fix: `.lock().unwrap_or_else(|e| e.into_inner())` (or `.lock().ok()` and log+skip). Trigger: Python-side panic (OOM, torch segfault) during a weight swap.

2. **LOW — `src/selfplay/evaluation.rs:270-335` — bootstrap path reruns after promotion #1 instead of switching to Elo gate.** Pool enumeration excludes the current champion version. After the first promotion, `best_v{N}.pt` is the only archive and gets excluded → pool is empty → the bootstrap / win-rate gate runs again. The Elo gate only activates after promotion #2. Behaviorally safe but contradicts the comment at 273-274 and the startup banner wording in `src/selfplay/selfplay.rs:173`.

3. **LOW — `src/selfplay/evaluation.rs:447,486` — cooldown counter can freeze if every opponent fails to load.** `total_games_since_last_promotion` only advances on completed games. If `load_weights` fails for every opponent in a cycle (`continue 'pool_loop`), the counter never increments and `promotion_cooldown_games` never clears. Edge case tied to persistent inference failures.

4. **LOW — `src/selfplay/evaluation.rs:264,440` — candidate_elo accumulates with opponents pinned at a fixed rating.** With pool_size K and all opponents at 1500, a candidate at 1510 after opponent 1 then faces opponent 2 at expected 0.514 — wins cost less than +16, losses cost more. The gate `candidate_elo > 1520` is reachable near ~60% win rate rather than the "20 Elo above a fresh 1500 opponent" framing in commit messages. Not a bug per the plan; worth clarifying.

5. **LOW — `scripts/run_baseline.sh:216-219` — LAST_CANDIDATE_ELO fallback breaks on empty input.** `awk '{last=$2} END{print last+0}'` emits `0` when the pipe is empty (`last` uninitialized), so `${LAST_CANDIDATE_ELO:-1500.0}` doesn't rescue, and the score term `(0 - 1500.0) * 0.05 = -75`. Guarded today by `if [ "$EVAL_CYCLES" -gt 0 ]`; latent if a future refactor decouples the two.

6. **LOW — `src/selfplay/pool.rs:30-39` — pool enumeration has no padding normalization or file-type check.** `u64::parse` treats `best_v3.pt` and `best_v003.pt` as the same version, allowing duplicate versions if a migration ever drops the `:03` padding. No `file_type().is_file()` check either; a directory or dangling symlink named `best_v123.pt` would be enumerated (fs::read fails later → WARN spam, not a crash). Not currently exploitable; `persist_champion_checkpoint` always uses `:03`.

7. **LOW — `src/selfplay/selfplay.rs:412-443` — opponent InferenceServer boots with random weights, no in-code guard.** `InferenceServer(config, device)` initializes a fresh model and the batcher is spawned immediately. Any inference request routed before the pool loop runs `load_weights` would use random weights. Currently safe because the opponent ChannelEvaluator is only used inside the pool branch after a weight swap, but there's no assertion to catch a future caller.

### Clean commits

`68b29ef`, `7b5dd87`, `aff97fb`, `7b53e5d` — no bugs. Elo math (`1/(1+10^((Rb-Ra)/400))`, positive K, score convention 1/0.5/0) is correct.

### Notes for future reviewers

- When this routine fires again, read the latest "Reviewed through" sha above and run `git log <sha>..HEAD` to find new commits. Review only those.
- Append a new `## YYYY-MM-DD — Reviewed through <sha>` section rather than editing past entries.
- Scope: bug-focus only. Style, naming, and coverage findings belong elsewhere.
