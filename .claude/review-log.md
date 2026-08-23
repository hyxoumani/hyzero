# Claude Review Log

Scheduled bug-review routine leaves entries here. Each entry names the commit reviewed through, the scope, and any findings that surfaced. Future runs pick up from the newest entry's SHA.

## 2026-08-23 — reviewed through bde4f9b

**Scope**: elo-promotion feature series (`git log 5f30ea8..bde4f9b`), bug-hunt only — no style/naming/doc notes.

### Findings (most severe first)

1. **`src/bin/selfplay.rs:144-151`** — Env parsers accept `HYZERO_ELO_K_FACTOR=0` and `HYZERO_POOL_SIZE=0` with no floor. `HYZERO_ELO_K_FACTOR=0` freezes `candidate_elo` at 1500.0, so the Elo gate never fires and promotion silently stops after the first archive lands. `HYZERO_POOL_SIZE=0` locks the system in bootstrap forever. The `filter(|&n: &usize| n >= 1)` pattern from `src/py/training.rs:435` is not applied.

2. **`src/selfplay/evaluation.rs:543, 282`** — `promote` stores `new_champ = self.challenger_evaluator.clone()`, an `Arc<ChannelEvaluator>` on the live challenger `InferenceServer`. `champion_backend` (set at `src/bin/selfplay.rs:529`) is never read in `run()`, so the "champion" tracks the challenger's continuously-updated weights instead of being frozen. Trigger: bootstrap re-enters (pool empty despite `champion_version > 0`); `champion_store.champion().await` returns the same evaluator as `challenger_evaluator`, so all `2·gps` games become challenger-vs-challenger, win_rate stalls at ~0.5, gate stuck.

3. **`src/selfplay/champion.rs:117-142` + `src/bin/selfplay.rs:381, 525`** — `promote()` uses whatever `latest_checkpoint_path` currently holds; training races ahead of eval and updates this Mutex between cycles. Trigger: training bumps to v8 while eval finishes v6 → `promote(new_champ, 6, ckpt=<v8 path>)` copies v8's file to `best_v006.pt`. Future Elo cycles then load "v6" weights that are actually v8. Likely pre-existing but now materially exercised by the Elo ladder.

4. **`src/selfplay/evaluation.rs:424, 462`** — PGN event numbering is per-opponent (`Game {game_idx+1}`), so with `pool_size=3, gps=4` one cycle emits three games each tagged `Eval Cycle N Game 1`, breaking `(cycle, game)` uniqueness for downstream tools.

5. **`scripts/run_baseline.sh:212-217`** — `awk '... END{print last+0}'` emits `0` when the pattern has zero hits, so the `${LAST_CANDIDATE_ELO:-1500.0}` fallback never fires. Today `EVAL_CYCLES>0` implies at least one ladder-match line, but a log-format drift would swing the composite score by `(0 − 1500) × 0.05 = −75`.

**Clean under review**: `src/selfplay/elo.rs` (Elo math), `src/selfplay/pool.rs` (archive enumeration), sign-flip on Black games in `evaluation.rs:468`, GIL/Mutex ordering around opponent `load_weights` (single holder of `opp_handle`, single writer to `opponent_server`).
