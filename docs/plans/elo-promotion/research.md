# Research — Pool-based Elo promotion

Replacing single-opponent win-rate gating (`win_rate >= 0.55`) in dual-model eval with a 3-opponent Elo ladder. K=3 archived champions held at fixed Elo=1500; candidate Elo updates per-game (K-factor=32); promote when candidate > 1520.

## 1. What exists today

**`EvaluationTask::new`** (`src/selfplay/evaluation.rs:89-108`)
```rust
pub fn new(
    precomputed: Arc<PrecomputedItems>,
    challenger_evaluator: Arc<dyn Evaluator>,
    model_version_rx: watch::Receiver<u64>,
    latest_checkpoint_path: Arc<Mutex<Option<PathBuf>>>,
    champion_store: Arc<ChampionStore>,
    config: EvaluationConfig,
) -> Self
```

**`EvaluationTask::run`** (`evaluation.rs:152`) — `pub async fn run(&mut self)`. Outer loop: wait on `model_version_rx`. Per cycle:
- Fetch `champion_eval = self.champion_store.champion().await` (one opponent).
- Two for-loops (`evaluation.rs:188-241`): `gps` games challenger=White, `gps` games challenger=Black. Each calls `play_game_dual(precomputed, white, black, game_config)`. Tally `ladder_wins/draws/losses` via `outcome.game_outcome` (>0.5 / <-0.5 thresholds), negating for Black-side.
- Compute `win_rate = (wins + 0.5*draws) / total_games` (line 244).
- Promotion gate (lines 262-284): `if win_rate >= self.config.promotion_threshold && cooldown_ok { self.champion_store.promote(new_champ, challenger_version, ckpt_path.as_ref()).await; ... }`.

**`EvaluationTask::write_pgn_game`** (`evaluation.rs:120-142`)
```rust
fn write_pgn_game(cycle: u64, game_num: usize, white_label: &str, black_label: &str, outcome: &DualGameOutcome)
```
Maps `outcome.game_outcome` → "1-0"/"0-1"/"1/2-1/2", delegates to `pgn::write_pgn_game`.

**`play_game_dual`** (`src/selfplay/game_task.rs:269-393`)
```rust
pub async fn play_game_dual(
    precomputed: Arc<PrecomputedItems>,
    white_evaluator: Arc<dyn Evaluator>,
    black_evaluator: Arc<dyn Evaluator>,
    config: GameConfig,
) -> DualGameOutcome
```
Cheap to call repeatedly: builds an `MCTSTree` per ply, no per-call inference-backend spawn. Picks evaluator by side-to-move (`game_task.rs:333-337`). `DualGameOutcome { game_outcome: f32 /* +1/-1/0 */, num_moves: usize, moves: Vec<String> }`.

**`ChampionStore::promote`** (`src/selfplay/champion.rs:72-112`)
```rust
pub async fn promote(&self, new_champion: Arc<dyn Evaluator>, new_version: u64, checkpoint_src: Option<&PathBuf>) -> u64
```
Swaps under `RwLock` write, bumps atomic version, optionally persists via `persist_champion_checkpoint` (`champion.rs:117-142` — writes `checkpoints/best.pt` + `checkpoints/best_v{NNN}.pt` with `{:03}` zero-pad), prunes oldest beyond `archive_depth`.

**`find_latest_archive_version`** (`src/bin/selfplay.rs:21-37`) — `fn find_latest_archive_version() -> Option<u64>`. Scans `checkpoints/`, parses `best_v{NUM}.pt`, returns max. **Bin-local** (private to `src/bin/selfplay.rs`).

## 2. Pool reachability

- `archive_depth=5` is hardcoded at `src/bin/selfplay.rs:389` (`ChampionStore::new_with_version(... , 5, ...)`). No env var. Need K=3, so 5 is sufficient — but the pool is the **last K of however-many exist**, not exactly 5.
- **No existing API to list `[(version, path)]` for archives.** `ChampionStore.archive_files` is private `RwLock<Vec<PathBuf>>` (`champion.rs:25`), tracks only files created in-process — useless for selecting pool members across restarts (though we're told no cross-run persistence, the current champion at version=v just promoted in cycle C may have older archives from cycles C-1, C-2 that aren't in `archive_files` if e.g. the binary restarted mid-run; defensively, scan disk).
- Directory: `checkpoints/`. Naming: `best_v{:03}.pt` (zero-padded), confirmed `champion.rs:92,121`.
- **Archives include the current champion**: `persist_champion_checkpoint` writes both `best.pt` AND `best_v{NNN}.pt` for the just-promoted version (`champion.rs:135-138`). Pool selection must exclude the **current champion's version** (`champion_store.version()`) to avoid candidate-vs-itself.
- Must write a new helper (e.g., `fn list_archives_for_pool(k: usize, exclude_version: u64) -> Vec<(u64, PathBuf)>`) that reuses the scan logic from `find_latest_archive_version`.

## 3. Codebase patterns

- **Env-var config**: `src/bin/selfplay.rs:97-139` — `env::var("HYZERO_FOO").ok().and_then(|v| v.parse().ok()).unwrap_or(defaults.foo)`. Defaults live in `RunConfig::default` (`selfplay.rs:73-88`). Field plumbed onto `EvaluationConfig` (`evaluation.rs:35-65`). Module-internal cached env (e.g., `gumbel_top_k` `game_task.rs:22-38`) uses `OnceLock` — fine for read-once knobs.
- **Logging**: `println!` to stdout with grep-friendly `[scope]` prefix and `k=v` tokens. Examples: `[eval] v{V} cycle={C} ladder_wins={W} ... win_rate={R:.3} champion_version={CV} ladder_match` (`evaluation.rs:246-255`); `[eval] promoted ...` (`278-283`); `[champion] saved ...` (`champion.rs:140`); `[selfplay] ...` for bin. Use `eprintln!` for warnings.
- **Errors**: No `anyhow`/`thiserror` in `Cargo.toml`. `std::io::Result` for fs ops (`champion.rs:117`); `.expect()` at bin startup; `.ok()`/`Err(e) => eprintln!(...)` for non-fatal in-loop failures. Pool-listing should swallow individual file-parse errors (log + skip) rather than fail-fast.
- **Tests**: Inline `#[cfg(test)] mod tests` at bottom of each `src/selfplay/*.rs`. No `tests/` integration dir. `tokio::test` for async, plain `#[test]` for pure logic. `tempfile = "3"` is a dev-dep available for fs tests; existing `replay_writer.rs` tests use it.

## 4. What can't change

- `Cargo.lock` read-only (CLAUDE.md).
- **`[eval] ... ladder_match` line** — parsed by `run_baseline.sh`. Required fields: `win_rate=` (line 192: regex `win_rate=([0-9.]+)`); presence of `ladder_match` token (cycle counter, line 175,188). New fields (`elo=`, `pool_size=`, etc.) are safe **appended** before `ladder_match`. Don't rename `win_rate` — script keys on it and writes `last_win_rate` to JSON.
- **`[eval] ... promoted ... champion_version=NN`** — script counts these (`PROMOTIONS`, line 178) and tracks `MAX_CHAMPION_VERSION` via field regex (lines 179-184). Keep the `promoted` token and the `champion_version=` field.
- **`pgn::write_pgn_game(path, event, white, black, result, moves)`** signature is general — no change needed if we want per-opponent labels in `[White]`/`[Black]` (already passes `&format!("champion v{cv}")`).
- **Re-exports** in `src/selfplay/mod.rs:9-16`: `EvaluationConfig`, `EvaluationTask`, `DualGameOutcome`, `ChampionStore` are all public. Field additions to `EvaluationConfig` are non-breaking only if defaults are populated.

## 5. What could break

- **Inference cost**: pool members are stale `best_v{NNN}.pt` files. The current champion runs through its own `champion_batcher` (PyO3Backend), set up at startup (`bin/selfplay.rs:259-275`). **No existing path spawns a batcher for arbitrary archive checkpoints.** Naive approach: load each pool member into a fresh `InferenceServer` + `PyO3Backend` + `InferenceBatcher` once per cycle = K extra Python instances + K batchers per cycle = heavy. Cheaper: build K batchers **once per pool refresh** (when a new archive lands) and hold `Arc<dyn Evaluator>` handles. Cheapest at correctness/risk tradeoff: single shared `champion_server` with `load_weights` swapped between games — but this serializes games and may race with the existing champion's batcher. Plan must call this out.
- **Cycle time**: today 8 games (gps=4, 2 sides), with K=3 → 24 games. At ~5-30s/game (50 sims, MCTS), cycles go from ~2-5min to ~6-15min. Nothing in `evaluation.rs` enforces a timeout, but `run_baseline.sh` runs for 1800s default — at 15min/cycle you may get only 1-2 cycles. Consider sequencing pool games to fewer total (e.g., 2 games per opponent × K=3 = 6, vs. 8 today).
- **Cooldown**: `promotion_cooldown_games` (today 0) counts `total_games_since_last_promotion` (`evaluation.rs:84,211,240,258-260,276`). Still meaningful — prevents flapping if Elo barely crosses 1520 then dips. Keep, but interpretation shifts (cooldown = games against pool, not games against single champion).
- **Early training (fewer than K archives)**: today eval runs from cycle 1 with `champion_version=0` (RandomEvaluator). Gating code at `evaluation.rs:156-166` only waits for a model-version bump, not for archives. Plan must specify behavior when `len(pool) < K`: options are (a) use pool of size N<K with same fixed-rating math, (b) fall back to single-opponent vs. current champion, (c) skip eval until pool fills. Recon flagged this as an open decision.
- **`last_evaluated_version` (`evaluation.rs:153`)**: in-process local, resets to 0 on restart. Not persisted. No new persistence requirement here — Elo state also resets per the user's "no cross-run persistence" decision.

## 6. Tests

- Inline pattern: add `#[cfg(test)] mod tests` to a new module (or extend `evaluation.rs::tests`).
- Elo math (`fn update_elo(candidate: f64, opponent_fixed: f64, score: f64, k: f64) -> f64`) is pure → table-driven `#[test]`:
  - `update_elo(1500.0, 1500.0, 1.0, 32.0)` → `1516.0` (win vs. equal).
  - `update_elo(1500.0, 1500.0, 0.0, 32.0)` → `1484.0` (loss vs. equal).
  - `update_elo(1500.0, 1500.0, 0.5, 32.0)` → `1500.0` (draw vs. equal).
  - `update_elo(1520.0, 1500.0, 0.0, 32.0)` → lower than 1500 (loss when ahead loses more).
- Pool-list helper: test against a `tempfile::tempdir()` (per `replay_writer.rs`) with synthesized `best_v001.pt..best_v007.pt` empty files; assert returns last K excluding the named "current" version. Cannot test inside `checkpoints/` (real-fs pollution).
- Integration: extend `test_evaluation_task_completes_one_cycle` to assert at least one game is played and the structured log fields emit (no actual log capture available — would need stdout redirection or a refactor to inject a logger). Pragma: keep the unit-test for Elo math + pool listing, defer end-to-end to manual `run_baseline.sh` verification.
