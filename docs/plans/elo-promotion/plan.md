# Plan: Elo-based dual-model promotion

## Approach

Replace single-opponent `win_rate >= 0.55` gating with a fixed-rating Elo ladder against the last K=3 archived champions (filesystem scan of `checkpoints/best_v*.pt`, excluding the current champion's own version). Add ONE extra opponent `InferenceServer`+`InferenceBatcher` pair at startup; for each pool member we reload that opponent server's weights via the existing Python `InferenceServer.load_weights(bytes)` path (already used at `src/bin/selfplay.rs:255,342`), then play `2 * games_per_side` ladder games against it while the candidate's Elo updates per-game (K=32, opponents pinned at 1500). Promote when candidate's final Elo > 1520; preserve every existing field in the `[eval] ... ladder_match` log line so `scripts/run_baseline.sh` parsers keep working, and append new fields (`candidate_elo`, `pool_size`, `opponents=...`, `pool_score`) before the `ladder_match` token.

## Subtasks

### 1. Elo math module

- Files: new `src/selfplay/elo.rs`; modify `src/selfplay/mod.rs` to add `pub mod elo;` and re-export `pub use elo::{expected_score, update_rating, INITIAL_RATING, K_FACTOR};`.
- Changes:
  - `pub const INITIAL_RATING: f32 = 1500.0;`
  - `pub const K_FACTOR: f32 = 32.0;`
  - `pub fn expected_score(r_a: f32, r_b: f32) -> f32` — returns `1.0 / (1.0 + 10f32.powf((r_b - r_a) / 400.0))`.
  - `pub fn update_rating(rating: f32, opp_rating: f32, score: f32, k: f32) -> f32` — returns `rating + k * (score - expected_score(rating, opp_rating))`. `score` is 1.0/0.5/0.0 for win/draw/loss.
- Tests (inline `#[cfg(test)] mod tests`):
  - `expected_equal_ratings_is_half` — `expected_score(1500.0, 1500.0)` within 1e-6 of 0.5.
  - `expected_higher_rating_above_half` — `expected_score(1600.0, 1500.0) > 0.5` and `< 1.0`.
  - `update_win_vs_equal_adds_16` — `update_rating(1500.0, 1500.0, 1.0, 32.0)` within 1e-3 of 1516.0.
  - `update_loss_vs_equal_subtracts_16` — `update_rating(1500.0, 1500.0, 0.0, 32.0)` within 1e-3 of 1484.0.
  - `update_draw_vs_equal_is_noop` — `update_rating(1500.0, 1500.0, 0.5, 32.0)` within 1e-6 of 1500.0.
  - `update_loss_when_ahead_loses_more` — `update_rating(1520.0, 1500.0, 0.0, 32.0) < 1484.0`.
  - `sequential_table_driven` — fold over `[1.0, 0.5, 1.0, 0.0, 1.0]` from 1500.0 vs. 1500.0 fixed opp; assert each intermediate value matches a hand-computed reference (precompute in test source; tolerance 1e-3).
- Dependencies: none.

### 2. Pool enumeration helper

- Files: new `src/selfplay/pool.rs`; modify `src/selfplay/mod.rs` to add `pub mod pool;` and re-export `pub use pool::latest_archive_versions;`.
- Changes:
  - `pub fn latest_archive_versions(checkpoints_dir: &Path, exclude_version: u64, k: usize) -> Vec<(u64, PathBuf)>` — read directory, parse `best_v{NNN}.pt` filenames using the same logic as `find_latest_archive_version` in `src/bin/selfplay.rs:21-37` (strip prefix `best_v`, strip suffix `.pt`, parse u64), skip entries where the parsed version equals `exclude_version`, sort newest-first by version, truncate to `k`. Returns empty vec on missing dir or no matches (no panic).
- Tests (inline):
  - `returns_empty_on_missing_dir` — pass nonexistent path, expect `vec![]`.
  - `returns_empty_when_no_matches` — tempdir with unrelated files (`foo.pt`, `best.pt`, `best_v.pt`), expect `vec![]`.
  - `orders_newest_first` — tempdir with empty `best_v001.pt..best_v007.pt`, call with `exclude_version=0, k=3`, assert versions `[7, 6, 5]` in that order and paths end with respective filenames.
  - `excludes_current_version` — same tempdir, call with `exclude_version=7, k=3`, assert versions `[6, 5, 4]`.
  - `truncates_to_k_when_more_available` — call with `k=2`, assert length 2 and newest two returned.
  - `returns_all_when_fewer_than_k` — tempdir with only `best_v001.pt, best_v002.pt`, call with `k=3`, assert length 2.
- Dependencies: none. (Uses `tempfile` per `replay_writer.rs` existing test pattern.)

### 3. Opponent inference setup

> Sub-investigation finding: `PyO3Backend` (`src/py/inference_backend.rs:12-26`) wraps a private `server: Py<PyAny>` field with no public accessor. The existing `load_weights` calls at `src/bin/selfplay.rs:255,342` are made via a **directly held `Py<PyAny>` clone of the server handle** (created at lines 174-179 / 240-247), not through the backend. We will follow the same pattern for the opponent: keep a `Py<PyAny>` clone of the opponent server in `src/bin/selfplay.rs` and pass it into `EvaluationTask` for hot-reload. No change to `PyO3Backend` API needed.

- Files: `src/bin/selfplay.rs`, `src/selfplay/evaluation.rs`.
- Changes:
  - In `src/bin/selfplay.rs`, around the existing champion-server block (after line 275, only when not on the Random fallback branch — but unconditionally for the opponent since pool may be empty initially; in the random branch, the opponent server starts uninitialized and is only loaded when the first archive appears):
    - Construct a fresh Python `InferenceServer` (mirror the champion construction at lines 222-249). Clone the `Py<PyAny>` into `opponent_server_handle` (analogous to `server_for_weights`).
    - Wrap it in `PyO3Backend::new(...)` and spawn an `InferenceBatcher` (mirror lines 259-275). Get an opponent `mpsc::Sender<InferenceRequest>` and construct `opponent_evaluator: Arc<dyn Evaluator> = Arc::new(ChannelEvaluator::new(opponent_tx))`.
    - Pass `opponent_evaluator` and `opponent_server_handle` (a `Py<PyAny>` wrapped in `Arc<Mutex<...>>` so the eval task can `call_method1(py, "load_weights", ...)` without owning the Python GIL contention surface) into `EvaluationTask` via a new builder method `with_opponent(opponent_evaluator, opponent_server_handle)`.
  - In `src/selfplay/evaluation.rs`:
    - Add fields to `EvaluationTask`: `opponent_evaluator: Option<Arc<dyn Evaluator>>`, `opponent_server_handle: Option<Arc<Mutex<Py<PyAny>>>>`. Both default `None` (preserves existing tests / construction sites). NOTE: adding a `Py<PyAny>` field requires the new `pyo3` dependency in `src/selfplay/evaluation.rs`. If the module-level pyo3 dependency is undesirable, abstract behind a trait: `trait WeightLoader { fn load(&self, bytes: &[u8]) -> Result<(), String>; }` implemented in `src/py/` and passed as `Arc<dyn WeightLoader>` — see fallback design note below.
    - Add `pub fn with_opponent(self, evaluator: Arc<dyn Evaluator>, server_handle: Arc<Mutex<Py<PyAny>>>) -> Self` builder (mirrors `with_champion_backend`).
  - Fallback design note (record in plan, decide during implementation): If pulling pyo3 into `evaluation.rs` is rejected on principle (currently `evaluation.rs` has no pyo3 import), introduce `pub trait WeightLoader: Send + Sync { fn load(&self, bytes: &[u8]); }` in `src/selfplay/inference.rs`; impl it on a new `src/py/weight_loader.rs::PyWeightLoader { server: Py<PyAny> }` that wraps `Python::attach(|py| server.call_method1(py, "load_weights", (PyBytes::new(py, bytes),)))`. `EvaluationTask` then holds `Option<Arc<dyn WeightLoader>>`. Costs ~30 lines extra; preserves module boundary. Recommended; the subtask test below assumes this shape.
- Tests:
  - Unit test in `src/py/weight_loader.rs` (or wherever the impl lives): `#[test] #[ignore = "requires hyzero Python package"]` — construct two `InferenceServer`s with the same config, dump weights from one trainer via Python (mirroring `python/tests/test_inference.py:102-138`), assert calling `WeightLoader::load(bytes)` followed by a `root_setup_batch` produces a different output than before-load (proves weights took effect). The full integration is exercised in subtask 4's manual smoke test.
- Dependencies: subtask 1 not required (no Elo math here); 2 not required (pool list passed in from caller). Sequencing only matters for plumbing order in subtask 4.

### 4. EvaluationTask refactor — per-opponent ladder

- Files: `src/selfplay/evaluation.rs`.
- Changes:
  - Extend `EvaluationConfig` with: `pub elo_k_factor: f32`, `pub pool_size: usize`, `pub promotion_elo_delta: f32`, `pub opponent_initial_elo: f32`, `pub checkpoints_dir: PathBuf`. `Default` populates from constants: K=32.0, pool_size=3, delta=20.0, initial_elo=1500.0, dir=`PathBuf::from("checkpoints")`. Keep `promotion_threshold` field (deprecated but tolerated; unused by new logic).
  - Refactor `EvaluationTask::run`:
    1. Build pool: `let pool = pool::latest_archive_versions(&self.config.checkpoints_dir, self.champion_store.version(), self.config.pool_size);`.
    2. **Empty-pool fallback (subtask 7)**: if `pool.is_empty()`, treat the live champion as a single opponent — call the legacy single-opponent path (still played via `play_game_dual` with `self.champion_store.champion().await`), but score it through Elo (one opponent at 1500.0). This preserves the metric definition.
    3. For each `(version, ckpt_path)` in pool: (a) read the file bytes via `std::fs::read(ckpt_path)`; on Err, `eprintln!("[eval] WARN: failed to read pool member v{version}: {e}")` and `continue` (skip — don't fail the cycle). (b) Call `weight_loader.load(&bytes)` (the opponent's `WeightLoader`). (c) Play `2 * games_per_side` games against `opponent_evaluator` using the existing white/black alternation (lines 188-241). For each game, accumulate into `ladder_wins/draws/losses` (pool-aggregated), and update `candidate_elo` per game via `update_rating(candidate_elo, opponent_initial_elo, game_score, k)` where game_score is challenger-perspective (already computed at lines 206-209 / 234-239).
    4. Extract a pure helper for unit testing: `pub(crate) fn compute_candidate_elo_from_results(initial: f32, opp_initial: f32, k: f32, scores: &[f32]) -> f32` — takes a slice of challenger-perspective game scores (1.0/0.5/0.0) and returns the final candidate Elo by sequential `update_rating` application. The main loop computes the slice during play; the helper is called both inline and from tests.
    5. Compute `pool_score = (ladder_wins as f32 + 0.5 * ladder_draws as f32) / total_games as f32` (kept for log output; not part of gating).
    6. Compute `win_rate = pool_score` (same value; kept under the legacy field name so `run_baseline.sh` line 192 still extracts).
  - New log line (single emission per cycle, append-only after existing fields, keep `ladder_match` token last):
    ```
    [eval] v{challenger_version} cycle={cycle} ladder_wins={w} ladder_draws={d} ladder_losses={l} \
      win_rate={win_rate:.3} champion_version={cv} candidate_elo={elo:.1} pool_size={n} \
      opponents={v1},{v2},{v3} pool_score={pool_score:.3} ladder_match
    ```
    Note: `champion_version` continues to be `self.champion_store.version()` (the current best); `opponents=` is comma-separated archive versions (or `none` if pool empty); `candidate_elo` is the post-cycle candidate Elo (Elo state is per-cycle, no persistence per user decision).
  - Promotion gate replacement (formerly lines 262-284):
    ```rust
    let cooldown_ok = self.total_games_since_last_promotion >= self.config.promotion_cooldown_games
        || self.config.promotion_cooldown_games == 0;
    if candidate_elo > self.config.opponent_initial_elo + self.config.promotion_elo_delta && cooldown_ok {
        // existing promote() call unchanged
    }
    ```
    Existing `[eval] promoted champion_version=... win_rate=...` line stays — script counts it (line 178). Append `candidate_elo={elo:.1}` to it for traceability.
- Tests (inline at bottom of `evaluation.rs`):
  - `compute_candidate_elo_empty_scores_returns_initial` — `compute_candidate_elo_from_results(1500.0, 1500.0, 32.0, &[]) == 1500.0`.
  - `compute_candidate_elo_all_wins_against_equal` — 8 wins vs. 1500 with K=32 starting at 1500: assert > 1520 (proves the promotion threshold is reachable in a clean sweep).
  - `compute_candidate_elo_50_percent_against_equal_is_noop` — alternating `[1.0, 0.0, 1.0, 0.0]`: final rating within ~1 Elo of 1500.0 (not exactly, due to compounding — assert `|final - 1500.0| < 1.0`).
  - `compute_candidate_elo_all_losses_against_equal` — 8 losses with K=32: assert < 1480 (symmetric check).
  - `evaluation_config_defaults_have_elo_fields` — extend existing `test_evaluation_config_defaults` to assert `elo_k_factor == 32.0`, `pool_size == 3`, `promotion_elo_delta == 20.0`, `opponent_initial_elo == 1500.0`.
- Dependencies: subtasks 1, 2, 3 must land first (uses `update_rating`, `latest_archive_versions`, opponent plumbing).

### 5. Config plumbing

- Files: `src/bin/selfplay.rs`.
- Changes:
  - In `RunConfig::default` (around lines 73-88), add fields: `elo_k_factor: f32` (default 32.0), `pool_size: usize` (default 3), `promotion_elo_delta: f32` (default 20.0), `opponent_initial_elo: f32` (default 1500.0).
  - In the env-var parsing block (around lines 97-139), add:
    - `HYZERO_POOL_SIZE` → `pool_size` (usize)
    - `HYZERO_PROMOTION_ELO_DELTA` → `promotion_elo_delta` (f32)
    - `HYZERO_ELO_K_FACTOR` → `elo_k_factor` (f32)
    - `HYZERO_OPPONENT_INITIAL_ELO` → `opponent_initial_elo` (f32)
  - For `HYZERO_PROMOTION_THRESHOLD`: keep parsing it onto `promotion_threshold` (field remains for back-compat) but emit `eprintln!("[selfplay] WARN: HYZERO_PROMOTION_THRESHOLD is deprecated; gating now uses Elo. Use HYZERO_PROMOTION_ELO_DELTA instead.")` when the env var is set explicitly (use `env::var("HYZERO_PROMOTION_THRESHOLD").is_ok()` to detect set-ness, not default fallback). The value is otherwise unused.
  - Wire all four new fields onto `EvaluationConfig` at lines 395-403.
  - Update the `[selfplay] Starting evaluation ladder ...` print line (around 405-408) to log `pool_size`, `promotion_elo_delta` in place of `threshold` (keep `threshold` echoed for the deprecation period).
- Tests:
  - Inline `#[cfg(test)]` in `src/bin/selfplay.rs` (note: bin crate doesn't expose tests easily — verify existing pattern; if no tests exist in this file, factor `RunConfig::from_env() -> RunConfig` into a pure function and unit-test that). Tests cover: `from_env_returns_defaults_when_unset` (clear env, assert defaults); `from_env_parses_pool_size_override` (set `HYZERO_POOL_SIZE=5`, assert field == 5); `from_env_parses_elo_delta_override` (set `HYZERO_PROMOTION_ELO_DELTA=30.0`, assert field == 30.0). Use `std::env::set_var`/`remove_var` with the standard test isolation caveat — if the codebase uses `serial_test` for env tests, follow that pattern; otherwise mark `#[test] #[ignore = "env tests run serially"]` and document.
- Dependencies: subtask 4 (consumes the new `EvaluationConfig` fields).

### 6. run_baseline.sh extraction

- Files: `scripts/run_baseline.sh`.
- Changes:
  - Add CANDIDATE_ELO extraction (parallel to existing win_rate parsing at lines 188-195):
    ```bash
    CANDIDATE_ELO_SUMMARY=$(awk '/\[eval\].*ladder_match/{
        cycle++
        elo = "1500.0"
        for (i=1; i<=NF; i++) {
            if ($i ~ /^candidate_elo=/) { split($i, a, "="); elo = a[2] }
        }
        print cycle, elo
    }' "$LOG_FILE")
    LAST_CANDIDATE_ELO=$(echo "$CANDIDATE_ELO_SUMMARY" | awk '{last=$2} END{print last+0}')
    LAST_CANDIDATE_ELO=${LAST_CANDIDATE_ELO:-1500.0}
    ```
    Insert inside the existing `if [ "$EVAL_CYCLES" -gt 0 ]; then` block (around line 186); when no eval cycles, default `LAST_CANDIDATE_ELO=1500.0` in the `else` branch (line 211).
  - Add to the Results section (echo around line 245): `echo "  Last candidate Elo:  $LAST_CANDIDATE_ELO"`.
  - Add to JSON heredoc (after `last_win_rate` line 281): `"last_candidate_elo": $LAST_CANDIDATE_ELO,`.
  - Composite SCORE update (line 223-231): add `(last_candidate_elo - 1500.0) * elo_score_weight` term. Propose `ELO_SCORE_WEIGHT=0.05` default (so 20 Elo gain ≈ 1.0 score-point, comparable to one promotion at weight 2.0 = half a promotion). Final formula:
    ```awk
    score = (init_loss - policy_loss) + (promotions * weight) - (avg_len / 100) + (last_candidate_elo - 1500.0) * elo_score_weight;
    ```
    Add `ELO_SCORE_WEIGHT=${HYZERO_ELO_SCORE_WEIGHT:-0.05}` to the Configuration block (around line 13).
  - **Preserve** all existing extractors: `LAST_WIN_RATE` (line 204), `PROMOTIONS` (line 178), `MAX_CHAMPION_VERSION` (line 179-184), `EVAL_CYCLES` (line 175). They keep extracting because subtask 4 preserves those exact field names in the log line.
- Tests:
  - `shellcheck scripts/run_baseline.sh` runs clean (existing dev convention).
  - Synthetic log line dry-grep: pipe a known log fragment through the new awk block and assert extracted value:
    ```bash
    echo '[eval] v3 cycle=1 ladder_wins=4 ladder_draws=2 ladder_losses=2 win_rate=0.625 champion_version=1 candidate_elo=1524.7 pool_size=3 opponents=2,1,0 pool_score=0.625 ladder_match' | awk '/ladder_match/{for(i=1;i<=NF;i++) if($i~/^candidate_elo=/){split($i,a,"="); print a[2]}}'
    ```
    Expected output: `1524.7`. Run this as a one-liner verification step in the plan's CI-equivalent (the test plan section).
- Dependencies: subtask 4 must define the log line shape; subtask 6 is otherwise standalone (can be drafted in parallel with subtask 5, sequenced behind 4 only for the final test).

### 7. Early-training fallback

- Files: `src/selfplay/evaluation.rs` (subsumed by subtask 4 but called out explicitly here).
- Changes:
  - In `EvaluationTask::run` at the start of each cycle, after building the pool: if `pool.is_empty()`, log `[eval] pool empty — falling back to single-opponent vs. current champion v{cv}` and use `self.champion_store.champion().await` as the single opponent. Score through Elo identically (one opponent at 1500.0, K=32). This keeps `candidate_elo` defined.
  - If `0 < pool.len() < pool_size`, proceed normally with the smaller pool — `latest_archive_versions` already truncates gracefully (subtask 2's `returns_all_when_fewer_than_k` test confirms).
  - `pool_size=` field in log line reports actual `pool.len()` (0 when empty-pool fallback, real count otherwise).
  - `opponents=` field reports `none` when empty, otherwise comma-separated versions.
- Tests:
  - Covered by subtask 2's `returns_all_when_fewer_than_k` (graceful partial pool).
  - Covered by subtask 4's `compute_candidate_elo_*` tests (Elo math works for any number of games against the fixed-1500 opponent).
  - One additional inline test in `evaluation.rs`: `eval_log_format_with_empty_pool` — a string-formatting test that asserts the log line shape includes `pool_size=0 opponents=none` when called with an empty pool (extract the formatting into a `fn format_eval_line(...)` helper if needed).
- Dependencies: subtask 4 (this is a behavior carve-out inside subtask 4's refactor).

## Testing strategy

- Unit tests: subtasks 1, 2, 4, 5, 7 each have inline `#[cfg(test)]` tests. Subtask 3's pyo3-touching test is `#[ignore]`-gated (matches existing convention in `src/py/inference_backend.rs:325-326`).
- Integration / build: `cargo build --release --bin selfplay` must succeed; `cargo test` must pass (excluding the `#[ignore]` pyo3 tests; the project does not run them by default — verify against existing CI behavior).
- Manual smoke test (low-cost):
  1. Set `HYZERO_EVAL_SIMS=10`, `HYZERO_GAMES_PER_SIDE=1`, `HYZERO_MAX_GAME_LENGTH=20` (if such an env exists; otherwise just trust eval-sims). Pre-seed `checkpoints/` with 3 small `best_v001.pt..best_v003.pt` files (copy `mate_pretrained.pt` 3 times).
  2. Run `target/release/selfplay` for ~60s, kill, grep log for `ladder_match`.
  3. Confirm: log line has `candidate_elo=`, `pool_size=3`, `opponents=3,2,1` (or similar), `win_rate=`, `champion_version=`, `ladder_match` (terminal token).
  4. Pipe the captured line through the new awk extractor (subtask 6's synthetic test). Confirm `LAST_CANDIDATE_ELO` is set to a float in [0, 3000] range.
- End-to-end: `bash scripts/run_baseline.sh 60` (1-min smoke). Confirm `logs/baseline_score.json` includes `last_candidate_elo` field with a numeric value, no `errors` count increase, no panics in log.
- Regression: confirm `run_baseline.sh` still extracts `last_win_rate`, `promotions`, `max_champion_version`, `eval_cycles` — all four fields preserved in the new log line by subtask 4.

## Rollback

- Plan rests on additive changes (new modules: `elo`, `pool`; new `EvaluationConfig` fields with defaults; new env vars; new log fields appended before the `ladder_match` token).
- The deprecated `HYZERO_PROMOTION_THRESHOLD` becomes a no-op with a warning — pre-existing scripts that set it continue to run (loudly but successfully).
- Rollback procedure: revert the feature branch (`git revert` the merge commit, or `git reset` if not yet merged). No data migration. No on-disk schema changes. Checkpoint files (`best.pt`, `best_v*.pt`) are unaffected — pool enumeration is read-only.
- Risk surface: subtask 3's opponent batcher adds ~1 model's worth of VRAM permanently for the run. If VRAM-constrained, fallback is to spawn the opponent batcher on CPU device (load `HYZERO_OPPONENT_DEVICE=cpu` from env, plumbed into the `InferenceServer` constructor). Document this as a follow-up if subtask 4's smoke test reveals OOM on the target GPU.
