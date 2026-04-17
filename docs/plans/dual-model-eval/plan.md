# Plan: Dual-Model Evaluation (Champion-Challenger Ladder)

## Approach

Replace the broken `decisive_ratio` signal with a champion-challenger ladder: one dedicated
tokio task (reserved from the N self-play slots) runs eval matches in a continuous loop,
always against the current champion. The champion starts as `RandomEvaluator` (no checkpoint
required) and is promoted when the challenger's win rate exceeds a threshold. Promotions
snapshot the current training checkpoint to `checkpoints/best.pt` and atomically swap the
champion evaluator. The scoring metric becomes **champion version reached during the run**,
which monotonically signals genuine progress independent of opponent strength fluctuation.
`play_game_dual` (separate white/black evaluators) is the shared primitive; both self-play
eval games and ladder games use it.

---

## Architecture Diagram

```
tokio runtime
│
├── [inference batcher — challenger]          mpsc channel (256)
│     PyO3Backend → InferenceServer (live weights, updated each version bump)
│     Consumers: N-1 self-play tasks + 1 eval task (challenger side of ladder games)
│
├── [champion batcher — frozen]               mpsc channel (64)
│     PyO3Backend → ChampionInferenceServer (frozen weights, loaded from best.pt)
│     Consumer: 1 eval task (champion side of ladder games)
│     At startup: RandomBackend (no Python needed until first promotion)
│
├── [training thread]
│     PyTrainingThread::run()
│     publishes: weight_tx (watch), version_tx (watch)
│     side-effect: saves checkpoints/model_vNNNNNN.pt periodically
│
├── [weight loader — challenger]
│     watches weight_rx → calls server.load_weights() on live InferenceServer
│
├── [self-play coordinator]                   N-1 persistent game tasks
│     each: loop { play_game(challenger_eval, ...) → send trajectory }
│
└── [eval task — champion-challenger ladder]  1 persistent task (reserved slot)
      ChampionStore (Arc<RwLock<Arc<dyn Evaluator>>>) ← read champion from here
      loop {
        play_game_dual(challenger_eval, champion_eval, ...) × M games as White
        play_game_dual(champion_eval, challenger_eval, ...) × M games as Black
        compute win_rate = (wins + 0.5*draws) / (2*M)
        if win_rate >= PROMOTION_THRESHOLD and cooldown elapsed:
          snapshot training checkpoint → best.pt.tmp → fsync → rename → best.pt
          load best.pt into ChampionInferenceServer (GIL acquire once)
          champion_store.write() = Arc::new(ChannelEvaluator::new(champion_tx))
          increment champion_version, log promotion
          reset win tracking
      }
```

Key distinction from the prior plan:
- `N-1` self-play tasks (not N): one slot is permanently reserved for eval.
- Challenger uses the **existing** live inference batcher (no contention change for self-play).
- Champion uses a **separate** batcher backed by either `RandomBackend` (pre-promotion) or
  a frozen `ChampionInferenceServer` (post-promotion). These two batchers share no state.
- The GIL is acquired at most once per promotion event (to call `champion_server.load_weights`),
  not continuously. No GIL interference with normal self-play batching.

---

## Subtasks

### 1. Add `play_game_dual` to `game_task.rs`

**Files**: `src/selfplay/game_task.rs`, `src/selfplay/mod.rs`

**Changes**:

Add a new public async function below `play_game`:

```rust
pub async fn play_game_dual(
    precomputed: Arc<PrecomputedItems>,
    white_evaluator: Arc<dyn Evaluator>,
    black_evaluator: Arc<dyn Evaluator>,
    model_version: u64,
    config: GameConfig,
) -> GameTrajectory
```

Body: identical to `play_game`, except the evaluator selected per turn:
```rust
let turn_evaluator = if side_to_move == Color::White {
    white_evaluator.clone()
} else {
    black_evaluator.clone()
};
let (hidden_state, policy, value) = turn_evaluator.root_setup(&observation, &legal_mask).await;
// ... tree built with turn_evaluator ...
tree.run_simulations(turn_evaluator.as_ref()).await;
```

The current `play_game` remains unchanged (self-play still uses it).

Update `src/selfplay/mod.rs` re-export:
```rust
pub use game_task::{GameConfig, play_game, play_game_dual};
```

**Tests** (in `game_task.rs` `#[cfg(test)]` block):
- `test_play_game_dual_random_vs_random`: two `RandomEvaluator`s, verify non-empty trajectory
  and valid outcome. Mirrors `test_play_game_completes`.
- `test_play_game_dual_outcome_perspective`: verify `game_outcome` in {-1.0, 0.0, 1.0}.

**Dependencies**: none.

---

### 2. Add `ChampionStore` abstraction

**Files**: `src/selfplay/champion.rs` (new file), `src/selfplay/mod.rs`

**Purpose**: A thread-safe, swappable holder of the current champion evaluator. The eval
task reads from it; the promotion protocol writes to it atomically.

```rust
use std::sync::{Arc, RwLock};
use crate::mcts::evaluator::Evaluator;

/// Holds the current champion evaluator behind a readers-writer lock.
/// Readers (eval task) acquire a read guard per eval cycle.
/// Writer (promotion logic) acquires a write guard at most once per promotion.
pub struct ChampionStore {
    inner: RwLock<Arc<dyn Evaluator>>,
    pub champion_version: std::sync::atomic::AtomicU32,  // monotonically increasing
}

impl ChampionStore {
    pub fn new(seed: Arc<dyn Evaluator>) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(seed),
            champion_version: std::sync::atomic::AtomicU32::new(0),
        })
    }

    pub fn current(&self) -> Arc<dyn Evaluator> {
        self.inner.read().unwrap().clone()
    }

    /// Atomically replace the champion. Returns new champion version number.
    pub fn promote(&self, new_champion: Arc<dyn Evaluator>) -> u32 {
        let mut guard = self.inner.write().unwrap();
        *guard = new_champion;
        self.champion_version.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
    }
}
```

Add to `src/selfplay/mod.rs`:
```rust
pub mod champion;
pub use champion::ChampionStore;
```

**Tests**: `test_champion_store_swap` — construct with `RandomEvaluator`, promote to a
second `RandomEvaluator`, verify `champion_version` increments.

**Dependencies**: none.

---

### 3. Add second `InferenceServer` for champion weights (Python side + Rust batcher)

**Files**: `src/bin/selfplay.rs`, `python/hyzero/inference/server.py`

**Python side**: No changes to `server.py` are needed. `InferenceServer` already supports
`load_weights(state_dict_bytes: bytes)`. We instantiate a second Python `InferenceServer`
object in `selfplay.rs` for the champion. It is constructed once and then updated only on
promotion.

**Rust side in `selfplay.rs`**:

At startup, create the champion batcher backed by `RandomBackend` (no Python object yet):

```rust
// Champion batcher starts with random backend (pre-promotion)
let (champion_tx, champion_rx) = mpsc::channel::<InferenceRequest>(64);
let champion_backend = Box::new(RandomBackend::new(hidden_channels));
let mut champion_batcher = InferenceBatcher::new(champion_rx, champion_backend,
    BatcherConfig { max_batch_size: 8, batch_timeout_ms: 10 });
tokio::spawn(async move { champion_batcher.run().await; });

let champion_evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
let champion_store = ChampionStore::new(champion_evaluator);
```

At promotion time (inside the eval task), create the Python `InferenceServer` if not yet
created, load `best.pt`, and push a new `ChannelEvaluator(champion_tx)` into the store.
The champion batcher's backend cannot be hot-swapped (it is moved into the batcher loop).

**Resolution**: Use a double-indirection approach — the champion batcher's backend is a
`SwappableBackend` that wraps `Arc<RwLock<Box<dyn InferenceBackend>>>`. Promotion writes
a new `PyO3Backend` into this slot. See Subtask 4 for the swap protocol.

**New type** (`src/selfplay/inference.rs`):

```rust
/// Backend that delegates to an inner backend which can be swapped at runtime.
pub struct SwappableBackend {
    inner: Arc<std::sync::Mutex<Box<dyn InferenceBackend>>>,
}
impl SwappableBackend {
    pub fn new(initial: Box<dyn InferenceBackend>) -> (Self, Arc<std::sync::Mutex<Box<dyn InferenceBackend>>>) {
        let slot = Arc::new(std::sync::Mutex::new(initial));
        (Self { inner: slot.clone() }, slot)
    }
}
impl InferenceBackend for SwappableBackend {
    fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) {
        self.inner.lock().unwrap().evaluate_batch(requests);
    }
}
```

The `slot` Arc is passed to the eval task so it can replace the backend on promotion.

**Tests**: `test_swappable_backend_switches` — start with `RandomBackend`, send a request
(expect random reply), swap to another `RandomBackend`, send again (expect reply). Verifies
lock/swap mechanics without Python.

**Dependencies**: Subtask 2 must complete first.

---

### 4. Checkpoint snapshot and atomic promotion protocol

**Files**: `src/selfplay/evaluation.rs` (new `PromotionProtocol` impl), `src/bin/selfplay.rs`

**Promotion trigger**: After every M challenger games vs current champion, compute:
```
win_rate = (wins + 0.5 * draws) / (2 * M)
```
If `win_rate >= PROMOTION_THRESHOLD` AND `games_since_last_promotion >= MIN_COOLDOWN_GAMES`:
promote.

**Threshold recommendation**: 55% over 40 games (20 as White + 20 as Black).
- Binomial significance: with p=0.55 vs H0: p=0.5, P(X >= 24 | n=40) ≈ 0.12. Single-tail
  p<0.05 would require ~26/40 wins (65%). For our tight budget, 55% over 40 is a pragmatic
  threshold — it filters obvious regressions without demanding statistical certainty.
- Rationale: 40 games × ~4s each = ~160s per eval cycle, well within budget. AlphaZero's
  400-game threshold is tuned for production; 40 games gives enough signal to distinguish
  RandomEvaluator-level (near 100%) from a genuine improvement over a neural net champion.

**Constants** (tunable via env var):
```rust
const PROMOTION_THRESHOLD: f64 = 0.55;
const EVAL_GAMES_PER_SIDE: usize = 20;          // HYZERO_EVAL_GAMES_PER_SIDE
const MIN_COOLDOWN_GAMES: usize = 80;           // HYZERO_EVAL_COOLDOWN_GAMES
const STALL_WARN_CYCLES: usize = 20;            // warn after 20 non-promoting cycles
```

**Atomic file replacement**:
```rust
fn snapshot_checkpoint(src_path: &Path) -> std::io::Result<()> {
    let tmp = Path::new("checkpoints/best.pt.tmp");
    let dst = Path::new("checkpoints/best.pt");
    std::fs::copy(src_path, tmp)?;
    // fsync the tmp file to ensure durability before rename
    let file = std::fs::OpenOptions::new().write(true).open(tmp)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(tmp, dst)?;  // atomic on POSIX
    Ok(())
}
```

`src_path` is the most recent checkpoint written by `PyTrainingThread` (read from a
shared `Arc<Mutex<Option<PathBuf>>>` called `latest_checkpoint`). The training loop
updates this after every successful checkpoint save. The eval task reads it under lock
before snapshotting — this ensures we never copy a partially-written file.

**Champion server warm-up at promotion**:
```rust
// On promotion event, inside the eval task:
Python::attach(|py| {
    // If champion_server not yet created, create it now
    if champion_py_server.is_none() {
        let cls = PyModule::import(py, "hyzero.inference.server")
            .unwrap().getattr("InferenceServer").unwrap();
        let srv = cls.call1((config_obj.clone(), "cpu")).unwrap().unbind();
        *champion_py_server = Some(srv);
        // Swap the champion batcher backend to use the new Python server
        let new_backend: Box<dyn InferenceBackend> = Box::new(
            PyO3Backend::new(champion_py_server.as_ref().unwrap().clone_ref(py), hidden_channels)
        );
        *champion_backend_slot.lock().unwrap() = new_backend;
    }
    // Load best.pt weights into the champion server
    let path = "checkpoints/best.pt";
    let bytes = std::fs::read(path).unwrap();
    let py_bytes = PyBytes::new(py, &bytes);
    champion_py_server.as_ref().unwrap()
        .call_method1(py, "load_weights", (py_bytes,)).unwrap();
});
```

After the load, update `champion_store` with `ChannelEvaluator::new(champion_tx.clone())`.
In-flight eval games that started before the swap complete with the old backend (the old
`ChannelEvaluator` still holds a sender to the old batcher queue — but the batcher itself
is still running and will flush its queue). No in-flight game sees a mixed-weight mid-game.

**Versioning**: Each champion gets a tagged copy:
```rust
let tag_path = format!("checkpoints/best_v{:03}.pt", new_champion_version);
std::fs::copy("checkpoints/best.pt", &tag_path).ok(); // non-fatal if fails
```
Keep `checkpoint_keep_best: usize = 5` — prune oldest `best_vNNN.pt` files beyond the
window. Rolling logic matches the existing training checkpoint pruning.

**Anti-churn cooldown**: `MIN_COOLDOWN_GAMES` (default 80 = 2 full eval cycles) prevents
back-to-back promotions. The eval task tracks `games_since_last_promotion: usize` and
only promotes if it exceeds the cooldown.

**Stall detection**: If `stall_cycles` (non-promoting eval cycles) exceeds `STALL_WARN_CYCLES`,
log:
```
[eval] WARNING: no promotion in {stall_cycles} eval cycles — training may be stuck
```
This is observable in the run log and can be monitored by the orchestrator.

**Tests**:
- `test_snapshot_checkpoint_atomic` — write a temp file, call `snapshot_checkpoint`, verify
  `best.pt` exists and `best.pt.tmp` does not.
- `test_promotion_threshold` — unit test the win-rate comparison logic.

**Dependencies**: Subtasks 1, 2, 3.

---

### 5. Rework `EvaluationTask` into continuous champion-challenger loop

**Files**: `src/selfplay/evaluation.rs`

**New `EvaluationTask` struct**:

```rust
pub struct EvaluationTask {
    precomputed: Arc<PrecomputedItems>,
    challenger_evaluator: Arc<dyn Evaluator>,  // live model (ChannelEvaluator → live batcher)
    champion_store: Arc<ChampionStore>,
    model_version_rx: watch::Receiver<u64>,
    latest_checkpoint: Arc<Mutex<Option<PathBuf>>>,  // set by training thread after each ckpt save
    champion_backend_slot: Arc<Mutex<Box<dyn InferenceBackend>>>,  // for SwappableBackend swap
    champion_tx: mpsc::Sender<InferenceRequest>,  // to champion batcher
    config: EvaluationConfig,
    // Transient state (not in struct — local to `run`):
    //   champion_py_server: Option<Py<PyAny>>
    //   games_since_last_promotion: usize
    //   stall_cycles: usize
}
```

**New `EvaluationConfig`**:

```rust
pub struct EvaluationConfig {
    pub eval_games_per_side: usize,       // default 20 (20W + 20B = 40 total)
    pub promotion_threshold: f64,          // default 0.55
    pub min_cooldown_games: usize,         // default 80
    pub stall_warn_cycles: usize,          // default 20
    pub num_simulations: u32,             // MCTS sims per eval game (same as before)
    pub temperature_moves: u32,
}
```

Remove `eval_interval_steps` and `eval_games` — the eval task now runs continuously,
not on a version-threshold trigger.

**`EvaluationTask::run` loop**:

```
loop:
  champion_eval = champion_store.current()   // clone Arc cheaply
  play 20 games as White (challenger=White, champion=Black)
  play 20 games as Black (challenger=Black, champion=White)
  compute win_rate (challenger-perspective)
  log: [eval] v{version} champion_v{cv} wins={} draws={} losses={} win_rate={:.3} challenge

  if win_rate >= PROMOTION_THRESHOLD and cooldown satisfied:
    snapshot best.pt (atomic)
    load into champion Python server (GIL once)
    champion_store.promote(ChannelEvaluator(champion_tx))
    log: [eval] PROMOTION champion_v{new_cv} from training_v{version}
    reset stall_cycles = 0
    games_since_last_promotion = 0
  else:
    stall_cycles += 1
    games_since_last_promotion += 40
    if stall_cycles >= STALL_WARN_CYCLES:
      log: [eval] WARNING: no promotion in {stall_cycles} cycles

  // No sleep — immediately start next cycle
```

**`latest_checkpoint` handoff**: The training loop (in `PyTrainingThread::run`) must
update `latest_checkpoint` after each checkpoint save. Add a parameter to `PyTrainingThread`
or pass it via a dedicated watch channel:

```rust
// In PyTrainingThread, after successful save:
let mut latest = latest_checkpoint.lock().unwrap();
*latest = Some(path);
```

This is the cleanest interface — eval reads `latest_checkpoint` once per promotion rather
than every cycle (no hot path).

**Remove** the old version-threshold watcher loop — the new loop is unconditionally
continuous.

**Keep** the self-play `[eval]` log lines (`white_wins`, `decisive_ratio`, etc.) as a
separate periodic report for observability. Retain `play_game`-based self-play eval at
`eval_interval_steps` alongside the new challenge log. This ensures backward log
compatibility while the ladder is being established. Cut this once the ladder is proven.

**Tests**:
- `test_challenge_cycle_completes`: create `EvaluationTask` with two `RandomEvaluator`s
  (challenger and champion both random). Run one cycle (eval_games_per_side=1). Verify
  log output contains `challenge` tag. No promotion expected (win_rate ≈ 0.5, below 0.55).
- `test_promotion_triggers`: set `promotion_threshold=0.0` (always promote). Verify
  `champion_store.champion_version` increments after one cycle with stub evaluators.

**Dependencies**: Subtasks 1, 2, 3, 4.

---

### 6. Wire new design in `selfplay.rs`

**Files**: `src/bin/selfplay.rs`

**Changes**:

1. Create `latest_checkpoint: Arc<Mutex<Option<PathBuf>>>`. Pass into `PyTrainingThread`
   as an additional constructor argument (or pass separately to `run()`).

2. Spawn `max_concurrent_games - 1` self-play tasks instead of `max_concurrent_games`.
   Reserve one slot for the eval task. Change `SelfPlayConfig::max_concurrent_games` to
   read from env `HYZERO_GAMES` but subtract 1 internally before spawning:
   ```rust
   let selfplay_concurrent = config.max_concurrent_games.saturating_sub(1).max(1);
   ```
   Document this in a comment.

3. Create the champion batcher with `SwappableBackend`:
   ```rust
   let (champion_tx, champion_rx) = mpsc::channel(64);
   let (swappable, slot) = SwappableBackend::new(Box::new(RandomBackend::new(hidden_channels)));
   let mut champion_batcher = InferenceBatcher::new(champion_rx, Box::new(swappable),
       BatcherConfig { max_batch_size: 8, batch_timeout_ms: 10 });
   tokio::spawn(async move { champion_batcher.run().await; });
   ```

4. Create `ChampionStore` seeded with `Arc::new(RandomEvaluator)`.

5. Build `EvaluationTask` with the new fields. Spawn as:
   ```rust
   tokio::spawn(async move { eval_task.run().await; });
   ```

6. Add new `RunConfig` fields:
   ```rust
   eval_games_per_side: usize,     // HYZERO_EVAL_GAMES_PER_SIDE, default 20
   promotion_threshold: f64,        // HYZERO_PROMOTION_THRESHOLD, default 0.55
   min_cooldown_games: usize,       // HYZERO_EVAL_COOLDOWN_GAMES, default 80
   ```

Remove old `eval_interval_steps`, `eval_games`, `eval_num_simulations` from `RunConfig`
(they are replaced — but keep `eval_num_simulations` → rename to `eval_sims` for clarity).

**Dependencies**: Subtasks 2, 3, 4, 5.

---

### 7. Update `scripts/run_baseline.sh` — ladder metric extraction

**Files**: `scripts/run_baseline.sh`

**New primary metric**: `champion_version` (number of promotions during the run).

**Extract champion version** from `[eval] PROMOTION` lines:
```bash
CHAMPION_VERSION=$(awk '/\[eval\].*PROMOTION/{n++} END{print n+0}' "$LOG_FILE")
```

**Extract challenge win rate** (last eval cycle before run ends, for observability):
```bash
LAST_WIN_RATE=$(awk '/\[eval\].*challenge/{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^win_rate=/) { split($i, a, "="); wr = a[2] }
    }
} END{ print wr+0 }' "$LOG_FILE")
LAST_WIN_RATE=${LAST_WIN_RATE:-0.0}
```

**New score formula**:
```
score = (8.55 - final_policy_loss) + (champion_version * 2.0) - (avg_game_length / 100)
```
Rationale: each promotion is worth 2.0 score points. A run achieving 3 promotions
(+6.0) vs 1 promotion (+2.0) cleanly separates experiments by training progress rate.
The multiplier 2.0 is chosen so that 3-5 promotions in a typical 1800s run produce
a range of 6-10, comparable in magnitude to the policy-loss component (~5.0 at baseline).

**Also keep `decisive_ratio`** in JSON output for backward observability:
```json
"metrics": {
    ...existing...,
    "champion_version": $CHAMPION_VERSION,
    "last_win_rate_vs_champion": $LAST_WIN_RATE,
    "decisive_ratio": $DECISIVE_RATIO        // kept for backward audit
}
```

**Update CLAUDE.md metric section**: formula → `(8.55 - final_policy_loss) + (champion_version * 2.0) - (avg_game_length / 100)`, baseline = "TBD after ladder lands".

**Dependencies**: Subtask 5 (log format established).

---

### 8. Smoke test

**Files**: `scripts/smoke_dual_eval.sh` (new file)

```bash
#!/usr/bin/env bash
# Smoke test: run selfplay for 120s and verify challenge log line appears.
set -euo pipefail
TIMEOUT=120
LOG=$(mktemp /tmp/hyzero_smoke_XXXXXX.log)

HYZERO_GAMES=2 \
HYZERO_EVAL_GAMES_PER_SIDE=1 \
HYZERO_EVAL_SIMS=2 \
HYZERO_PROMOTION_THRESHOLD=0.0 \
target/release/selfplay > "$LOG" 2>&1 &
PID=$!
sleep "$TIMEOUT"
kill -TERM $PID 2>/dev/null || true
wait $PID 2>/dev/null || true

if grep -q 'PROMOTION' "$LOG"; then
    echo "PASS: promotion log line found"
    rm "$LOG"
    exit 0
else
    echo "FAIL: promotion not found in log"
    cat "$LOG"
    rm "$LOG"
    exit 1
fi
```

`HYZERO_PROMOTION_THRESHOLD=0.0` forces promotion on the first eval cycle regardless of
win rate — tests the full promotion pathway without needing the challenger to actually
beat the champion.

**Dependencies**: Subtasks 1-7.

---

## Promotion Protocol (summary)

| Parameter | Value | Env var |
|-----------|-------|---------|
| Games per eval cycle | 40 (20W + 20B) | `HYZERO_EVAL_GAMES_PER_SIDE=20` |
| Promotion threshold | 55% win rate | `HYZERO_PROMOTION_THRESHOLD=0.55` |
| Cooldown | 80 games between promotions | `HYZERO_EVAL_COOLDOWN_GAMES=80` |
| Stall warning | 20 non-promoting cycles | hardcoded constant |
| Checkpoint save | atomic rename via `.tmp` | |
| Champion file | `checkpoints/best.pt` | |
| Champion archive | `checkpoints/best_v{N:03}.pt`, keep 5 | |

**Binomial note**: At 40 games and p=0.5 null, P(wins >= 22) ≈ 0.24 (one-tailed). This
means a 55% threshold with 40 games has roughly 24% false-positive rate per cycle. This
is acceptable because: (a) the seed champion is RandomEvaluator — the first promotion
happens at ~100% win rate, not 55%, so the statistical question only arises for
neural-net-vs-neural-net transitions; (b) the cooldown and anti-churn logic prevent
oscillation even if an occasional false promotion occurs; (c) for a 1800s training run,
precision matters less than sensitivity (we want to detect real progress, not miss it).
Use `HYZERO_EVAL_GAMES_PER_SIDE=50` for validation runs when tighter statistical
guarantees are needed (55% over 100 games is one-tailed p ≈ 0.14, adequate).

---

## Scoring Formula

### Formula

```
score = (8.55 - final_policy_loss) + (champion_version * 2.0) - (avg_game_length / 100)
```

### Why `champion_version` and not `win_rate_vs_champion`

`win_rate_vs_champion` hovers near 50% at equilibrium: a challenger that is slightly
stronger than the champion wins 55-60%, then becomes the new champion, and the next
challenger (the same improving training run) wins 55-60% again. In steady-state, the
metric is nearly constant regardless of absolute model strength improvement. This is the
same symmetry-collapse problem as `decisive_ratio`, just one level up.

`champion_version` (= number of successful promotions) does not collapse. Each promotion
represents a verified step of progress (the challenger beat the champion at the threshold).
Faster learning → more promotions per 1800s → higher score. The metric is monotonically
correlated with progress speed.

### Scale calibration

At baseline, expect 2-4 promotions in 1800s:
- 0 promotions: model never beats RandomEvaluator (severe failure, score = policy_loss_term)
- 1 promotion: beats random, then plateaus (early training)
- 3 promotions: active learning throughout
- 5+ promotions: strong fast learner

With multiplier 2.0: range 0-10 added, comparable to the policy-loss term (~4-5 at baseline).

### Historical baseline incompatibility

This formula is **not comparable** to the old `decisive_ratio`-based score (6.78 baseline).
Re-establish baseline from scratch after the ladder lands. Update `CLAUDE.md` and
`logs/baseline_score.json`.

---

## Risk / Failure Modes

### Risk 1: Stalled promotion (model never beats RandomEvaluator)

If the model with a dead value head cannot beat random, `champion_version` stays at 0.
Score = `(8.55 - policy_loss) - (avg_length/100)` ≈ 3.5-5.0, far below any model that
achieves even one promotion. This is a useful diagnostic signal, not a silent failure.
Mitigation: stall warning logs after 20 cycles; orchestrator escalates.

### Risk 2: Champion inference weight-swap race

The training loop writes `checkpoints/model_vNNNNNN.pt` while the eval task may be
reading the most recent path from `latest_checkpoint`. Race resolution: the training loop
updates `latest_checkpoint` only after a successful `fsync` + close of the checkpoint file.
The eval task reads `latest_checkpoint` under mutex only at promotion time, never during
active game play. Two concurrent promotions cannot occur because there is exactly one eval
task.

### Risk 3: In-flight eval games during promotion

Games that are already in progress when a promotion triggers continue to completion using
their pre-promotion evaluator clones (they hold their own `Arc<dyn Evaluator>` through the
game). The `ChampionStore` write guard affects only the next call to `champion_store.current()`,
which happens at the start of the next eval cycle. No game sees mid-game weight changes.

### Risk 4: RandomEvaluator-as-seed edge cases

At startup, `champion_store` holds `Arc::new(RandomEvaluator)`. The eval task calls
`champion_store.current()` and gets `RandomEvaluator`. `play_game_dual` dispatches it as
Black (or White) — it issues `root_setup` calls, returning uniform policy and zero value.
MCTS with a uniform prior degrades gracefully to near-uniform action selection. No panic,
no GIL access, no Python dependency. The first promotion transitions from `RandomEvaluator`
to `ChannelEvaluator(champion_tx)`, which routes through `SwappableBackend` (now backed by
`PyO3Backend` pointing at `ChampionInferenceServer`). The swap is transparent to callers.

### Risk 5: Dead value head interaction with ladder

With `value=0.0000` (confirmed self-referential bootstrap bug), MCTS degrades to prior
sampling for the challenger. The challenger may still beat RandomEvaluator (policy prior
learns to avoid obvious blunders), but progression through neural-net champions will
be slow because the value head provides no tactical depth signal. The ladder metric
is still useful: it distinguishes "beats random" (1-2 promotions) from "beats prior
champion" (3+ promotions). The value-head fix (outcome targets) is a separate experiment
that will amplify the benefit of the ladder, not a prerequisite.

### Risk 6: PyO3 `champion_server` creation at promotion time

The first `champion_py_server` is created inside the eval tokio task by calling
`Python::attach(|py| ...)`. This is the same pattern as `selfplay.rs` startup. GIL is
held only for the duration of the attach call; it does not block the challenger batcher
(which runs in a separate blocking thread via PyO3's spawn_blocking under the hood).
Acquisition latency: ~1-2ms. This happens at most once per promotion (rare event).

### Risk 7: Separate champion batcher adds GPU memory if both use CUDA

On CPU (current default), two `InferenceServer` instances consume 2× model RAM
(~10-50MB depending on hidden_channels). On CUDA, if ever used, the champion server
holds frozen weights in VRAM. Accept: the champion server is small (eval only, not
actively training) and CUDA is not the current target device.

---

## Files Modified

| File | ~LOC changed | Language | Scope |
|------|-------------|----------|-------|
| `src/selfplay/game_task.rs` | +60 (new function + 2 tests) | Rust | New `play_game_dual` |
| `src/selfplay/champion.rs` | +50 (new file) | Rust | `ChampionStore` abstraction |
| `src/selfplay/evaluation.rs` | +120 (full eval loop rewrite + 2 tests) | Rust | Champion-challenger loop |
| `src/selfplay/inference.rs` | +40 (`SwappableBackend` + 1 test) | Rust | Hot-swappable backend |
| `src/selfplay/mod.rs` | +3 (re-exports) | Rust | Module wiring |
| `src/py/training.rs` | +10 (pass `latest_checkpoint` to training thread) | Rust | Checkpoint path sharing |
| `src/bin/selfplay.rs` | +60 (champion batcher, store, new config fields) | Rust | Binary wiring |
| `scripts/run_baseline.sh` | +25 (new extraction + formula) | Bash | Score extraction |
| `CLAUDE.md` | +5 (metric update) | Markdown | Documentation |
| `scripts/smoke_dual_eval.sh` | +30 (new file) | Bash | Smoke test |

Total Rust: ~343 LOC. Total shell: ~55 LOC. No Python changes. No `Cargo.lock`,
`pyproject.toml`, or `docs/wiki/` changes.

---

## Time Budget Impact

**Standing cost per 1800s run**:

The eval task runs continuously (not per-cycle triggered). With `HYZERO_EVAL_GAMES_PER_SIDE=20`:
- 40 games per cycle × ~4s/game = ~160s per cycle
- Self-play has 1 fewer concurrent game (N-1 instead of N). With N=4 default, 3 self-play
  games now run. Throughput reduction: ~25% fewer self-play games per unit time.
- Eval cycle frequency: 1 cycle per ~160s → ~11 eval cycles in 1800s

**Trade-off**: fewer self-play games → slower replay buffer fill → fewer training steps.
But eval quality is higher (40 games gives cleaner signal vs prior 10 games). This is a
deliberate trade: we are optimizing for promotion signal quality over raw training throughput.

**If budget is tight**: Set `HYZERO_GAMES=5` (4 self-play + 1 eval) or reduce
`HYZERO_EVAL_GAMES_PER_SIDE=10` (20 games per cycle, ~80s per cycle). The eval task
cost is the dominant new overhead; self-play games cost is reduced, not added.

**Constraint compliance**: The +10 min cap applies to *added* wall-clock vs the prior
design (which had 10 self-play eval games per cycle, adding ~2-3 min). The champion-
challenger ladder replaces those 10 games with 40 challenger games — net added time is
~(40-10) × 4s × cycles = ~30s × 11 cycles = ~5.5 min. Within the +10 min cap.

---

## Open Questions for the User

1. **N=4 game slot allocation**: The plan reserves 1 slot for eval, leaving 3 for
   self-play. Should `HYZERO_GAMES` default stay at 4 (= 3 self-play + 1 eval) or should
   it be bumped to 5 to maintain 4 concurrent self-play games? The default environment
   has limited parallelism; bumping to 5 may increase GIL contention.

2. **Promotion threshold for first neural-net-vs-neural-net match**: 55%/40 games is
   calibrated for detecting genuine improvement. For the very first neural-net-vs-neural-net
   promotion (champion_v1 vs challenger), should the threshold be higher (e.g., 60%) since
   the champion is still weak and noise is high? Or is a flat 55% for all promotions acceptable?

3. **Archive policy**: Keep last 5 `best_vNNN.pt` files by default. Is rollback to a
   prior champion ever needed? If yes, should we keep more (e.g., 10)? These files are
   ~5-50MB each depending on model size.

4. **Scoring formula multiplier**: `champion_version * 2.0` is calibrated for an expected
   3-5 promotions per 1800s run. If in practice the model promotes very slowly (0-1 per run),
   the multiplier may need to increase. Should the formula be finalized after first baseline,
   or locked in now?
