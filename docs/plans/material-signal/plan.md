# Plan: Material Signal — Cap Outcome and Adjudication

## Approach

Replace the 300-move-cap draw (`game_outcome = 0`) with a material-based signal (`tanh(Δmaterial/5.0)`), and add early-game adjudication when one side is materially dominant for 10 consecutive plies. Both changes are confined to `src/selfplay/game_task.rs` with a helper function added in the same file. A new `scripts/run_training.sh` enables long "keep training" runs without checkpoint cleanup.

---

## Section 1: Current State

### Outcome-writing locations

**`src/selfplay/game_task.rs`** — `play_game()` function:

- Line 174–176: move-cap break
  ```rust
  if turn_count >= MAX_GAME_LENGTH {
      // Treat as draw — prevents runaway games
      break;
  }
  ```
- Lines 258–262: outcome assigned (White-absolute convention)
  ```rust
  let game_outcome = match board.result() {
      GameResult::Checkmate(Color::White) => 1.0,   // White won
      GameResult::Checkmate(Color::Black) => -1.0,  // Black won
      _ => 0.0,                                     // Draw or other
  };
  ```
- Lines 265–267: last step's `reward` field set to `game_outcome`
- Line 269–274: `GameTrajectory { steps, game_outcome, model_version }` returned

**`src/selfplay/game_task.rs`** — `play_game_dual()` function (eval only, no training buffer):

- Line 74: same `MAX_GAME_LENGTH` constant (also `const MAX_GAME_LENGTH: usize = 300`)
- Lines 133–137: same White-absolute outcome pattern, also falls to `0.0` on cap

**The move cap is defined locally** (not in `GameConfig` or `SelfPlayConfig`) — two `const MAX_GAME_LENGTH: usize = 300` instances, one in each function.

### Sign convention — CRITICAL

`game_outcome` in `GameTrajectory` is **absolute White-perspective**: +1 = White wins, −1 = Black wins, 0 = draw.

The trainer (`src/py/training.rs:112–136`) converts this to side-to-move perspective **at training time** using observation plane 101:
```rust
let root_white_to_move = steps[0].observation.planes[101 * 64] > 0.5;
let root_side_sign: f32 = if root_white_to_move { 1.0 } else { -1.0 };
// ...
let ply_flip: f32 = if k % 2 == 0 { 1.0 } else { -1.0 };
let outcome_in_step_perspective = sample.game_outcome * root_side_sign * ply_flip;
```

**Conclusion**: `game_outcome` in `GameTrajectory` must remain in absolute White-perspective. The trainer handles the per-step perspective flip. The material-based cap outcome must follow the same convention: positive = White material advantage.

The wiki mistakes log entry at line 394–414 that said this conversion "is NOT done anywhere" is **stale** — the fix is already in `training.rs:136`. The wiki needs updating (flag for context-keeper).

### Outcome conventions: how many exist?

Two conventions exist in the codebase:
1. **Absolute White-perspective**: used in `GameTrajectory.game_outcome` and everywhere in `game_task.rs`
2. **Side-to-move perspective**: derived in `training.rs` at batch assembly time from plane 101

There is no third convention. The new `compute_material_diff` return value must be in absolute White-perspective (positive = White ahead), matching convention 1.

### Material counting helpers

There are **no existing material-counting helpers** in `src/game/` or `src/pieces/`. The `is_insufficient_material()` method in `src/game/board.rs:1047` counts pieces for draw detection but does not compute a signed material balance.

Access pattern needed: `board.player1.pieces_bb[PieceType::X as usize].count_ones()` for White, `board.player2.pieces_bb[PieceType::X as usize].count_ones()` for Black. The `PieceType` enum order is `[Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5]` with piece values `[1, 3, 3, 5, 9, 0]`.

`board.player1` = White, `board.player2` = Black (confirmed by `Player::init_player(true)` = White in `game_task.rs:153–154`).

---

## Section 2: Code Changes

### Helper function — `compute_material_diff`

Add to the bottom of `src/selfplay/game_task.rs`, above the `#[cfg(test)]` block:

```rust
/// Compute material balance (White - Black) in centipawn equivalents.
/// Standard piece values: P=1, N=3, B=3, R=5, Q=9, K=0.
/// Returns positive if White is ahead, negative if Black is ahead.
/// Uses the bitboard count from player1 (White) and player2 (Black).
fn compute_material_diff(board: &GameBoard) -> i32 {
    const VALUES: [i32; 6] = [1, 3, 3, 5, 9, 0]; // Pawn,Knight,Bishop,Rook,Queen,King
    let mut white_mat = 0i32;
    let mut black_mat = 0i32;
    for (i, &val) in VALUES.iter().enumerate() {
        white_mat += val * board.player1.pieces_bb[i].count_ones() as i32;
        black_mat += val * board.player2.pieces_bb[i].count_ones() as i32;
    }
    white_mat - black_mat
}
```

Note: `pieces_bb` is `[u64; 6]` indexed 0..5 as Pawn..King. `count_ones()` = popcount. This function does not require `pub` — it stays file-private.

### Edit 1: Add adjudication state to `play_game()`

**File**: `src/selfplay/game_task.rs`
**Location**: inside `play_game()`, after `let mut turn_count: usize = 0;` (line 158)

Add two state variables:
```rust
let mut adj_counter: u32 = 0; // Consecutive plies with |Δmaterial| >= 6
const ADJ_THRESHOLD: i32 = 6;
const ADJ_PLIES: u32 = 10;
```

### Edit 2: Move-cap with material outcome (replaces the draw)

**File**: `src/selfplay/game_task.rs`
**Location**: `play_game()`, replace lines 174–176:

OLD:
```rust
        if turn_count >= MAX_GAME_LENGTH {
            // Treat as draw — prevents runaway games
            break;
        }
```

NEW (keep comment, change semantics):
```rust
        if turn_count >= MAX_GAME_LENGTH {
            // Material-based outcome at cap: tanh(Δ/5) in White-absolute convention.
            // Trainer converts to side-to-move perspective via plane 101.
            break; // game_outcome will be set from material after the loop
        }
```

(The actual outcome value is set post-loop — see Edit 4.)

### Edit 3: Adjudication check inside the game loop

**File**: `src/selfplay/game_task.rs`
**Location**: `play_game()`, immediately after the `turn_count >= MAX_GAME_LENGTH` check, before the `encode_board` call:

```rust
        // Adjudication: end early if one side is materially dominant for N consecutive plies.
        {
            let delta = compute_material_diff(&board);
            if delta.abs() >= ADJ_THRESHOLD {
                adj_counter += 1;
                if adj_counter >= ADJ_PLIES {
                    // Adjudicated: use sign of material advantage as outcome
                    // (White-absolute convention, matching checkmate outcome encoding)
                    let adj_outcome = if delta > 0 { 1.0f32 } else { -1.0f32 };
                    // Apply outcome to last step reward and return early
                    if let Some(last) = steps.last_mut() {
                        last.reward = adj_outcome;
                    }
                    eprintln!(
                        "[selfplay] adjudicated turn={} delta={} adj_outcome={}",
                        turn_count, delta, adj_outcome
                    );
                    return GameTrajectory {
                        steps,
                        game_outcome: adj_outcome,
                        model_version,
                    };
                }
            } else {
                adj_counter = 0; // Reset if dominance drops below threshold
            }
        }
```

**Adjudication state machine**:
- `adj_counter` starts at 0.
- Each ply: compute `delta = white_material - black_material`.
- If `|delta| >= 6`: increment `adj_counter`.
- If `|delta| < 6`: reset `adj_counter = 0` (must be **sustained** for 10 plies; any ply where material dips below 6 resets the counter).
- If `adj_counter >= 10`: fire adjudication immediately (do not wait for the ply to complete).
- Adjudicated outcome: `+1.0` if white ahead, `-1.0` if black ahead. White-absolute convention.

The adjudication check must happen **before** the MCTS and move-application code, so we can return early without a partially-recorded step. It checks the position as entered, not after the move.

### Edit 4: Post-loop material outcome (replacing draw-at-cap)

**File**: `src/selfplay/game_task.rs`
**Location**: `play_game()`, replace lines 258–262:

OLD:
```rust
    // Determine game outcome
    let game_outcome = match board.result() {
        GameResult::Checkmate(Color::White) => 1.0,  // White won
        GameResult::Checkmate(Color::Black) => -1.0, // Black won
        _ => 0.0,                                    // Draw or other
    };
```

NEW:
```rust
    // Determine game outcome.
    // For genuine checkmates/stalemates/draws: use rule-based result.
    // For 300-move cap: substitute material-based proxy (White-absolute convention).
    let game_outcome = match board.result() {
        GameResult::Checkmate(Color::White) => 1.0,
        GameResult::Checkmate(Color::Black) => -1.0,
        GameResult::Ongoing => {
            // Hit the 300-move cap. Use tanh(Δmaterial / 5.0) as outcome signal.
            let delta = compute_material_diff(&board);
            (delta as f32 / 5.0).tanh()
        }
        _ => 0.0, // Stalemate, 50-move, insufficient material, threefold
    };
```

Note: `board.result()` returns `GameResult::Ongoing` when the loop exits via the move-cap break (no terminal condition was set). All other exits — checkmate, stalemate, etc. — set the game_result before breaking.

The `last.reward = game_outcome` assignment on lines 265–267 is **unchanged** — it already reads from `game_outcome` after it's computed.

### Adjudication logging convention

Use `eprintln!("[selfplay] adjudicated ...")` so it appears in stderr, matching the pattern of `[py_training]` and `[eval]` log prefixes used by the rest of the system. Grep pattern for log extraction: `\[selfplay\] adjudicated`.

### `play_game_dual()` — no changes needed

Eval games do not go to the training buffer. The cap-draw in `play_game_dual()` only affects win_rate counting. Adjudication could theoretically be added here too (to make eval games faster), but the task brief specifies only `play_game()` as target. Leave `play_game_dual()` unchanged.

---

## Section 3: New Script — `scripts/run_training.sh`

### Full content

```bash
#!/usr/bin/env bash
set -euo pipefail
set +m

# ── Configuration ──────────────────────────────────────────────
DURATION=${1:-7200}          # 2 hours default
SIMS=${HYZERO_SIMS:-80}
EVAL_SIMS=${HYZERO_EVAL_SIMS:-50}
GAMES=${HYZERO_GAMES:-8}
VALUE_BETA=${HYZERO_VALUE_OUTCOME_BETA:-0.3}
PROMOTION_THRESHOLD=${HYZERO_PROMOTION_THRESHOLD:-0.55}
CHAMPION_SCORE_WEIGHT=${HYZERO_CHAMPION_SCORE_WEIGHT:-2.0}
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
JSON_FILE="${LOG_DIR}/training_${TIMESTAMP}.json"

mkdir -p "$LOG_DIR"

echo "=== hyzero Training Run ==="
echo "Duration: ${DURATION}s"
echo "SIMS=${SIMS}, EVAL_SIMS=${EVAL_SIMS}, GAMES=${GAMES}, beta=${VALUE_BETA}"
echo "Log: ${LOG_FILE}"
echo "(Checkpoints are preserved — this is keep-training mode)"

# ── Build ──────────────────────────────────────────────────────
echo "[1/4] Building release binary..."
cargo build --release --bin selfplay 2>&1 | tail -1

# ── Run — NO checkpoint cleanup ────────────────────────────────
echo "[2/4] Running selfplay for ${DURATION}s (checkpoints preserved)..."
if [ -d checkpoints ]; then
    echo "  Existing checkpoints:"
    ls checkpoints/best*.pt 2>/dev/null || echo "  (none)"
fi

HYZERO_SIMS=$SIMS \
HYZERO_EVAL_SIMS=$EVAL_SIMS \
HYZERO_GAMES=$GAMES \
HYZERO_VALUE_OUTCOME_BETA=$VALUE_BETA \
HYZERO_PROMOTION_THRESHOLD=$PROMOTION_THRESHOLD \
HYZERO_CHAMPION_SCORE_WEIGHT=$CHAMPION_SCORE_WEIGHT \
target/release/selfplay > "$LOG_FILE" 2>&1 &
PID=$!
sleep "$DURATION"
kill -TERM $PID 2>/dev/null || true
sleep 2
kill -KILL $PID 2>/dev/null || true
set +e
wait $PID 2>/dev/null
set -e

# ── Extract metrics ────────────────────────────────────────────
echo "[3/4] Extracting metrics..."

GAMES_PLAYED=$(awk '/\[py_training\].*Game received/{n++} END{print n+0}' "$LOG_FILE")
TRAIN_STEPS=$(awk '/\[py_training\].*step [0-9]/{n++} END{print n+0}' "$LOG_FILE")
_LAST_TRAIN_LINE=$(awk '/\[py_training\].*step [0-9]/{line=$0} END{print line}' "$LOG_FILE")
LAST_POLICY=$(printf '%s\n' "$_LAST_TRAIN_LINE" | sed -n 's/.*policy=\([0-9.]*\).*/\1/p')
LAST_POLICY=${LAST_POLICY:-0.0}
LAST_LOSS=$(printf '%s\n' "$_LAST_TRAIN_LINE" | sed -n 's/.*total=\([0-9.]*\).*/\1/p')
LAST_LOSS=${LAST_LOSS:-0.0}

PROMOTIONS=$(awk '/\[eval\].*promoted/{n++} END{print n+0}' "$LOG_FILE")
MAX_CHAMPION_VERSION=$(awk '/\[eval\].*promoted/{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^champion_version=/) { split($i, a, "="); v=a[2]+0; if(v>max) max=v }
    }
} END{print max+0}' "$LOG_FILE")

# Adjudication rate: adjudicated games / total games
ADJUDICATED=$(awk '/\[selfplay\] adjudicated/{n++} END{print n+0}' "$LOG_FILE")
ADJ_RATE=$(awk "BEGIN { if ($GAMES_PLAYED > 0) printf \"%.4f\", $ADJUDICATED / $GAMES_PLAYED; else print \"0.0000\" }")

AVG_GAME_LEN=$(awk '/\[py_training\].*Game received/{split($0,a,"received: "); split(a[2],b," "); sum+=b[1]; n++} END{if(n>0) printf "%.1f", sum/n; else print "0"}' "$LOG_FILE")

ERRORS=$(awk 'tolower($0) ~ /error|panic/{n++} END{print n+0}' "$LOG_FILE")
GIT_COMMIT=$(git rev-parse --short HEAD)

echo ""
echo "=== Results ==="
echo "  Games played:        $GAMES_PLAYED"
echo "  Training steps:      $TRAIN_STEPS"
echo "  Final policy loss:   $LAST_POLICY"
echo "  Avg game length:     $AVG_GAME_LEN"
echo "  Promotions:          $PROMOTIONS"
echo "  Adjudicated games:   $ADJUDICATED (rate: $ADJ_RATE)"
echo "  Errors:              $ERRORS"

# ── Write JSON summary ─────────────────────────────────────────
echo "[4/4] Writing JSON summary..."
cat > "$JSON_FILE" << EOF
{
    "timestamp": "$TIMESTAMP",
    "git_commit": "$GIT_COMMIT",
    "duration_s": $DURATION,
    "metrics": {
        "games_played": $GAMES_PLAYED,
        "training_steps": $TRAIN_STEPS,
        "final_policy_loss": $LAST_POLICY,
        "final_total_loss": $LAST_LOSS,
        "avg_game_length": $AVG_GAME_LEN,
        "promotions": $PROMOTIONS,
        "max_champion_version": ${MAX_CHAMPION_VERSION:-0},
        "adjudicated_games": $ADJUDICATED,
        "adjudication_rate": $ADJ_RATE,
        "errors": $ERRORS
    },
    "config": {
        "sims": $SIMS,
        "eval_sims": $EVAL_SIMS,
        "concurrent_games": $GAMES,
        "value_outcome_beta": $VALUE_BETA,
        "promotion_threshold": $PROMOTION_THRESHOLD,
        "champion_score_weight": $CHAMPION_SCORE_WEIGHT
    },
    "log_file": "$LOG_FILE"
}
EOF

echo "  JSON written to: $JSON_FILE"
echo "  Log saved to: $LOG_FILE"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "WARNING: $ERRORS errors detected — check log"
fi

echo ""
echo "TRAINING RUN COMPLETE"
```

### Differences from `run_baseline.sh`

| Property | `run_baseline.sh` | `run_training.sh` |
|----------|-------------------|-------------------|
| Default duration | 1800s (30 min) | 7200s (2 hours) |
| Checkpoint cleanup | Deletes `model_v*.pt` at start | None — preserves all checkpoints |
| Sims | reads `HYZERO_SIMS` (default 40) | Sets `HYZERO_SIMS=80` |
| Eval sims | reads `HYZERO_EVAL_SIMS` (default 25) | Sets `HYZERO_EVAL_SIMS=50` |
| Concurrent games | reads `HYZERO_GAMES` (default 5) | Sets `HYZERO_GAMES=8` |
| beta | reads `HYZERO_VALUE_OUTCOME_BETA` (default 0.1) | Sets `VALUE_BETA=0.3` |
| Output file | `logs/baseline_score.json` (overwrites) | `logs/training_{timestamp}.json` (timestamped) |
| Score formula | Computes composite score | Does not compute score (not a controlled experiment) |
| Adjudication metric | Not present | Extracted from `[selfplay] adjudicated` log lines |

The summary JSON is produced by awk/sed parsing of the same log file format as `run_baseline.sh` — no separate metrics collection endpoint needed.

---

## Section 4: Tests

### 4.1 Unit test: `compute_material_diff`

**File**: `src/selfplay/game_task.rs`, inside `#[cfg(test)] mod tests`

```rust
#[test]
fn test_compute_material_diff_starting_position() {
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    let p1 = Player::init_player(true);
    let p2 = Player::init_player(false);
    let board = GameBoard::init_game_board(precomputed, p1, p2);
    // Starting position is symmetric — material diff should be 0.
    assert_eq!(compute_material_diff(&board), 0);
}

#[test]
fn test_compute_material_diff_asymmetric() {
    use crate::game::fen::board_from_fen;
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    // White has queen+king, Black has king only → Δ = 9
    let (board, _, _) =
        board_from_fen("8/8/8/8/8/8/8/K6k w - - 0 1", precomputed.clone())
            .expect("valid fen");
    // Wait — K vs K has no queen. Use a position with queen advantage.
    // White queen + king vs black king: "K6Q/8/8/8/8/8/8/7k w - - 0 1" — Δ = 9
    let (board2, _, _) =
        board_from_fen("K6Q/8/8/8/8/8/8/7k w - - 0 1", precomputed)
            .expect("valid fen");
    assert_eq!(compute_material_diff(&board2), 9);
}
```

**Note**: The FEN-parsing helper in `src/game/fen.rs` returns `(GameBoard, Color, u32)`. Use `board_from_fen` already imported in the existing test at line 663.

### 4.2 Unit test: adjudication state machine (logic only, no async)

Since adjudication is embedded in the async `play_game()` function, test via a helper that mirrors the counter logic:

```rust
#[test]
fn test_adjudication_counter_logic() {
    // Simulate the counter state machine: reset on drop below threshold
    const ADJ_THRESHOLD: i32 = 6;
    const ADJ_PLIES: u32 = 10;
    let deltas = [7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7]; // drops at index 3, resets
    let mut counter = 0u32;
    let mut adjudicated = false;
    for delta in deltas {
        if delta.abs() >= ADJ_THRESHOLD {
            counter += 1;
            if counter >= ADJ_PLIES {
                adjudicated = true;
                break;
            }
        } else {
            counter = 0;
        }
    }
    // Counter reset at index 3, then 7 more plies (indices 4-10) → counter=7, not 10
    assert!(!adjudicated, "Should not adjudicate: reset dropped counter");
    assert_eq!(counter, 7);

    // Now try without a break: 10 consecutive plies >= threshold
    let deltas2 = [8i32; 10];
    counter = 0;
    adjudicated = false;
    for delta in deltas2 {
        if delta.abs() >= ADJ_THRESHOLD {
            counter += 1;
            if counter >= ADJ_PLIES {
                adjudicated = true;
                break;
            }
        } else {
            counter = 0;
        }
    }
    assert!(adjudicated, "Should adjudicate after 10 consecutive plies");
}
```

### 4.3 Integration test: cap produces non-zero outcome

**File**: `src/selfplay/game_task.rs`, inside `#[cfg(test)]`

Add a test using `RandomEvaluator` with a tiny sim count and a crafted FEN where one side is a queen up. Because `RandomEvaluator` always plays randomly and the 300-move cap will eventually trigger in a capped test game (or the asymmetric position produces non-zero result at cap), this verifies the outcome is non-zero for asymmetric positions:

```rust
#[tokio::test]
async fn test_material_cap_outcome_nonzero_for_asymmetric_position() {
    // This test verifies the tanh(Δ/5) formula produces non-zero outcome at cap
    // by directly calling compute_material_diff on an asymmetric board.
    use crate::game::fen::board_from_fen;
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    // White ahead by a queen (Δ = 9) → tanh(9/5) ≈ 0.964
    let (board, _, _) =
        board_from_fen("K6Q/8/8/8/8/8/8/7k w - - 0 1", precomputed)
            .expect("valid fen");
    let delta = compute_material_diff(&board);
    let outcome = (delta as f32 / 5.0).tanh();
    assert!(outcome > 0.9, "Expected outcome ~0.96, got {outcome}");
    assert!(outcome <= 1.0, "Outcome must be tanh-bounded to <= 1.0");
}
```

### 4.4 Existing tests to verify

Run `cargo test` and confirm these still pass:
- `test_play_game_completes` — checks that `game_outcome` is `+1`, `-1`, or `0`. **This will still hold**: `tanh()` returns values in `(-1, 1)`, never `0` or `±1` exactly for `delta != 0`, but the test only checks membership in `{1.0, -1.0, 0.0}`. **This test will FAIL** for cap-terminated games because `tanh(non-zero) != 1.0, -1.0, or 0.0`. Update the assertion:
  ```rust
  assert!(
      trajectory.game_outcome.abs() <= 1.0,
      "Game outcome should be in [-1, 1], got {}",
      trajectory.game_outcome
  );
  ```
- `test_play_game_dual_completes` — eval games are unchanged, test still passes.
- `test_dual_game_outcome_sign_convention` — eval games are unchanged, still passes.
- `test_legal_moves_starting_position`, `test_legal_moves_promotion_position` — unrelated, pass.

---

## Section 5: Validation Approach

### 5.1 Smoke test: force adjudication at low threshold

Temporarily build with `ADJ_THRESHOLD = 2` and `ADJ_PLIES = 3` (or pass as env vars if implemented), run a 60s session, grep for adjudication events:

```bash
# Build with lowered threshold (edit constants in game_task.rs temporarily, or add env-var override)
HYZERO_SIMS=10 HYZERO_GAMES=2 timeout 60 target/release/selfplay 2>&1 | grep '\[selfplay\] adjudicated' | head -5
```

Expected: multiple `[selfplay] adjudicated turn=N delta=D adj_outcome=X` lines within 60 seconds with random-play games (where Δ fluctuates but may hit 2 repeatedly).

Alternatively (without modifying constants): run a 5-minute training run and check the log for at least one adjudication line:

```bash
bash scripts/run_training.sh 300
grep '\[selfplay\] adjudicated' logs/training_*.log | wc -l
```

For the production threshold (`|Δ| >= 6` for 10 plies), adjudications may be rare initially (random play rarely maintains a 6-piece advantage for 10 consecutive plies). The smoke test with `ADJ_THRESHOLD=2` is more reliable.

### 5.2 Full validation: run `scripts/run_training.sh 7200`

Report from the output JSON:
- `promotions` — should improve vs 3.66 baseline (more promotions = better play)
- `avg_game_length` — should decrease vs 300-move average (adjudicated or material-outcome games end earlier)
- `adjudication_rate` — confirm > 0% as self-play improves
- `final_policy_loss` — should not regress severely

**Validation checklist** (per experiment-protocol.md):
1. Promotions >= 0 (regression if 0, target > 3 for 2-hour run)
2. Early eval cycles (cycle 1–3): win_rate > 0.40
3. Game length: some decrease expected as material-outcome games terminate at cap with non-zero signal
4. Run twice if improvement < 1.5 points

### 5.3 Stale wiki entry

The mistakes.md entry "2026-04-15: Game Outcome Perspective — Absolute White vs Side-to-Move Relative (Unverified)" at lines 394–414 says the side-to-move conversion is "NOT done anywhere in the codebase." This is **incorrect** — `training.rs:136` already applies `root_side_sign * ply_flip`. Flag for context-keeper to update.

---

## Section 6: Risk and Rollback

### Risk 1: tanh at cap misaligns sign convention

**Symptom**: `compute_material_diff` returns positive for White advantage, but `game_outcome` in `GameTrajectory` is supposed to be White-perspective. These must match. If `player1` is not always White (a defensive assumption), the sign would be inverted.

**Verification**: `player1 = Player::init_player(true)` = White confirmed in `game_task.rs:153–154`. Safe.

**Mitigation**: Add a unit test asserting `compute_material_diff` returns 9 for a White queen + king vs black king position.

### Risk 2: tanh values in trainer — outcome range assumption

The trainer at `training.rs:141` blends: `(1.0 - beta) * step.root_value + beta * outcome_in_step_perspective`. The value head outputs `tanh` (range `[-1, 1]`). `tanh(Δ/5.0)` stays within `(-1, 1)` always, so no out-of-range issue. The trainer has no clamp on the outcome value; `tanh` is bounded.

### Risk 3: Adjudication too aggressive — caps blowouts too early

If `ADJ_THRESHOLD = 6` fires frequently in positions that are not actually decisive (e.g., after a queen blunder that gets recovered), the training buffer fills with false ±1 outcomes.

**Mitigation**: The 10-ply requirement and reset-on-drop logic prevents most false positives. Monitor `adjudication_rate` in the JSON. If > 20%, consider raising threshold to 9 (rook equivalent).

### Risk 4: NN learns to lose on purpose (sign inversion)

If the sign convention for `compute_material_diff` is wrong (returns negative for White advantage), the NN would learn to lose material. Symmetry helps: both sides face the same training data, so training should collapse symmetrically rather than train one side to lose. However, if there's a systematic bias (e.g., player1 always on top), the NN would learn to lose as White.

**Detection**: Check if White win rate in eval collapses after the change. If White win_rate < 0.3 consistently, suspect sign inversion.

### Risk 5: `test_play_game_completes` fails due to outcome range check

As noted in Section 4.4, the existing test asserts `game_outcome` is exactly `1.0`, `-1.0`, or `0.0`. The new `tanh` at cap breaks this. The test must be updated.

**Rollback**: If material signal regresses the score, revert by restoring the `_ => 0.0` arm in the `match board.result()` block and removing the adjudication check. The adjudication counter variables and `compute_material_diff` can remain (they become dead code until the feature is re-enabled).

---

## Subtasks

### 1. Add `compute_material_diff` helper
- **Files**: `src/selfplay/game_task.rs`
- **Changes**: Add `fn compute_material_diff(board: &GameBoard) -> i32` before `#[cfg(test)]`
- **Tests**: `test_compute_material_diff_starting_position`, `test_compute_material_diff_asymmetric`
- **Dependencies**: none

### 2. Adjudication state and loop changes in `play_game()`
- **Files**: `src/selfplay/game_task.rs`
- **Changes**: Add `adj_counter`, `ADJ_THRESHOLD`, `ADJ_PLIES` constants; insert adjudication check block inside the game loop; replace cap-break comment
- **Tests**: `test_adjudication_counter_logic`
- **Dependencies**: subtask 1 (calls `compute_material_diff`)

### 3. Post-loop material outcome at cap
- **Files**: `src/selfplay/game_task.rs`
- **Changes**: Replace `_ => 0.0` arm with `GameResult::Ongoing => { tanh... }` arm; add `GameResult::Ongoing` match arm
- **Tests**: `test_material_cap_outcome_nonzero_for_asymmetric_position`; update `test_play_game_completes` assertion
- **Dependencies**: subtask 1 (calls `compute_material_diff`)

### 4. New training script
- **Files**: `scripts/run_training.sh` (new file)
- **Changes**: Write script as specified in Section 3
- **Tests**: Manual smoke test: `bash scripts/run_training.sh 60` — verify log and JSON created, no errors
- **Dependencies**: none (independent of Rust changes)

## Testing Strategy

End-to-end verification:
1. `cargo test` — all existing tests pass (with updated `test_play_game_completes` assertion)
2. `bash scripts/smoke_dual_eval.sh` — baseline self-play still produces promotions
3. `bash scripts/run_training.sh 300` — 5-min run: JSON produced, `adjudication_rate` field present, log contains at least some `[selfplay] adjudicated` lines (may be 0 at low rate; acceptable)
4. `bash scripts/run_baseline.sh 1800` (fresh start, `rm -f checkpoints/best*.pt`) — score should be >= 3.66 (current regression baseline); ideally recovers toward 14.51 historical peak
