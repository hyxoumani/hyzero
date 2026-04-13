# hyzero Development Roadmap

Ordered increments. Each batch is a full session of work — multiple files, meaningful
architectural change, justified by a 30-minute baseline run (`bash scripts/run_baseline.sh 1800`).

**Current baseline score**: 4.78 (commit c1e5cdc, 2026-04-13)

**Score formula**: `(8.55 - final_policy_loss) + (decisive_ratio * 10) - (avg_game_length / 100)`

---

## Batch 1: Representation Overhaul

Bundle all observation/action shape-breaking changes into one shot so we only retrain once.

### History Planes
Expand board encoding from 19 → 103 planes (current position + 7 past positions).
The representation network currently sees a single snapshot — no temporal context.
Adding history lets the network learn repetition patterns, piece mobility trends,
and positional momentum. AlphaZero uses 8 past positions for this reason.

- Add position history ring buffer to game_task (stores last 7 board states)
- Update `encode_board()` to accept history and fill 12 planes per past position
- Update `RepresentationNetwork` input channels: 19 → 103
- Update config defaults

### Underpromotion
Expand action space from 4096 → 4672. Currently only queen promotion is supported.
Knight promotion (especially with check) is a critical tactical pattern that the
engine literally cannot find right now.

- Add 3 underpromotion plane sets (knight/bishop/rook × 8 files × 8 target ranks)
- Update `move_to_action`, `action_to_move`, `action_to_notation`
- Update `PredictionNetwork` output layer: 4096 → 4672
- Update all visit distribution padding in training pipeline

### Legal Move Masking
Before softmax, set logits of illegal actions to -inf. The network currently wastes
capacity learning to suppress 4000+ illegal moves per position — masking makes every
gradient step focus on choosing among legal options.

- Mask illegal moves in `play_game()` before MCTS policy initialization
- Pass legal move mask through inference pipeline
- Apply mask in Python inference server before softmax

### Files
`src/data/encoding.rs`, `src/data/types.rs`, `src/selfplay/game_task.rs`,
`python/hyzero/config.py`, `python/hyzero/models/representation.py`,
`python/hyzero/models/prediction.py`, `python/hyzero/training/trainer.py`,
`python/hyzero/inference/server.py`

### Expected Impact
Faster loss convergence, richer input signal, correct promotion handling,
less wasted network capacity on illegal move suppression.

---

## Batch 2: Search & MCTS Improvements

Make each move's search stronger without touching the network architecture.

### Tree Reuse
Currently the MCTS tree is discarded after every move (transient). After selecting
action A, keep the subtree rooted at child(A). On the next move, after the opponent
plays B, look up child(B) and reuse it. Saves ~50% of MCTS simulations per move.
Fall back to a fresh tree if the subtree is missing (unexpected move).

- Modify `play_game()` loop to pass previous tree into next iteration
- Add `MCTSTree::reuse_subtree(action)` method
- Memory management: prune detached branches after reuse

### Legal Move Policy Masking in MCTS
Renormalize the policy distribution over legal moves only at the tree root, so MCTS
doesn't waste simulations exploring impossible actions. Complements Batch 1's network-
level masking — this is the search-level equivalent.

### Temperature Schedule Tuning
Current implementation is a hard cutoff: temperature=1.0 for the first N moves, then
0.01. Switch to smooth exponential decay. Add configurable exploration parameters.

### Pondering (stretch)
Think during the opponent's turn using the reused tree. Only valuable once tree reuse
is implemented.

### Files
`src/mcts/tree.rs`, `src/mcts/node.rs`, `src/mcts/puct.rs`,
`src/selfplay/game_task.rs`

### Expected Impact
More decisive games, shorter game lengths, better use of the simulation budget.

---

## Batch 3: Training Pipeline Hardening

Make training smarter per gradient step.

### Learning Rate Scheduling
Cosine decay or step schedule per MuZero paper. Current fixed LR means early steps
overshoot and late steps plateau without convergence. Standard approach: warm up for
100 steps, then cosine decay to 1/10th of peak LR.

### Loss Weighting Rebalance
Policy loss currently dominates at 99.9% of total loss. Value and reward heads
essentially don't train. Scale value/reward losses up (e.g., 10x) so all three
network heads learn simultaneously.

### Priority Replay Sampling
Weight the replay buffer toward longer and more recent trajectories instead of
uniform random. Fresh games from the current model version are more informative
than stale games from early random play.

### Training-to-Self-Play Ratio
Currently hardcoded at 4 gradient steps per game. MuZero paper uses much higher
ratios. Make this configurable and experiment with 8-16 steps per game.

### Files
`python/hyzero/training/trainer.py`, `python/hyzero/config.py`,
`src/data/replay_buffer.rs`, `src/py/training.rs`

### Expected Impact
Lower final loss, value/reward heads start learning, less wasted gradient computation,
faster convergence to useful policy.

---

## Batch 4: Tactical Strength Metric

Actually measure chess ability, not just pipeline health. The current baseline score
(4.78) reflects pipeline efficiency — policy loss drop, game throughput, decisive
ratio. It does not measure whether the engine plays good chess.

### Puzzle Test Suite
Curate 50-100 tactical positions with known best moves (forks, pins, back-rank
mates, discovered checks, promotion tactics). Source from standard puzzle databases
(Lichess puzzles, Win at Chess, etc.).

### Strength Evaluator Script
`scripts/eval_strength.py` — loads a checkpoint, runs inference on each puzzle
position, checks if the top policy move (or top-3) matches the known solution.
Reports accuracy percentage.

### Composite Metric Update
Add puzzle accuracy as a weighted component of the training score:
```
score = (8.55 - policy_loss) + (decisive_ratio * 10) - (avg_length / 100) + (puzzle_accuracy * 20)
```
This ensures the autoresearch loop optimizes for actual play quality, not just
loss numbers.

### Frozen Baseline Opponent
Save an early checkpoint as a permanent baseline opponent. Play the current model
against it and track win rate. Win rate against a fixed opponent is a direct Elo
proxy without needing external engines.

### Files
New `scripts/eval_strength.py`, new `data/puzzles.json`,
update `scripts/run_baseline.sh`, update `CLAUDE.md` metric section

### Expected Impact
Real chess strength signal. Autoresearch loop can optimize for tactical play.
Clear threshold for "this model actually learned chess."

---

## Batch 5: UCI Protocol & Playability

Make the engine usable by humans and other engines.

### UCI Binary
New `src/bin/uci.rs` implementing the Universal Chess Interface protocol. Parses
standard UCI commands: `uci`, `isready`, `position fen/startpos moves`, `go`,
`bestmove`, `quit`.

### Time Control
Convert UCI time parameters (`wtime`, `btime`, `winc`, `binc`, `movetime`) into
a simulation budget per move. Simple formula: allocate time proportional to
remaining clock, convert to simulation count based on measured sims/second.

### Checkpoint Loading
UCI binary loads a trained model checkpoint on startup. Supports both CPU and GPU
inference. Command-line flag: `--model checkpoints/model_v000100.pt`.

### External Testing
Test against Stockfish at fixed depth via `cutechess-cli` or equivalent. Measure
Elo rating against a range of Stockfish depths (1-5). This gives a concrete,
comparable strength number.

### Files
New `src/bin/uci.rs`, update `Cargo.toml`, update `CLAUDE.md` commands section

### Expected Impact
Engine is playable in any chess GUI. Can benchmark Elo against real engines.
Demo-able to humans. Foundation for online play (Lichess bot).

---

## Order

```
Batch 1 (Representation) → Batch 2 (Search) → Batch 3 (Training) → Batch 4 (Metrics) → Batch 5 (UCI)
```

Each builds on the previous:
1. Fix the representation before serious training
2. Make search effective with the new representation
3. Make training efficient with the improved search
4. Measure real strength once the model can actually play
5. Make it playable once it's strong enough to be interesting
