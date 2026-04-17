# Plan: C=128 Model Capacity Experiment

## Hypothesis

The current model (hidden_channels=64, 4 ResBlocks, ~1.3M parameters) has insufficient
representational capacity for chess. Policy loss at 4.32 after 900s reflects the model
hitting a hard capacity ceiling, not a data or optimization problem. Doubling
hidden_channels to 128 scales ResBlock parameter count as C^2 (4x per block), pushing
total parameters from ~1.3M to ~3.8M. This gives the network room to encode meaningful
positional patterns, which directly reduces policy loss — the dominant metric component.

Expected policy_loss improvement: 4.32 → ~3.0-3.5 (delta +0.8 to +1.3). Combined with
shorter games from a sharper policy (avg_length improvement worth ~0.3-0.5), total
expected delta: **+1.5 to +3.0**. A +1.5 delta would be comfortably above the ±1.0
noise floor.

## Why it beats the noise floor (±1.0)

Policy loss is a smooth, deterministic function of gradient steps × model capacity. More
capacity = lower loss floor, not more variance. The prior failed attempt (a2be1de) was
blocked by a hardcoded `PyO3Backend::new(server, 64)` shape mismatch, not a hypothesis
failure — the fix in e44583a resolves this. All prior failed experiments (loss reweighting
+0.13, recency replay variable) were noise-range because they did not increase capacity.
Capacity is the current binding constraint.

## Risk

**Throughput reduction**: Inference time per call scales as O(C^2) for conv layers, so
C=64→128 gives ~4x slower per inference call. In 900s with 4 concurrent games at 50 sims
each, baseline produces ~59 games and ~472 training steps. At 4x slower inference, expect
~25-35 games and ~200-280 steps. The metric may still improve if per-step quality gain
(lower policy_loss per step from more capacity) outpaces the throughput reduction.

**Worst case**: If games drop to <20 (>65% throughput reduction), the policy_loss
component improves but the decisive_ratio component degrades (fewer games = weaker
self-play signal, more random outcomes). Score could land in the 3-4 range.

**Fallback**: If throughput is severe, reduce num_res_blocks from 4 to 2 while keeping
C=128. This recovers ~2x throughput (4 ResBlocks × C^2 → 2 ResBlocks × C^2) at modest
capacity cost. The plan supports this via config.

**Stale checkpoints**: C=64 checkpoints in `checkpoints/` are shape-incompatible with
C=128 layers. They MUST be deleted before running — loading them would crash the trainer.

## Subtasks

### 1. Update DEFAULT_CONFIG to hidden_channels=128

- **Files**: `python/hyzero/config.py`
- **Changes**: Change `"hidden_channels": 64` to `"hidden_channels": 128`. No other
  config changes needed — all three networks (h, g, f) read `hidden_channels` from
  config and scale automatically. The Rust binary reads `hidden_channels` from Python
  config at startup via e44583a (no Rust change needed).
- **Tests**: `cd python && pytest` — existing forward-pass tests instantiate from
  DEFAULT_CONFIG and will exercise the new size automatically.
- **Dependencies**: none

### 2. Delete stale checkpoints

- **Files**: `checkpoints/model_v*.pt` (currently: 000013, 000019, 000026, 000032, 000038)
- **Changes**: `rm -f checkpoints/model_v*.pt` — must run before the baseline.
  These checkpoints were saved with C=64 layer shapes (Conv2d(103→64) in RepNet,
  Linear(128→4672) in PredNet) and are incompatible with C=128 model dimensions.
- **Tests**: None
- **Dependencies**: none (run before subtask 3)

### 3. Run baseline and collect result

- **Files**: none
- **Changes**: `bash scripts/run_baseline.sh 900` then
  `python3 -c "import json; print(json.load(open('logs/baseline_score.json'))['score'])"`
- **Monitor**: Watch `games_completed` in logs. If <30 games complete, the throughput
  hit is severe — rerun with `num_res_blocks=2` (add to config.py) as fallback.
- **Dependencies**: Subtasks 1 and 2

## Testing Strategy

1. `cd python && pytest` — all model, trainer, and inference tests pass with C=128.
   Tests are parameterized from DEFAULT_CONFIG so they exercise new shapes directly.

2. `cargo test` — Rust tests are unaffected by Python config. Confirm 82 tests still pass.

3. Run `bash scripts/run_baseline.sh 900` (after checkpoint wipe). Decision gate:
   - score ≥ 5.5 (delta ≥ +1.4): keep — above noise floor, meaningful improvement
   - score 4.1–5.5: ambiguous — check `games_completed`. If <30 games: throughput is
     bottleneck; retry with `num_res_blocks=2`. If ≥40 games: genuine noise or capacity
     didn't help — discard.
   - score < 4.1 and games_completed ≥ 40: discard — capacity didn't overcome overhead.

## Expected Score Delta

Baseline: 4.13 (policy_loss=4.32, decisive_ratio=0.20, avg_game_length=210.1)

Expected at C=128:
- policy_loss → ~3.0–3.5 → component gain +0.8 to +1.3
- avg_game_length → ~170–190 (sharper policy shortens games) → component gain +0.2 to +0.4
- decisive_ratio → 0.10–0.25 (uncertain; fewer games increases variance) → ±0.0 to +0.5

**Central estimate: +2.0 delta → score ~6.1**
**Conservative floor: +1.5 delta → score ~5.6**
**Worst case (severe throughput hit): -0.5 to +0.5 → score ~3.6–4.6**

## Notes

- Prior attempt a2be1de was blocked by hardcoded `PyO3Backend::new(server, 64)` in
  `src/bin/selfplay.rs`. This caused a shape mismatch at runtime (Rust expected 64-channel
  hidden states, Python produced 128-channel). Fixed in e44583a — Rust now reads
  `hidden_channels` from DEFAULT_CONFIG at startup.
- The MCTS node masking in `src/mcts/node.rs:27-39` already renormalizes priors over
  legal actions only. "Legal move policy masking in MCTS" from the roadmap is already
  implemented — the roadmap item is stale and should be removed or marked done.
