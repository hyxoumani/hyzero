# Plan: Larger Model (hidden_channels 64 → 128)

## Hypothesis

The current model is tiny (hidden_channels=64, 4 ResBlocks, ~500K parameters). Doubling
hidden_channels to 128 quadruples the parameter count of every ResBlock (conv weights scale
as C²) while tripling the representation network stem. This gives the network 4-5x more
representational capacity to learn chess patterns, directly reducing the policy loss floor.

Policy loss is the dominant training_score component (weight 1.0, currently 4.32 at end of
900s run). A larger model should reduce policy loss to ~3.0-3.5 within 900s — a delta of
+0.8 to +1.3 from this component alone. Alongside any improvement in value head quality
(larger value head in PredictionNetwork: hidden_channels→128 for the internal linear), the
combined effect should reach +1.5 to +3.0 score.

**Why it beats the noise floor (±1.0)**: Policy loss improvement is continuous and scales
predictably with model capacity; even modest reductions (+0.8) combine with game-length
reductions (fewer random moves once policy is sharper) for a compound effect. The 10x loss
weight experiment (+0.13, noise) failed because it did not increase network capacity —
it just redistributed gradient signal across the same small model.

**Risk**: Slower inference per call reduces game throughput. In 900s with 4 concurrent
games at 50 sims, ~59 games complete. If the larger model cuts throughput by 30%, we get
~41 games and ~330 training steps instead of 472. The metric may still improve if each
training step is higher quality (better policy from more capacity). Fallback: if fewer
games dominate, reduce sims to 30 or increase concurrent_games.

**Checkpoint wipe**: Changing hidden_channels changes all layer shapes. Old checkpoints
are incompatible and must be deleted before running.

## Subtasks

### 1. Update DEFAULT_CONFIG to hidden_channels=128
- **Files**: `python/hyzero/config.py`
- **Changes**: Change `"hidden_channels": 64` to `"hidden_channels": 128`.
  The `num_res_blocks` stays at 4. All three networks (h, g, f) read `hidden_channels`
  from config and will automatically use the new size.
- **Tests**: No new tests needed — the existing `python/tests/test_models.py` forward-pass
  tests instantiate from DEFAULT_CONFIG and will exercise the new size automatically.
- **Dependencies**: none

### 2. Verify inference server tensor shape compatibility
- **Files**: `src/py/inference.rs` (read-only audit)
- **Changes**: None — the inference server receives raw bytes from `get_weights()` and
  passes NumPy arrays in/out. Hidden state shape `[B, hidden_channels, 8, 8]` is opaque
  to Rust (stored as `Vec<f32>` in `HiddenState`). Confirm `HiddenState::new(64)` in
  tests uses a literal size unrelated to the model channel count (it is just the flat
  buffer size for test stubs). No Rust changes needed.
- **Tests**: Run `cargo test` to confirm existing tests still pass with Python config change.
- **Dependencies**: Subtask 1

### 3. Delete stale checkpoints
- **Files**: `checkpoints/` directory
- **Changes**: Remove any `model_v*.pt` files before running the baseline. Old checkpoints
  were saved with C=64 layer shapes; loading them into C=128 models will crash.
  `scripts/run_baseline.sh` does not auto-clean checkpoints, so this must be done manually
  before the run: `rm -f checkpoints/model_v*.pt`
- **Tests**: None
- **Dependencies**: Subtask 1

## Testing Strategy

1. `cd python && pytest` — all existing tests (forward pass shapes, trainer loss, policy
   masking) must pass. The tests instantiate from DEFAULT_CONFIG so they directly exercise
   hidden_channels=128.

2. Quick sanity: `cargo test` — Rust tests do not import Python config, so they are
   unaffected. Confirm 82 tests still pass.

3. Run `bash scripts/run_baseline.sh 900` after deleting stale checkpoints. Expected
   outcome:
   - `last_policy_loss` ≤ 3.5 (down from 4.32 baseline)
   - `score` ≥ 5.5 (up from 4.13 baseline, delta ≥ +1.5)

4. If score < 5.5 but policy_loss did improve: check `games_completed` vs 59 baseline.
   If games dropped >30% (i.e. <41), the throughput hit outweighed the capacity gain.
   Fallback: set `hidden_channels=128` with `num_res_blocks=2` to recover speed.

## Parameter Count Estimate

Current (C=64, 4 ResBlocks per net):
  - RepNet: stem (103→64, 3×3) = 103×64×9 ≈ 59K; 4 ResBlocks × 2×(64×64×9) = 295K → ~354K
  - DynNet: stem (67→64, 3×3) = 67×64×9 ≈ 39K; 4 ResBlocks → 295K; reward head ≈ 1K → ~335K
  - PredNet: policy head ≈ 64×2×1 + 128×4672 ≈ 0.6M; value head ≈ 64×1×1 + 64×64 + 64×1 ≈ 5K → ~605K
  - Total: ~1.3M parameters

New (C=128, 4 ResBlocks per net):
  - RepNet: stem (103→128) ≈ 118K; 4 ResBlocks × 2×(128×128×9) ≈ 1.18M → ~1.3M
  - DynNet: stem (131→128) ≈ 150K; 4 ResBlocks ≈ 1.18M; reward head ≈ 1K → ~1.33M
  - PredNet: policy head ≈ 128×2 + 256×4672 ≈ 1.2M; value head ≈ 128×128 + 128 ≈ 17K → ~1.2M
  - Total: ~3.8M parameters (~3x current)

Inference time per batch scales roughly as O(C²) for conv layers, so expect ~4x slower
per-call, partially amortized by the batch_size=256 training batch (same overhead).
Self-play inference (single-position eval) will feel the full slowdown.
