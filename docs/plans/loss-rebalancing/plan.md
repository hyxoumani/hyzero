# Plan: Loss Weight Rebalancing

## Approach
Add `value_loss_weight` and `reward_loss_weight` scalar multipliers to `DEFAULT_CONFIG`
and apply them at the single line in `Trainer.train_batch` where the three averaged
losses are summed. No other code paths are touched.

## Subtasks

### 1. Add weight keys to DEFAULT_CONFIG
- **Files**: `python/hyzero/config.py`
- **Changes**: Add two keys to the dict:
  ```python
  "value_loss_weight": 1.0,   # scale factor applied to avg_value_loss before summing
  "reward_loss_weight": 1.0,  # scale factor applied to avg_reward_loss before summing
  ```
  Set to `1.0` (no-op defaults) so existing callers that don't pass these keys are unaffected.
- **Tests**: None needed — config is a plain dict, tested implicitly by trainer tests.
- **Dependencies**: none

### 2. Read weights in Trainer.__init__ and apply in train_batch
- **Files**: `python/hyzero/training/trainer.py`
- **Changes**:

  In `__init__`, after the optimizer is constructed, store the two scalars:
  ```python
  self.value_loss_weight: float = float(cfg.get("value_loss_weight", 1.0))
  self.reward_loss_weight: float = float(cfg.get("reward_loss_weight", 1.0))
  ```

  In `train_batch`, replace line 122:
  ```python
  # before
  total_loss = avg_policy_loss + avg_value_loss + avg_reward_loss

  # after
  total_loss = (
      avg_policy_loss
      + self.value_loss_weight * avg_value_loss
      + self.reward_loss_weight * avg_reward_loss
  )
  ```

  The returned dict (lines 129-134) stays unchanged — individual losses are still
  reported unweighted so they remain interpretable in logs.
- **Tests**: Add two tests to `python/tests/test_training.py` (see Testing Strategy).
- **Dependencies**: Subtask 1 must complete first (keys must exist in DEFAULT_CONFIG).

## Testing Strategy

Two new tests in `python/tests/test_training.py`:

**test_loss_weights_applied**: Confirm that passing `value_loss_weight=10.0` and
`reward_loss_weight=10.0` makes `total_loss > policy_loss` by more than the
unweighted case. Run two trainers from the same seed — one default, one with weights —
compare `(total_loss - policy_loss)` between them. With random targets the
value/reward MSE terms will be non-trivial, so the weighted version must produce a
measurably larger gap.

**test_loss_weights_default_unchanged**: Confirm that a Trainer constructed without
explicit weight keys in the config still produces `total_loss == policy_loss +
value_loss + reward_loss` (within fp tolerance). This is a regression guard: the
default `1.0` weights must not change existing behavior.

End-to-end verification: After implementing, run `cd python && pytest` to confirm all
8 tests pass (6 existing + 2 new). Then run `bash scripts/run_baseline.sh 1800` with
`value_loss_weight=10.0, reward_loss_weight=10.0` in `DEFAULT_CONFIG` and compare
score against 4.78 baseline.
