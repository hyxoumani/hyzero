---
name: Tablebase Supervision
description: Research findings for Syzygy tablebase injection to break value-head distributional collapse; batch integration design, encoding notes, insertion point
type: project
---

## Summary

Value head confirmed distributionally collapsed (kqk_value ≈ −0.012, stuck 650+ steps).
Plan is to inject 3-4-5-man Syzygy TB positions as exact ±1 supervision into training batches.

**Why:** With Q≈0, MCTS reduces to prior sampling. Forcing value head to see ±1 decisive positions
breaks the zero-attractor loop without touching MCTS or reward architecture.

## Key findings

### Batch assembly is Rust-side only today

`assemble_batch_arrays` in `src/py/training.rs:103` builds all batch numpy arrays from
`ReplayBuffer` samples and passes them to `trainer.train_batch(batch_dict)` via PyO3.
Python never does batch assembly today. TB mix must happen at the top of `train_batch`
in Python BEFORE tensor conversion: concatenate a Python-built TB sub-batch (last n_tb
rows) into the dict, then proceed as normal.

### `scripts/reward_probe.py` does not exist

The task brief references this file for `encode_board_python` and `encode_action_spatial`.
It does not exist on disk. The Python encoder must be written from scratch in
`python/hyzero/data/board_encoder.py`, mirroring `src/data/encoding.rs:encode_board`.
Ground truth for KQK position: `_build_kqk_white_winning_obs()` in `trainer.py:106`.

### python-chess is available (not in pyproject.toml as dep)

`chess.syzygy.open_tablebase` is importable (`python3 -c "import chess.syzygy"` passes).
It is NOT listed in `pyproject.toml` dependencies (only torch, numpy). Need to add it
or document it as a build-time dep for the TB script.

### Trainer K-step shape contract

- Replay batch: K=5 (`unroll_k=5` in `from_default_config`)
- TB samples use K_TB=1 but must be zero-padded to K=5 for concatenation
- `train_batch` reads `k_steps = actions.shape[1]` — so padding must match replay K
- Reward target at step 1 = +1 for mating actions (Option B)
- Consistency loss must be zeroed for TB rows (their obs slots 1..K are all-zeros)

### Consistency loss zeroing

Block at `trainer.py:737-750`. Must mask TB rows by index before accumulating
cosine-similarity terms. TB rows have zero in `obs_all[:, 1..K]` — if included, they
would push g toward "match zero latent after mating-action", which is harmful.

### WDL polarity (python-chess convention)

`chess.syzygy.probe_wdl(board)` returns WDL from SIDE-TO-MOVE perspective:
- +2 = STM wins (forced mate)
- +1 = STM wins (cursed win)
- 0  = draw
- -1 = STM loses (blessed loss)
- -2 = STM loses (forced)
Map: `target_value = +1 if wdl > 0 else (-1 if wdl < 0 else 0)`.
No POV flip needed — already from STM perspective, which matches hyzero value convention.

### Plan file location

`docs/plans/tablebase-supervision/plan.md`

## How to apply

When implementing TB supervision, check plan.md for exact file:line insertion points.
The TB mix must not affect existing replay-sample code paths when `HYZERO_TABLEBASE_FRAC=0.0`.
