# Plan: Batch 1 — Representation Overhaul

## Approach

Bundle the three shape-breaking changes (history planes, underpromotion, legal-move
masking) into a single implementation session so the model is retrained once with the
full AlphaZero-style representation. All three changes share a common pattern: update
the Rust constants/encoding first, then propagate through PyO3 bridge shapes, then
Python network/training code, then tests. The dependency chain is strictly linear:
action space must be updated before masking can be wired in, and history must be
wired in before the network input channel count can be updated.

---

## 1. Context Summary

### Current state

`encode_board()` (`src/data/encoding.rs:6`) produces a 19-plane snapshot:
- Planes 0-5: White pieces (Pawn..King)
- Planes 6-11: Black pieces
- Planes 12-15: Castling rights (constant planes)
- Plane 16: En passant target (one-hot)
- Plane 17: Side-to-move
- Plane 18: Halfmove clock (normalized)

Stored in `BoardObservation { planes: Vec<f32> }` (`src/data/types.rs:22`).
`NUM_OBS_PLANES = 19` (`src/data/types.rs:11`).

Action space: `NUM_ACTIONS = 4096` (`src/data/types.rs:8`). Encoding is
`from_sq * 64 + to_sq`. Only queen promotion is emitted by `move_to_action`
(`src/data/encoding.rs:66`) and `get_legal_moves` (`src/selfplay/game_task.rs:163`).
`action_to_notation` (`src/selfplay/game_task.rs:142`) appends `q` suffix only;
`action_to_move` (`src/data/encoding.rs:72`) always returns `Queen` for promotions.

### Call graph for encode_board → inference

```
game_task.rs:64   encode_board(&board, side_to_move)  -> BoardObservation
game_task.rs:72   evaluator.root_setup(&observation)   -> (HiddenState, Policy, value)
  inference.rs:130 ChannelEvaluator::root_setup()       -> InferenceRequest::RootSetup
  inference_backend.rs:96-110                            -> numpy [B,19,8,8] (reshape hardcoded at line 106)
  server.py:65-76  h(obs_t) -> f(hidden) -> softmax     -> policies [B,4096]
  inference_backend.rs:135   policy_stride = NUM_ACTIONS
```

### Call graph for training

```
py/training.rs:42  obs_stride = 19 * 64  (hardcoded)
py/training.rs:127 obs_arr.reshape([b, 19, 8, 8])  (hardcoded)
py/training.rs:133 pol_arr.reshape([b, kp1, NUM_ACTIONS])
trainer.py:79      obs [B,19,8,8] -> h -> f -> policy_logits [B,4096]
trainer.py:95-110  _policy_loss over logits [B,4096] vs targets [B,K+1,4096]
```

### File:line reference map

| Location | Current value | Change needed |
|---|---|---|
| `src/data/types.rs:8` | `NUM_ACTIONS = 4096` | `NUM_ACTIONS = 4672` |
| `src/data/types.rs:11` | `NUM_OBS_PLANES = 19` | `NUM_OBS_PLANES = 103` |
| `src/data/types.rs:22-33` | `BoardObservation::default()` size | Grows with constant |
| `src/data/encoding.rs:6-62` | single-snapshot `encode_board` | add `history` param |
| `src/data/encoding.rs:66-68` | `move_to_action` queen-only | add underpromo plane encoding |
| `src/data/encoding.rs:72-118` | `action_to_move` queen-only | decode underpromo range |
| `src/selfplay/game_task.rs:64` | `encode_board(&board, side)` | pass history buffer |
| `src/selfplay/game_task.rs:142-158` | `action_to_notation` queen-only | decode underpromo suffix |
| `src/selfplay/game_task.rs:163-236` | `get_legal_moves` queen-only | emit all 4 promotions |
| `src/selfplay/game_task.rs:97-104` | `StepRecord` construction | include legal_mask field |
| `src/py/inference_backend.rs:98,106` | `b * 19 * 64`, `reshape [b,19,8,8]` | use `NUM_OBS_PLANES` |
| `src/py/inference_backend.rs:135,236` | `policy_stride = NUM_ACTIONS` | auto via constant |
| `src/py/training.rs:42` | `obs_stride = 19 * 64` | use `NUM_OBS_PLANES * 64` |
| `src/py/training.rs:127` | `reshape [b,19,8,8]` | use `NUM_OBS_PLANES` |
| `python/hyzero/config.py:6-7` | `input_planes=19, num_actions=4096` | `103, 4672` |
| `python/hyzero/models/representation.py:16` | `input_planes=19` | `input_planes=103` |
| `python/hyzero/models/prediction.py:18` | `num_actions=4096` | `num_actions=4672` |
| `python/hyzero/inference/server.py:65-76` | softmax over all logits | mask before softmax |
| `python/hyzero/inference/server.py:98-110` | softmax over all logits | mask before softmax |
| `python/hyzero/training/trainer.py:95-112` | unmasked policy loss | mask before log_softmax |
| `python/tests/test_training.py:17,20` | hardcoded `19`, `4096` | use config constants |
| `python/tests/test_inference.py:116-118,145-147` | hardcoded `19`, `4096` | use config constants |

---

## 2. Complete File List With Changes

### Rust files

**`src/data/types.rs`** — constants and types
- Change `NUM_ACTIONS = 4096` to `NUM_ACTIONS = 4672`
- Change `NUM_OBS_PLANES = 19` to `NUM_OBS_PLANES = 103`
- Add `NUM_HISTORY_POSITIONS: usize = 8` (current + 7 past)
- Add `NUM_UNDERPROMO_ACTIONS: usize = 576` (3 pieces × 8 files × 24 moves)
- `BoardObservation::default()` is derived from `NUM_OBS_PLANES * 64` — no change needed
- Delta: +4 lines
- Dependencies: must be done FIRST — all other files import these constants

**`src/data/encoding.rs`** — board encoding and action encoding
- Change `encode_board(board, side_to_move)` signature to
  `encode_board(board, side_to_move, history: &[BoardSnapshot])` where
  `BoardSnapshot` is a new lightweight struct (just piece bitboards per color)
  OR accept `Option<&[GameBoard]>` for the ring buffer slice
- Planes 0-11 become the CURRENT position piece planes (unchanged layout)
- Planes 12-23: past position 1 piece planes (12 piece planes, castling/EP dropped for history)
- ... repeated for positions 2-7 (planes 24-95)
- Planes 96-99: castling rights (current only)
- Plane 100: en passant target (current only)
- Plane 101: side-to-move
- Plane 102: halfmove clock
- Total: 8×12 + 7 = 103 planes
- Expand `move_to_action`: if `mv.promotion_piece_type` is Knight/Bishop/Rook, return
  `4096 + underpromo_index(mv)` where `underpromo_index` encodes piece×file×direction
- Expand `action_to_move`: if `action >= 4096`, decode underpromotion
- Update `action_to_notation`: if `action >= 4096`, append `n`/`b`/`r` suffix
- Update `num_actions()` function to return `NUM_ACTIONS`
- Delta: +80 lines
- Dependencies: `types.rs` constants must be updated first

**`src/selfplay/game_task.rs`** — game play loop
- Add `history_buffer: VecDeque<BoardSnapshot>` (capacity 7) to `play_game`
  internal state
- After each board update, push current position snapshot to buffer
- Pass `history_buffer.make_contiguous()` slice to `encode_board`
- In `get_legal_moves`, add 3 more promotion options per promotion square
  (currently only Queen; add Knight, Bishop, Rook) — each becomes a separate
  action in the returned Vec
- In `action_to_notation`: delegate to updated `encoding.rs` version or extend
  `q` suffix logic to also emit `n`/`b`/`r` for underpromotion action range
- Add `legal_mask: Vec<bool>` to `StepRecord` construction for use by masking;
  OR compute mask inline and pass to MCTS root setup
- Delta: +60 lines
- Dependencies: `encoding.rs` changes must be complete

**`src/py/inference_backend.rs`** — PyO3 RootSetup batch
- Line 98: change `b * 19 * 64` to `b * NUM_OBS_PLANES * 64`
- Line 106: change `arr.reshape([b, 19, 8, 8])` to
  `arr.reshape([b, NUM_OBS_PLANES as usize, 8, 8])`
- For legal-move mask: `InferenceRequest::RootSetup` needs a `legal_mask: Vec<bool>`
  field (Vec of length NUM_ACTIONS). Build numpy bool array `[B, NUM_ACTIONS]`, pass
  as second arg to `root_setup_batch(obs, mask)`
- `policy_stride = NUM_ACTIONS` is already via constant — no change
- Delta: +25 lines
- Dependencies: `types.rs` and `inference.rs` struct changes

**`src/selfplay/inference.rs`** — InferenceRequest and ChannelEvaluator
- Add `legal_mask: Vec<bool>` to `InferenceRequest::RootSetup`
- Update `ChannelEvaluator::root_setup` signature to accept mask
- Update `Evaluator` trait: `root_setup(obs, legal_mask) -> ...`
- `RandomBackend::evaluate_batch` just ignores the mask
- Delta: +15 lines
- Dependencies: `types.rs` first; this change affects all callers of `Evaluator`

**`src/mcts/evaluator.rs`** (need to check, likely simple trait def)
- Update `Evaluator::root_setup` trait method signature to include mask
- Delta: +2 lines
- Dependencies: `inference.rs`

**`src/py/training.rs`** — batch assembly for Python trainer
- Line 42: change `let obs_stride = 19 * 64;` to `let obs_stride = NUM_OBS_PLANES * 64;`
- Line 127: change `obs_arr.reshape([b, 19, 8, 8])` to
  `obs_arr.reshape([b, NUM_OBS_PLANES as usize, 8, 8])`
- `pol_stride = NUM_ACTIONS` already via constant — no change
- Add `legal_masks` field to `BatchArrays` (flat `Vec<bool>` of size `B * NUM_ACTIONS`)
- Pass masks to Python as bool numpy array `[B, NUM_ACTIONS]` in `batch_dict`
- Update comments/docstrings: `[B, 19, 8, 8]` → `[B, 103, 8, 8]`, `4096` → `4672`
- Delta: +20 lines
- Dependencies: `types.rs` constants

### Python files

**`python/hyzero/config.py`** — hyperparameters
- `"input_planes": 19` → `"input_planes": 103`
- `"num_actions": 4096` → `"num_actions": 4672`
- Delta: 2 lines changed, 0 added

**`python/hyzero/models/representation.py`** — RepresentationNetwork
- Default arg `input_planes: int = 19` → `input_planes: int = 103`
- Docstring update: `Input: [B, input_planes, 8, 8] (default input_planes=103)`
- No structural change needed — `nn.Conv2d(input_planes, ...)` is already parameterized
- Delta: 2 lines changed

**`python/hyzero/models/prediction.py`** — PredictionNetwork
- Default arg `num_actions: int = 4096` → `num_actions: int = 4672`
- Comment: `linear (128->4096)` → `linear (128->4672)`
- No structural change — `nn.Linear(2 * board_size, num_actions)` already parameterized
- Delta: 2 lines changed

**`python/hyzero/inference/server.py`** — InferenceServer
- `root_setup_batch(observations, legal_masks)` — add mask parameter
  - Signature: `observations: [B, 103, 8, 8]`, `legal_masks: np.ndarray [B, 4672] bool`
  - Before softmax: `policy_logits[~legal_masks] = float('-inf')`
  - This ensures `softmax` concentrates entirely on legal moves
- Same treatment for `expand_leaf_batch` — NOTE: at leaf nodes in latent space we do
  NOT have a legal move mask (we're in learned latent space, not real board state).
  Leave `expand_leaf_batch` unmasked. Only `root_setup_batch` receives the real-board mask.
- Update docstrings from `[B, 19, 8, 8]` → `[B, 103, 8, 8]`, `[B, 4096]` → `[B, 4672]`
- Delta: +15 lines

**`python/hyzero/training/trainer.py`** — Trainer
- `train_batch` receives batch dict that now includes `"legal_masks": [B, NUM_ACTIONS] bool`
- In `_policy_loss`, apply mask before computing cross-entropy:
  - Mask logits: `logits = logits.masked_fill(~legal_masks, float('-inf'))`
  - Mask targets: normalize targets over legal moves only (divide by sum)
  - Then standard cross-entropy: `-sum(targets * log_softmax(logits))`
  - This ensures gradient doesn't push logits at illegal positions
- Update docstrings for `observations` shape
- Delta: +20 lines
- Dependencies: `config.py` changed first so default ctor uses new shapes

### Test files

**`python/tests/test_training.py`**
- `make_random_batch`: change hardcoded `19` to `INPUT_PLANES` (already imported
  via config), `4096` to `NUM_ACTIONS` (already imported)
- Line 17: `np.random.randn(batch_size, 19, 8, 8)` → use `INPUT_PLANES`
- Line 20: `np.full((..., 4096), ...)` → use `NUM_ACTIONS`
- Delta: 2 lines changed

**`python/tests/test_inference.py`**
- Lines 116, 118, 145, 147: same hardcoded 19/4096 pattern → use config constants
- Delta: 4 lines changed

**`src/py/training.rs` tests (test_batch_assembly_shapes, test_batch_assembly_pads_short_policies)**
- `b * 19 * 64` assertions → use `NUM_OBS_PLANES`
- `kp1 * NUM_ACTIONS` patterns — already use constant, no change
- Docstring comments with `4096` → update
- Delta: 4 lines changed

### Read-only files (must NOT touch)
- `Cargo.lock`
- `python/pyproject.toml`
- `logs/`
- `docs/wiki/`

---

## 3. Shape Transition Table

| Boundary | Before | After |
|---|---|---|
| `encode_board()` output | `Vec<f32>` len=1216 (19×64) | `Vec<f32>` len=6592 (103×64) |
| `BoardObservation.planes` | 1216 floats | 6592 floats |
| Rust → PyO3 `RootSetup` flat buffer | `B × 1216` floats | `B × 6592` floats |
| PyO3 reshape in `inference_backend.rs` | `[B, 19, 8, 8]` | `[B, 103, 8, 8]` |
| Legal mask in `InferenceRequest` | absent | `Vec<bool>` len=4672 |
| Legal mask numpy | absent | `[B, 4672] bool` |
| Python `root_setup_batch` obs input | `[B, 19, 8, 8]` | `[B, 103, 8, 8]` |
| Python policy logits (pre-softmax) | `[B, 4096]` | `[B, 4672]` |
| Python policy output (post-softmax) | `[B, 4096]` | `[B, 4672]` masked |
| `expand_leaf_batch` policy output | `[B, 4096]` | `[B, 4672]` (unmasked) |
| `Policy` type in Rust | `Vec<f32>` len=4096 | `Vec<f32>` len=4672 |
| MCTS node prior array | indexed by ActionIndex | indexed by ActionIndex (range now 0..4671) |
| MCTS visit distribution | sparse over 4096 | sparse over 4672 |
| `StepRecord.visit_distribution` | max len 4096 | max len 4672 |
| Batch assembly `pol_stride` in training.rs | 4096 (`NUM_ACTIONS`) | 4672 (via constant) |
| PyO3 `reshape policies_np` | `[B, K+1, 4096]` | `[B, K+1, 4672]` |
| Trainer `train_batch` obs | `[B, 19, 8, 8]` | `[B, 103, 8, 8]` |
| Trainer `train_batch` target_policies | `[B, K+1, 4096]` | `[B, K+1, 4672]` |
| Trainer `_policy_loss` logits | `[B, 4096]` | `[B, 4672]` masked |
| Checkpoint on disk | `h` state dict: Conv2d(19→64) | `h` state dict: Conv2d(103→64) |
| Checkpoint on disk | `f` state dict: Linear(128→4096) | `f` state dict: Linear(128→4672) |

Note on `expand_leaf_batch` masking: at depth > 0 the engine operates in the learned
latent space — there is no real board from which to derive legal moves. Masking there
would require tracking board state through K-step unroll, which is out of scope for
Batch 1. Leave `expand_leaf_batch` unmasked.

---

## 4. Migration Concerns

### Checkpoint incompatibility
Any existing checkpoint in `checkpoints/` has incompatible weight shapes:
- `h` (RepresentationNetwork): stem `Conv2d(19, 64)` → new `Conv2d(103, 64)` — different
  weight tensor shape, cannot be loaded
- `f` (PredictionNetwork): `Linear(128, 4096)` → `Linear(128, 4672)` — different shape

**Action required**: `rm -rf checkpoints/` before running `bash scripts/run_baseline.sh 900`.
This must be documented at the top of `plan.md` and checked by the orchestrator before
issuing the baseline run command. A `load_checkpoint` attempt on a pre-Batch-1 file will
raise a PyTorch shape mismatch error — non-silent, so it will be caught immediately.

### Tests that hardcode 19 planes or 4096 actions

Rust tests (will fail to compile after `NUM_OBS_PLANES` and `NUM_ACTIONS` change):
- `src/py/training.rs:463-465` — `b * 19 * 64` assertion string literal
- `src/py/training.rs:475-477` — `4096` in assertion string literal
- `src/py/training.rs:501` — comment `"only 10 entries (not 4096)"`
- `src/selfplay/inference.rs:173` — `policy.len(), NUM_ACTIONS` — already uses constant, OK
- `src/selfplay/inference.rs:195` — same pattern, OK

Python tests (will produce wrong shapes and assertion errors):
- `python/tests/test_training.py:17` — `np.random.randn(batch_size, 19, 8, 8)`
- `python/tests/test_training.py:20` — `np.full((..., 4096), 1.0/4096, ...)`
- `python/tests/test_inference.py:116` — `np.random.randn(4, 19, 8, 8)`
- `python/tests/test_inference.py:118` — `np.full((4, 4, 4096), 1.0/4096, ...)`
- `python/tests/test_inference.py:145` — `np.random.randn(4, 19, 8, 8)`
- `python/tests/test_inference.py:147` — `np.full((4, 4, 4096), 1.0/4096, ...)`

These are MINOR fixes: replace literals with config constants that are already imported
at the top of each test file.

### Assertions and debug_assert! that check shapes

- `src/py/inference_backend.rs:137-143` — runtime size check `b * hidden_stride` vs
  `hidden_data.len()`. This is fine — `hidden_stride = hidden_channels * 64` uses
  `self.hidden_channels`, not the obs plane count.
- `src/py/inference_backend.rs:144-150` — `b * policy_stride` check. `policy_stride =
  NUM_ACTIONS` — will automatically be 4672 after constant update. Fine.
- `src/py/training.rs:53-59` — `debug_assert!(steps.len() > unroll_k, ...)` — unaffected.
- `src/py/training.rs:463-477` — assert string literals with `19` and `4096` — these
  are in test code, need updating for correctness of the message (not for compilation).

### Underpromotion action index design

AlphaZero uses 8×8×73 = 4672. The 73 planes per square are:
- 56 queen-type moves (7 distances × 8 directions)
- 8 knight moves
- 9 underpromotion moves (3 pieces × 3 directions: straight, left diagonal, right diagonal)

For hyzero's simpler `from×to` encoding (`action = from_sq * 64 + to_sq`), the base
4096 already covers all queen and normal pawn moves. The additional 576 slots encode:
- Piece type: 0=Knight, 1=Bishop, 2=Rook (3 types)
- Promotion pawn file: 0-7 (8 files)
- Move direction: 0=straight, 1=left-capture, 2=right-capture (3 directions, but only
  valid captures go diagonal, straight goes to file 0-7 at back rank)

Proposed layout: `4096 + piece_type * 192 + file * 24 + direction * 8 + to_rank_offset`
— but the simplest correct encoding is:
`4096 + (piece_type * 8 * 8) + (from_file * 8) + to_file` where piece_type is 0/1/2
for Knight/Bishop/Rook. This gives 3×64 = 192 slots per promotion piece type... but
pawn promotions are only from rank 7→8 (white) or rank 2→1 (black), so `from_file`
is sufficient to distinguish. A pawn on file `f` can promote straight (to file f) or
capture diagonally (to file f±1). So:

**Proposed simple encoding**:
```
underpromo_action = 4096
                    + piece_idx * 64        (0=Knight, 1=Bishop, 2=Rook)
                    + from_file * 8
                    + to_file               (from_file or from_file ± 1)
```
This gives 3×64 = 192 slots, total = 4096 + 192 = 4288. But the roadmap says 4672.

**AlphaZero canonical 4672**: 4096 base + 3 pieces × 8 from_files × 8 to_files +
overcount handled by 8×8=64 per piece not 8×8=64. Actually 3×192 = 576, and 4096
+ 576 = 4672 matches. The 192 per piece type = 8 from_files × 24, but 24 = 3 target
positions (straight, left, right) × 8... This doesn't cleanly divide.

**Resolved**: AlphaZero's 576 comes from the 73 move-type planes applied only to
promotion positions. For this simpler `from×to` encoding:
- 3 non-queen piece types × 8 from-files × 24 target-slots per file-rank = doesn't work

**Practical design decision needed**: The simplest encoding that gives 4672 total and
is unambiguous is: reserve 576 slots, one per (piece_type, from_sq, to_sq) where
`from_sq` is always on rank 7 (white) or rank 1 (black), and `to_sq` is rank 8 or
rank 0. Since there are exactly 3 × (8 straight + 16 diagonal = 24 targets per color
but at most 3 files per from_sq)... In practice there are at most 48 legal promotion
targets per color (8 files × up to 3 destinations × 2 piece variants = complex).

**Orchestrator decision required**: The exact underpromotion index formula must be
confirmed before implementation. Options:
1. Keep `from×to` encoding: action = from_sq×64 + to_sq for queen promotions,
   plus a 3-bit extension embedded in a new action range 4096-4671 encoding
   (piece_type, from_sq_rank-relative, to_sq). There are only 3×8×3=72 legal
   white underpromotion moves and 72 black underpromotion moves, so 144 total
   possible underpromotion actions needed (not 576).
2. Use the AlphaZero 4672 formula directly. The 576 extra slots includes many
   illegal/unused combinations. The network must learn to suppress unused slots.
   This is equivalent to how the 4096 base already has many illegal from×to pairs.

**Recommendation**: Use option 2 (4096 + 576 = 4672) where the 576 encodes
`piece_type * (8 files * 24 moves_per_file)` = `piece_type * 192`. The `moves_per_file`
can be 8 ranks × 3 directions = 24. Many of these 576 will be illegal — the network
will learn to suppress them just as it suppresses illegal base moves. This matches
the roadmap spec and avoids needing to enumerate exactly which moves are legal at
encoding time.

---

## 5. Implementation Order

The strict ordering below ensures the codebase is never in a broken intermediate state:

### Step 1 — Update constants (no breakage yet)
Add new constants alongside the old ones in `src/data/types.rs`:
```
NUM_ACTIONS = 4672           (was 4096)
NUM_OBS_PLANES = 103         (was 19)
NUM_HISTORY_POSITIONS = 8
```
After this step, `cargo build` will still succeed — nothing references the new
value yet. All existing code still compiles.

### Step 2 — Underpromotion: encoding functions
Update `src/data/encoding.rs`:
- `move_to_action`: add underpromotion arm
- `action_to_move`: add underpromotion decode
- Add `encode_underpromo_action` and `decode_underpromo_action` helpers
- Add tests for each new promotion type (knight/bishop/rook, both colors)

`cargo test` will pass all existing tests; new tests for underpromotion are added here.

### Step 3 — get_legal_moves: emit all 4 promotions
Update `src/selfplay/game_task.rs::get_legal_moves`:
- For pawn on promotion rank, emit 4 moves: Queen (existing), Knight, Bishop, Rook
- Each uses `move_to_action` which now correctly maps to 0-4095 (queen) or 4096-4671
- Update `action_to_notation` to handle underpromotion range

`cargo test` — game tests still pass. The legal move count at startpos is still 20
(no promotions at start). A position with a pawn on 7th rank now produces 4× moves
instead of 1× — verify with a test.

### Step 4 — History: BoardSnapshot type and ring buffer
Add `BoardSnapshot` (lightweight: just piece bitboards, castling flags, EP target,
side-to-move) to `src/data/types.rs`. Update `encode_board` signature:
```rust
pub fn encode_board(
    board: &GameBoard,
    side_to_move: Color,
    history: &[BoardSnapshot],  // 0..7 past positions, oldest first
) -> BoardObservation
```
Implement the 103-plane layout. If `history.len() < 7`, fill missing past positions
with all-zeros planes.

Update `play_game` in `src/selfplay/game_task.rs`:
- Add `let mut history: VecDeque<BoardSnapshot> = VecDeque::with_capacity(7);`
- Before each `encode_board` call: `let hist_slice = history.make_contiguous();`
- After each move, push current board snapshot, pop front if len > 7

`cargo build` — compiles with new signature. The observation size is now 103×64=6592
floats per position.

### Step 5 — Wire legal-move mask through inference pipeline
Update `InferenceRequest::RootSetup` to carry `legal_mask: Vec<bool>` (len=NUM_ACTIONS).
Update `Evaluator::root_setup` signature. Update `ChannelEvaluator`, `RandomBackend`
(ignore mask), `play_game` call site (build mask from `legal_actions`).

`cargo build` — must update all `root_setup` call sites. Tests use `RandomEvaluator`
which ignores the mask, so existing game tests still pass.

### Step 6 — PyO3 bridge: propagate new shapes
Update `src/py/inference_backend.rs`:
- Replace hardcoded `19` with `NUM_OBS_PLANES`
- Pass `legal_masks` numpy array to Python `root_setup_batch`

Update `src/py/training.rs`:
- Replace hardcoded `19` with `NUM_OBS_PLANES`
- Add `legal_masks` to `BatchArrays` and `assemble_batch_arrays`
- Pass `legal_masks` in batch dict to Python trainer

`cargo build` — at this point the Rust side expects `root_setup_batch(obs, mask)`.
Python will break until step 7.

### Step 7 — Python: config, models, inference, trainer
All Python changes in one step to keep Python in a runnable state:
1. `config.py`: update `input_planes=103, num_actions=4672`
2. `models/representation.py`: update default `input_planes=103`
3. `models/prediction.py`: update default `num_actions=4672`
4. `inference/server.py`: add mask parameter, apply masking before softmax
5. `training/trainer.py`: update docstrings, apply mask in `_policy_loss`

`cd python && pytest` — must pass after this step.

### Step 8 — Fix all hardcoded literal tests
- `python/tests/test_training.py`: replace `19` and `4096` literals
- `python/tests/test_inference.py`: replace `19` and `4096` literals
- `src/py/training.rs` test assertion messages: update strings (non-breaking, just clarity)

### Step 9 — Full validation
`cargo test`, `cargo clippy -- -D warnings`, `cd python && pytest`, short self-play
smoke test.

---

## 6. Test Strategy

### Tests that will break (must be fixed in Step 8)

| File | Line | Current | After fix |
|---|---|---|---|
| `python/tests/test_training.py` | 17 | `19` literal | `INPUT_PLANES` from config |
| `python/tests/test_training.py` | 20 | `4096` literal | `NUM_ACTIONS` from config |
| `python/tests/test_inference.py` | 116 | `19` literal | `INPUT_PLANES` |
| `python/tests/test_inference.py` | 118 | `4096` literal | `NUM_ACTIONS` |
| `python/tests/test_inference.py` | 145 | `19` literal | `INPUT_PLANES` |
| `python/tests/test_inference.py` | 147 | `4096` literal | `NUM_ACTIONS` |

All these tests already import `INPUT_PLANES` and `NUM_ACTIONS` from `DEFAULT_CONFIG`
at lines 12-13 and 18-19 respectively — the fix is a 1-line literal substitution per case.

### Rust tests that will need updating

- `src/py/training.rs:test_batch_assembly_shapes` — assertion message strings with
  `19 * 64` and `4096` need updating. The logic uses `NUM_OBS_PLANES` and `NUM_ACTIONS`
  constants — just the assertion message strings need text update.

### New tests to add

**Underpromotion encoding (`src/data/encoding.rs` tests)**:
- `test_move_to_action_knight_promotion` — White pawn e7→e8 with promotion=Knight
  should return action in range [4096, 4672)
- `test_move_to_action_bishop_promotion` — same for Bishop
- `test_move_to_action_rook_promotion` — same for Rook
- `test_action_to_move_knight_underpromo_roundtrip` — encode then decode returns
  same Move (from/to/piece_type)
- `test_action_to_move_black_underpromo` — Black pawn h2→h1 knight promotion

**Legal moves with promotion (`src/selfplay/game_task.rs` tests)**:
- `test_legal_moves_promotion_position` — set up a position with white pawn on e7,
  enemy free rank, verify `get_legal_moves` returns 4× the expected promotion squares
  (queen + 3 underpromotion options each) plus other legal moves

**History encoding (`src/data/encoding.rs` tests)**:
- `test_encode_board_no_history` — with empty history slice, planes 12-95 are all zeros,
  planes 96-102 contain current position's castling/EP/etc (same as old behavior for
  those planes)
- `test_encode_board_with_history` — provide 3 past snapshots; verify planes 12-23 match
  past position 1's pieces, planes 24-35 match past position 2, etc.
- `test_encode_board_plane_count` — output length is exactly `NUM_OBS_PLANES * 64`

**Masking (`python/tests/test_inference.py` additions)**:
- `test_root_setup_batch_masked_policy_sums_to_one` — pass a mask that allows 5 actions;
  verify returned policy sums to 1.0 and illegal positions have probability ~0
- `test_root_setup_batch_all_masked_raises` — all-false mask; should not produce NaN
  (softmax of all-inf → NaN, must handle gracefully — at minimum document behavior)

**Masking in training (`python/tests/test_training.py` additions)**:
- `test_train_batch_with_masks` — add `"legal_masks"` key to batch dict with random
  bool masks; verify training completes without NaN losses

### End-to-end verification

1. `cargo test` — all 82 passing tests must continue passing + new encoding/history tests
2. `cargo clippy -- -D warnings` — zero warnings required
3. `cd python && pytest` — all tests pass with updated shapes
4. Short self-play smoke test: run `cargo run --release --bin selfplay` for 60 seconds;
   verify logs show games completing, trajectories being added to replay buffer,
   training steps running (policy loss reported as finite). This exercises the full
   Rust→Python→Rust inference and training loop with the new shapes.

---

## 7. Risks and Mitigation

### Risk 1: OOM from 5× larger input tensors
`BoardObservation` grows from 1216 to 6592 floats (5.4×). The replay buffer holds up
to 10,000 trajectories. If each game averages 100 moves, that is 10,000 × 100 × 6592
× 4 bytes ≈ 26 GB for observations alone.

**Mitigation**: Store history as references or indices rather than copying full boards
into every `BoardObservation`. The `BoardObservation` in `StepRecord` should store
only the 103-plane encoding once (the current observation, computed once at record
time). History snapshots in the ring buffer are transient per-game and don't persist
to `StepRecord`. This is the current behavior — `encode_board` is called once per
step and the result stored. The size change is real: from ~4.9 GB to ~26 GB for
10k trajectories at 100 steps each.

**Alternative**: Reduce `max_replay_trajectories` in `PyTrainingThread::from_default_config`
from 10,000 to 2,000 as part of this batch, trading off replay diversity for memory.
Or store raw board snapshots in `StepRecord` and lazily encode before training — but
this requires a schema change to the replay buffer and `StepRecord` serialization.

**Decision needed (orchestrator)**: Accept larger memory footprint, or reduce
`max_replay_trajectories`? For the 15-minute baseline run (`run_baseline.sh 900`),
game throughput is limited — the buffer may not fill to 10k anyway. Proceed with
default 10k and observe actual memory usage.

### Risk 2: History ring buffer off-by-one
The ring buffer fills from index 0 (oldest) to index 6 (most recent). `encode_board`
must pass the history slice such that index 0 is the OLDEST position (written first
into planes 12-23) and index len-1 is the MOST RECENT past position (written into
the highest history plane block). Reversing this would invert temporal ordering.

**Mitigation**: Add an explicit test (`test_encode_board_temporal_order`) that plays
3 moves and verifies that planes 12-23 reflect the position AFTER move 1, not move 3.

### Risk 3: Underpromotion action index collision
If the underpromotion index formula wraps around or overlaps with base 4096 range due
to an off-by-one, `action_to_move` will decode the wrong piece type. Knight/Bishop/Rook
could silently interchange.

**Mitigation**: Add roundtrip tests for all 3 piece types and both colors. Also add a
test that verifies decoded piece type == expected piece type (not just any valid piece).

### Risk 4: Masking NaN when all legal moves are masked
If `legal_mask` is all-False (bug), softmax of all `-inf` produces NaN, which
propagates to all downstream computations.

**Mitigation**: Add a guard in `server.py::root_setup_batch`:
```python
if not legal_masks.any(axis=-1).all():
    # At least one position has no legal moves — fall back to uniform
    fallback = ~legal_masks.any(axis=-1, keepdims=True)
    legal_masks = legal_masks | fallback  # allow all moves for empty-mask positions
```
Log a warning when this fires. In practice, `legal_masks` should never be all-False
because the game checks `legal_actions.is_empty()` before requesting inference.

### Risk 5: Training slowdown from 5× larger input
With 103 input planes instead of 19, the first convolutional layer processes 5.4×
more data per forward pass. On CPU (development), this will slow training noticeably.
On GPU (production), this is less significant but still measurable.

**Mitigation**: This is expected and acceptable for correctness — the whole point of
the history planes is richer signal. Document in the baseline run log that per-step
time increases. The 15-minute baseline window (`run_baseline.sh 900`) may produce
fewer training steps than the previous 30-minute run — adjust expectations for
baseline score comparison accordingly.

### Risk 6: Stale checkpoints loaded by accident
`checkpoints/` contains pre-Batch-1 model files. If a future operator resumes with
`--resume-checkpoint`, PyTorch will raise a shape mismatch error. This error is
loud (Python exception) not silent, but could waste debugging time.

**Mitigation**: As part of this batch, rename or delete all files in `checkpoints/`
at the start of the first baseline run. Document this in the commit message:
"BREAKING: old checkpoints incompatible, wipe checkpoints/ before resuming."

---

## Subtasks Summary

| # | Name | Files | Dependencies |
|---|---|---|---|
| 1 | Update constants | `src/data/types.rs` | None |
| 2 | Underpromotion encoding | `src/data/encoding.rs` | 1 |
| 3 | Legal move generation | `src/selfplay/game_task.rs` | 1, 2 |
| 4 | History planes encoding | `src/data/encoding.rs`, `src/data/types.rs` | 1 |
| 5 | Legal-mask pipeline (Rust) | `src/selfplay/inference.rs`, `src/selfplay/game_task.rs`, `src/mcts/evaluator.rs` | 3 |
| 6 | PyO3 bridge shape update | `src/py/inference_backend.rs`, `src/py/training.rs` | 1, 4, 5 |
| 7 | Python network+training update | `python/hyzero/config.py`, `models/representation.py`, `models/prediction.py`, `inference/server.py`, `training/trainer.py` | 6 |
| 8 | Fix test literals | `python/tests/test_training.py`, `python/tests/test_inference.py`, `src/py/training.rs` test messages | 7 |

**Disjoint file sets**: Steps 2 and 4 both touch `src/data/encoding.rs` — they must
be sequential (encode underpromotion first, then history). The implementer may combine
steps 2+4 into a single `encoding.rs` edit to avoid two passes.
