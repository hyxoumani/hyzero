# Plan: Syzygy Tablebase Supervision

## Approach

Inject Syzygy 3-4-5-man tablebase positions as exact ±1 supervision samples into
every training batch to break the value-head distributional collapse (kqk_value
≈ −0.012, stuck since step ~650). A fraction `HYZERO_TABLEBASE_FRAC` (default 0.0,
first run 0.1) of each batch is replaced with TB pseudo-trajectories using Option B:
synthesize a 1-step rollout where `a_0` = mating action and `r_1` = +1. This directly
exercises the reward head via g while letting the value head see ±1 targets it cannot
avoid. Consistency loss is zeroed for TB samples to avoid forcing the trunk to match
arbitrary latents.

---

## Key Data-Flow Facts

The Rust side assembles batches in `assemble_batch_arrays` (`src/py/training.rs:103`)
from `ReplayBuffer` samples, then calls `train_batch_python` which converts to numpy
and calls `trainer.train_batch(batch_dict)`. The Python side never does batch assembly
itself — today. The TB mix must happen in Python, *after* the Rust side delivers its
portion of the batch, by concatenating a Python-built TB sub-batch into the same dict
before the forward pass runs. The cleanest insertion point is at the top of
`trainer.train_batch`, before any tensor conversion.

Batch shape contract (K=5 for replay samples, K_TB=1 for TB samples):

```
observations:    [B, K+1, 102, 8, 8]
actions:         [B, K,   3,  8, 8]
target_policies: [B, K+1, 4672]
target_values:   [B, K+1]
target_rewards:  [B, K+1]
legal_masks:     [B, 4672] bool   (optional)
is_tablebase:    [B] bool          (NEW — present only when TB samples exist)
```

TB samples use K_TB=1. They must be zero-padded in the batch axis to match the replay
K (trainer reads `actions.shape[1]` as k_steps). Padding approach: replicate the
root obs in all K+1 slots; pad actions with zeros; pad targets with zeros for k>1;
set `is_tablebase[i]=True`. The trainer already sums policy/value loss over all K+1
steps, so zeros at padded steps contribute zero MSE (target=0, value output≈0 initially
gives low gradient — acceptable, since value at root is what matters).

---

## Subtasks

### 1. Extract Python encoder module

**Files**: create `python/hyzero/data/__init__.py`, create `python/hyzero/data/board_encoder.py`

**Changes**:

The brief references `scripts/reward_probe.py` for `encode_board_python` and
`encode_action_spatial` — that file does not exist on disk. The canonical Python
encoder must be written from scratch, mirroring `src/data/encoding.rs:encode_board`
and `encode_action_spatial_for_color`.

`board_encoder.py` must implement:

- `encode_board_python(board: chess.Board) -> np.ndarray` — returns `[102, 8, 8]`
  float32. Side-to-move perspective (AlphaZero convention). Plane layout:
  - Groups 0–7 (12 planes each): current-player pieces (planes 0–5) + opponent
    pieces (planes 6–11). Group 0 = current position; groups 1–7 = history (all zeros
    for TB samples with no history). Black-to-move: rank-mirror all square indices
    (`sq → 63 - sq`).
  - Planes 96–99: castling rights (current-player ks, qs; opponent ks, qs). Fill entire
    8×8 plane with 1.0 if right exists.
  - Plane 100: en-passant target square (one-hot, rank-mirrored for Black).
  - Plane 101: halfmove clock / 100.
  - **Must match Rust byte-for-byte**: verify against `_build_kqk_white_winning_obs()`
    in `trainer.py:106-137` (that function IS the ground truth for KQK-white-to-move).

- `encode_action_spatial(action: int, white_to_move: bool) -> np.ndarray` — returns
  `[3, 8, 8]` float32. Mirrors `encode_action_spatial_for_color` in
  `src/data/encoding.rs`. Plane layout: plane 0 = from-square, plane 1 = to-square,
  plane 2 = promotion flag. Rank-flip for black (`!white_to_move`).

- `action_from_move(move: chess.Move, board: chess.Board) -> int` — converts a
  python-chess Move to the hyzero 4672-action integer (from_sq * 64 + to_sq for
  queen/default promotions; underpromo encoding for knight/bishop/rook promos).

**Tests**: `python/tests/test_tablebase.py` — `test_tablebase_encoding_roundtrip`
(KQK position encodes identically to `_build_kqk_white_winning_obs` in trainer).

**Dependencies**: none

---

### 2. Build tablebase cache script

**Files**: create `scripts/build_tablebase_cache.py`

**Changes**:

One-time precompute script. Run with `HYZERO_TABLEBASE_PATH=/path/to/syzygy python3 scripts/build_tablebase_cache.py`.

Download source: `https://tablebase.lichess.ovh/` (3-4-5-man `.rtbw`/`.rtbz` files,
~400MB). Cache location: `data/syzygy/cache.pkl` (outside repo; add `data/` to
`.gitignore` if not already there).

Algorithm:

1. Open TB: `tb = chess.syzygy.open_tablebase(os.environ["HYZERO_TABLEBASE_PATH"])`.
2. Enumerate N_TOTAL=500_000 positions across these endgame classes (piece sets in
   python-chess notation):
   - KQK: 80k, KRK: 80k, KBBK: 40k, KBNK: 40k, KPK: 80k
   - KRKP: 60k, KQKR: 60k, KRKR: 60k
3. For each class, generate positions by:
   a. Place kings on two non-adjacent squares (distance > 1). Reject adjacent.
   b. Place remaining pieces on random empty squares.
   c. Build `chess.Board` from FEN; verify `board.is_valid()`.
   d. Reject if the side NOT to move is in check (illegal position).
   e. Try both sides to move.
4. For each valid position, probe:
   - `wdl = tb.probe_wdl(board)` — returns int ∈ {-2, -1, 0, 1, 2}
   - Map to value target: `target_value = +1 if wdl > 0 else (-1 if wdl < 0 else 0)`
     (side-to-move POV; python-chess WDL is already from STM perspective — verified
     by: positive wdl when STM wins).
   - `dtz = tb.probe_dtz(board)` — distance to zero.
   - Legal moves: `list(board.legal_moves)`.
   - Mating moves (mate-in-1): moves where `board.gives_check(m)` AND after push the
     position is checkmate. Collect as `mating_actions: list[int]`.
   - Optimal policy: moves with `|dtz| == min(|dtz_after_move|)`. Uniform distribution
     over optimal moves (others 0). If `probe_dtz` raises, fall back to uniform over
     all legal moves.
5. Serialize each position as a `TBSample` dataclass (see Subtask 3).
6. Pickle list to `HYZERO_TABLEBASE_CACHE_PATH` (default `data/syzygy/cache.pkl`).

Note: wrap all TB probes in try/except — some positions (underpromotion pieces in
wrong rank, etc.) can raise `chess.syzygy.MissingTableError`. Skip those.

**Dependencies**: Subtask 1 (needs `action_from_move`)

---

### 3. Tablebase loader and batch builder

**Files**: create `python/hyzero/data/tablebase.py`

**Changes**:

```python
@dataclass
class TBSample:
    fen: str                             # position FEN
    target_value: float                  # ±1 or 0 (STM POV)
    mating_actions: list[int]            # action indices of mate-in-1 moves (may be empty)
    optimal_actions: list[int]           # action indices of optimal-DTZ moves
    all_legal_actions: list[int]         # all legal action indices
```

```python
class TablebaseCache:
    def __init__(self, path: str) -> None: ...   # loads pickle
    def sample(self, n: int) -> list[TBSample]: ...  # random.sample without replacement
    def __len__(self) -> int: ...
```

```python
def build_tb_batch(
    samples: list[TBSample],
    k_steps: int,                      # must match replay batch K (e.g. 5)
    num_actions: int = 4672,
) -> dict[str, np.ndarray]:
```

`build_tb_batch` produces arrays shaped for `k_steps` (same K as replay), padded:

- `observations [N, K+1, 102, 8, 8]`: step 0 = real encode; steps 1..K = zeros.
- `actions [N, K, 3, 8, 8]`: step 0 = encode mating move if exists else encode best
  optimal move; steps 1..K-1 = zeros.
- `target_policies [N, K+1, 4672]`: step 0 = uniform over optimal_actions (0 elsewhere);
  steps 1..K = zeros.
- `target_values [N, K+1]`: step 0 = target_value; steps 1..K = 0.0.
- `target_rewards [N, K+1]`: step 0 = 0.0; step 1 = +1.0 if mating_actions else 0.0;
  steps 2..K = 0.0.
- `legal_masks [N, 4672]` bool: True for all_legal_actions at step 0.
- `is_tablebase [N]` bool: all True.

Design choice (Option B): reward target at step 1 is +1 when a mating action exists.
This means `g(h(pos), mating_action)` gets gradient toward reward=+1 — directly
repairing the reward head. When no mate-in-1 exists, reward target at step 1 = 0
(neutral; still forces value head via step-0 target).

**Dependencies**: Subtask 1

---

### 4. Trainer integration

**Files**: `python/hyzero/training/trainer.py`

**Changes**:

**`__init__` (around line 386-458)**:

Add after existing init:

```python
tb_path = os.environ.get("HYZERO_TABLEBASE_PATH")
tb_cache_path = os.environ.get("HYZERO_TABLEBASE_CACHE_PATH", "data/syzygy/cache.pkl")
self._tb_cache: TablebaseCache | None = None
if tb_path is not None:
    from hyzero.data.tablebase import TablebaseCache
    if os.path.exists(tb_cache_path):
        self._tb_cache = TablebaseCache(tb_cache_path)
        print(f"[trainer] tablebase cache loaded: {len(self._tb_cache)} positions")
    else:
        print(f"[trainer] WARN: HYZERO_TABLEBASE_PATH set but cache not found at {tb_cache_path}; TB supervision disabled")

self._tb_frac = float(os.environ.get("HYZERO_TABLEBASE_FRAC", "0.0"))
```

**`train_batch` (top of method, around line 477-510)**:

Before any tensor conversion, insert TB mixing:

```python
batch, tb_indices = self._maybe_mix_tb_samples(batch)
```

New private method `_maybe_mix_tb_samples`:

```python
def _maybe_mix_tb_samples(
    self, batch: dict
) -> tuple[dict, set[int]]:
    """Replace tb_frac fraction of batch with TB samples.

    Returns updated batch dict and set of indices that are TB samples
    (used to zero consistency loss for those rows).
    """
    if self._tb_cache is None or self._tb_frac <= 0.0:
        return batch, set()

    b = batch["observations"].shape[0]
    k_steps = batch["actions"].shape[1]
    n_tb = max(1, int(b * self._tb_frac))
    n_tb = min(n_tb, b)

    from hyzero.data.tablebase import build_tb_batch
    tb_samples = self._tb_cache.sample(n_tb)
    tb_dict = build_tb_batch(tb_samples, k_steps=k_steps)

    # Replace last n_tb rows of the replay batch with TB rows.
    tb_indices = set(range(b - n_tb, b))
    merged = {}
    for key in ("observations", "actions", "target_policies",
                "target_values", "target_rewards"):
        merged[key] = np.concatenate(
            [batch[key][:b - n_tb], tb_dict[key]], axis=0
        )
    # legal_masks: optional in replay batch, always present in TB
    replay_masks = batch.get("legal_masks")
    if replay_masks is not None:
        merged["legal_masks"] = np.concatenate(
            [replay_masks[:b - n_tb], tb_dict["legal_masks"]], axis=0
        )
    else:
        merged["legal_masks"] = tb_dict["legal_masks"]  # only TB rows have masks
    merged["is_tablebase"] = np.zeros(b, dtype=bool)
    merged["is_tablebase"][b - n_tb:] = True
    return merged, tb_indices
```

**Consistency loss block (around lines 737-750)**:

Wrap the consistency accumulation so TB rows (where `obs_all[:, k_idx]` is zeros) are
excluded:

```python
if consistency_weight > 0 and k_steps > 0:
    is_tb_mask = batch.get("is_tablebase")
    is_tb_tensor = (
        torch.from_numpy(is_tb_mask).to(self.device)
        if is_tb_mask is not None else None
    )
    for k_idx in range(1, k_steps + 1):
        dyn_latent_k = hidden_states[k_idx]
        p1 = self.h.predict(self.h.project(dyn_latent_k))
        obs_k = obs_all[:, k_idx]
        target_latent = self.h(obs_k)
        p2 = self.h.project(target_latent).detach()
        cos_sim = F.cosine_similarity(p1, p2, dim=-1)  # [B]
        if is_tb_tensor is not None:
            cos_sim = cos_sim[~is_tb_tensor]  # exclude TB rows
        if cos_sim.numel() > 0:
            consistency_loss = consistency_loss + (1 - cos_sim.mean())
    if k_steps > 0:
        consistency_loss = consistency_loss / k_steps
```

Also strip `is_tablebase` from batch before the existing tensor-conversion block to
avoid passing it to numpy→tensor conversion (it's only used in Python):

```python
# Pop is_tablebase before tensor conversion (Python-only field).
is_tb_mask = batch.pop("is_tablebase", None)
```

**Dependencies**: Subtask 3

---

### 5. Tests

**Files**: `python/tests/test_tablebase.py`

**Changes**:

Four tests (all run without a real TB; use hand-crafted `TBSample` objects):

1. `test_tablebase_encoding_roundtrip`:
   Build FEN for KQK (white Ke1, Qa2, black Ke8, white to move). Call
   `encode_board_python(board)`. Compare to `_build_kqk_white_winning_obs` from trainer
   (already validated against Rust). Assert `np.allclose(result, expected, atol=1e-6)`.

2. `test_tablebase_value_target_sign`:
   KQK white-to-move: WDL returned by python-chess is +2 → `target_value = +1`.
   Black-to-move (same material, flipped): WDL = −2 → `target_value = −1` from black's
   POV. Verify by constructing `TBSample` manually and checking sign matches convention:
   a positive value means side-to-move is winning.

3. `test_tablebase_reward_per_action`:
   Construct a simple `TBSample` with two `mating_actions=[42]` and
   `all_legal_actions=[42, 99]`. Call `build_tb_batch([sample], k_steps=5)`.
   Assert `target_rewards[0, 1] == +1.0` (step 1 reward for the mating action path).
   Assert `target_rewards[0, 0] == 0.0` and `target_rewards[0, 2:] == 0.0`.

4. `test_mixed_batch_shapes`:
   Create a Trainer with a mock `_tb_cache` that returns 2 `TBSample` objects.
   Create a replay batch of size 8, k_steps=5. Monkeypatch `_tb_frac=0.25`.
   Call `trainer._maybe_mix_tb_samples(batch)`. Assert merged shapes are correct:
   `observations: [8, 6, 102, 8, 8]`, `is_tablebase: [8]` with last 2 True.

**Dependencies**: Subtasks 1, 3, 4

---

## Env Vars

| Variable | Default | Notes |
|----------|---------|-------|
| `HYZERO_TABLEBASE_PATH` | unset | Directory with `.rtbw`/`.rtbz` files. TB disabled if unset. |
| `HYZERO_TABLEBASE_FRAC` | `0.0` | Fraction of each batch from TB. Recommended first run: `0.1`. |
| `HYZERO_TABLEBASE_CACHE_PATH` | `data/syzygy/cache.pkl` | Path to pickled position cache. |

---

## Tablebase Download

```bash
# 3-4-5-man Syzygy (~400MB total)
mkdir -p data/syzygy
cd data/syzygy
# WDL tables (required for probe_wdl)
for pieces in KQvK KRvK KBBvK KBNvK KPvK KRvKP KQvKR KRvKR; do
  wget -q "https://tablebase.lichess.ovh/tables/standard/3-4-5/${pieces}.rtbw"
done
# DTZ tables (required for probe_dtz; needed for optimal-move policy targets)
for pieces in KQvK KRvK KBBvK KBNvK KPvK KRvKP KQvKR KRvKR; do
  wget -q "https://tablebase.lichess.ovh/tables/standard/3-4-5/${pieces}.rtbz"
done
```

After download, build the cache:

```bash
HYZERO_TABLEBASE_PATH=data/syzygy python3 scripts/build_tablebase_cache.py
```

---

## Testing Strategy

### Unit (no TB required)

```bash
cd python && pytest python/tests/test_tablebase.py -v
```

All four tests must pass without a real tablebase; they use hand-constructed samples.

### Smoke run (TB required, ~120s)

```bash
HYZERO_TABLEBASE_PATH=data/syzygy \
HYZERO_TABLEBASE_FRAC=0.1 \
HYZERO_TABLEBASE_CACHE_PATH=data/syzygy/cache.pkl \
bash scripts/smoke_dual_eval.sh
```

Grep for `[kqk_value]` in output. Target: value > +0.5 after 1000 steps (currently
−0.012).

### Full validation (1800s)

```bash
HYZERO_TABLEBASE_PATH=data/syzygy \
HYZERO_TABLEBASE_FRAC=0.1 \
bash scripts/run_baseline.sh 1800
```

---

## Success Criteria (first 1000 steps from `best_v1489.pt`)

| Metric | Before | Target |
|--------|--------|--------|
| `[kqk_value]` | −0.012 | > +0.5 |
| `[start_value]` | ~0 | ±0.2 range (non-trivial) |
| policy_loss | ≤4.0 | ≤4.0 (no regression) |

If `kqk_value` does not move toward +1 after 10% TB in batches over 1000 steps, the
value head architecture itself has a problem that tablebase injection cannot fix (likely
the tanh saturation + zero-initialized bias in `PredictionNetwork.value_head`).

---

## Stale Wiki Finding

`docs/wiki/neural-networks.md` and `docs/wiki/board-encoding.md` both state 19 input
planes. The actual codebase uses **102 planes** (8 history groups × 12 + 6 game state).
`config.py:input_planes=102`, `trainer.py` uses `[B, 102, 8, 8]` throughout. The wiki
"Network Shapes" table is stale: `h: Conv2d(19→64, ...)` should read `102→128`. Flag
for context-keeper.
