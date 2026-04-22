"""MuZero training loop with K-step unroll, loss computation, and checkpointing."""

import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def _diag_print(msg: str) -> None:
    """Write a diagnostic line to stdout via multiple fallback paths.

    PyO3 may redirect Python's sys.stdout. This helper tries multiple output
    channels so diagnostic lines appear in the selfplay log regardless of how
    PyO3 initialises the Python interpreter's stdio.

    Channels tried in order:
      1. print() to sys.stdout with flush (goes through Python's stdout object)
      2. sys.__stdout__ (original unwrapped stdout)
      3. sys.__stderr__ (stderr, also captured by `2>&1`)
      4. os.write(1, ...) and os.write(2, ...) — raw fd writes
    """
    line = msg + "\n"
    # Try Python-level streams first (most compatible).
    for stream in (sys.stdout, sys.__stdout__, sys.stderr, sys.__stderr__):
        if stream is not None:
            try:
                stream.write(line)
                stream.flush()
                return
            except Exception:
                pass
    # Raw fd fallback.
    encoded = line.encode("utf-8", errors="replace")
    for fd in (1, 2):
        try:
            os.write(fd, encoded)
            return
        except OSError:
            pass

from hyzero.config import DEFAULT_CONFIG
from hyzero.models.representation import RepresentationNetwork
from hyzero.models.dynamics import DynamicsNetwork
from hyzero.models.prediction import PredictionNetwork


def _flip_obs_planes(obs: torch.Tensor) -> torch.Tensor:
    """Port of Rust flip_obs_planes from src/data/encoding.rs:353-399.

    Flips a board observation to the opponent's perspective:
      - For each of the 8 history groups (12 planes each, base = group*12):
          planes 0-5 (my pieces) swap with planes 6-11 (opp pieces), both rank-mirrored.
      - Castling planes (constant-fill, no rank mirror): 96↔98, 97↔99.
      - Plane 100 (en passant): rank-mirrored.
      - Plane 101 (halfmove clock): unchanged.

    Args:
        obs: [N, 102, 8, 8] float32 tensor (on any device).

    Returns:
        [N, 102, 8, 8] float32 tensor with the same device/dtype.
    """
    N = obs.shape[0]
    out = torch.zeros_like(obs)

    # Helper: rank-mirror a [N, 8, 8] spatial plane by reversing rank dimension (dim=1).
    def rank_mirror(plane: torch.Tensor) -> torch.Tensor:
        # plane: [N, 8, 8]; flip along rank (dim=1)
        return plane.flip(1)

    # 8 history groups × 12 planes each
    for group in range(8):
        base = group * 12
        for pt in range(6):
            my_idx = base + pt          # planes 0-5 within group
            opp_idx = base + 6 + pt     # planes 6-11 within group

            my_plane = obs[:, my_idx]    # [N, 8, 8]
            opp_plane = obs[:, opp_idx]  # [N, 8, 8]

            # my-pieces go to opp slot, rank-mirrored
            out[:, opp_idx] = rank_mirror(my_plane)
            # opp-pieces go to my slot, rank-mirrored
            out[:, my_idx] = rank_mirror(opp_plane)

    # Castling planes: swap pairs (no rank mirror)
    # 96 (my kingside) ↔ 98 (opp kingside)
    # 97 (my queenside) ↔ 99 (opp queenside)
    out[:, 96] = obs[:, 98]
    out[:, 97] = obs[:, 99]
    out[:, 98] = obs[:, 96]
    out[:, 99] = obs[:, 97]

    # Plane 100: en passant — rank-mirror
    out[:, 100] = rank_mirror(obs[:, 100])

    # Plane 101: halfmove clock — unchanged
    out[:, 101] = obs[:, 101]

    return out


def _build_kqk_white_winning_obs(device: str) -> torch.Tensor:
    """Build the KQ-vs-K position observation [1, 102, 8, 8] (white to move, white winning).

    White: King on e1 (sq 4), Queen on a2 (sq 8).
    Black: King on e8 (sq 60).
    White to move, empty history, no castling, no en passant, halfmove 0.

    Plane layout (group 0 only; groups 1-7 are zeros):
      Plane 4  (my queen):   a2 = rank 1, file 0  → obs[0, 4, 1, 0] = 1
      Plane 5  (my king):    e1 = rank 0, file 4  → obs[0, 5, 0, 4] = 1
      Plane 11 (opp king):   e8 = rank 7, file 4  → obs[0, 11, 7, 4] = 1
      All castling planes (96-99): zeros (no castling rights in this position).
      Plane 100: zeros (no en passant).
      Plane 101: zeros (halfmove 0).

    Args:
        device: Torch device string.

    Returns:
        [1, 102, 8, 8] float32 tensor.
    """
    obs = torch.zeros(1, 102, 8, 8, dtype=torch.float32, device=device)
    # Plane 4: my queen at a2 (rank=1, file=0); sq 8 = rank*8+file = 1*8+0
    obs[0, 4, 1, 0] = 1.0
    # Plane 5: my king at e1 (rank=0, file=4); sq 4 = 0*8+4
    obs[0, 5, 0, 4] = 1.0
    # Plane 11: opp king at e8 (rank=7, file=4); sq 60 = 7*8+4
    obs[0, 11, 7, 4] = 1.0
    # Castling planes 96-99: zeros (no castling in this endgame position)
    # Plane 100: no en passant (zeros already)
    # Plane 101: halfmove clock = 0 (zeros already)
    return obs


def _build_k_vs_kq_white_losing_obs(device: str) -> torch.Tensor:
    """Build the K-vs-KQ position observation [1, 102, 8, 8] (white to move, white losing).

    White: King on e1 (sq 4).
    Black: King on e8 (sq 60), Queen on a7 (sq 48).
    White to move, white is down a queen.

    Plane layout (group 0 only; groups 1-7 are zeros):
      Plane 5  (my king):    e1 = rank 0, file 4  → obs[0, 5, 0, 4] = 1
      Plane 10 (opp queen):  a7 = rank 6, file 0  → obs[0, 10, 6, 0] = 1 (sq 48 = 6*8+0)
      Plane 11 (opp king):   e8 = rank 7, file 4  → obs[0, 11, 7, 4] = 1
      All castling planes (96-99): zeros.
      Plane 100: zeros (no en passant).
      Plane 101: zeros (halfmove 0).

    Args:
        device: Torch device string.

    Returns:
        [1, 102, 8, 8] float32 tensor.
    """
    obs = torch.zeros(1, 102, 8, 8, dtype=torch.float32, device=device)
    # Plane 5: my king at e1 (rank=0, file=4); sq 4 = 0*8+4
    obs[0, 5, 0, 4] = 1.0
    # Plane 10: opp queen at a7 (rank=6, file=0); sq 48 = 6*8+0
    obs[0, 10, 6, 0] = 1.0
    # Plane 11: opp king at e8 (rank=7, file=4); sq 60 = 7*8+4
    obs[0, 11, 7, 4] = 1.0
    # Castling planes 96-99: zeros (no castling in this endgame position)
    # Plane 100: no en passant (zeros already)
    # Plane 101: halfmove clock = 0 (zeros already)
    return obs


def _build_start_obs(device: str) -> torch.Tensor:
    """Build the standard chess starting position observation [1, 102, 8, 8].

    Piece-type plane index order matches pieces_bb in src/game/playerobj.rs:
      0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King

    Layout (group 0, White-to-move POV; all other 7 history groups are zeros):
      Plane 0  (my pawns):    rank 1 (row index 1), all 8 files
      Plane 1  (my knights):  b1=sq1, g1=sq6
      Plane 2  (my bishops):  c1=sq2, f1=sq5
      Plane 3  (my rooks):    a1=sq0, h1=sq7
      Plane 4  (my queen):    d1=sq3
      Plane 5  (my king):     e1=sq4
      Plane 6  (opp pawns):   rank 6 (row index 6), all 8 files
      Plane 7  (opp knights): b8=sq57, g8=sq62
      Plane 8  (opp bishops): c8=sq58, f8=sq61
      Plane 9  (opp rooks):   a8=sq56, h8=sq63
      Plane 10 (opp queen):   d8=sq59
      Plane 11 (opp king):    e8=sq60
      Planes 12-95: zeros (history steps 1-7 empty)
      Planes 96-99: ones (all castling rights)
      Plane 100: zeros (no en passant)
      Plane 101: zeros (halfmove clock = 0)

    Squares are row-major: sq = rank*8 + file (rank 0 = rank-1 in chess notation).

    Args:
        device: Torch device string.

    Returns:
        [1, 102, 8, 8] float32 tensor.
    """
    obs = torch.zeros(1, 102, 8, 8, dtype=torch.float32, device=device)

    # --- my pieces (White, group 0, planes 0-5) ---
    # Plane 0: my pawns at rank 1, all files
    obs[0, 0, 1, :] = 1.0
    # Plane 1: my knights at b1 (rank=0, file=1) and g1 (rank=0, file=6)
    obs[0, 1, 0, 1] = 1.0
    obs[0, 1, 0, 6] = 1.0
    # Plane 2: my bishops at c1 (rank=0, file=2) and f1 (rank=0, file=5)
    obs[0, 2, 0, 2] = 1.0
    obs[0, 2, 0, 5] = 1.0
    # Plane 3: my rooks at a1 (rank=0, file=0) and h1 (rank=0, file=7)
    obs[0, 3, 0, 0] = 1.0
    obs[0, 3, 0, 7] = 1.0
    # Plane 4: my queen at d1 (rank=0, file=3)
    obs[0, 4, 0, 3] = 1.0
    # Plane 5: my king at e1 (rank=0, file=4)
    obs[0, 5, 0, 4] = 1.0

    # --- opp pieces (Black, group 0, planes 6-11) ---
    # Plane 6: opp pawns at rank 6, all files
    obs[0, 6, 6, :] = 1.0
    # Plane 7: opp knights at b8 (rank=7, file=1) and g8 (rank=7, file=6)
    obs[0, 7, 7, 1] = 1.0
    obs[0, 7, 7, 6] = 1.0
    # Plane 8: opp bishops at c8 (rank=7, file=2) and f8 (rank=7, file=5)
    obs[0, 8, 7, 2] = 1.0
    obs[0, 8, 7, 5] = 1.0
    # Plane 9: opp rooks at a8 (rank=7, file=0) and h8 (rank=7, file=7)
    obs[0, 9, 7, 0] = 1.0
    obs[0, 9, 7, 7] = 1.0
    # Plane 10: opp queen at d8 (rank=7, file=3)
    obs[0, 10, 7, 3] = 1.0
    # Plane 11: opp king at e8 (rank=7, file=4)
    obs[0, 11, 7, 4] = 1.0

    # Planes 96-99: castling rights, all ones
    obs[0, 96, :, :] = 1.0
    obs[0, 97, :, :] = 1.0
    obs[0, 98, :, :] = 1.0
    obs[0, 99, :, :] = 1.0

    # Plane 100: no en passant (zeros already)
    # Plane 101: halfmove clock = 0 (zeros already)

    return obs


def _parse_lr_schedule_env() -> tuple[str, int, float]:
    """Parse LR schedule env vars, returning (schedule, t_max, eta_min).

    Env vars:
        HYZERO_LR_SCHEDULE:      "none" (default) or "cosine". Unknown value warns and uses "none".
        HYZERO_LR_COSINE_T_MAX:  int, total annealing steps. Default 5000. Clamped to [100, 1_000_000].
        HYZERO_LR_COSINE_ETA_MIN: float, minimum LR. Default 1e-5. Clamped to [0.0, 1e-2].

    Returns:
        Tuple of (schedule_name, t_max, eta_min).
    """
    # --- schedule name ---
    raw_sched = os.environ.get("HYZERO_LR_SCHEDULE", "none").strip().lower()
    if raw_sched not in ("none", "cosine"):
        print(
            f"[trainer] WARNING: HYZERO_LR_SCHEDULE={raw_sched!r} is not valid"
            " (expected 'none' or 'cosine'); using 'none'",
            file=sys.stderr,
        )
        raw_sched = "none"

    # --- T_max ---
    raw_tmax = os.environ.get("HYZERO_LR_COSINE_T_MAX")
    t_max = 5000
    if raw_tmax is not None:
        try:
            t_max = int(raw_tmax)
        except (ValueError, TypeError):
            print(
                f"[trainer] WARNING: HYZERO_LR_COSINE_T_MAX={raw_tmax!r} is not a valid int;"
                " using default 5000",
                file=sys.stderr,
            )
            t_max = 5000
        else:
            clamped = max(100, min(1_000_000, t_max))
            if clamped != t_max:
                print(
                    f"[trainer] WARNING: HYZERO_LR_COSINE_T_MAX={t_max} clamped to {clamped}",
                    file=sys.stderr,
                )
            t_max = clamped

    # --- eta_min ---
    raw_eta = os.environ.get("HYZERO_LR_COSINE_ETA_MIN")
    eta_min = 1e-5
    if raw_eta is not None:
        try:
            eta_min = float(raw_eta)
        except (ValueError, TypeError):
            print(
                f"[trainer] WARNING: HYZERO_LR_COSINE_ETA_MIN={raw_eta!r} is not a valid float;"
                " using default 1e-5",
                file=sys.stderr,
            )
            eta_min = 1e-5
        else:
            clamped = max(0.0, min(1e-2, eta_min))
            if clamped != eta_min:
                print(
                    f"[trainer] WARNING: HYZERO_LR_COSINE_ETA_MIN={eta_min} clamped to {clamped}",
                    file=sys.stderr,
                )
            eta_min = clamped

    return raw_sched, t_max, eta_min


def _parse_loss_weight_env(name: str, default: float = 1.0) -> float:
    """Parse a loss weight env var, clamping to [0.0, 100.0].

    Args:
        name:    Environment variable name.
        default: Value to return on missing or unparseable input.

    Returns:
        Parsed and clamped float weight.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (ValueError, TypeError):
        print(
            f"[trainer] WARNING: {name}={raw!r} is not a valid float; using default {default}",
            file=sys.stderr,
        )
        return default
    clamped = max(0.0, min(100.0, value))
    if clamped != value:
        print(
            f"[trainer] WARNING: {name}={value} clamped to {clamped}",
            file=sys.stderr,
        )
    return clamped


def _reinit_value_head(prediction_network: torch.nn.Module) -> None:
    """Re-randomize the value-output layers of the prediction network.

    Preserves the trunk and policy head. Used to climb out of a collapsed
    value-head attractor (head has learned to output ~0 regardless of input)
    after changing the target scale (e.g. switching β=0.3→1.0).

    When ``HYZERO_REINIT_VALUE_BIAS`` is set to a non-zero float (e.g. 0.3),
    the *final* Linear layer's bias is initialised to that constant instead of
    zero.  This puts the initial output at ``tanh(bias_offset) ≈ +0.29`` for
    any input, breaking the tie toward the positive half-plane so that TB +1
    supervision and self-play can't pull it into a negative attractor.

    Only the *final* linear layer receives the offset — earlier biases would
    create cascaded ReLU effects that dilute the shift.  A value of ±1.0 would
    saturate tanh and kill the gradient, so keep it in the ±0.1–0.5 range.
    """
    bias_offset = float(os.environ.get("HYZERO_REINIT_VALUE_BIAS", "0.0"))

    # Look at the PredictionNetwork in hyzero/models/prediction.py to identify
    # the value-path module(s). Typical structure: a shared MLP + two heads
    # (policy_head, value_head). Reinit just value_head.
    if hasattr(prediction_network, 'value_head'):
        linear_layers = [m for m in prediction_network.value_head.modules() if isinstance(m, torch.nn.Linear)]
        for i, m in enumerate(linear_layers):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            if m.bias is not None:
                # Only the FINAL linear (before tanh) gets the bias offset.
                # Applying earlier biases creates cascaded ReLU effects that dilute the offset.
                is_final = (i == len(linear_layers) - 1)
                if is_final and bias_offset != 0.0:
                    torch.nn.init.constant_(m.bias, bias_offset)
                else:
                    torch.nn.init.zeros_(m.bias)

        if bias_offset != 0.0:
            print(f"[trainer] value head reinitialized with final-bias offset {bias_offset:+.3f}")
        else:
            print("[trainer] value head reinitialized (HYZERO_REINIT_VALUE_HEAD=1)")
    else:
        print("[trainer] WARN: prediction network has no .value_head attribute; cannot reinit")


class Trainer:
    """Manages MuZero training: K-step unroll, loss computation, and weight checkpointing.

    Attributes:
        config: Hyperparameter dictionary.
        device: Torch device string ("cpu" or "cuda").
        model_version: Number of train_batch calls completed.
        h: RepresentationNetwork.
        g: DynamicsNetwork.
        f: PredictionNetwork.
        optimizer: Shared AdamW optimizer over all three networks.
    """

    def __init__(self, config: dict = None, device: str = "cpu") -> None:
        self.config = config if config is not None else dict(DEFAULT_CONFIG)
        self.device = device

        cfg = self.config
        self.h = RepresentationNetwork(
            input_planes=cfg["input_planes"],
            hidden_channels=cfg["hidden_channels"],
            num_res_blocks=cfg["num_res_blocks"],
        ).to(device).train()

        self.g = DynamicsNetwork(
            hidden_channels=cfg["hidden_channels"],
            action_planes=cfg["action_planes"],
            num_res_blocks=cfg["num_res_blocks"],
        ).to(device).train()

        self.f = PredictionNetwork(
            hidden_channels=cfg["hidden_channels"],
            num_actions=cfg["num_actions"],
        ).to(device).train()

        all_params = (
            list(self.h.parameters())
            + list(self.g.parameters())
            + list(self.f.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        lr_schedule, lr_t_max, lr_eta_min = _parse_lr_schedule_env()
        if lr_schedule == "cosine":
            self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=lr_t_max,
                    eta_min=lr_eta_min,
                )
            )
            print(f"[trainer] LR schedule: cosine T_max={lr_t_max} eta_min={lr_eta_min}")
        else:
            self.lr_scheduler = None
            print(f"[trainer] LR schedule: none (fixed lr={cfg['lr']})")

        self.model_version: int = 0

        # Pre-built canonical-position observations for periodic value probes.
        # All shapes [1, 102, 8, 8]. Built once at init; device is fixed at construction.
        self._start_obs: torch.Tensor = _build_start_obs(device)
        self._kqk_obs: torch.Tensor = _build_kqk_white_winning_obs(device)
        self._kvk_queenless_obs: torch.Tensor = _build_k_vs_kq_white_losing_obs(device)

        # Checkmate counter state.
        # Incremented by notify_trajectory() each time a decisive (non-draw) game arrives.
        # Logged periodically in the train_batch diagnostic block.
        self._total_cm: int = 0
        self._cm_since_last_log: int = 0

        self.policy_loss_weight = _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT")
        self.value_loss_weight = _parse_loss_weight_env("HYZERO_VALUE_LOSS_WEIGHT")
        self.reward_loss_weight = _parse_loss_weight_env("HYZERO_REWARD_LOSS_WEIGHT")
        self.policy_entropy_weight = _parse_loss_weight_env(
            "HYZERO_POLICY_ENTROPY_WEIGHT", default=0.0
        )
        print(
            f"[trainer] loss weights: policy={self.policy_loss_weight:.2f}"
            f" value={self.value_loss_weight:.2f}"
            f" reward={self.reward_loss_weight:.2f}"
            f" entropy={self.policy_entropy_weight:.4f}"
        )

        # Syzygy tablebase supervision — optional, enabled by HYZERO_TABLEBASE_PATH.
        tb_path = os.environ.get("HYZERO_TABLEBASE_PATH")
        tb_cache_path = os.environ.get("HYZERO_TABLEBASE_CACHE_PATH", "data/syzygy/cache.pkl")
        self._tb_cache: object | None = None
        if tb_path is not None:
            from hyzero.data.tablebase import TablebaseCache
            if os.path.exists(tb_cache_path):
                self._tb_cache = TablebaseCache(tb_cache_path)
                print(f"[trainer] tablebase cache loaded: {len(self._tb_cache)} positions")
            else:
                print(
                    f"[trainer] WARN: HYZERO_TABLEBASE_PATH set but cache not found at"
                    f" {tb_cache_path!r}; TB supervision disabled"
                )
        self._tb_frac = float(os.environ.get("HYZERO_TABLEBASE_FRAC", "0.0"))

    def notify_trajectory(self, game_outcome: float, is_draw: bool) -> None:
        """Record a completed game trajectory for checkmate counting.

        Called by the Rust training loop (via PyO3) each time a trajectory arrives,
        BEFORE it is added to the replay buffer. Decisive games (is_draw=False and
        |game_outcome|=1.0) increment the checkmate counter.

        Args:
            game_outcome: White-absolute game result: +1.0=white wins, -1.0=black wins,
                          0.0=draw.
            is_draw:      True if the game ended non-decisively (stalemate, repetition,
                          50-move, insufficient material).
        """
        if not is_draw and abs(game_outcome) >= 0.999:
            self._total_cm += 1
            self._cm_since_last_log += 1

    def train_batch(self, batch: dict) -> dict:
        """Run one K-step unroll training step.

        Args:
            batch: Dictionary of numpy arrays:
                "observations":    [B, K+1, 102, 8, 8]  — all K+1 steps for consistency loss
                "actions":         [B, K, 3, 8, 8]
                "target_policies": [B, K+1, 4672]
                "target_values":   [B, K+1]
                "target_rewards":  [B, K+1]
                "legal_masks":     [B, 4672] bool (optional) — if present, illegal
                                   actions are masked to -inf in the policy loss so
                                   gradients do not push logits at illegal positions.

        Returns:
            dict with keys: total_loss, policy_loss, value_loss, reward_loss,
            consistency_loss, model_version, lr (all loss values are Python floats).
        """
        # Mix in tablebase supervision rows before tensor conversion.
        batch, tb_indices = self._maybe_mix_tb_samples(batch)

        # Pop is_tablebase before tensor conversion (Python-only field).
        is_tb_mask = batch.pop("is_tablebase", None)

        # Convert numpy arrays to tensors on the target device.
        # observations: [B, K+1, 102, 8, 8] — step 0 is root, steps 1..K for consistency
        obs_all = torch.from_numpy(batch["observations"]).to(self.device)      # [B, K+1, 102, 8, 8]
        obs = obs_all[:, 0]                                                     # [B, 102, 8, 8]
        actions = torch.from_numpy(batch["actions"]).to(self.device)           # [B, K, 3, 8, 8]
        tgt_policies = torch.from_numpy(batch["target_policies"]).to(self.device)  # [B, K+1, 4672]
        tgt_values = torch.from_numpy(batch["target_values"]).to(self.device)  # [B, K+1]
        tgt_rewards = torch.from_numpy(batch["target_rewards"]).to(self.device)  # [B, K+1]
        # Legal mask (root step only): [B, 4672] bool, or None if not provided.
        legal_mask_np = batch.get("legal_masks")
        legal_mask = (
            torch.from_numpy(legal_mask_np).to(self.device)
            if legal_mask_np is not None
            else None
        )

        k_steps = actions.shape[1]  # K

        self.optimizer.zero_grad()

        # Build a boolean mask for TB rows: True where the sample is a tablebase row.
        # Lifted here (before the unroll loop) so both the loss computation and the
        # consistency loss can share it without re-building.
        is_tb_tensor: torch.Tensor | None = (
            torch.from_numpy(is_tb_mask).to(self.device)
            if is_tb_mask is not None
            else None
        )

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)

        # Collect hidden states at each step for consistency loss computation.
        # hidden_states[k] = latent output of g after k dynamics steps
        # (hidden_states[0] = h(obs_root), hidden_states[k] = g(hidden_{k-1}, a_{k-1}))
        hidden_states: list[torch.Tensor] = []

        # Accumulate per-k predictions for diagnostic stats (measurement only, no grad impact).
        predicted_values_per_k: list[torch.Tensor] = []
        predicted_rewards_per_k: list[torch.Tensor] = []
        predicted_policy_logits_per_k: list[torch.Tensor] = []

        # Step 0: encode observation, predict (policy, value).
        hidden = self.h(obs)  # [B, hidden_channels, 8, 8]
        hidden_states.append(hidden)
        policy_logits, value = self.f(hidden)  # [B, 4672], [B, 1]

        predicted_values_per_k.append(value)
        # No reward at step 0; use a placeholder zeros tensor so indices stay aligned.
        predicted_rewards_per_k.append(torch.zeros_like(value))
        predicted_policy_logits_per_k.append(policy_logits)

        # Policy loss at step 0 — apply legal mask if provided.
        # ALL samples contribute at step 0 (no TB masking at root).
        total_policy_loss = total_policy_loss + self._policy_loss(
            policy_logits, tgt_policies[:, 0], legal_mask
        )
        # Value loss at step 0 — all samples contribute.
        total_value_loss = total_value_loss + F.mse_loss(value.squeeze(-1), tgt_values[:, 0])

        # Steps 1..K: unroll dynamics.
        for k in range(1, k_steps + 1):
            action_plane = actions[:, k - 1]  # [B, 3, 8, 8]
            hidden, reward = self.g(hidden, action_plane)  # [B, hidden_channels, 8, 8], [B, 1]
            hidden_states.append(hidden)

            # MuZero: scale gradient at dynamics boundary (Appendix G) to stabilize K-step unroll
            hidden.register_hook(lambda grad: grad * 0.5)

            policy_logits, value = self.f(hidden)  # [B, 4672], [B, 1]

            predicted_values_per_k.append(value)
            predicted_rewards_per_k.append(reward)
            predicted_policy_logits_per_k.append(policy_logits)

            # For TB rows at k >= 1, targets are zero-padded (not real signals).
            # Mask them out so padded zeros don't dilute the step-0 TB supervision.
            # Non-TB rows always contribute at all steps.
            if is_tb_tensor is not None:
                non_tb = (~is_tb_tensor).float()  # [B], 1.0 for replay, 0.0 for TB
                non_tb_count = non_tb.sum().clamp(min=1.0)

                # Policy loss at k >= 1: mask TB rows.
                per_sample_pol = self._policy_loss_per_sample(policy_logits, tgt_policies[:, k])
                total_policy_loss = total_policy_loss + (per_sample_pol * non_tb).sum() / non_tb_count

                # Value loss at k >= 1: mask TB rows.
                per_sample_val = (value.squeeze(-1) - tgt_values[:, k]) ** 2  # [B]
                total_value_loss = total_value_loss + (per_sample_val * non_tb).sum() / non_tb_count

                # Reward loss: step 1 has a real mating-action target for TB rows → all samples.
                # Steps k >= 2 are zero-padded for TB → mask them.
                per_sample_rwd = (reward.squeeze(-1) - tgt_rewards[:, k]) ** 2  # [B]
                if k >= 2:
                    total_reward_loss = total_reward_loss + (per_sample_rwd * non_tb).sum() / non_tb_count
                else:
                    # k == 1: TB step-1 reward carries the real mating-action signal.
                    total_reward_loss = total_reward_loss + per_sample_rwd.mean()
            else:
                # No TB rows in this batch — standard unmasked losses.
                # No mask for latent steps (operating in learned latent space, not real board)
                total_policy_loss = total_policy_loss + self._policy_loss(policy_logits, tgt_policies[:, k])
                total_value_loss = total_value_loss + F.mse_loss(value.squeeze(-1), tgt_values[:, k])
                total_reward_loss = total_reward_loss + F.mse_loss(reward.squeeze(-1), tgt_rewards[:, k])

        # -----------------------------------------------------------------------
        # DIAGNOSTIC INSTRUMENTATION — measurements only, no training impact.
        # Wrapped in try/except so any instrumentation error is logged but does
        # not abort the training step. Uses _diag_print (os.write to fd 1) to
        # bypass any PyO3 sys.stdout redirection.
        # -----------------------------------------------------------------------
        # Probe: write to a temp file to bypass all stdout redirection.
        try:
            with open("/tmp/hyzero_diag_probe.txt", "a") as _pf:
                _pf.write(f"[diag_reached] step={self.model_version}\n")
                _pf.flush()
        except Exception:
            pass
        _diag_print(f"[diag_reached] step={self.model_version}")
        try:
            with torch.no_grad():
                # 1. Per-k target and prediction stats (every call, cheap).
                n_steps_diag = k_steps + 1  # 0..K inclusive
                val_parts: list[str] = []
                rwd_parts: list[str] = []
                pol_parts: list[str] = []
                for k in range(n_steps_diag):
                    tgt_v = tgt_values[:, k]
                    pred_v = predicted_values_per_k[k].squeeze(-1)
                    mse_v = F.mse_loss(pred_v, tgt_v).item()
                    val_parts.append(
                        f"k{k} tgt={tgt_v.mean().item():.4f}±{tgt_v.std().item():.4f}"
                        f" pred={pred_v.mean().item():.4f}±{pred_v.std().item():.4f}"
                        f" mse={mse_v:.5f}"
                    )

                    tgt_r = tgt_rewards[:, k]
                    pred_r = predicted_rewards_per_k[k].squeeze(-1)
                    mse_r = F.mse_loss(pred_r, tgt_r).item()
                    rwd_parts.append(
                        f"k{k} tgt={tgt_r.mean().item():.4f}±{tgt_r.std().item():.4f}"
                        f" pred={pred_r.mean().item():.4f}±{pred_r.std().item():.4f}"
                        f" mse={mse_r:.5f}"
                    )

                    logits_k = predicted_policy_logits_per_k[k]
                    probs_k = F.softmax(logits_k, dim=-1)
                    log_probs_k = F.log_softmax(logits_k, dim=-1)
                    entropy_k = (-probs_k * log_probs_k).sum(dim=-1).mean().item()
                    top1_k = probs_k.max(dim=-1).values.mean().item()
                    tgt_p = tgt_policies[:, k]
                    tgt_log_p = tgt_p.clamp(min=1e-9).log()
                    tgt_entropy_k = (-tgt_p * tgt_log_p).sum(dim=-1).mean().item()
                    pol_parts.append(
                        f"k{k} tgt_entropy={tgt_entropy_k:.4f}"
                        f" pred_entropy={entropy_k:.4f}"
                        f" pred_top1={top1_k:.4f}"
                    )

                _diag_print(f"[val_stats] step={self.model_version} " + " | ".join(val_parts))
                _diag_print(f"[reward_stats] step={self.model_version} " + " | ".join(rwd_parts))
                _diag_print(f"[policy_stats] step={self.model_version} " + " | ".join(pol_parts))

                # Periodic probes every 50 calls.
                if self.model_version % 50 == 0:
                    # 2. Color-symmetry probe on real obs from the current batch.
                    obs_one = obs[0:1]                             # [1, 102, 8, 8]
                    flipped_one = _flip_obs_planes(obs_one)        # [1, 102, 8, 8]

                    # f(h(obs)) and f(h(flipped_obs)) — only extract value (index 1).
                    v1 = self.f(self.h(obs_one))[1].item()        # scalar
                    v2 = self.f(self.h(flipped_one))[1].item()    # scalar
                    v_sum = v1 + v2
                    ratio = v_sum / (abs(v1) + 1e-9)
                    _diag_print(
                        f"[sym_probe] step={self.model_version}"
                        f" v(obs)={v1:.4f} v(flip)={v2:.4f}"
                        f" sum={v_sum:.4f} v_plus_vflip_over_abs_v={ratio:.4f}"
                    )

                    # Batch symmetry probe (up to 10 samples).
                    n_batch = min(10, obs.shape[0])
                    obs_batch = obs[0:n_batch]                         # [N, 102, 8, 8]
                    flipped_batch = _flip_obs_planes(obs_batch)        # [N, 102, 8, 8]

                    vals_batch = self.f(self.h(obs_batch))[1].squeeze(-1)         # [N]
                    vals_flipped = self.f(self.h(flipped_batch))[1].squeeze(-1)   # [N]
                    neg_vals_flipped = -vals_flipped                               # [N]

                    # Pearson correlation between v_batch and -v_flipped_batch.
                    if n_batch > 1:
                        vb_mean = vals_batch.mean()
                        nvf_mean = neg_vals_flipped.mean()
                        cov = ((vals_batch - vb_mean) * (neg_vals_flipped - nvf_mean)).mean()
                        std_vb = vals_batch.std(unbiased=False).clamp(min=1e-9)
                        std_nvf = neg_vals_flipped.std(unbiased=False).clamp(min=1e-9)
                        corr = (cov / (std_vb * std_nvf)).item()
                    else:
                        corr = float("nan")
                    mean_sum_batch = (vals_batch + vals_flipped).mean().item()
                    _diag_print(
                        f"[sym_probe_batch] step={self.model_version}"
                        f" corr_v_neg_vflip={corr:.4f}"
                        f" mean_sum={mean_sum_batch:.4f} (N={n_batch})"
                    )

                    # 3. Target distribution histogram over 10 bins in [-1.0, +1.0].
                    flat_tgt = tgt_values.flatten()
                    bin_edges = torch.linspace(-1.0, 1.0, steps=11, device=self.device)
                    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    counts = torch.zeros(10, dtype=torch.long, device=self.device)
                    for b_idx in range(10):
                        lo = bin_edges[b_idx]
                        hi = bin_edges[b_idx + 1]
                        if b_idx < 9:
                            counts[b_idx] = ((flat_tgt >= lo) & (flat_tgt < hi)).sum()
                        else:
                            # Last bin is inclusive on right edge
                            counts[b_idx] = ((flat_tgt >= lo) & (flat_tgt <= hi)).sum()
                    hist_parts = ", ".join(
                        f"{bin_mids[b_idx].item():.1f}:{counts[b_idx].item()}"
                        for b_idx in range(10)
                    )
                    _diag_print(f"[tgt_hist] step={self.model_version} {hist_parts}")

                    # 4. Canonical-position value probes.
                    # start_value: starting position (material-balanced).
                    start_obs = self._start_obs  # [1, 102, 8, 8]
                    start_v = self.f(self.h(start_obs))[1].item()
                    _diag_print(f"[start_value] step={self.model_version} v={start_v:.4f}")

                    # kqk_value: KQ-vs-K (white to move, white winning).
                    kqk_v = self.f(self.h(self._kqk_obs))[1].item()
                    _diag_print(f"[kqk_value] step={self.model_version} v={kqk_v:.4f}")

                    # kvk_queenless_value: K-vs-KQ (white to move, white down a queen).
                    kvk_v = self.f(self.h(self._kvk_queenless_obs))[1].item()
                    _diag_print(
                        f"[kvk_queenless_value] step={self.model_version} v={kvk_v:.4f}"
                    )

                    # value_spread: discrimination signal — how far each position deviates
                    # from the starting position. Should grow as the value head learns
                    # material from the checkmate stream.
                    kqk_minus_start = kqk_v - start_v
                    kvk_minus_start = kvk_v - start_v
                    _diag_print(
                        f"[value_spread] step={self.model_version}"
                        f" kqk_minus_start={kqk_minus_start:.4f}"
                        f" kvk_minus_start={kvk_minus_start:.4f}"
                    )

                    # 5. Checkmate counter.
                    _diag_print(
                        f"[cm_count] step={self.model_version}"
                        f" total_cm={self._total_cm}"
                        f" cm_since_last_log={self._cm_since_last_log}"
                    )
                    self._cm_since_last_log = 0
        except Exception as _diag_exc:
            _diag_print(f"[diag_error] step={self.model_version} exception={_diag_exc!r}")
        # -----------------------------------------------------------------------

        # Average losses over K+1 steps for policy/value; rewards only over K steps (none at step 0).
        n_steps = k_steps + 1
        avg_policy_loss = total_policy_loss / n_steps
        avg_value_loss = total_value_loss / n_steps
        avg_reward_loss = total_reward_loss / k_steps

        # EfficientZero self-supervised consistency loss (Ye et al., NeurIPS 2021).
        # For each dynamics step k in 1..K, force:
        #   g(h(obs_{k-1}), a_{k-1}) ≈ h(obs_k)
        # via cosine similarity, with stop-gradient on the target (h(obs_k)) side.
        # This gives the dynamics network `g` a DIRECT training signal independent of `f`.
        consistency_weight = _parse_loss_weight_env("HYZERO_CONSISTENCY_LOSS_WEIGHT", default=0.5)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if consistency_weight > 0 and k_steps > 0:
            # is_tb_tensor was already built at the top of the training step.
            # TB rows have zero obs at steps 1..K, which would force the consistency
            # target toward a zero-latent and poison the dynamics network. Exclude them.
            for k_idx in range(1, k_steps + 1):
                # Dynamics branch: project -> predict (online branch, receives gradients)
                dyn_latent_k = hidden_states[k_idx]  # [B, C, 8, 8]
                p1 = self.h.predict(self.h.project(dyn_latent_k))  # [B, proj_dim]
                # Target branch: h(obs_k) projected with stop-gradient (no gradient flows here)
                obs_k = obs_all[:, k_idx]  # [B, 102, 8, 8]
                target_latent = self.h(obs_k)  # [B, C, 8, 8]
                p2 = self.h.project(target_latent).detach()  # [B, proj_dim], stop-grad
                # Cosine similarity loss: 1 - cos_sim (maximize similarity → minimize loss)
                cos_sim = F.cosine_similarity(p1, p2, dim=-1)  # [B]
                if is_tb_tensor is not None:
                    # Exclude TB rows: their obs at steps 1..K are zeros, not real positions.
                    cos_sim = cos_sim[~is_tb_tensor]
                if cos_sim.numel() > 0:
                    consistency_loss = consistency_loss + (1 - cos_sim.mean())
            if k_steps > 0:
                consistency_loss = consistency_loss / k_steps

        total_loss = (
            self.policy_loss_weight * avg_policy_loss
            + self.value_loss_weight * avg_value_loss
            + self.reward_loss_weight * avg_reward_loss
            + consistency_weight * consistency_loss
        )

        total_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.model_version += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "reward_loss": avg_reward_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "model_version": self.model_version,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy between logits and a target probability distribution.

        Optionally adds an entropy bonus −β·H(π) to penalize over-sharpening of the
        output distribution, controlled by HYZERO_POLICY_ENTROPY_WEIGHT (default 0.0,
        off). Equivalent to β·KL(π || Uniform) regularization up to an additive
        constant. Applies at every unroll step (root + K dynamics steps).

        Args:
            logits:      [B, num_actions]
            targets:     [B, num_actions]  (soft probability distribution summing to 1)
            legal_mask:  [B, num_actions] bool or None.
                         If provided, illegal positions are masked to -inf before
                         log_softmax so gradients do not push logits at illegal positions.

        Returns:
            Scalar tensor (mean over batch).
        """
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))
        log_probs = F.log_softmax(logits, dim=-1)
        # Replace -inf (from masked illegal actions) with 0.0 before multiplying by targets.
        # Since targets are always 0.0 at illegal positions, 0.0 * 0.0 = 0.0 is correct.
        # This avoids 0.0 * (-inf) = NaN in IEEE 754 arithmetic.
        log_probs = log_probs.nan_to_num(nan=0.0, neginf=0.0)
        ce_loss = -torch.sum(targets * log_probs, dim=-1).mean()

        if self.policy_entropy_weight > 0.0:
            # Entropy bonus: penalize over-sharp output distributions.
            # −β·H(π) = β·Σ π·log π  (equivalent to β·KL(π || Uniform) + const).
            # Illegal moves contribute 0: probs=0 from -inf logits, log_probs=0 from
            # nan_to_num → 0·0=0 (no NaN, no illegal-move pressure).
            probs = F.softmax(logits, dim=-1)
            neg_entropy = torch.sum(probs * log_probs, dim=-1).mean()
            return ce_loss + self.policy_entropy_weight * neg_entropy

        return ce_loss

    def _policy_loss_per_sample(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample cross-entropy between logits and a target probability distribution.

        Like _policy_loss but returns one scalar per batch element (shape [B]) rather
        than the batch mean. Used when a caller needs to apply a per-row loss mask
        before reducing (e.g. TB-row masking at k >= 1 padded steps).

        No legal_mask support (latent steps do not have real legal-move lists).
        No entropy bonus (applied at the scalar level by the caller).

        Args:
            logits:  [B, num_actions]
            targets: [B, num_actions]  (soft probability distribution summing to 1)

        Returns:
            [B] float32 tensor — per-sample cross-entropy.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.nan_to_num(nan=0.0, neginf=0.0)
        return -torch.sum(targets * log_probs, dim=-1)  # [B]

    def _maybe_mix_tb_samples(
        self, batch: dict
    ) -> tuple[dict, set[int]]:
        """Replace tb_frac fraction of batch with Syzygy tablebase samples.

        Inserts TB rows at the end of the batch (indices [b-n_tb, b)).
        TB rows have exact ±1 value targets and 1-step reward signal from
        mate-in-1 moves, providing hard gradient for the value and reward heads.

        Args:
            batch: Replay batch dict (numpy arrays, not yet converted to tensors).

        Returns:
            Tuple of (updated batch dict, set of TB row indices).
            The batch dict gains an 'is_tablebase' bool array [B].
        """
        if self._tb_cache is None or self._tb_frac <= 0.0:
            return batch, set()

        b = batch["observations"].shape[0]
        k_steps = batch["actions"].shape[1]
        n_tb = max(1, int(b * self._tb_frac))
        n_tb = min(n_tb, b)

        from hyzero.data.tablebase import build_tb_batch, build_tb_batch_trajectories
        tb_items = self._tb_cache.sample(n_tb)
        # Route to trajectory builder for trajectory-format caches; fall back
        # to the legacy snapshot builder otherwise. Both return the same dict
        # shape, differing only in the ``is_tablebase`` flag (True for
        # snapshots → trainer masks step-1..K losses; False for trajectories
        # → trainer treats rows identically to replay samples).
        if getattr(self._tb_cache, "is_trajectory_format", False):
            tb_dict = build_tb_batch_trajectories(tb_items, k_steps=k_steps)
        else:
            tb_dict = build_tb_batch(tb_items, k_steps=k_steps)

        # Replace last n_tb rows of the replay batch with TB rows.
        tb_indices = set(range(b - n_tb, b))
        merged: dict = {}
        for key in ("observations", "actions", "target_policies",
                    "target_values", "target_rewards"):
            merged[key] = np.concatenate(
                [batch[key][:b - n_tb], tb_dict[key]], axis=0
            )
        # legal_masks: optional in replay batch, always present in TB.
        replay_masks = batch.get("legal_masks")
        if replay_masks is not None:
            merged["legal_masks"] = np.concatenate(
                [replay_masks[:b - n_tb], tb_dict["legal_masks"]], axis=0
            )
        else:
            # Only TB rows have masks; replay rows get all-True masks as placeholder.
            # This avoids a None legal_mask downstream when TB is active.
            placeholder = np.ones((b - n_tb, tb_dict["legal_masks"].shape[1]), dtype=bool)
            merged["legal_masks"] = np.concatenate(
                [placeholder, tb_dict["legal_masks"]], axis=0
            )
        # ``is_tablebase`` controls whether the trainer applies the
        # step-1..K masking. Snapshot batches set it True (legacy behavior);
        # trajectory batches set it False so the K-step loss + consistency
        # loss apply in full. Trust what the TB builder emitted rather than
        # mechanically marking the suffix rows.
        merged["is_tablebase"] = np.zeros(b, dtype=bool)
        tb_flag = tb_dict.get("is_tablebase")
        if tb_flag is not None:
            merged["is_tablebase"][b - n_tb:] = tb_flag
        else:
            merged["is_tablebase"][b - n_tb:] = True
        return merged, tb_indices

    def get_weights(self) -> bytes:
        """Serialize network weights to bytes for inference server transfer.

        Does not include optimizer state or model_version — use save_checkpoint
        for full training state persistence.

        Returns:
            bytes object containing the serialized weights.
        """
        buf = io.BytesIO()
        torch.save(
            {
                "h": self.h.state_dict(),
                "g": self.g.state_dict(),
                "f": self.f.state_dict(),
            },
            buf,
        )
        return buf.getvalue()

    def save_checkpoint(self, path: str, eval_metrics: dict = None) -> None:
        """Save network weights, optimizer state, and metadata to disk.

        Args:
            path:         File path to write the checkpoint to.
            eval_metrics: Optional dict of evaluation metrics to persist.
        """
        checkpoint = {
            "h": self.h.state_dict(),
            "g": self.g.state_dict(),
            "f": self.f.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_version": self.model_version,
            "eval_metrics": eval_metrics,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict:
        """Restore network weights, optimizer state, and model_version from disk.

        Args:
            path: File path to load the checkpoint from.

        Returns:
            eval_metrics dict that was stored in the checkpoint (may be None).
        """
        # weights_only=False: checkpoint contains eval_metrics dict alongside tensors
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.h.load_state_dict(checkpoint["h"])
        self.g.load_state_dict(checkpoint["g"])
        self.f.load_state_dict(checkpoint["f"])
        # Optimizer state is optional and may not match — e.g. pretrain_dynamics.pt
        # only optimized h+g (f was frozen), so its param-group count differs from the
        # main trainer's h+g+f+projector optimizer. Fall back to the freshly-constructed
        # optimizer in that case; the RL loop will rebuild running statistics quickly.
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as e:
                print(
                    f"[trainer] optimizer state_dict mismatch ({e}); "
                    f"keeping fresh optimizer state (checkpoint={path})",
                    flush=True,
                )
        self.model_version = checkpoint.get("model_version", 0)
        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if os.environ.get('HYZERO_REINIT_VALUE_HEAD', '').strip() not in ('', '0', 'false', 'no'):
            _reinit_value_head(self.f)

        return checkpoint.get("eval_metrics")
