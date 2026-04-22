"""Tablebase position loader and batch builder for training supervision.

Two formats supported:

1. **Snapshot format (legacy)**: ``TBSample`` — single-position Syzygy probe with
   step-0 targets only; steps 1..K are zero-padded. Used by the original
   ``HYZERO_TABLEBASE_FRAC`` / ``is_tablebase`` masking code path in the trainer.

2. **Trajectory format (canonical MuZero-shaped)**: ``TBTrajectory`` — K-step
   rollout of optimal-DTZ play with Syzygy WDL re-probed at every step, terminal
   transition reward, and absorbing-state padding past checkmate. This matches
   what canonical MuZero gets from real self-play trajectories and supplies the
   reward head with many terminal-transition examples instead of the 0.29%
   mate-in-1 positions of the snapshot cache.

Shape contract for both builders (K = k_steps):
    observations:    [N, K+1, 102, 8, 8]  float32
    actions:         [N, K,   3,  8, 8]   float32
    target_policies: [N, K+1, 4672]        float32
    target_values:   [N, K+1]              float32
    target_rewards:  [N, K+1]              float32
    legal_masks:     [N, 4672]             bool
    is_tablebase:    [N]                   bool

Trajectory batches set ``is_tablebase = False`` so the trainer treats rows
identically to replay samples (full K-step loss + consistency). Snapshot
batches set ``is_tablebase = True`` to keep the legacy step-0-only masking.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass

import chess
import numpy as np

from hyzero.data.board_encoder import (
    encode_board_python,
    encode_action_spatial,
    action_from_move,
    NUM_ACTIONS,
)


@dataclass
class TBSample:
    """A single tablebase-probed position with supervision targets (legacy snapshot).

    Attributes:
        fen:              FEN string of the position.
        target_value:     ±1.0 or 0.0 from side-to-move POV (Syzygy WDL mapped).
        mating_actions:   Action indices of mate-in-1 moves (may be empty).
        optimal_actions:  Action indices achieving minimum |DTZ| (optimal play).
        all_legal_actions: All legal action indices at this position.
    """

    fen: str
    target_value: float
    mating_actions: list[int]
    optimal_actions: list[int]
    all_legal_actions: list[int]


# Sentinel for an absorbing-state step (post-terminal). The FEN is None; the
# encoder produces a zero observation, the action is a null (zeros), and all
# targets are zero. This matches the canonical MuZero absorbing-state padding
# used when a trajectory window extends past the true terminal.
ABSORBING_FEN: str | None = None


@dataclass
class TBTrajectory:
    """A K-step optimal-play rollout from a tablebase position.

    Each trajectory has K+1 "steps"; step 0 is the root TB position, and
    steps 1..K are produced by chaining Syzygy-optimal (minimum-DTZ) moves
    via python-chess. When the rollout lands on checkmate at step k*, the
    reward fires at that step and the remaining steps (k*+1..K) are filled
    with absorbing-state placeholders (zero observation, null action, zero
    targets).

    Attributes:
        fens:           Length-(K+1) list of FEN strings. ``None`` entries mark
                        absorbing-state steps past terminal.
        actions:        Length-K list of action indices taken at each real step.
                        Null actions (past terminal) use -1 as a sentinel.
        target_values:  Length-(K+1) list of Syzygy WDL values from each step's
                        STM POV (+1 winning, -1 losing, 0 drawn or absorbing).
        target_rewards: Length-(K+1) list. Entry k is the reward received
                        transitioning INTO step k (0 everywhere except the
                        mating-transition step, where it is +1 from the mover's
                        POV). Entry 0 is always 0 (no transition into the root).
        legal_actions:  Length-(K+1) list of per-step legal action index lists.
                        Empty for absorbing steps.
        optimal_actions: Length-(K+1) list of per-step Syzygy-optimal action
                        index lists (used to form policy targets). Empty for
                        absorbing steps.
        mate_step:      The step index k* at which the mating reward fires, or
                        ``None`` if no checkmate occurred within K plies.
    """

    fens: list[str | None]
    actions: list[int]
    target_values: list[float]
    target_rewards: list[float]
    legal_actions: list[list[int]]
    optimal_actions: list[list[int]]
    mate_step: int | None


class TablebaseCache:
    """Loaded tablebase position cache supporting random sampling.

    Auto-detects whether the pickle contains a list[TBSample] (snapshot cache)
    or list[TBTrajectory] (trajectory cache) by inspecting the first element.
    Use ``is_trajectory_format`` to branch on builder selection.

    Args:
        path: Path to a pickled list[TBSample] OR list[TBTrajectory]
              (built by build_tablebase_cache.py).
    """

    def __init__(self, path: str) -> None:
        # The cache was built by build_tablebase_cache.py which defines its
        # dataclasses in __main__. Unpickling would fail here because the module
        # path differs. Shim __main__ with our dataclasses before loading.
        import types

        _shim = types.ModuleType("__main__")
        _shim.TBSample = TBSample           # type: ignore[attr-defined]
        _shim.TBTrajectory = TBTrajectory   # type: ignore[attr-defined]
        import sys as _sys
        _prev = _sys.modules.get("__main__")
        _sys.modules["__main__"] = _shim  # type: ignore[assignment]
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        finally:
            if _prev is not None:
                _sys.modules["__main__"] = _prev
            else:
                del _sys.modules["__main__"]

        if not data:
            raise ValueError(f"TablebaseCache: empty cache at {path!r}")

        # Detect format by inspecting the first entry's attributes.
        first = data[0]
        self._is_trajectory: bool = hasattr(first, "fens")

        self._samples: list[TBSample] = []
        self._trajectories: list[TBTrajectory] = []

        if self._is_trajectory:
            for item in data:
                self._trajectories.append(TBTrajectory(
                    fens=list(item.fens),
                    actions=list(item.actions),
                    target_values=[float(v) for v in item.target_values],
                    target_rewards=[float(r) for r in item.target_rewards],
                    legal_actions=[list(a) for a in item.legal_actions],
                    optimal_actions=[list(a) for a in item.optimal_actions],
                    mate_step=item.mate_step,
                ))
        else:
            for item in data:
                self._samples.append(TBSample(
                    fen=item.fen,
                    target_value=float(item.target_value),
                    mating_actions=list(item.mating_actions),
                    optimal_actions=list(item.optimal_actions),
                    all_legal_actions=list(item.all_legal_actions),
                ))

    @property
    def is_trajectory_format(self) -> bool:
        """True if this cache stores K-step trajectories; False for single-position snapshots."""
        return self._is_trajectory

    def sample(self, n: int) -> list[TBSample] | list[TBTrajectory]:
        """Return n randomly sampled entries (with replacement if n > len).

        Returns a list of TBSample if the cache is snapshot-format, or a list
        of TBTrajectory if it is trajectory-format.
        """
        pool = self._trajectories if self._is_trajectory else self._samples
        if n >= len(pool):
            return random.choices(pool, k=n)  # type: ignore[return-value]
        return random.sample(pool, n)  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._trajectories if self._is_trajectory else self._samples)


def build_tb_batch(
    samples: list[TBSample],
    k_steps: int,
    num_actions: int = NUM_ACTIONS,
) -> dict[str, np.ndarray]:
    """Build a training batch dict from a list of TBSamples.

    Produces arrays shaped to match the replay batch K (k_steps). TB samples
    use a 1-step pseudo-trajectory (Option B from the plan): root position at
    step 0, mating action at step 0 action slot, reward=+1 at step 1 if a mating
    move exists. All steps > 1 are zero-padded.

    The value target at step 0 is the Syzygy WDL result (±1 or 0, STM POV).
    The policy target at step 0 is a uniform distribution over optimal_actions.

    Shape contract:
        observations:    [N, K+1, 102, 8, 8]  float32
        actions:         [N, K,   3,  8, 8]   float32
        target_policies: [N, K+1, 4672]        float32
        target_values:   [N, K+1]              float32
        target_rewards:  [N, K+1]              float32
        legal_masks:     [N, 4672]             bool
        is_tablebase:    [N]                   bool

    Args:
        samples:     List of TBSample objects to encode.
        k_steps:     Number of unroll steps K (must match replay batch; e.g. 5).
        num_actions: Size of the action space (default 4672).

    Returns:
        Dict of numpy arrays as described above.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("build_tb_batch: samples list is empty")

    observations    = np.zeros((n, k_steps + 1, 102, 8, 8), dtype=np.float32)
    actions         = np.zeros((n, k_steps,    3,  8, 8),   dtype=np.float32)
    target_policies = np.zeros((n, k_steps + 1, num_actions),  dtype=np.float32)
    target_values   = np.zeros((n, k_steps + 1),                dtype=np.float32)
    target_rewards  = np.zeros((n, k_steps + 1),                dtype=np.float32)
    legal_masks     = np.zeros((n, num_actions),                 dtype=bool)
    is_tablebase    = np.ones(n,                                  dtype=bool)

    for i, sample in enumerate(samples):
        board = chess.Board(sample.fen)
        white_to_move = (board.turn == chess.WHITE)

        # Encode root observation.
        root_obs = encode_board_python(board)  # [102, 8, 8]

        # observations: step 0 = real encode; steps 1..K = zeros.
        observations[i, 0] = root_obs
        # Steps 1..K are already zero from np.zeros initialization.

        # actions: step 0 = mating action if exists, else best optimal action.
        # steps 1..K-1 = zeros.
        chosen_actions = sample.mating_actions if sample.mating_actions else sample.optimal_actions
        if chosen_actions:
            act_idx = chosen_actions[0]
            actions[i, 0] = encode_action_spatial(act_idx, white_to_move)  # [3, 8, 8]
        # Actions at steps 1..K-1 are already zero.

        # target_policies: step 0 = uniform over optimal_actions.
        # steps 1..K = zeros.
        if sample.optimal_actions:
            policy_weight = 1.0 / len(sample.optimal_actions)
            for act in sample.optimal_actions:
                if 0 <= act < num_actions:
                    target_policies[i, 0, act] = policy_weight
        else:
            # Fallback: uniform over all legal actions.
            if sample.all_legal_actions:
                policy_weight = 1.0 / len(sample.all_legal_actions)
                for act in sample.all_legal_actions:
                    if 0 <= act < num_actions:
                        target_policies[i, 0, act] = policy_weight

        # target_values: step 0 = WDL target; steps 1..K = 0.0.
        target_values[i, 0] = sample.target_value

        # target_rewards: step 0 = 0.0; step 1 = +1.0 if mate-in-1 exists; steps 2..K = 0.0.
        if sample.mating_actions and k_steps >= 1:
            target_rewards[i, 1] = 1.0

        # legal_masks: True for all_legal_actions at step 0.
        for act in sample.all_legal_actions:
            if 0 <= act < num_actions:
                legal_masks[i, act] = True

    return {
        "observations":    observations,
        "actions":         actions,
        "target_policies": target_policies,
        "target_values":   target_values,
        "target_rewards":  target_rewards,
        "legal_masks":     legal_masks,
        "is_tablebase":    is_tablebase,
    }


def build_tb_batch_trajectories(
    trajectories: list[TBTrajectory],
    k_steps: int,
    num_actions: int = NUM_ACTIONS,
) -> dict[str, np.ndarray]:
    """Build a training batch from a list of TBTrajectories.

    Produces arrays in the same shape contract as ``build_tb_batch``, but with
    REAL K+1-step supervision at every step: each trajectory step encodes the
    actual board arrived at via optimal-DTZ play, Syzygy WDL as the value
    target, and the terminal reward fires at the actual mate step (not
    mechanically at step 1). Absorbing steps past terminal are zero-padded in
    the canonical MuZero fashion so the network can learn the absorbing
    dynamics under the trainer's K-step unroll.

    ``is_tablebase`` is set to ``False`` on the output so the trainer applies
    the full replay-style K-step loss (no step-1..K masking, consistency loss
    active). This is the intended behavior for trajectory-format supervision.

    Shape contract (same as build_tb_batch):
        observations:    [N, K+1, 102, 8, 8]  float32
        actions:         [N, K,   3,  8, 8]   float32
        target_policies: [N, K+1, 4672]        float32
        target_values:   [N, K+1]              float32
        target_rewards:  [N, K+1]              float32
        legal_masks:     [N, 4672]             bool  (step-0 legality)
        is_tablebase:    [N]                   bool  (all False)

    Args:
        trajectories: List of TBTrajectory objects to encode.
        k_steps:      Number of unroll steps K. Trajectories must be length
                      (K+1) in fens / values / rewards / policies; K in actions.
        num_actions:  Size of the action space (default 4672).

    Returns:
        Dict of numpy arrays as described above.
    """
    n = len(trajectories)
    if n == 0:
        raise ValueError("build_tb_batch_trajectories: trajectories list is empty")

    observations    = np.zeros((n, k_steps + 1, 102, 8, 8), dtype=np.float32)
    actions         = np.zeros((n, k_steps,    3,  8, 8),   dtype=np.float32)
    target_policies = np.zeros((n, k_steps + 1, num_actions),  dtype=np.float32)
    target_values   = np.zeros((n, k_steps + 1),                dtype=np.float32)
    target_rewards  = np.zeros((n, k_steps + 1),                dtype=np.float32)
    legal_masks     = np.zeros((n, num_actions),                 dtype=bool)
    is_tablebase    = np.zeros(n,                                 dtype=bool)

    for i, traj in enumerate(trajectories):
        if len(traj.fens) != k_steps + 1:
            raise ValueError(
                f"trajectory {i}: fens length {len(traj.fens)} != k_steps+1 {k_steps + 1}"
            )
        if len(traj.actions) != k_steps:
            raise ValueError(
                f"trajectory {i}: actions length {len(traj.actions)} != k_steps {k_steps}"
            )

        # Step 0..K: encode observation, value, reward, policy targets.
        # Note: value/reward targets are always copied from the trajectory;
        # only the observation and policy target depend on whether this step
        # is an absorbing (FEN=None) state. The mating reward fires at the
        # absorbing step immediately after the mating move — so r=+1 must
        # land at that step even though FEN is None there.
        for k in range(k_steps + 1):
            target_values[i, k] = traj.target_values[k]
            target_rewards[i, k] = traj.target_rewards[k]

            fen = traj.fens[k]
            if fen is None:
                # Absorbing step: zero observation, zero policy target. Value
                # and reward targets were already set above.
                continue

            # Real step: encode the board.
            board = chess.Board(fen)
            observations[i, k] = encode_board_python(board)

            # Policy target: uniform over Syzygy-optimal actions at this step.
            opt = traj.optimal_actions[k] if k < len(traj.optimal_actions) else []
            if opt:
                w = 1.0 / len(opt)
                for act in opt:
                    if 0 <= act < num_actions:
                        target_policies[i, k, act] = w
            elif traj.legal_actions[k]:
                # Fallback: uniform over legal (e.g. if DTZ probing failed).
                legal_k = traj.legal_actions[k]
                w = 1.0 / len(legal_k)
                for act in legal_k:
                    if 0 <= act < num_actions:
                        target_policies[i, k, act] = w

        # Step 0..K-1: encode the action taken at each transition.
        for k in range(k_steps):
            action_idx = traj.actions[k]
            fen_k = traj.fens[k]
            if action_idx < 0 or fen_k is None:
                # Null action at absorbing step: leave zeros.
                continue
            board_k = chess.Board(fen_k)
            white_to_move = (board_k.turn == chess.WHITE)
            actions[i, k] = encode_action_spatial(action_idx, white_to_move)

        # legal_masks: legal actions at the root step (k=0) — used for policy head masking.
        root_legal = traj.legal_actions[0] if traj.legal_actions else []
        for act in root_legal:
            if 0 <= act < num_actions:
                legal_masks[i, act] = True

    return {
        "observations":    observations,
        "actions":         actions,
        "target_policies": target_policies,
        "target_values":   target_values,
        "target_rewards":  target_rewards,
        "legal_masks":     legal_masks,
        "is_tablebase":    is_tablebase,
    }
