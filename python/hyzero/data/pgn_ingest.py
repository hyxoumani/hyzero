"""PGN → training-record ingest for external-corpus warm-start.

Reads a PGN stream via ``chess.pgn.read_game``, replays each game, and emits
K-step training trajectories whose batch-dict rows match exactly what the
trainer's ``_maybe_mix_tb_samples`` injects (same keys / shapes / dtypes), so
ingested rows ride the SAME mixing/merge injection point in ``train_batch``.

Differences from the tablebase trajectory source (``build_tb_batch_trajectories``):

  * Policy target is a ONE-HOT on the move actually played (optionally
    label-smoothed over the legal moves via ``HYZERO_PGN_POLICY_SMOOTH``),
    rather than uniform-over-Syzygy-optimal. These are good targets, so PGN
    rows set ``tb_policy_mask = False`` (full-weight policy CE).
  * Value target is the game result from each step's side-to-move POV
    (+1 win / 0 draw / -1 loss), optionally decayed toward the root by
    ``HYZERO_PGN_VALUE_DISCOUNT ** plies_to_end`` (default 1.0 = plain outcome).
  * Reward targets are all zero: PGN warm-start supervises value + policy only;
    the reward head is left to self-play / tablebase supervision.
  * The 8 lc0-style repetition planes (102-109) are handled: while replaying the
    game we track position recurrence, and the current-position repetition plane
    (102) is set to 1.0 when the position has occurred earlier in the game. The
    history repetition planes (103-109) stay zero because ``encode_board_python``
    only fills the group-0 (current) position; there are no populated history
    slots to flag.

Cache format mirrors the tablebase cache (``HYZERO_TABLEBASE_CACHE_PATH``-style):
a pickled ``list[PGNTrajectory]`` loaded by ``PGNCache`` with the same
``__main__`` unpickling shim, and sampled the same way.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from dataclasses import dataclass

import chess
import chess.pgn
import numpy as np

from hyzero.data.board_encoder import (
    encode_board_python,
    encode_action_spatial,
    action_from_move,
    NUM_ACTIONS,
)

# Index of the current-position repetition plane (lc0-style planes 102-109;
# 102 = group-0 / current position). See board_encoder.py plane layout.
REP_PLANE_CURRENT = 102

# Absorbing-state sentinel: a FEN of None marks a post-terminal padding step
# (zero observation, null action, zero targets) — same convention as the
# tablebase trajectory builder.
ABSORBING_FEN: str | None = None


@dataclass
class PGNTrajectory:
    """A K-step window of an external PGN game.

    Each trajectory has K+1 steps; step 0 is the window root, steps 1..K chain
    the actual moves played in the game. When the window extends past the game's
    terminal position the remaining steps are absorbing-state placeholders.

    Attributes:
        fens:           Length-(K+1) list of FEN strings. ``None`` marks an
                        absorbing (post-terminal) step.
        actions:        Length-K list of action indices for the move played at
                        each transition. ``-1`` for absorbing / terminal steps.
        policy_actions: Length-(K+1) list of the played-move action index at
                        each step (the one-hot policy target). ``-1`` when no
                        move is played from that step (terminal / absorbing).
        target_values:  Length-(K+1) list of result-from-STM-POV values,
                        optionally decayed by the value discount.
        target_rewards: Length-(K+1) list, all 0.0 (PGN supervises value+policy).
        legal_actions:  Length-(K+1) list of per-step legal action index lists.
                        Empty for absorbing steps.
        rep_flags:      Length-(K+1) list of bools: True if that step's position
                        had already occurred earlier in the game.
    """

    fens: list[str | None]
    actions: list[int]
    policy_actions: list[int]
    target_values: list[float]
    target_rewards: list[float]
    legal_actions: list[list[int]]
    rep_flags: list[bool]


def _policy_smooth() -> float:
    """Label-smoothing weight for PGN policy targets (HYZERO_PGN_POLICY_SMOOTH)."""
    try:
        return float(os.environ.get("HYZERO_PGN_POLICY_SMOOTH", "0.0"))
    except ValueError:
        return 0.0


def _value_discount() -> float:
    """Per-ply value decay toward the root (HYZERO_PGN_VALUE_DISCOUNT, default 1.0)."""
    try:
        return float(os.environ.get("HYZERO_PGN_VALUE_DISCOUNT", "1.0"))
    except ValueError:
        return 1.0


def _result_to_white_outcome(result: str) -> float | None:
    """Map a PGN result header to a White-absolute outcome, or None if unknown."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result == "1/2-1/2":
        return 0.0
    return None


def _game_min_elo(game: chess.pgn.Game) -> int | None:
    """Return the lower of the two players' Elo headers, or None if absent/unparseable."""
    elos: list[int] = []
    for key in ("WhiteElo", "BlackElo"):
        raw = game.headers.get(key, "").strip()
        if raw.isdigit():
            elos.append(int(raw))
    if len(elos) != 2:
        return None
    return min(elos)


def game_to_trajectories(
    game: chess.pgn.Game,
    k_steps: int,
    skip_first_n_plies: int = 0,
) -> list[PGNTrajectory]:
    """Replay one PGN game into per-position K-step trajectories.

    One trajectory is emitted per eligible root position (every position from
    ``skip_first_n_plies`` up to the last position that has a move played).
    Repetition flags are tracked across the whole game before windowing.

    Args:
        game:               Parsed ``chess.pgn.Game``.
        k_steps:            Unroll length K (must match the trainer batch K).
        skip_first_n_plies: Number of opening plies to skip as window roots.

    Returns:
        List of ``PGNTrajectory`` (possibly empty for unusable games).
    """
    outcome = _result_to_white_outcome(game.headers.get("Result", "*"))
    if outcome is None:
        return []

    board = game.board()
    discount = _value_discount()

    # Replay the mainline, recording per-position state. ``fens[i]`` is the
    # position BEFORE move i; ``move_actions[i]`` is the action of move i;
    # ``rep_flags[i]`` marks whether fens[i] recurred earlier in the game.
    fens: list[str] = []
    turns: list[bool] = []            # True == white to move at that position
    move_actions: list[int] = []      # action index of the move played from fens[i]
    legal_lists: list[list[int]] = []
    rep_flags: list[bool] = []

    seen: dict[object, int] = {}

    def record_position() -> None:
        key = board._transposition_key()
        prior = seen.get(key, 0)
        rep_flags.append(prior >= 1)
        seen[key] = prior + 1
        fens.append(board.fen())
        turns.append(board.turn == chess.WHITE)
        legal_lists.append([action_from_move(m, board) for m in board.legal_moves])

    for move in game.mainline_moves():
        record_position()
        move_actions.append(action_from_move(move, board))
        board.push(move)

    # Terminal position (no outgoing move).
    record_position()

    n = len(fens)                     # positions P0..P_{n-1}; P_{n-1} is terminal
    n_moves = len(move_actions)       # == n - 1
    trajectories: list[PGNTrajectory] = []

    # A window root must have a move played from it (so its policy target is
    # defined); the terminal position is never a root.
    for start in range(skip_first_n_plies, n_moves):
        w_fens: list[str | None] = []
        w_actions: list[int] = []
        w_policy: list[int] = []
        w_values: list[float] = []
        w_rewards: list[float] = []
        w_legal: list[list[int]] = []
        w_rep: list[bool] = []

        for j in range(k_steps + 1):
            idx = start + j
            if idx < n:
                w_fens.append(fens[idx])
                stm_sign = 1.0 if turns[idx] else -1.0
                plies_to_end = (n - 1) - idx
                w_values.append(outcome * stm_sign * (discount ** plies_to_end))
                w_policy.append(move_actions[idx] if idx < n_moves else -1)
                w_legal.append(legal_lists[idx])
                w_rep.append(rep_flags[idx])
            else:
                # Absorbing padding past terminal.
                w_fens.append(ABSORBING_FEN)
                w_values.append(0.0)
                w_policy.append(-1)
                w_legal.append([])
                w_rep.append(False)
            w_rewards.append(0.0)

        for j in range(k_steps):
            idx = start + j
            w_actions.append(move_actions[idx] if idx < n_moves else -1)

        trajectories.append(PGNTrajectory(
            fens=w_fens,
            actions=w_actions,
            policy_actions=w_policy,
            target_values=w_values,
            target_rewards=w_rewards,
            legal_actions=w_legal,
            rep_flags=w_rep,
        ))

    return trajectories


def build_pgn_batch(
    trajectories: list[PGNTrajectory],
    k_steps: int,
    num_actions: int = NUM_ACTIONS,
) -> dict[str, np.ndarray]:
    """Build a training batch from a list of PGNTrajectories.

    Produces arrays in the SAME shape contract as
    ``tablebase.build_tb_batch_trajectories`` so the trainer's shared merge path
    accepts them unchanged:

        observations:    [N, K+1, 110, 8, 8]  float32
        actions:         [N, K,   3,  8, 8]   float32
        target_policies: [N, K+1, 4672]        float32
        target_values:   [N, K+1]              float32
        target_rewards:  [N, K+1]              float32
        legal_masks:     [N, 4672]             bool  (step-0 legality)
        is_tablebase:    [N]                   bool  (all False)
        tb_policy_mask:  [N]                   bool  (all False)

    Policy targets are one-hot on the played move, optionally label-smoothed over
    the legal moves by ``HYZERO_PGN_POLICY_SMOOTH``. The current-position
    repetition plane (102) is set per step from each trajectory's rep flags.

    Args:
        trajectories: List of PGNTrajectory objects to encode.
        k_steps:      Unroll length K. Trajectories must be length (K+1) in
                      fens / values / rewards / policy_actions; K in actions.
        num_actions:  Size of the action space (default 4672).

    Returns:
        Dict of numpy arrays as described above.
    """
    n = len(trajectories)
    if n == 0:
        raise ValueError("build_pgn_batch: trajectories list is empty")

    smooth = _policy_smooth()

    observations    = np.zeros((n, k_steps + 1, 110, 8, 8), dtype=np.float32)
    actions         = np.zeros((n, k_steps,    3,  8, 8),   dtype=np.float32)
    target_policies = np.zeros((n, k_steps + 1, num_actions),  dtype=np.float32)
    target_values   = np.zeros((n, k_steps + 1),                dtype=np.float32)
    target_rewards  = np.zeros((n, k_steps + 1),                dtype=np.float32)
    legal_masks     = np.zeros((n, num_actions),                 dtype=bool)
    is_tablebase    = np.zeros(n,                                 dtype=bool)
    # PGN rows carry one-hot played-move policy targets — good supervision, so
    # they are NOT downweighted: tb_policy_mask stays False (full-weight CE).
    tb_policy_mask  = np.zeros(n,                                 dtype=bool)

    for i, traj in enumerate(trajectories):
        if len(traj.fens) != k_steps + 1:
            raise ValueError(
                f"trajectory {i}: fens length {len(traj.fens)} != k_steps+1 {k_steps + 1}"
            )
        if len(traj.actions) != k_steps:
            raise ValueError(
                f"trajectory {i}: actions length {len(traj.actions)} != k_steps {k_steps}"
            )

        for k in range(k_steps + 1):
            target_values[i, k] = traj.target_values[k]
            target_rewards[i, k] = traj.target_rewards[k]

            fen = traj.fens[k]
            if fen is None:
                continue

            board = chess.Board(fen)
            obs = encode_board_python(board)
            if traj.rep_flags[k]:
                # Current position recurred earlier in the game — set the
                # group-0 repetition plane.
                obs[REP_PLANE_CURRENT, :, :] = 1.0
            observations[i, k] = obs

            # Policy target: one-hot on the played move, optionally smoothed.
            played = traj.policy_actions[k]
            legal_k = traj.legal_actions[k]
            if played is not None and played >= 0:
                legal_valid = [a for a in legal_k if 0 <= a < num_actions]
                if smooth > 0.0 and legal_valid:
                    share = smooth / len(legal_valid)
                    for a in legal_valid:
                        target_policies[i, k, a] = share
                    if 0 <= played < num_actions:
                        target_policies[i, k, played] += 1.0 - smooth
                elif 0 <= played < num_actions:
                    target_policies[i, k, played] = 1.0

        for k in range(k_steps):
            action_idx = traj.actions[k]
            fen_k = traj.fens[k]
            if action_idx < 0 or fen_k is None:
                continue
            board_k = chess.Board(fen_k)
            white_to_move = (board_k.turn == chess.WHITE)
            actions[i, k] = encode_action_spatial(action_idx, white_to_move)

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
        "tb_policy_mask":  tb_policy_mask,
    }


class PGNCache:
    """Loaded PGN warm-start cache supporting random sampling.

    Mirrors ``tablebase.TablebaseCache``: a pickled ``list[PGNTrajectory]`` is
    loaded with the same ``__main__`` unpickling shim (the ingest CLI may pickle
    the dataclass from ``__main__``), and ``sample`` returns rows with
    replacement when ``n`` exceeds the pool size.

    Args:
        path: Path to a pickled ``list[PGNTrajectory]`` (built by pgn_ingest).
        seed: Optional seed for the sampling RNG. ``None`` (default) seeds from
            system entropy — the original non-reproducible behavior; passing an
            int makes ``sample`` reproducible.
    """

    def __init__(self, path: str, seed: int | None = None) -> None:
        import types

        # Dedicated RNG so sampling is reproducible when seeded and never
        # perturbs (or is perturbed by) the global ``random`` state.
        self._rng = random.Random(seed)

        _shim = types.ModuleType("__main__")
        _shim.PGNTrajectory = PGNTrajectory  # type: ignore[attr-defined]
        _prev = sys.modules.get("__main__")
        sys.modules["__main__"] = _shim  # type: ignore[assignment]
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        finally:
            if _prev is not None:
                sys.modules["__main__"] = _prev
            else:
                del sys.modules["__main__"]

        if not data:
            raise ValueError(f"PGNCache: empty cache at {path!r}")

        self._trajectories: list[PGNTrajectory] = [
            PGNTrajectory(
                fens=list(item.fens),
                actions=list(item.actions),
                policy_actions=list(item.policy_actions),
                target_values=[float(v) for v in item.target_values],
                target_rewards=[float(r) for r in item.target_rewards],
                legal_actions=[list(a) for a in item.legal_actions],
                rep_flags=[bool(r) for r in item.rep_flags],
            )
            for item in data
        ]

    def sample(self, n: int) -> list[PGNTrajectory]:
        """Return n randomly sampled trajectories (with replacement if n > len)."""
        pool = self._trajectories
        if n >= len(pool):
            return self._rng.choices(pool, k=n)
        return self._rng.sample(pool, n)

    def __len__(self) -> int:
        return len(self._trajectories)


def ingest_pgn_stream(
    stream,
    k_steps: int,
    min_elo: int | None = None,
    max_games: int | None = None,
    skip_first_n_plies: int = 0,
) -> tuple[list[PGNTrajectory], dict[str, int]]:
    """Read a PGN text stream and build training trajectories.

    Args:
        stream:             An open text file object positioned at PGN start.
        k_steps:            Unroll length K.
        min_elo:            Skip games whose lower player Elo is below this
                            (only when both Elo headers are present).
        max_games:          Stop after this many ACCEPTED games (None = all).
        skip_first_n_plies: Opening plies to skip as window roots.

    Returns:
        Tuple ``(trajectories, stats)`` where stats has keys
        ``games_read``, ``games_accepted``, ``games_skipped``, ``positions``.
    """
    trajectories: list[PGNTrajectory] = []
    games_read = games_accepted = games_skipped = 0

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games_read += 1

        if min_elo is not None:
            low = _game_min_elo(game)
            if low is not None and low < min_elo:
                games_skipped += 1
                continue

        game_trajs = game_to_trajectories(
            game, k_steps=k_steps, skip_first_n_plies=skip_first_n_plies
        )
        if not game_trajs:
            games_skipped += 1
            continue

        trajectories.extend(game_trajs)
        games_accepted += 1
        if max_games is not None and games_accepted >= max_games:
            break

    stats = {
        "games_read": games_read,
        "games_accepted": games_accepted,
        "games_skipped": games_skipped,
        "positions": len(trajectories),
    }
    return trajectories, stats


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a PGN corpus into a warm-start training cache."
    )
    parser.add_argument("pgn_path", help="Path to the input PGN file.")
    parser.add_argument("out_path", help="Path to write the pickled cache.")
    parser.add_argument(
        "--k-steps", type=int, default=5,
        help="Unroll length K (must match the trainer batch K; default 5).",
    )
    parser.add_argument(
        "--min-elo", type=int, default=None,
        help="Skip games whose lower player Elo is below this (needs Elo headers).",
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Stop after this many accepted games.",
    )
    parser.add_argument(
        "--skip-first-n-plies", type=int, default=0,
        help="Opening plies to skip as window roots (default 0).",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print a summary of games / positions / skipped after ingest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    with open(args.pgn_path, "r", encoding="utf-8", errors="replace") as f:
        trajectories, stats = ingest_pgn_stream(
            f,
            k_steps=args.k_steps,
            min_elo=args.min_elo,
            max_games=args.max_games,
            skip_first_n_plies=args.skip_first_n_plies,
        )

    with open(args.out_path, "wb") as f:
        pickle.dump(trajectories, f)

    if args.stats:
        print(
            f"[pgn_ingest] games_read={stats['games_read']} "
            f"games_accepted={stats['games_accepted']} "
            f"games_skipped={stats['games_skipped']} "
            f"positions={stats['positions']} "
            f"k_steps={args.k_steps} out={args.out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
