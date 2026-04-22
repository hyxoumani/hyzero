#!/usr/bin/env python3
"""Build a K-step Syzygy tablebase TRAJECTORY cache for canonical-MuZero-shaped
supervision.

Unlike ``build_tablebase_cache.py`` (which stores single-position snapshots),
this script rolls each TB position forward up to K optimal-DTZ plies via
python-chess, re-probes Syzygy WDL at every step, and lands a terminal reward
at the actual checkmate step — giving the reward head many real mate-transition
examples instead of the ~0.29% mate-in-1 positions available from snapshots.

Usage:
    HYZERO_TABLEBASE_PATH=/path/to/syzygy \
    HYZERO_TABLEBASE_CACHE_PATH=data/syzygy/cache_trajectories.pkl \
    HYZERO_TB_N_TOTAL=200000 \
    HYZERO_TB_K_STEPS=5 \
    python3 scripts/build_tablebase_trajectory_cache.py

Environment variables:
    HYZERO_TABLEBASE_PATH:       Directory containing .rtbw/.rtbz files (required).
    HYZERO_TABLEBASE_CACHE_PATH: Output path for pickle. Default: data/syzygy/cache_trajectories.pkl.
    HYZERO_TB_N_TOTAL:           Total trajectories to generate. Default: 200000.
    HYZERO_TB_K_STEPS:           K-step unroll length (must match replay buffer K). Default: 5.

Endgame class composition matches ``build_tablebase_cache.py``.
"""

from __future__ import annotations

import os
import sys
import pickle
import random
from dataclasses import dataclass

# Ensure hyzero package is importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import chess
import chess.syzygy

from hyzero.data.board_encoder import action_from_move


# ─── Config ───────────────────────────────────────────────────────────────────

TB_PATH = os.environ.get("HYZERO_TABLEBASE_PATH")
CACHE_PATH = os.environ.get(
    "HYZERO_TABLEBASE_CACHE_PATH", "data/syzygy/cache_trajectories.pkl"
)
N_TOTAL = int(os.environ.get("HYZERO_TB_N_TOTAL", "200000"))
K_STEPS = int(os.environ.get("HYZERO_TB_K_STEPS", "5"))

ENDGAME_CLASSES: list[tuple[str, int]] = [
    ("KQK",  80_000),
    ("KRK",  80_000),
    ("KBBK", 40_000),
    ("KBNK", 40_000),
    ("KPK",  80_000),
    ("KRKP", 60_000),
    ("KQKR", 60_000),
    ("KRKR", 60_000),
]


# ─── TBTrajectory dataclass (must match python/hyzero/data/tablebase.py) ─────

@dataclass
class TBTrajectory:
    fens: list[str | None]              # length K+1; None marks absorbing step
    actions: list[int]                  # length K; -1 for null (absorbing) action
    target_values: list[float]          # length K+1; Syzygy WDL from each step's STM POV
    target_rewards: list[float]         # length K+1; +1 at mate-transition step, 0 else
    legal_actions: list[list[int]]      # length K+1; [] for absorbing
    optimal_actions: list[list[int]]    # length K+1; [] for absorbing
    mate_step: int | None               # step at which mating reward fires, or None


# ─── King placement (Chebyshev >1) ────────────────────────────────────────────

def _king_distance(sq1: int, sq2: int) -> int:
    r1, f1 = sq1 // 8, sq1 % 8
    r2, f2 = sq2 // 8, sq2 % 8
    return max(abs(r1 - r2), abs(f1 - f2))


def _place_kings() -> tuple[int, int]:
    while True:
        k1 = random.randint(0, 63)
        k2 = random.randint(0, 63)
        if k1 != k2 and _king_distance(k1, k2) > 1:
            return k1, k2


# ─── Root position generators (build a valid python-chess Board) ─────────────

def _try_board(pieces: list[tuple[int, chess.PieceType, chess.Color]],
               turn: chess.Color) -> chess.Board | None:
    """Place the given pieces on an empty board; return it if valid, else None."""
    board = chess.Board(None)
    for sq, piece_type, color in pieces:
        board.set_piece_at(sq, chess.Piece(piece_type, color))
    board.turn = turn
    board.castling_rights = 0
    if not board.is_valid():
        return None
    # Reject positions where side NOT to move is in check.
    b2 = board.copy()
    b2.turn = not board.turn
    if b2.is_check():
        return None
    return board


def _gen_root_kqk() -> chess.Board | None:
    wk, bk = _place_kings()
    wq = random.choice([s for s in range(64) if s != wk and s != bk])
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (wq, chess.QUEEN, chess.WHITE),
                       (bk, chess.KING, chess.BLACK)], turn)


def _gen_root_krk() -> chess.Board | None:
    wk, bk = _place_kings()
    wr = random.choice([s for s in range(64) if s != wk and s != bk])
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (wr, chess.ROOK, chess.WHITE),
                       (bk, chess.KING, chess.BLACK)], turn)


def _gen_root_kbbk() -> chess.Board | None:
    wk, bk = _place_kings()
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2:
        return None
    b1, b2 = random.sample(empty, 2)
    # Different coloured squares for mate possibility.
    if (b1 + b1 // 8) % 2 == (b2 + b2 // 8) % 2:
        return None
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (b1, chess.BISHOP, chess.WHITE),
                       (b2, chess.BISHOP, chess.WHITE),
                       (bk, chess.KING, chess.BLACK)], turn)


def _gen_root_kbnk() -> chess.Board | None:
    wk, bk = _place_kings()
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2:
        return None
    bs, ns = random.sample(empty, 2)
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (bs, chess.BISHOP, chess.WHITE),
                       (ns, chess.KNIGHT, chess.WHITE),
                       (bk, chess.KING, chess.BLACK)], turn)


def _gen_root_kpk() -> chess.Board | None:
    wk, bk = _place_kings()
    pawn_sqs = [s for s in range(8, 56) if s != wk and s != bk]
    if not pawn_sqs:
        return None
    wp = random.choice(pawn_sqs)
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (wp, chess.PAWN, chess.WHITE),
                       (bk, chess.KING, chess.BLACK)], turn)


def _gen_root_krkp() -> chess.Board | None:
    wk, bk = _place_kings()
    occ = {wk, bk}
    empty = [s for s in range(64) if s not in occ]
    if not empty:
        return None
    wr = random.choice(empty)
    occ.add(wr)
    pawn_sqs = [s for s in range(8, 56) if s not in occ]
    if not pawn_sqs:
        return None
    bp = random.choice(pawn_sqs)
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (wr, chess.ROOK, chess.WHITE),
                       (bk, chess.KING, chess.BLACK),
                       (bp, chess.PAWN, chess.BLACK)], turn)


def _gen_root_kqkr() -> chess.Board | None:
    wk, bk = _place_kings()
    occ = {wk, bk}
    empty = [s for s in range(64) if s not in occ]
    if len(empty) < 2:
        return None
    wq, br = random.sample(empty, 2)
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (wq, chess.QUEEN, chess.WHITE),
                       (bk, chess.KING, chess.BLACK),
                       (br, chess.ROOK, chess.BLACK)], turn)


def _gen_root_krkr() -> chess.Board | None:
    wk, bk = _place_kings()
    occ = {wk, bk}
    empty = [s for s in range(64) if s not in occ]
    if len(empty) < 2:
        return None
    wr, br = random.sample(empty, 2)
    turn = random.choice([chess.WHITE, chess.BLACK])
    return _try_board([(wk, chess.KING, chess.WHITE),
                       (wr, chess.ROOK, chess.WHITE),
                       (bk, chess.KING, chess.BLACK),
                       (br, chess.ROOK, chess.BLACK)], turn)


_ROOT_GENERATORS = {
    "KQK":  _gen_root_kqk,
    "KRK":  _gen_root_krk,
    "KBBK": _gen_root_kbbk,
    "KBNK": _gen_root_kbnk,
    "KPK":  _gen_root_kpk,
    "KRKP": _gen_root_krkp,
    "KQKR": _gen_root_kqkr,
    "KRKR": _gen_root_krkr,
}


# ─── Trajectory builder (the core logic) ─────────────────────────────────────

def _wdl_to_value(wdl: int) -> float:
    """Map Syzygy WDL int ∈ {-2,-1,0,1,2} to target value ∈ {-1, 0, +1}."""
    if wdl > 0:
        return 1.0
    if wdl < 0:
        return -1.0
    return 0.0


def _find_mate_in_1_moves(board: chess.Board) -> list[chess.Move]:
    """Return all legal moves that deliver checkmate in one."""
    result: list[chess.Move] = []
    for m in board.legal_moves:
        if not board.gives_check(m):
            continue
        board.push(m)
        if board.is_checkmate():
            result.append(m)
        board.pop()
    return result


def _find_optimal_moves(tb: chess.syzygy.Tablebase,
                        board: chess.Board) -> list[chess.Move]:
    """Return all legal moves achieving minimum |DTZ| (optimal play).

    Falls back to all legal moves if DTZ probing fails.
    """
    legal = list(board.legal_moves)
    if not legal:
        return []

    # Prefer mate-in-1 if available.
    mates = _find_mate_in_1_moves(board)
    if mates:
        return mates

    dtz_scores: list[tuple[int, chess.Move]] = []
    for m in legal:
        board.push(m)
        try:
            dtz = tb.probe_dtz(board)
        except Exception:
            dtz = 99_999
        board.pop()
        dtz_scores.append((abs(dtz), m))

    if not dtz_scores:
        return legal
    min_dtz = min(d for d, _ in dtz_scores)
    return [m for d, m in dtz_scores if d == min_dtz]


def _build_trajectory(tb: chess.syzygy.Tablebase,
                      root_board: chess.Board,
                      k_steps: int) -> TBTrajectory | None:
    """Roll forward k_steps optimal plies from root_board; return TBTrajectory.

    Returns None if the root is already terminal or fails initial validation.
    Terminates the rollout early on checkmate (reward fires, rest absorbing) or
    on stalemate/draw-by-rule (no reward, rest absorbing).
    """
    fens: list[str | None]              = [None] * (k_steps + 1)
    actions: list[int]                  = [-1]   * k_steps
    target_values: list[float]          = [0.0]  * (k_steps + 1)
    target_rewards: list[float]         = [0.0]  * (k_steps + 1)
    legal_actions: list[list[int]]      = [[] for _ in range(k_steps + 1)]
    optimal_actions: list[list[int]]    = [[] for _ in range(k_steps + 1)]
    mate_step: int | None = None

    board = root_board.copy()
    # Probe root.
    try:
        wdl_root = tb.probe_wdl(board)
    except Exception:
        return None
    # Skip positions where root is immediately drawn by insufficient material
    # (they don't teach anything useful) and skip immediate-terminal roots.
    if board.is_checkmate() or board.is_stalemate():
        return None

    # Step 0: record.
    fens[0] = board.fen()
    target_values[0] = _wdl_to_value(wdl_root)
    legal_k = list(board.legal_moves)
    legal_actions[0] = [action_from_move(m, board) for m in legal_k]
    opt_k = _find_optimal_moves(tb, board)
    optimal_actions[0] = [action_from_move(m, board) for m in opt_k]

    # Walk forward up to k_steps plies.
    for k in range(k_steps):
        if not opt_k:
            # No legal moves — terminal without mate (stalemate) or already
            # handled. Pad absorbing from here.
            break

        # Choose an optimal move (uniformly at random among optimals).
        chosen = random.choice(opt_k)
        actions[k] = action_from_move(chosen, board)

        # Push and inspect the resulting position.
        board.push(chosen)

        # Did we just deliver checkmate?
        if board.is_checkmate():
            # Reward fires at step k+1 (transition INTO the mated state).
            target_rewards[k + 1] = 1.0
            mate_step = k + 1
            # fens[k+1] left as None → absorbing observation in the encoder.
            # target_values[k+1] and later remain 0 (absorbing).
            break

        # Did we reach a non-mate terminal (stalemate / insufficient / 50-move / 3-fold)?
        if (board.is_stalemate()
                or board.is_insufficient_material()
                or board.is_fifty_moves()
                or board.is_repetition(3)):
            # No reward (draw). Remaining steps absorbing.
            break

        # Still mid-game — probe + record step k+1.
        try:
            wdl_next = tb.probe_wdl(board)
        except Exception:
            break
        fens[k + 1] = board.fen()
        target_values[k + 1] = _wdl_to_value(wdl_next)
        legal_next = list(board.legal_moves)
        legal_actions[k + 1] = [action_from_move(m, board) for m in legal_next]
        opt_next = _find_optimal_moves(tb, board)
        optimal_actions[k + 1] = [action_from_move(m, board) for m in opt_next]

        # Advance loop variable for next iteration.
        opt_k = opt_next

    return TBTrajectory(
        fens=fens,
        actions=actions,
        target_values=target_values,
        target_rewards=target_rewards,
        legal_actions=legal_actions,
        optimal_actions=optimal_actions,
        mate_step=mate_step,
    )


# ─── Per-class generation loop ────────────────────────────────────────────────

def _generate_class(name: str,
                    root_fn,
                    tb: chess.syzygy.Tablebase,
                    n_target: int,
                    k_steps: int) -> list[TBTrajectory]:
    trajectories: list[TBTrajectory] = []
    attempts = 0
    max_attempts = n_target * 30
    while len(trajectories) < n_target and attempts < max_attempts:
        attempts += 1
        root = root_fn()
        if root is None:
            continue
        traj = _build_trajectory(tb, root, k_steps)
        if traj is not None:
            trajectories.append(traj)
    return trajectories


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if TB_PATH is None:
        print("ERROR: HYZERO_TABLEBASE_PATH must be set", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(TB_PATH):
        print(f"ERROR: HYZERO_TABLEBASE_PATH={TB_PATH!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"[traj_cache] Opening tablebase at {TB_PATH!r}")
    try:
        tb = chess.syzygy.open_tablebase(TB_PATH)
    except Exception as e:
        print(f"ERROR: Failed to open tablebase: {e}", file=sys.stderr)
        sys.exit(1)

    default_total = sum(n for _, n in ENDGAME_CLASSES)
    scale = N_TOTAL / default_total
    scaled_classes = [(n, max(1, int(cnt * scale))) for n, cnt in ENDGAME_CLASSES]
    print(f"[traj_cache] Target: {N_TOTAL} trajectories total (scale={scale:.2f}, K={K_STEPS})")

    all_trajectories: list[TBTrajectory] = []
    for name, target_n in scaled_classes:
        root_fn = _ROOT_GENERATORS[name]
        print(f"[traj_cache] Generating {target_n} trajectories for {name} ...", flush=True)
        class_trajs = _generate_class(name, root_fn, tb, target_n, K_STEPS)
        print(
            f"[traj_cache]   Got {len(class_trajs)} for {name} "
            f"(mate_in_K: {sum(1 for t in class_trajs if t.mate_step is not None)})"
        )
        all_trajectories.extend(class_trajs)

    # Aggregate stats.
    total = len(all_trajectories)
    mate_count = sum(1 for t in all_trajectories if t.mate_step is not None)
    mate_by_step: dict[int | None, int] = {}
    for t in all_trajectories:
        mate_by_step[t.mate_step] = mate_by_step.get(t.mate_step, 0) + 1
    print(f"[traj_cache] Total trajectories: {total}")
    print(f"[traj_cache] With mate transition: {mate_count} ({100.0 * mate_count / max(1, total):.1f}%)")
    print(f"[traj_cache] mate_step histogram: {sorted(mate_by_step.items(), key=lambda p: (p[0] is None, p[0]))}")

    cache_dir = os.path.dirname(CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    print(f"[traj_cache] Writing cache to {CACHE_PATH!r} ...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(all_trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[traj_cache] Done. Cache contains {total} trajectories.")


if __name__ == "__main__":
    main()
