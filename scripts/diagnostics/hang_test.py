#!/usr/bin/env python3
"""KQvK queen-hang prior-mass test.

For five PINNED K+Q vs K positions, forward-pass the network's policy head
(masked to legal moves) and report how much prior probability mass the raw
policy places on moves that HANG the queen — i.e. queen moves to a square the
lone enemy king can legally capture (turning a win into a bare-king draw). Also
classifies the top policy move as ``hang`` / ``mate`` / ``safe``.

The five FENs are pinned (not regenerated per run) so results are directly
comparable across checkpoints and sessions — the ad-hoc probes reconstructed
them each time, which made runs non-comparable.

Load the value/policy heads with the SAME config the checkpoint was trained
under (mirrors value_ladder.py):

    HYZERO_VALUE_HEAD       scalar | categorical   (default scalar)
    HYZERO_MOVES_LEFT_HEAD  0 | 1                  (default 0)

Usage:
    python3 scripts/diagnostics/hang_test.py checkpoints/best.pt --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import numpy as np
import torch
import chess

from hyzero.config import DEFAULT_CONFIG
from hyzero.inference.server import InferenceServer
from hyzero.data.board_encoder import encode_board_python, action_from_move

NUM_ACTIONS = 4672

# Five pinned KQvK positions, White (K+Q) to move. In each the queen is far from
# its own king, so many queen moves land next to the bare black king undefended
# and are therefore hangs — a discriminating prior-mass test.
PINNED_FENS = [
    ("q_far_corner_king", "7k/8/8/8/8/8/8/KQ6 w - - 0 1"),
    ("q_far_center_king", "8/8/8/4k3/8/8/8/KQ6 w - - 0 1"),
    ("q_split_center", "8/8/8/8/3k4/8/8/6QK w - - 0 1"),
    ("q_file_opposition", "3k4/8/8/8/8/8/8/3QK3 w - - 0 1"),
    ("q_near_side_king", "8/8/2k5/8/8/8/8/KQ6 w - - 0 1"),
]


def load_server(ckpt_path: str, device: str) -> InferenceServer:
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    with open(ckpt_path, "rb") as handle:
        srv.load_weights(handle.read())
    return srv


@torch.no_grad()
def _policy(srv: InferenceServer, board: chess.Board) -> np.ndarray:
    """Return the legal-masked, renormalized policy over the full action space.

    ``root_setup_batch`` with ``legal_masks=None`` returns an UNMASKED softmax,
    so re-mask to legal moves here — hang mass must be measured over legal moves.
    """
    obs = encode_board_python(board)
    out = srv.root_setup_batch(obs[None, :].astype(np.float32), None)
    probs = out[1][0].astype(np.float64)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[action_from_move(m, board)] = True
    masked = np.where(mask, probs, 0.0)
    total = masked.sum()
    if total > 0:
        masked = masked / total
    return masked


def _hanging_actions(board: chess.Board) -> set[int]:
    """Action indices of legal queen moves the enemy king can then capture.

    A move counts as a hang only if it is not itself checkmate (a mate leaves
    the opponent no reply). The lone enemy king capturing the queen collapses
    KQvK to a drawn bare-king position. ``action_from_move`` is evaluated against
    the pre-move board (before the queen move is pushed).
    """
    hangs: set[int] = set()
    for mv in board.legal_moves:
        piece = board.piece_at(mv.from_square)
        if piece is None or piece.piece_type != chess.QUEEN:
            continue
        action = action_from_move(mv, board)
        board.push(mv)
        is_mate = board.is_checkmate()
        capturable = any(r.to_square == mv.to_square for r in board.legal_moves)
        board.pop()
        if not is_mate and capturable:
            hangs.add(action)
    return hangs


def _classify_top(board: chess.Board, probs: np.ndarray, hangs: set[int]) -> str:
    legal = [(action_from_move(m, board), m) for m in board.legal_moves]
    legal.sort(key=lambda am: -probs[am[0]])
    top_action, top_move = legal[0]
    tmp = board.copy()
    tmp.push(top_move)
    if tmp.is_checkmate():
        return "mate"
    if top_action in hangs:
        return "hang"
    return "safe"


def probe(ckpt_path: str, device: str) -> dict:
    srv = load_server(ckpt_path, device)
    positions = []
    hang_masses = []
    top_classes = []
    for label, fen in PINNED_FENS:
        board = chess.Board(fen)
        probs = _policy(srv, board)
        hangs = _hanging_actions(board)
        hang_mass = float(sum(probs[a] for a in hangs))
        top_class = _classify_top(board, probs, hangs)
        hang_masses.append(hang_mass)
        top_classes.append(top_class)
        positions.append(
            {
                "label": label,
                "fen": fen,
                "n_hanging_moves": len(hangs),
                "hang_prior_mass": hang_mass,
                "top_move_class": top_class,
            }
        )

    return {
        "checkpoint": os.path.basename(ckpt_path),
        "value_head": os.environ.get("HYZERO_VALUE_HEAD", "scalar"),
        "positions": positions,
        "mean_hang_prior_mass": float(np.mean(hang_masses)) if hang_masses else 0.0,
        "top_move_hang_count": sum(1 for c in top_classes if c == "hang"),
        "top_move_mate_count": sum(1 for c in top_classes if c == "mate"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KQvK queen-hang prior-mass test.")
    parser.add_argument("ckpt", help="path to the checkpoint (.pt)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.ckpt, args.device)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
