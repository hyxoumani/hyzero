#!/usr/bin/env python3
"""Deep analysis of WHICH mating patterns erode vs survive under RL training.

For each of our verified mate-in-1 probes, show:
  - Top 8 legal moves with probabilities
  - Whether each "top" move is a WINNING move (keeps +1 WDL) vs drawing/losing
  - Policy entropy
  - Result: is erosion "forgot mating" or "spread across multiple winning moves"?
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch
import chess

from hyzero.config import DEFAULT_CONFIG
from hyzero.inference.server import InferenceServer
from hyzero.data.board_encoder import encode_board_python, action_from_move

NUM_ACTIONS = 4672

POSITIONS = [
    ("back-rank Ra8#",       "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),
    ("Scholar's Mate Qxf7#", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
    ("KQK Qg8# h8 king",     "7k/8/6K1/8/8/8/Q7/8 w - - 0 1"),
    ("KQK Qg8# b8 king",     "1k6/8/1K6/8/8/8/Q7/8 w - - 0 1"),
    ("Qe8# corridor",        "4k3/8/3K4/8/8/8/8/4Q3 w - - 0 1"),
    ("Qa7# king march",      "7k/8/6K1/8/8/8/Q7/8 w - - 0 1"),
    ("Qxh2+ (Black)",        "6k1/8/8/8/8/2q5/6PP/6K1 b - - 0 1"),
]


def load_server(ckpt_path: str, device="cpu") -> InferenceServer:
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    srv.h.load_state_dict(ckpt["h"])
    srv.g.load_state_dict(ckpt["g"])
    srv.f.load_state_dict(ckpt["f"])
    srv.h.eval(); srv.g.eval(); srv.f.eval()
    return srv


@torch.no_grad()
def policy_probs(srv: InferenceServer, board: chess.Board) -> np.ndarray:
    obs = encode_board_python(board)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(srv.device)
    h = srv.h(obs_t)
    logits, _ = srv.f(h)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[action_from_move(m, board)] = True
    masked = logits[0].cpu().numpy().copy()
    masked[~mask] = -1e9
    e = np.exp(masked - masked.max())
    return e / e.sum()


def classify_moves(board: chess.Board) -> dict[chess.Move, str]:
    """Classify each legal move as 'mate' | 'winning' | 'equal' | 'losing'.

    winning/equal/losing based on whether the position after the move is
    clearly winning (lots of extra material vs opponent) or not.
    """
    PIECE_VALUE = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
                   chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
    def material_diff(b: chess.Board, side: chess.Color) -> int:
        mine = sum(v * len(b.pieces(p, side)) for p, v in PIECE_VALUE.items())
        theirs = sum(v * len(b.pieces(p, not side)) for p, v in PIECE_VALUE.items())
        return mine - theirs

    mover = board.turn
    pre_diff = material_diff(board, mover)

    out = {}
    for m in board.legal_moves:
        board.push(m)
        try:
            if board.is_checkmate():
                out[m] = "MATE"
            elif board.is_stalemate() or board.is_insufficient_material():
                out[m] = "draw"
            else:
                post_diff = material_diff(board, mover)
                # If mover had a huge advantage and still does, it's a winning move.
                if pre_diff >= 3 and post_diff >= pre_diff - 1:
                    out[m] = "winning"
                elif post_diff >= pre_diff:
                    out[m] = "keeps_material"
                else:
                    out[m] = "drops_material"
        finally:
            board.pop()
    return out


def analyze_position(srv: InferenceServer, label: str, fen: str) -> None:
    board = chess.Board(fen)
    probs = policy_probs(srv, board)

    classes = classify_moves(board)
    move_data = []
    for m in board.legal_moves:
        a = action_from_move(m, board)
        move_data.append((board.san(m), m, probs[a], classes[m]))
    move_data.sort(key=lambda x: -x[2])

    # Policy entropy (over legal moves only)
    p_legal = np.array([d[2] for d in move_data])
    p_legal = p_legal[p_legal > 1e-12]
    entropy = -(p_legal * np.log(p_legal)).sum()
    n_legal = len(list(board.legal_moves))
    max_entropy = np.log(n_legal)

    # Count winning-like moves
    winning_count = sum(1 for d in move_data if d[3] in ("MATE", "winning"))
    mate_prob = sum(d[2] for d in move_data if d[3] == "MATE")
    winning_prob = sum(d[2] for d in move_data if d[3] in ("MATE", "winning"))

    print(f"--- {label} ({n_legal} legal, {winning_count} winning-ish) ---")
    print(f"  entropy: {entropy:.3f} / {max_entropy:.3f} = {100*entropy/max_entropy:.0f}% of max")
    print(f"  P(mate move): {mate_prob:.3f}")
    print(f"  P(any winning move): {winning_prob:.3f}")
    print(f"  Top 8:")
    for san, m, p, cls in move_data[:8]:
        marker = "★" if cls == "MATE" else ("+" if cls == "winning" else " ")
        print(f"    {marker} {san:>7s} {p:6.3f} [{cls}]")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    srv = load_server(args.ckpt)
    for label, fen in POSITIONS:
        analyze_position(srv, label, fen)


if __name__ == "__main__":
    main()
