#!/usr/bin/env python3
"""Test whether the network's raw policy head recognizes mate-in-1 moves.

For a suite of canonical mate-in-1 positions (verified against python-chess),
forward-pass the network, mask to legal moves, and report:
  - Rank of the mating move in the policy distribution
  - Probability assigned to the mating move
  - Network's value head output (should be close to +1 for White-to-move mates)

No MCTS involved — this tests whether the policy head learned
"mating-move recognition" directly. If raw policy already ranks mating moves high,
MCTS will amplify. If raw policy treats them as ordinary moves, MCTS needs
many simulations to discover the mate via value-backup.

Usage:
    python3 scripts/test_mate_in_1.py --ckpt checkpoints/backup_kqk_peak.pt
    python3 scripts/test_mate_in_1.py --ckpt checkpoints/backup_regressed_step5018.pt
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch
import chess

from hyzero.config import DEFAULT_CONFIG
from hyzero.inference.server import InferenceServer
from hyzero.data.board_encoder import (
    encode_board_python,
    action_from_move,
)

NUM_ACTIONS = 4672


# Curated mate-in-1 positions. All verified with python-chess to have EXACTLY
# one mating move available to the side to move. Drawn from classic patterns.
MATE_IN_1_POSITIONS = [
    # (label, FEN, expected mating SAN) — expected SAN is just documentation,
    # we actually verify by finding the unique move where board.is_checkmate() holds.
    ("back-rank Ra8#",      "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",             "Ra8#"),
    ("Qh7# smother",         "6rk/6pp/8/8/8/8/6PP/3Q2K1 w - - 0 1",            "Qxh7+ ... not M1"),
    ("Scholar's Mate Qxf7#", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "Qxf7#"),
    ("KQK Qg8# h8 king",     "7k/8/6K1/8/8/8/Q7/8 w - - 0 1",                   "Qg8#"),
    ("KQK Qg8# b8 king",     "1k6/8/1K6/8/8/8/Q7/8 w - - 0 1",                  "Qg8#"),
    ("KQK Qa8# edge",        "k7/8/1K6/8/8/8/Q7/8 w - - 0 1",                   "Qa8#"),
    ("Re8#",                 "6k1/5pp1/7p/8/8/8/6PP/R3R1K1 w - - 0 1",          "Re8#"),
    ("Qe8#",                 "4k3/8/3K4/8/8/8/8/4Q3 w - - 0 1",                "Qe8#"),
    ("Qg7# with K support",  "r1b2rk1/p3pp1p/3p1np1/8/2PQ4/2N5/PP3PPP/3RR1K1 w - - 0 1", "Qxg7#"),
    ("Nf7# smother",         "6rk/6pp/5N2/8/8/8/8/6K1 w - - 0 1",              "Nxh6+"),
    ("R back-rank a1",       "r3k3/8/8/8/8/8/8/1K6 b - - 0 1",                 "Ra1#"),
    ("Qa7# king march",      "7k/8/6K1/8/8/8/Q7/8 w - - 0 1",                  "Qh2#"),
    ("Qxh2+",                "6k1/8/8/8/8/2q5/6PP/6K1 b - - 0 1",              "Qxh8? wait"),
]


def filter_verified_m1s(positions):
    """Keep only positions where a unique mating move exists for side-to-move."""
    verified = []
    for label, fen, _ in positions:
        try:
            board = chess.Board(fen)
        except Exception as e:
            print(f"[skip] {label}: invalid FEN ({e})")
            continue
        mates = []
        for m in board.legal_moves:
            board.push(m)
            if board.is_checkmate():
                mates.append(m)
            board.pop()
        if len(mates) == 1:
            verified.append((label, fen, board, mates[0]))
        else:
            print(f"[skip] {label}: found {len(mates)} mating moves, need exactly 1")
    return verified


def load_server(ckpt_path: str, device: str = "cpu") -> InferenceServer:
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    srv.h.load_state_dict(ckpt["h"])
    srv.g.load_state_dict(ckpt["g"])
    srv.f.load_state_dict(ckpt["f"])
    srv.h.eval(); srv.g.eval(); srv.f.eval()
    return srv


@torch.no_grad()
def eval_position(srv: InferenceServer, board: chess.Board) -> tuple[float, np.ndarray, torch.Tensor]:
    obs = encode_board_python(board)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(srv.device)
    h_state = srv.h(obs_t)
    logits, value = srv.f(h_state)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[action_from_move(m, board)] = True
    masked = logits[0].cpu().numpy().copy()
    masked[~mask] = -1e9
    e = np.exp(masked - masked.max())
    probs = e / e.sum()
    return float(value.item()), probs, h_state


@torch.no_grad()
def eval_move_reward(srv: InferenceServer, h_state: torch.Tensor, action: int,
                     white_to_move: bool) -> tuple[float, float]:
    """Forward pass through g to get (predicted reward, next-state value).

    Returns (reward_after_move, value_from_opponent_pov).
    A correct mate-in-1 transition should yield reward ≈ +1 (mover wins)
    and the next-state value should be near -1 (opponent is mated = losing).
    """
    from hyzero.data.board_encoder import encode_action_spatial
    action_planes = encode_action_spatial(action, white_to_move)  # [3, 8, 8]
    action_t = torch.from_numpy(action_planes).unsqueeze(0).to(srv.device)  # [1, 3, 8, 8]
    next_h, reward = srv.g(h_state, action_t)
    _, next_value = srv.f(next_h)
    return float(reward.item()), float(next_value.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"Loading: {args.ckpt}")
    srv = load_server(args.ckpt, device=args.device)

    verified = filter_verified_m1s(MATE_IN_1_POSITIONS)
    print(f"\nVerified mate-in-1 positions: {len(verified)}")
    hdr = (f"{'#':>2} {'label':25s} {'turn':5s} {'val':>7s} {'mate_rnk':>8s} "
           f"{'mate_p':>7s} {'top1':>8s} {'top1_p':>7s} {'g_rwd(mate)':>11s} {'g_val(aftermate)':>16s}")
    print(hdr)

    top1_hit = 0
    top3_hit = 0
    top5_hit = 0
    reward_recognized = 0
    for idx, (label, fen, board, mate_move) in enumerate(verified):
        value, probs, h_state = eval_position(srv, board)
        mate_action = action_from_move(mate_move, board)
        mate_prob = float(probs[mate_action])

        # Rank among legal moves
        legal_actions = [action_from_move(m, board) for m in board.legal_moves]
        legal_probs = [(a, probs[a]) for a in legal_actions]
        legal_probs.sort(key=lambda x: -x[1])
        ranks = {a: r + 1 for r, (a, _) in enumerate(legal_probs)}
        mate_rank = ranks[mate_action]

        top1_action, top1_prob = legal_probs[0]
        top1_move = next(m for m in board.legal_moves if action_from_move(m, board) == top1_action)
        top1_san = board.san(top1_move)
        mate_san = board.san(mate_move)

        # g-network check: what reward + next-value does the dynamics net predict
        # for the actual mating move?
        white_to_move = board.turn == chess.WHITE
        g_reward, g_next_value = eval_move_reward(srv, h_state, mate_action, white_to_move)

        turn = "W" if board.turn else "B"
        label_short = label[:24]
        print(f"{idx:>2} {label_short:25s} {turn:5s} {value:+7.3f} {mate_rank:>8d} "
              f"{mate_prob:>7.3f} {top1_san:>8s} {top1_prob:>7.3f} "
              f"{g_reward:>+11.3f} {g_next_value:>+16.3f}")

        if mate_rank == 1: top1_hit += 1
        if mate_rank <= 3: top3_hit += 1
        if mate_rank <= 5: top5_hit += 1
        if g_reward > 0.5: reward_recognized += 1

    n = len(verified)
    print(f"\n=== MATE-FINDING SUMMARY ===")
    print(f"Policy top-1  (picks mate):          {top1_hit}/{n} ({100*top1_hit/n:.0f}%)")
    print(f"Policy top-3  (mate in top 3):       {top3_hit}/{n} ({100*top3_hit/n:.0f}%)")
    print(f"Policy top-5  (mate in top 5):       {top5_hit}/{n} ({100*top5_hit/n:.0f}%)")
    print(f"Reward head  (g predicts +>0.5 reward for mating move): "
          f"{reward_recognized}/{n} ({100*reward_recognized/n:.0f}%)")
    print(f"\nBaseline (uniform random over avg legal moves): ~{100/30:.0f}% top-1, ~{100*3/30:.0f}% top-3")


if __name__ == "__main__":
    main()
