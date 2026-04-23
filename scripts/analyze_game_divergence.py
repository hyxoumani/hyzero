#!/usr/bin/env python3
"""Compare how two checkpoints play + evaluate chess positions.

Given two trained checkpoints (e.g., peak vs regressed), this script:
  1. For a small set of canonical positions, prints each net's value head
     output and top-5 policy moves.
  2. Plays a short greedy game (argmax of masked policy, no MCTS) from the
     initial position with EACH net and logs per-ply divergence.

Use this to see concretely how the value head collapse affects move choice.

Usage:
    python3 scripts/analyze_game_divergence.py \\
        --peak checkpoints/backup_kqk_peak.pt \\
        --regressed checkpoints/backup_regressed_step5018.pt \\
        --plies 20
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
    encode_action_spatial,
    action_from_move,
)

NUM_ACTIONS = 4672


def load_server(ckpt_path: str, device: str = "cpu") -> InferenceServer:
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    srv.h.load_state_dict(ckpt["h"])
    srv.g.load_state_dict(ckpt["g"])
    srv.f.load_state_dict(ckpt["f"])
    srv.h.eval(); srv.g.eval(); srv.f.eval()
    return srv


@torch.no_grad()
def root_eval(srv: InferenceServer, board: chess.Board) -> tuple[float, np.ndarray]:
    """Forward pass only. Returns (value_scalar, policy_4672 post-softmax)."""
    obs = encode_board_python(board)                    # [102, 8, 8] float32
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(srv.device)  # [1, 102, 8, 8]
    h = srv.h(obs_t)
    logits, value = srv.f(h)
    # Mask illegal moves BEFORE softmax (avoid NaN).
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[action_from_move(m, board)] = True
    masked_logits = logits[0].cpu().numpy().copy()
    masked_logits[~mask] = -1e9
    # Softmax
    exp = np.exp(masked_logits - masked_logits.max())
    probs = exp / exp.sum()
    return float(value.item()), probs


def top_k_moves(probs: np.ndarray, board: chess.Board, k: int = 5) -> list[tuple[str, float, int]]:
    """Return top-k (san_move, probability, action_idx), legal-only."""
    pairs = []
    for m in board.legal_moves:
        a = action_from_move(m, board)
        pairs.append((board.san(m), probs[a], a))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:k]


def canonical_probe_set() -> list[tuple[str, chess.Board]]:
    positions = []
    # Initial position
    positions.append(("initial", chess.Board()))
    # After 1.e4
    b = chess.Board(); b.push_san("e4")
    positions.append(("after 1.e4", b))
    # After 1.e4 e5
    b = chess.Board(); b.push_san("e4"); b.push_san("e5")
    positions.append(("after 1.e4 e5", b))
    # The exact KQK probe position
    positions.append(("KQK probe (K-e1 Q-a2 K-e8)",
                      chess.Board("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1")))
    # Random other KQK
    positions.append(("KQK random (K-a1 Q-d4 K-h8)",
                      chess.Board("7k/8/8/8/3Q4/8/8/K7 w - - 0 1")))
    # KRK
    positions.append(("KRK (K-e1 R-a1 K-e8)",
                      chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")))
    # KPK
    positions.append(("KPK (K-e1 P-e4 K-e8)",
                      chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")))
    # K-vs-KQ (white losing)
    positions.append(("K-vs-KQ (K-e1 vs K-e8 Q-a2)",
                      chess.Board("4k3/q7/8/8/8/8/8/4K3 w - - 0 1")))
    # A simple middlegame
    positions.append(("middlegame (Italian)",
                      chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")))
    # Mate-in-1 for white
    positions.append(("mate-in-1 (white to move)",
                      chess.Board("6k1/6p1/7p/8/8/8/6PP/R5K1 w - - 0 1")))
    return positions


def compare_positions(peak: InferenceServer, regressed: InferenceServer) -> None:
    print("=" * 90)
    print("POSITION-BY-POSITION EVALUATION COMPARISON")
    print("=" * 90)
    for label, board in canonical_probe_set():
        v_peak, p_peak = root_eval(peak, board)
        v_regr, p_regr = root_eval(regressed, board)
        print(f"\n-- {label} (turn: {'W' if board.turn else 'B'}) --")
        print(f"  FEN: {board.fen()}")
        print(f"  value:  peak={v_peak:+.4f}   regressed={v_regr:+.4f}   Δ={v_regr - v_peak:+.4f}")
        tp = top_k_moves(p_peak, board, 5)
        tr = top_k_moves(p_regr, board, 5)
        print(f"  peak top-5:      {', '.join(f'{m}({p:.3f})' for m,p,_ in tp)}")
        print(f"  regressed top-5: {', '.join(f'{m}({p:.3f})' for m,p,_ in tr)}")
        same_top1 = tp[0][0] == tr[0][0] if tp and tr else None
        print(f"  same top-1? {same_top1}")


def play_greedy_game(srv: InferenceServer, plies: int, label: str,
                     start_fen: str | None = None) -> list[dict]:
    """Play a game by always picking the top-1 move of the masked policy. No MCTS."""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    moves_log = []
    for ply in range(plies):
        if board.is_game_over():
            break
        v, p = root_eval(srv, board)
        tops = top_k_moves(p, board, 5)
        if not tops:
            break
        best_san = tops[0][0]
        # Apply the move
        move = next(m for m in board.legal_moves if board.san(m) == best_san)
        moves_log.append({
            "ply": ply + 1,
            "fen_before": board.fen(),
            "player": "W" if board.turn else "B",
            "value": v,
            "top1_san": best_san,
            "top1_prob": tops[0][1],
            "top5": tops,
        })
        board.push(move)
    # Final state
    moves_log.append({
        "ply": len(moves_log) + 1,
        "fen_before": board.fen(),
        "player": "W" if board.turn else "B",
        "value": root_eval(srv, board)[0] if not board.is_game_over() else None,
        "terminal": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    })
    return moves_log


def compare_greedy_games(peak: InferenceServer, regressed: InferenceServer,
                         plies: int) -> None:
    print("\n" + "=" * 90)
    print(f"GREEDY-POLICY GAME FROM INITIAL POSITION ({plies} plies each)")
    print("=" * 90)
    g_peak = play_greedy_game(peak, plies, "peak")
    g_regr = play_greedy_game(regressed, plies, "regressed")

    def to_pgn(moves):
        san_list = [m["top1_san"] for m in moves if "top1_san" in m]
        out = ""
        for i, san in enumerate(san_list):
            if i % 2 == 0:
                out += f"{i//2 + 1}. "
            out += san + " "
        return out.strip()

    print(f"\nPeak game:      {to_pgn(g_peak)}")
    print(f"Regressed game: {to_pgn(g_regr)}")

    # Find first divergence
    same_san = 0
    for a, b in zip(g_peak, g_regr):
        if a.get("top1_san") != b.get("top1_san"):
            break
        same_san += 1
    print(f"\nFirst divergence at ply {same_san + 1}")
    print(f"\nPer-ply value trajectory:")
    print(f"  {'ply':>3} {'peak_san':>10} {'peak_v':>8}   {'regr_san':>10} {'regr_v':>8}")
    for a, b in zip(g_peak, g_regr):
        if "top1_san" in a and "top1_san" in b:
            mark = " " if a["top1_san"] == b["top1_san"] else "*"
            print(f"  {a['ply']:>3}{mark} {a['top1_san']:>10} {a['value']:+.4f}   "
                  f"{b['top1_san']:>10} {b['value']:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak", default="checkpoints/backup_kqk_peak.pt")
    ap.add_argument("--regressed", default="checkpoints/backup_regressed_step5018.pt")
    ap.add_argument("--plies", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"Loading PEAK checkpoint: {args.peak}")
    peak = load_server(args.peak, device=args.device)
    print(f"Loading REGRESSED checkpoint: {args.regressed}")
    regressed = load_server(args.regressed, device=args.device)

    compare_positions(peak, regressed)
    compare_greedy_games(peak, regressed, args.plies)


if __name__ == "__main__":
    main()
