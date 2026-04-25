#!/usr/bin/env python3
"""Play N games from the initial position using the network's raw masked
policy — no MCTS. Compare argmax vs temperature-sampled games to see whether
the learned policy has enough breadth to produce diverse games.

Interpretation:
  - If argmax game is the ONLY one we get, the net's top-1 is always the
    same move (expected — deterministic).
  - If temp=1 sampled games are also all identical or nearly so, the net's
    policy is effectively a delta — entropy reg isn't propagating.
  - If temp=1 sampled games diverge meaningfully, the raw policy IS broad;
    then the bottleneck is MCTS re-sharpening it via PUCT.
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


def load_server(ckpt_path: str, device: str = "cpu") -> InferenceServer:
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    srv.h.load_state_dict(ckpt["h"])
    srv.g.load_state_dict(ckpt["g"])
    srv.f.load_state_dict(ckpt["f"])
    srv.h.eval(); srv.g.eval(); srv.f.eval()
    return srv


@torch.no_grad()
def masked_probs(srv: InferenceServer, board: chess.Board,
                 temperature: float) -> tuple[np.ndarray, float]:
    obs = encode_board_python(board)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(srv.device)
    h = srv.h(obs_t)
    logits, value = srv.f(h)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in board.legal_moves:
        mask[action_from_move(m, board)] = True
    lg = logits[0].cpu().numpy().copy()
    lg[~mask] = -1e9
    if temperature == 0.0:
        probs = np.zeros_like(lg)
        probs[lg.argmax()] = 1.0
    else:
        lg = lg / max(temperature, 1e-6)
        e = np.exp(lg - lg.max())
        probs = e / e.sum()
    return probs, float(value.item())


def play_one(srv: InferenceServer, max_plies: int, temperature: float,
             rng: np.random.Generator) -> dict:
    board = chess.Board()
    san_moves: list[str] = []
    top1_probs: list[float] = []
    entropies: list[float] = []
    for _ in range(max_plies):
        if board.is_game_over():
            break
        probs, _ = masked_probs(srv, board, temperature)
        legal_p = probs[probs > 0]
        entropies.append(float(-(legal_p * np.log(legal_p + 1e-12)).sum()))
        top1_probs.append(float(legal_p.max()))
        if temperature == 0.0:
            a = int(probs.argmax())
        else:
            a = int(rng.choice(NUM_ACTIONS, p=probs))
        # Find matching move
        chosen = None
        for m in board.legal_moves:
            if action_from_move(m, board) == a:
                chosen = m
                break
        if chosen is None:
            break
        san_moves.append(board.san(chosen))
        board.push(chosen)
    return {
        "san_moves": san_moves,
        "result": board.result() if board.is_game_over() else "*",
        "n_plies": len(san_moves),
        "mean_top1_prob": float(np.mean(top1_probs)) if top1_probs else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


def first_divergence_ply(games: list[list[str]]) -> int:
    if not games:
        return 0
    min_len = min(len(g) for g in games)
    for i in range(min_len):
        mv = games[0][i]
        if any(g[i] != mv for g in games):
            return i + 1
    return min_len + 1  # all identical up to min length


def unique_prefix_counts(games: list[list[str]], max_ply: int = 20) -> list[int]:
    out = []
    for p in range(1, max_ply + 1):
        prefixes = {tuple(g[:p]) for g in games if len(g) >= p}
        out.append(len(prefixes))
    return out


def report(label: str, games: list[dict]) -> None:
    print(f"\n=== {label} ===")
    for i, g in enumerate(games):
        moves_str = " ".join(
            f"{i//2+1}.{g['san_moves'][i]}" if i % 2 == 0 else g['san_moves'][i]
            for i in range(min(20, len(g['san_moves'])))
        )
        print(f"  [{i:2d}] ({g['result']:>7s}, {g['n_plies']:3d} plies, "
              f"top1̄={g['mean_top1_prob']:.3f}, H̄={g['mean_entropy']:.2f}) {moves_str}")

    san_lists = [g["san_moves"] for g in games]
    div = first_divergence_ply(san_lists)
    print(f"  first divergence at ply {div}")
    uniq = unique_prefix_counts(san_lists, max_ply=20)
    print(f"  distinct prefixes by ply (1..20): {uniq}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    srv = load_server(args.ckpt, device=args.device)

    # Argmax (deterministic) — do once.
    rng = np.random.default_rng(args.seed)
    greedy = [play_one(srv, args.max_plies, temperature=0.0, rng=rng)]
    report("GREEDY (argmax, 1 game - deterministic)", greedy)

    # Sampled at temperature=1.
    rng = np.random.default_rng(args.seed)
    sampled_t1 = [play_one(srv, args.max_plies, temperature=1.0, rng=rng)
                  for _ in range(args.n_games)]
    report(f"SAMPLED temperature=1.0 ({args.n_games} games)", sampled_t1)

    # Sampled at temperature=0.5 (what happens if we sharpen a bit)
    rng = np.random.default_rng(args.seed)
    sampled_t05 = [play_one(srv, args.max_plies, temperature=0.5, rng=rng)
                   for _ in range(args.n_games)]
    report(f"SAMPLED temperature=0.5 ({args.n_games} games)", sampled_t05)


if __name__ == "__main__":
    main()
