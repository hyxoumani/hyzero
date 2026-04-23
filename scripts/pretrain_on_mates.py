#!/usr/bin/env python3
"""Targeted supervised pretraining on mate-in-1 positions.

Generates random mate-in-1 positions (KQK, KRK, queen-mate-with-king, etc.),
then fine-tunes h/g/f on them with explicit supervision:
    obs  (pre-mate)  → value target = +1 (mover winning)
                     → policy target = one-hot on mating action
    action (mate)     → reward target = +1 (mating transition)
    next_hidden       → value target = -1 (opponent POV of being mated)

This directly attacks the "reward head dead on mates" failure mode found via
scripts/test_mate_in_1.py (~0.03 predicted reward on mating moves even with
value head at +1). After pretraining, re-run test_mate_in_1.py to measure
how many mates the policy finds and how close g's reward prediction is to +1.

Usage:
    python3 scripts/pretrain_on_mates.py \\
        --in-ckpt checkpoints/backup_kqk_peak.pt \\
        --out-ckpt checkpoints/mate_pretrained.pt \\
        --n-positions 10000 --steps 2000
"""
from __future__ import annotations
import argparse, os, sys, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch
import torch.nn.functional as F
import chess

from hyzero.config import DEFAULT_CONFIG
from hyzero.data.board_encoder import (
    encode_board_python, encode_action_spatial, action_from_move,
)

NUM_ACTIONS = 4672


# ─── Mate generator ───────────────────────────────────────────────────────────

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


def _try_board(pieces, turn) -> chess.Board | None:
    b = chess.Board(None)
    for sq, pt, color in pieces:
        b.set_piece_at(sq, chess.Piece(pt, color))
    b.turn = turn
    b.castling_rights = 0
    if not b.is_valid():
        return None
    b2 = b.copy()
    b2.turn = not b.turn
    if b2.is_check():
        return None
    return b


def _find_unique_mating_move(board: chess.Board):
    mates = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            mates.append(m)
        board.pop()
    return mates[0] if len(mates) == 1 else None


def generate_kqk_mate(rng) -> tuple[chess.Board, chess.Move] | None:
    wk, bk = _place_kings()
    avail = [s for s in range(64) if s != wk and s != bk]
    wq = rng.choice(avail)
    turn = chess.WHITE
    b = _try_board([(wk, chess.KING, chess.WHITE),
                    (wq, chess.QUEEN, chess.WHITE),
                    (bk, chess.KING, chess.BLACK)], turn)
    if b is None:
        return None
    m = _find_unique_mating_move(b)
    return (b, m) if m else None


def generate_krk_mate(rng) -> tuple[chess.Board, chess.Move] | None:
    wk, bk = _place_kings()
    avail = [s for s in range(64) if s != wk and s != bk]
    wr = rng.choice(avail)
    turn = chess.WHITE
    b = _try_board([(wk, chess.KING, chess.WHITE),
                    (wr, chess.ROOK, chess.WHITE),
                    (bk, chess.KING, chess.BLACK)], turn)
    if b is None:
        return None
    m = _find_unique_mating_move(b)
    return (b, m) if m else None


def generate_back_rank_mate(rng) -> tuple[chess.Board, chess.Move] | None:
    """White with rook, black king on 8th rank with pawn barrier."""
    wk_sq = rng.randint(0, 15)  # White king on ranks 1-2
    bk_file = rng.randint(5, 7)  # Black king on g8/h8 area
    bk = 56 + bk_file
    # Three pawns in front of king
    pawns = [48 + f for f in (5, 6, 7) if 48 + f != bk_file + 48]
    # White rook somewhere on the a-file or 1st rank
    wr_sq = rng.choice([0, 1, 2, 3])  # rank 1 or 2 — must deliver mate by checking back rank
    turn = chess.WHITE
    pieces = [(wk_sq, chess.KING, chess.WHITE),
              (wr_sq, chess.ROOK, chess.WHITE),
              (bk, chess.KING, chess.BLACK)]
    for p in pawns:
        pieces.append((p, chess.PAWN, chess.BLACK))
    b = _try_board(pieces, turn)
    if b is None:
        return None
    m = _find_unique_mating_move(b)
    return (b, m) if m else None


def generate_mate_in_1(n: int, rng: random.Random) -> list[tuple[chess.Board, chess.Move]]:
    """Generate up to n mate-in-1 positions by rejection sampling."""
    out = []
    gens = [generate_kqk_mate, generate_krk_mate, generate_back_rank_mate]
    attempts = 0
    while len(out) < n and attempts < n * 100:
        attempts += 1
        g = rng.choice(gens)
        r = g(rng)
        if r is not None:
            out.append(r)
    return out


# ─── Network wrapper (load Trainer to get full optimizer + heads) ────────────

def load_trainer(ckpt_path: str, device: str):
    from hyzero.training.trainer import Trainer
    t = Trainer(DEFAULT_CONFIG, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    t.h.load_state_dict(ckpt["h"])
    t.g.load_state_dict(ckpt["g"])
    t.f.load_state_dict(ckpt["f"])
    # Do NOT load optimizer — we want fresh Adam for the new loss distribution.
    t.model_version = ckpt.get("model_version", 0)
    return t


# ─── Training step ────────────────────────────────────────────────────────────

def batch_to_tensors(batch: list[tuple[chess.Board, chess.Move]], device: str):
    """Convert a batch of (board, mating_move) into tensors for pretraining."""
    B = len(batch)
    obs = np.zeros((B, 102, 8, 8), dtype=np.float32)
    action_planes = np.zeros((B, 3, 8, 8), dtype=np.float32)
    action_indices = np.zeros(B, dtype=np.int64)
    for i, (board, move) in enumerate(batch):
        obs[i] = encode_board_python(board)
        white_to_move = (board.turn == chess.WHITE)
        action_indices[i] = action_from_move(move, board)
        action_planes[i] = encode_action_spatial(action_indices[i], white_to_move)
    return (torch.from_numpy(obs).to(device),
            torch.from_numpy(action_planes).to(device),
            torch.from_numpy(action_indices).to(device))


def train_step(trainer, batch, device, optimizer) -> dict:
    obs_t, action_t, action_idx_t = batch_to_tensors(batch, device)
    B = obs_t.shape[0]

    # Forward: h → f at root
    h_state = trainer.h(obs_t)
    logits_root, value_root = trainer.f(h_state)

    # Forward: g one step (applying mating move)
    next_hidden, reward_pred = trainer.g(h_state, action_t)
    _, value_next = trainer.f(next_hidden)

    # Targets
    target_value_root = torch.full((B,), 1.0, device=device)     # mover winning
    target_reward    = torch.full((B,), 1.0, device=device)      # mate transition
    target_value_next = torch.full((B,), -1.0, device=device)    # opponent losing post-mate
    # target_policy: one-hot on mating action
    target_policy = F.one_hot(action_idx_t, num_classes=NUM_ACTIONS).float()

    # Losses
    loss_value  = F.mse_loss(value_root.squeeze(-1), target_value_root)
    loss_reward = F.mse_loss(reward_pred.squeeze(-1), target_reward)
    loss_vnext  = F.mse_loss(value_next.squeeze(-1), target_value_next)
    # Policy: cross-entropy with one-hot → −log(p[mate])
    log_probs = F.log_softmax(logits_root, dim=-1)
    loss_policy = -(log_probs * target_policy).sum(dim=-1).mean()

    loss_total = loss_value + loss_reward + loss_vnext + loss_policy
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    return {
        "total": float(loss_total.item()),
        "value": float(loss_value.item()),
        "reward": float(loss_reward.item()),
        "vnext": float(loss_vnext.item()),
        "policy": float(loss_policy.item()),
        "pred_value": float(value_root.mean().item()),
        "pred_reward": float(reward_pred.mean().item()),
        "pred_vnext": float(value_next.mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ckpt", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--n-positions", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-file", action="store_true",
                    help="Load puzzles from HYZERO_MATE_PUZZLES file (default: data/mate_puzzles.pkl) instead of generating")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # If --puzzle-file is provided, load from it; else generate on-the-fly.
    puzzles = []
    puzzle_file = os.environ.get("HYZERO_MATE_PUZZLES", "data/mate_puzzles.pkl")
    if args.use_file and os.path.exists(puzzle_file):
        print(f"Loading puzzles from {puzzle_file}")
        import pickle
        with open(puzzle_file, "rb") as f:
            raw = pickle.load(f)
        # raw is list[(fen, uci)]; convert to list[(board, move)]
        for fen, uci in raw:
            try:
                b = chess.Board(fen)
                m = chess.Move.from_uci(uci)
                if m in b.legal_moves:
                    puzzles.append((b, m))
            except Exception:
                pass
        print(f"  loaded {len(puzzles)} puzzles")
        if args.n_positions > 0 and len(puzzles) > args.n_positions:
            puzzles = rng.sample(puzzles, args.n_positions)
            print(f"  subsampled to {len(puzzles)}")
    else:
        print(f"Generating {args.n_positions} mate-in-1 positions on-the-fly...")
        t0 = time.time()
        puzzles = generate_mate_in_1(args.n_positions, rng)
        print(f"  produced {len(puzzles)} in {time.time()-t0:.1f}s")

    print(f"Loading {args.in_ckpt}")
    trainer = load_trainer(args.in_ckpt, args.device)
    trainer.h.train(); trainer.g.train(); trainer.f.train()

    params = (list(trainer.h.parameters())
              + list(trainer.g.parameters())
              + list(trainer.f.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.lr)

    print(f"Pretraining {args.steps} steps, batch={args.batch_size}, lr={args.lr}")
    for step in range(args.steps):
        batch = rng.sample(puzzles, min(args.batch_size, len(puzzles)))
        metrics = train_step(trainer, batch, args.device, optimizer)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}: total={metrics['total']:.4f} "
                  f"value={metrics['value']:.4f} "
                  f"reward={metrics['reward']:.4f} "
                  f"vnext={metrics['vnext']:.4f} "
                  f"policy={metrics['policy']:.4f} | "
                  f"pred_value={metrics['pred_value']:+.3f} "
                  f"pred_reward={metrics['pred_reward']:+.3f} "
                  f"pred_vnext={metrics['pred_vnext']:+.3f}")

    # Save checkpoint in same format as the trainer's own save_checkpoint.
    print(f"\nSaving to {args.out_ckpt}")
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    ckpt_out = {
        "h": trainer.h.state_dict(),
        "g": trainer.g.state_dict(),
        "f": trainer.f.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_version": trainer.model_version,
        "mate_pretrain_step": args.steps,
        "mate_pretrain_positions": len(puzzles),
    }
    torch.save(ckpt_out, args.out_ckpt)
    print("Done.")


if __name__ == "__main__":
    main()
