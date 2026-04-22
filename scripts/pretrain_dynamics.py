#!/usr/bin/env python3
"""Dynamics pretraining: train h + g jointly via SimSiam consistency on
(s, a, s') tuples, with f frozen at random init.

Before any RL or self-play, this frontloads the dynamics-learning compute:
- h and g learn to represent board transitions consistently
- Consistency loss is the ONLY training signal (no policy/value/reward loss)
- f stays random — will be trained during subsequent main RL training

Output: checkpoint that can be loaded by the main trainer as a warm-start init.

Env vars:
  HYZERO_PRETRAIN_DATA     — path to (fen, uci) pickle (default: data/pretrain_dynamics.pkl)
  HYZERO_PRETRAIN_CKPT     — output checkpoint path (default: checkpoints/pretrain_dynamics.pt)
  HYZERO_PRETRAIN_STEPS    — max training steps (default: 20000)
  HYZERO_PRETRAIN_BATCH    — batch size (default: 256)
  HYZERO_PRETRAIN_LR       — learning rate (default: 0.001)
  HYZERO_PRETRAIN_TARGET   — cos_sim target for early stop (default: 0.85)
  HYZERO_PRETRAIN_VAR_REG  — variance-regularization weight (default: 0.01)
  HYZERO_PRETRAIN_VAL_EVERY — validation frequency (default: 500 steps)
  HYZERO_PRETRAIN_LOG_EVERY — log frequency (default: 50 steps)
  HYZERO_PRETRAIN_DEVICE   — torch device (default: cpu)
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from hyzero.training.trainer import Trainer
from hyzero.data.board_encoder import (
    encode_board_python,
    encode_action_spatial,
    action_from_move,
)


DATA_PATH = os.environ.get("HYZERO_PRETRAIN_DATA", "data/pretrain_dynamics.pkl")
CKPT_PATH = os.environ.get("HYZERO_PRETRAIN_CKPT", "checkpoints/pretrain_dynamics.pt")
MAX_STEPS = int(os.environ.get("HYZERO_PRETRAIN_STEPS", "20000"))
BATCH_SIZE = int(os.environ.get("HYZERO_PRETRAIN_BATCH", "256"))
LR = float(os.environ.get("HYZERO_PRETRAIN_LR", "0.001"))
COS_TARGET = float(os.environ.get("HYZERO_PRETRAIN_TARGET", "0.85"))
VAR_REG_WEIGHT = float(os.environ.get("HYZERO_PRETRAIN_VAR_REG", "0.01"))
VAL_EVERY = int(os.environ.get("HYZERO_PRETRAIN_VAL_EVERY", "500"))
LOG_EVERY = int(os.environ.get("HYZERO_PRETRAIN_LOG_EVERY", "50"))
DEVICE = os.environ.get("HYZERO_PRETRAIN_DEVICE", "cpu")


def encode_tuple(fen: str, uci: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Encode one (fen, uci) tuple into (obs_before, action_planes, obs_after).

    Returns None on any encoding error (illegal move, bad FEN, etc.).
    """
    try:
        board = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return None
        white_to_move = (board.turn == chess.WHITE)
        action_idx = action_from_move(mv, board)
        obs_before = encode_board_python(board)
        action_planes = encode_action_spatial(action_idx, white_to_move)
        board.push(mv)
        obs_after = encode_board_python(board)
        return obs_before, action_planes, obs_after
    except Exception:
        return None


def build_batch(tuples: list[tuple[str, str]], indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the given tuple indices into a batch of tensors."""
    obs_b_list, act_list, obs_a_list = [], [], []
    for idx in indices:
        fen, uci = tuples[idx]
        enc = encode_tuple(fen, uci)
        if enc is None:
            continue  # skip malformed
        ob, ap, oa = enc
        obs_b_list.append(ob)
        act_list.append(ap)
        obs_a_list.append(oa)
    if not obs_b_list:
        return None, None, None
    obs_b = torch.from_numpy(np.stack(obs_b_list)).to(DEVICE)
    act = torch.from_numpy(np.stack(act_list)).to(DEVICE)
    obs_a = torch.from_numpy(np.stack(obs_a_list)).to(DEVICE)
    return obs_b, act, obs_a


@torch.no_grad()
def validate(trainer: Trainer, val_tuples: list[tuple[str, str]], n_samples: int = 512) -> dict:
    """Compute avg cos_sim between g-unrolled and h-encoded latents on held-out data."""
    trainer.h.eval(); trainer.g.eval()
    indices = random.sample(range(len(val_tuples)), min(n_samples, len(val_tuples)))
    obs_b, act, obs_a = build_batch(val_tuples, indices)
    if obs_b is None:
        return {"cos_sim": float("nan"), "n": 0}

    z_s = trainer.h(obs_b)
    z_pred, _ = trainer.g(z_s, act)
    z_target = trainer.h(obs_a)

    # Raw latent cos_sim (not through projector — this is what MCTS cares about).
    cos_raw = F.cosine_similarity(z_pred.flatten(1), z_target.flatten(1), dim=-1).mean().item()

    # Projected cos_sim (training-space metric, for monitoring loss correlate).
    p1 = trainer.h.predict(trainer.h.project(z_pred))
    p2 = trainer.h.project(z_target)
    cos_proj = F.cosine_similarity(p1, p2, dim=-1).mean().item()

    # Variance of h output across batch — collapse detector.
    # Flatten each sample's latent and take std across samples.
    z_flat = z_target.flatten(1)  # [B, D]
    per_dim_std = z_flat.std(dim=0)  # [D]
    var_metric = per_dim_std.mean().item()

    trainer.h.train(); trainer.g.train()
    return {
        "cos_sim_raw": cos_raw,
        "cos_sim_proj": cos_proj,
        "h_var": var_metric,
        "n": obs_b.shape[0],
    }


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH!r}", file=sys.stderr)
        sys.exit(1)

    print(f"[pretrain] loading dataset from {DATA_PATH!r} ...", flush=True)
    with open(DATA_PATH, "rb") as f:
        all_tuples: list[tuple[str, str]] = pickle.load(f)
    print(f"[pretrain] loaded {len(all_tuples)} tuples", flush=True)

    # Train / val split: last 10k as held-out.
    val_size = min(10000, len(all_tuples) // 10)
    train_tuples = all_tuples[:-val_size]
    val_tuples = all_tuples[-val_size:]
    print(f"[pretrain] train={len(train_tuples)} val={len(val_tuples)}", flush=True)

    # Fresh trainer with random-init networks.
    trainer = Trainer(device=DEVICE)
    print(f"[pretrain] initialized fresh trainer on device={DEVICE}", flush=True)

    # Freeze f (prediction network) — it's not trained during pretraining.
    for p in trainer.f.parameters():
        p.requires_grad = False
    # Also zero out f's existing optimizer state if any (trainer constructed the optimizer
    # over all three networks; we need a new optimizer over h + g only).
    h_g_params = list(trainer.h.parameters()) + list(trainer.g.parameters())
    optimizer = torch.optim.Adam(h_g_params, lr=LR)
    print(f"[pretrain] optimizer over {len(h_g_params)} tensor groups "
          f"(h+g only; f frozen)", flush=True)

    # Set nets to train mode (h and g); f stays in either mode, doesn't matter.
    trainer.h.train(); trainer.g.train()

    # Training loop.
    t_start = time.time()
    best_cos = -1.0
    for step in range(1, MAX_STEPS + 1):
        indices = random.sample(range(len(train_tuples)), BATCH_SIZE)
        obs_b, act, obs_a = build_batch(train_tuples, indices)
        if obs_b is None or obs_b.shape[0] < 2:
            continue  # malformed batch, skip

        # Forward.
        z_s = trainer.h(obs_b)
        z_pred, _ = trainer.g(z_s, act)
        z_target = trainer.h(obs_a)

        # Direct cosine on raw latents (not through SimSiam projector). This is the
        # metric MCTS actually cares about — g's output matching h-encoded ground
        # truth in raw latent space. SimSiam projects to a reduced space where
        # alignment can be easy without raw alignment (we confirmed this).
        # The stop-gradient on the target keeps the SimSiam asymmetry for
        # collapse avoidance.
        z_pred_flat   = z_pred.flatten(1)             # [B, D]
        z_target_flat = z_target.flatten(1).detach()  # [B, D] stop-grad on target branch
        cos = F.cosine_similarity(z_pred_flat, z_target_flat, dim=-1)
        consistency_loss = (1 - cos).mean()

        # Variance regularization: penalize low per-dim std of h outputs (collapse guard).
        # Critical for direct-cosine approach — without it, h could collapse to
        # a constant and cos would be 1 trivially.
        z_t_flat_full = z_target.flatten(1)  # [B, D]
        per_dim_std = z_t_flat_full.std(dim=0)  # [D]
        # Hinge penalty: if per-dim std falls below 1.0, incur a cost.
        var_penalty = F.relu(1.0 - per_dim_std).mean()

        loss = consistency_loss + VAR_REG_WEIGHT * var_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            rate = step / max(elapsed, 1e-9)
            print(
                f"[pretrain] step {step:>6d} loss={loss.item():.5f} "
                f"consistency={consistency_loss.item():.5f} "
                f"var_pen={var_penalty.item():.5f} "
                f"rate={rate:.1f}/s",
                flush=True,
            )

        if step % VAL_EVERY == 0 or step == 1:
            val = validate(trainer, val_tuples)
            print(
                f"[pretrain] VAL step={step} cos_raw={val['cos_sim_raw']:.4f} "
                f"cos_proj={val['cos_sim_proj']:.4f} "
                f"h_var={val['h_var']:.4f} "
                f"n={val['n']}",
                flush=True,
            )
            if val["cos_sim_raw"] > best_cos:
                best_cos = val["cos_sim_raw"]
                # Save checkpoint when we hit a new best.
                ckpt_dir = os.path.dirname(CKPT_PATH)
                if ckpt_dir:
                    os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "h": trainer.h.state_dict(),
                    "g": trainer.g.state_dict(),
                    "f": trainer.f.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_version": 0,  # main training starts from 0
                    "pretrain_step": step,
                    "pretrain_cos_sim_raw": val["cos_sim_raw"],
                    "pretrain_cos_sim_proj": val["cos_sim_proj"],
                }, CKPT_PATH)
                print(f"[pretrain]   saved checkpoint @ step={step} cos_raw={best_cos:.4f}", flush=True)

            # Early stop if target reached.
            if val["cos_sim_raw"] >= COS_TARGET:
                print(f"[pretrain] target cos_sim_raw ≥ {COS_TARGET} reached; stopping.", flush=True)
                break

            # Abort on collapse signal — but only after a warmup period, because a
            # freshly-initialized network with default BN running stats will naturally
            # show low h_var before it has trained on any data.
            if step > 2000 and val["h_var"] < 0.05:
                print(f"[pretrain] COLLAPSE DETECTED (h_var={val['h_var']:.4f} < 0.05); aborting.", flush=True)
                sys.exit(2)

    total = time.time() - t_start
    print(f"[pretrain] done. total time={total:.1f}s best_cos_raw={best_cos:.4f}", flush=True)
    print(f"[pretrain] checkpoint saved to {CKPT_PATH}", flush=True)


if __name__ == "__main__":
    main()
