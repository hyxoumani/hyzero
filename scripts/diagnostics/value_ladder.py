#!/usr/bin/env python3
"""KQvK value-by-DTZ ladder probe.

Samples winning K+Q vs K positions (strong side to move), probes each position's
Syzygy distance-to-zeroing (DTZ), forward-passes the network's value head, and
reports the mean predicted value per DTZ bucket plus the Pearson/Spearman
correlation between DTZ and value.

A well-shaped value head produces a monotone ladder: positions close to mate
(small DTZ) score near +1 and far-from-mate positions score lower, giving a
strong NEGATIVE dtz-vs-value correlation. A flat ladder (corr near 0) is the
value-signal-starvation signature the campaign chased.

The value head must be loaded with the SAME head configuration the checkpoint
was trained under. Set these to match the training run:

    HYZERO_VALUE_HEAD       scalar | categorical   (default scalar)
    HYZERO_MOVES_LEFT_HEAD  0 | 1                  (default 0)

Usage:
    HYZERO_VALUE_HEAD=categorical \
    python3 scripts/diagnostics/value_ladder.py checkpoints/best.pt \
        --tb data/syzygy --samples 400 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import numpy as np
import torch
import chess
import chess.syzygy

from hyzero.config import DEFAULT_CONFIG
from hyzero.inference.server import InferenceServer
from hyzero.data.board_encoder import encode_board_python

# DTZ buckets: (label, lo, hi) inclusive; the last is open-ended.
_BUCKETS = [
    ("dtz_1_2", 1, 2),
    ("dtz_3_5", 3, 5),
    ("dtz_6_10", 6, 10),
    ("dtz_11_15", 11, 15),
    ("dtz_15plus", 16, 10_000),
]


def load_server(ckpt_path: str, device: str) -> InferenceServer:
    """Load an InferenceServer from a checkpoint, honoring the head-config env."""
    srv = InferenceServer(dict(DEFAULT_CONFIG), device=device)
    with open(ckpt_path, "rb") as handle:
        srv.load_weights(handle.read())
    return srv


def _king_distance(a: int, b: int) -> int:
    return max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))


def _gen_kqvk(rng: random.Random) -> chess.Board | None:
    """Random valid KQvK position with the strong (White) side to move."""
    wk = rng.randint(0, 63)
    bk = rng.randint(0, 63)
    if wk == bk or _king_distance(wk, bk) <= 1:
        return None
    wq = rng.choice([s for s in range(64) if s not in (wk, bk)])
    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = 0
    if not board.is_valid():
        return None
    # Reject positions where the side NOT to move is already in check.
    probe = board.copy()
    probe.turn = chess.BLACK
    if probe.is_check():
        return None
    if board.is_game_over():
        return None
    return board


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    xr = _rankdata(x)
    yr = _rankdata(y)
    return _pearson(xr, yr)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (matches scipy.stats.rankdata default)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # Average ties.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def probe(
    ckpt_path: str,
    tb_path: str,
    samples: int,
    device: str,
    seed: int = 0,
    batch_size: int = 256,
) -> dict:
    rng = random.Random(seed)
    srv = load_server(ckpt_path, device)
    tb = chess.syzygy.open_tablebase(tb_path)

    dtzs: list[int] = []
    boards: list[chess.Board] = []
    attempts = 0
    max_attempts = samples * 200
    while len(boards) < samples and attempts < max_attempts:
        attempts += 1
        board = _gen_kqvk(rng)
        if board is None:
            continue
        try:
            wdl = tb.probe_wdl(board)
            if wdl <= 0:
                continue
            dtz = tb.probe_dtz(board)
        except Exception:
            continue
        if dtz is None or dtz <= 0:
            continue
        boards.append(board)
        dtzs.append(int(dtz))

    values: list[float] = []
    for i in range(0, len(boards), batch_size):
        chunk = boards[i : i + batch_size]
        obs = np.stack([encode_board_python(b) for b in chunk]).astype(np.float32)
        out = srv.root_setup_batch(obs, None)
        values.extend(float(v) for v in out[2])

    dtz_arr = np.asarray(dtzs, dtype=np.float64)
    val_arr = np.asarray(values, dtype=np.float64)

    buckets = {}
    for label, lo, hi in _BUCKETS:
        sel = (dtz_arr >= lo) & (dtz_arr <= hi)
        n = int(sel.sum())
        buckets[label] = {
            "n": n,
            "mean_value": float(val_arr[sel].mean()) if n else None,
        }

    return {
        "checkpoint": os.path.basename(ckpt_path),
        "value_head": os.environ.get("HYZERO_VALUE_HEAD", "scalar"),
        "moves_left_head": os.environ.get("HYZERO_MOVES_LEFT_HEAD", "0"),
        "samples": len(boards),
        "buckets": buckets,
        "pearson_dtz_value": _pearson(dtz_arr, val_arr) if len(boards) > 1 else 0.0,
        "spearman_dtz_value": _spearman(dtz_arr, val_arr) if len(boards) > 1 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KQvK value-by-DTZ ladder probe.")
    parser.add_argument("ckpt", help="path to the checkpoint (.pt)")
    parser.add_argument("--tb", default="data/syzygy", help="Syzygy tablebase dir")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = probe(args.ckpt, args.tb, args.samples, args.device, args.seed)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
