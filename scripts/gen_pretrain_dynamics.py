#!/usr/bin/env python3
"""Generate (s, a, s') tuples for dynamics pretraining via random-play chess.

Output format: pickle of list[(fen_before, move_uci)] — the next board is
reconstructed at training time by applying the move. Storing move_uci avoids
the need for a reverse action-index encoder and keeps each tuple tiny.

Diversity strategy:
  - Playouts start from the initial position and continue until game-over or
    a cap on plies.
  - At each visited position, sample K random legal moves to RECORD (not play).
  - Play ONE random move to advance to the next position.
  - This yields ~K tuples per ply × ~50 plies per game = ~200-500 tuples per
    game, covering a mix of opening/middlegame/endgame positions across playouts.

Env vars:
  HYZERO_PRETRAIN_OUTPUT   — output path (default: data/pretrain_dynamics.pkl)
  HYZERO_PRETRAIN_N        — target tuple count (default: 2000000)
  HYZERO_PRETRAIN_K        — tuples recorded per position (default: 3)
  HYZERO_PRETRAIN_MAX_PLIES — max plies per game before restart (default: 200)
  HYZERO_PRETRAIN_SEED     — PRNG seed (default: 42)
"""

from __future__ import annotations

import os
import sys
import pickle
import random
import time

import chess


OUTPUT = os.environ.get("HYZERO_PRETRAIN_OUTPUT", "data/pretrain_dynamics.pkl")
N_TARGET = int(os.environ.get("HYZERO_PRETRAIN_N", "2000000"))
K_PER_POS = int(os.environ.get("HYZERO_PRETRAIN_K", "3"))
MAX_PLIES = int(os.environ.get("HYZERO_PRETRAIN_MAX_PLIES", "200"))
SEED = int(os.environ.get("HYZERO_PRETRAIN_SEED", "42"))


def generate() -> list[tuple[str, str]]:
    random.seed(SEED)
    tuples: list[tuple[str, str]] = []
    n_games = 0
    n_positions = 0
    t_start = time.time()

    while len(tuples) < N_TARGET:
        board = chess.Board()
        n_games += 1
        for _ in range(MAX_PLIES):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            if not legal:
                break
            n_positions += 1

            # Record up to K random legal moves from this position.
            n_record = min(K_PER_POS, len(legal))
            for mv in random.sample(legal, n_record):
                tuples.append((board.fen(), mv.uci()))
                if len(tuples) >= N_TARGET:
                    break
            if len(tuples) >= N_TARGET:
                break

            # Advance the game by playing one random legal move.
            board.push(random.choice(legal))

        if n_games % 200 == 0:
            elapsed = time.time() - t_start
            rate = len(tuples) / max(elapsed, 1e-9)
            print(
                f"[gen] games={n_games} positions={n_positions} "
                f"tuples={len(tuples)} ({100 * len(tuples) / N_TARGET:.1f}%) "
                f"rate={rate:.0f}/s elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(
        f"[gen] DONE games={n_games} positions={n_positions} "
        f"tuples={len(tuples)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return tuples


def main() -> None:
    tuples = generate()

    # Shuffle so playouts don't cluster in the dataset.
    print("[gen] shuffling …", flush=True)
    random.shuffle(tuples)

    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[gen] writing {len(tuples)} tuples to {OUTPUT!r} …", flush=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(tuples, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"[gen] done. file size: {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
