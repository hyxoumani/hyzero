#!/usr/bin/env python3
"""Generate real middlegame FENs via Stockfish-vs-Stockfish play.

Plays fixed-depth Stockfish games from standard openings, extracts positions
between move 15 and 30 where at least 6 captures have occurred (ensuring
structural middlegame, not just ply count).

Env vars:
  HYZERO_MG_OUTPUT        — output FEN file (default: data/middlegame_stockfish.txt)
  HYZERO_MG_N             — target count (default: 40000)
  HYZERO_MG_DEPTH         — Stockfish depth per move (default: 6 — fast, ~50ms/move)
  HYZERO_MG_MIN_CAPTURES  — min captures by extraction point (default: 6)
  HYZERO_MG_PER_GAME      — positions extracted per game (default: 4)
  HYZERO_MG_STOCKFISH_BIN — path to stockfish binary (default: stockfish)
  HYZERO_MG_WORKERS       — parallel engines (default: 8)
  HYZERO_MG_SEED          — PRNG seed (default: 42)

Stockfish at depth 6 produces opening book-like moves early and real middlegame
structure by move 15-20. We seed each game with a random opening move to avoid
all games going down the same line.
"""

from __future__ import annotations
import os
import sys
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import chess.engine


OUTPUT = os.environ.get("HYZERO_MG_OUTPUT", "data/middlegame_stockfish.txt")
N_TARGET = int(os.environ.get("HYZERO_MG_N", "40000"))
DEPTH = int(os.environ.get("HYZERO_MG_DEPTH", "6"))
MIN_CAPTURES = int(os.environ.get("HYZERO_MG_MIN_CAPTURES", "6"))
PER_GAME = int(os.environ.get("HYZERO_MG_PER_GAME", "4"))
STOCKFISH_BIN = os.environ.get("HYZERO_MG_STOCKFISH_BIN", "stockfish")
WORKERS = int(os.environ.get("HYZERO_MG_WORKERS", "8"))
SEED = int(os.environ.get("HYZERO_MG_SEED", "42"))


def play_one_game(worker_id: int, game_seed: int) -> list[str]:
    """Play one Stockfish-vs-Stockfish game, return list of middlegame FENs."""
    rng = random.Random(game_seed)
    fens_out: list[str] = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)
    except Exception as e:
        print(f"[worker {worker_id}] engine start failed: {e}", file=sys.stderr)
        return fens_out

    try:
        board = chess.Board()
        # Random first move to diversify (Stockfish is deterministic at fixed depth).
        legal = list(board.legal_moves)
        board.push(rng.choice(legal))

        captures_so_far = 0
        # Play up to ply 60 (move 30); extract positions between ply 28–60 with ≥6 captures.
        for ply in range(1, 60):
            if board.is_game_over():
                break
            # Count captures by comparing piece count to 32 initial pieces.
            captures_so_far = 32 - len(board.piece_map())
            # Extraction window: plies 28-60 (moves 14-30) AND ≥MIN_CAPTURES captures.
            if 28 <= ply <= 60 and captures_so_far >= MIN_CAPTURES:
                if rng.random() < 0.25:  # don't grab every eligible ply
                    fens_out.append(board.fen())
                    if len(fens_out) >= PER_GAME:
                        break

            # Stockfish move with some randomness (MultiPV=3, pick weighted).
            try:
                result = engine.play(board, chess.engine.Limit(depth=DEPTH))
                if result.move is None:
                    break
                board.push(result.move)
            except Exception:
                break
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    return fens_out


def run_batch(worker_id: int, n_games: int, base_seed: int) -> list[str]:
    """Run a sequential batch of games in one process (one engine instance)."""
    rng = random.Random(base_seed + worker_id * 9973)
    collected: list[str] = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)
    except Exception as e:
        print(f"[w{worker_id}] engine start failed: {e}", file=sys.stderr)
        return collected

    try:
        for g in range(n_games):
            board = chess.Board()
            legal = list(board.legal_moves)
            # Two random opening moves to diversify.
            board.push(rng.choice(legal))
            if not board.is_game_over():
                legal = list(board.legal_moves)
                board.push(rng.choice(legal))

            for ply in range(2, 60):
                if board.is_game_over():
                    break
                captures = 32 - len(board.piece_map())
                if 28 <= ply <= 60 and captures >= MIN_CAPTURES:
                    if rng.random() < 0.25:
                        collected.append(board.fen())
                try:
                    result = engine.play(board, chess.engine.Limit(depth=DEPTH))
                    if result.move is None:
                        break
                    board.push(result.move)
                except Exception:
                    break
    finally:
        try:
            engine.quit()
        except Exception:
            pass
    return collected


def main() -> None:
    random.seed(SEED)
    # Estimate: ~PER_GAME positions per game, so aim for N_TARGET / PER_GAME games.
    # Allow 2× headroom since some games yield fewer.
    n_games_total = max(N_TARGET // PER_GAME * 2, 1000)
    n_per_worker = (n_games_total + WORKERS - 1) // WORKERS

    print(f"[mg] target={N_TARGET} fens, depth={DEPTH}, workers={WORKERS}, "
          f"games_per_worker={n_per_worker}, min_captures={MIN_CAPTURES}", flush=True)
    t0 = time.time()

    all_fens: list[str] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_batch, w, n_per_worker, SEED) for w in range(WORKERS)]
        for i, fut in enumerate(as_completed(futures)):
            fens = fut.result()
            all_fens.extend(fens)
            elapsed = time.time() - t0
            print(f"[mg] worker done ({i+1}/{WORKERS}): +{len(fens)} fens, "
                  f"total={len(all_fens)} ({elapsed:.0f}s)", flush=True)
            if len(all_fens) >= N_TARGET:
                # Can't cleanly cancel the in-flight workers; let them finish.
                pass

    # Deduplicate and shuffle.
    all_fens = list(dict.fromkeys(all_fens))  # preserve order, dedupe
    random.shuffle(all_fens)
    if len(all_fens) > N_TARGET:
        all_fens = all_fens[:N_TARGET]

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w") as f:
        for fen in all_fens:
            f.write(fen + "\n")

    print(f"[mg] wrote {len(all_fens)} FENs to {OUTPUT} in {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
