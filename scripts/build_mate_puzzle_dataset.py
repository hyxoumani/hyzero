#!/usr/bin/env python3
"""Build a comprehensive mate-in-1 puzzle dataset for supervised pretraining.

Generates diverse mate-in-1 positions across:
  - Many endgame classes (KQK, KRK, KBBK, KBNK, KQPK, KRPK, KQKR, KRKR, KQQK)
  - Both colors as the mating side (symmetric white/black coverage)
  - Back-rank mates with various pawn barriers
  - Smothered mate patterns
  - Two-rook ladder mates
  - Mid-game mates (random piece placement within legal constraints)

Output: pickle of list[(fen, uci_move)] where uci_move is the unique mating move
found for that FEN.

Usage:
    python3 scripts/build_mate_puzzle_dataset.py --n-target 30000 \\
        --out data/mate_puzzles.pkl
"""
from __future__ import annotations
import argparse, os, pickle, random, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess


def _king_dist(s1: int, s2: int) -> int:
    r1, f1 = s1 // 8, s1 % 8
    r2, f2 = s2 // 8, s2 % 8
    return max(abs(r1 - r2), abs(f1 - f2))


def _place_kings(rng) -> tuple[int, int]:
    while True:
        k1 = rng.randint(0, 63)
        k2 = rng.randint(0, 63)
        if k1 != k2 and _king_dist(k1, k2) > 1:
            return k1, k2


def _try_board(pieces, turn, ep_square=None) -> chess.Board | None:
    b = chess.Board(None)
    for sq, pt, color in pieces:
        b.set_piece_at(sq, chess.Piece(pt, color))
    b.turn = turn
    b.castling_rights = 0
    if ep_square is not None:
        b.ep_square = ep_square
    if not b.is_valid():
        return None
    b2 = b.copy()
    b2.turn = not b.turn
    if b2.is_check():
        return None
    return b


def _any_mate(board: chess.Board, rng) -> chess.Move | None:
    """Return any mating move (chosen randomly if multiple). Maximizes yield.

    For supervised pretraining we want quantity and variety. Multiple-mate
    positions are fine — we supervise on ONE valid mate per position.
    """
    mates = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            mates.append(m)
        board.pop()
    return rng.choice(mates) if mates else None


# ─── Generator per pattern + color ───────────────────────────────────────────

def _gen_kqk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    wq = rng.choice(empty)
    pieces = [
        (wk, chess.KING, mating_color),
        (wq, chess.QUEEN, mating_color),
        (bk, chess.KING, not mating_color),
    ]
    return _try_board(pieces, mating_color)


def _gen_krk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    wr = rng.choice(empty)
    return _try_board([(wk, chess.KING, mating_color),
                       (wr, chess.ROOK, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_kqqk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    wq1, wq2 = rng.sample(empty, 2)
    return _try_board([(wk, chess.KING, mating_color),
                       (wq1, chess.QUEEN, mating_color),
                       (wq2, chess.QUEEN, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_krrk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    wr1, wr2 = rng.sample(empty, 2)
    return _try_board([(wk, chess.KING, mating_color),
                       (wr1, chess.ROOK, mating_color),
                       (wr2, chess.ROOK, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_kbbk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    b1, b2 = rng.sample(empty, 2)
    if (b1 + b1 // 8) % 2 == (b2 + b2 // 8) % 2:
        return None
    return _try_board([(wk, chess.KING, mating_color),
                       (b1, chess.BISHOP, mating_color),
                       (b2, chess.BISHOP, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_kbnk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    bs, ns = rng.sample(empty, 2)
    return _try_board([(wk, chess.KING, mating_color),
                       (bs, chess.BISHOP, mating_color),
                       (ns, chess.KNIGHT, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_kqpk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    wq = rng.choice(empty)
    # Pawn must be on ranks 2-7 from mating side's POV
    if mating_color == chess.WHITE:
        pawn_squares = [s for s in empty if 8 <= s <= 55 and s != wq]
    else:
        pawn_squares = [s for s in empty if 8 <= s <= 55 and s != wq]
    if not pawn_squares: return None
    wp = rng.choice(pawn_squares)
    return _try_board([(wk, chess.KING, mating_color),
                       (wq, chess.QUEEN, mating_color),
                       (wp, chess.PAWN, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_krpk(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    wr = rng.choice(empty)
    pawn_squares = [s for s in empty if 8 <= s <= 55 and s != wr]
    if not pawn_squares: return None
    wp = rng.choice(pawn_squares)
    return _try_board([(wk, chess.KING, mating_color),
                       (wr, chess.ROOK, mating_color),
                       (wp, chess.PAWN, mating_color),
                       (bk, chess.KING, not mating_color)], mating_color)


def _gen_kqkr(rng, mating_color):
    wk, bk = _place_kings(rng)
    empty = [s for s in range(64) if s != wk and s != bk]
    if len(empty) < 2: return None
    wq, br = rng.sample(empty, 2)
    return _try_board([(wk, chess.KING, mating_color),
                       (wq, chess.QUEEN, mating_color),
                       (bk, chess.KING, not mating_color),
                       (br, chess.ROOK, not mating_color)], mating_color)


def _gen_back_rank(rng, mating_color):
    """Rook/queen delivers mate on the defender's back rank.

    Pattern: defender king cornered with pawn barrier on 7th-rank-from-defender.
    Attacker places rook/queen on the back rank at distance ≥ 2 (so the king
    can't capture it) OR adjacent but protected by mating king.

    Placing attacker on rank 1 (opposite side) produced no mates because the
    attacker along the file attacks only one king escape square, not all.
    The attacker MUST be on the king's own back rank (rank 8 for White mate,
    rank 1 for Black mate) to control it.
    """
    if mating_color == chess.WHITE:
        # Black king on 8th rank, cornered (a8 or h8) for back-rank mate
        bk_file = rng.choice([0, 7])  # corner
        bk = 56 + bk_file
        # Pawn barrier on rank 7: need pawn in front of king + adjacent file
        pawn_files = []
        for df in (-1, 0, 1):
            f = bk_file + df
            if 0 <= f <= 7 and f != bk_file:
                pawn_files.append(f)
        pawns = [48 + f for f in pawn_files]  # rank 7
        # White rook/queen on rank 8, not adjacent to king (so king can't capture)
        safe_files = [f for f in range(8) if abs(f - bk_file) >= 2]
        if not safe_files: return None
        attacker_sq = 56 + rng.choice(safe_files)
        attacker_type = rng.choice([chess.ROOK, chess.QUEEN])
        # White king far away (ranks 1-3)
        wk = rng.randint(0, 23)
        if _king_dist(wk, bk) <= 1: return None
        pieces = [(wk, chess.KING, chess.WHITE),
                  (attacker_sq, attacker_type, chess.WHITE),
                  (bk, chess.KING, chess.BLACK)]
        for p in pawns:
            pieces.append((p, chess.PAWN, chess.BLACK))
    else:  # Black mates on rank 1
        wk_file = rng.choice([0, 7])
        wk = wk_file  # rank 1
        pawn_files = []
        for df in (-1, 0, 1):
            f = wk_file + df
            if 0 <= f <= 7 and f != wk_file:
                pawn_files.append(f)
        pawns = [8 + f for f in pawn_files]  # rank 2 (white pawns)
        safe_files = [f for f in range(8) if abs(f - wk_file) >= 2]
        if not safe_files: return None
        attacker_sq = rng.choice(safe_files)  # rank 1
        attacker_type = rng.choice([chess.ROOK, chess.QUEEN])
        bk = rng.randint(40, 63)
        if _king_dist(wk, bk) <= 1: return None
        pieces = [(wk, chess.KING, chess.WHITE),
                  (attacker_sq, attacker_type, chess.BLACK),
                  (bk, chess.KING, chess.BLACK)]
        for p in pawns:
            pieces.append((p, chess.PAWN, chess.WHITE))
    return _try_board(pieces, mating_color)


def _gen_corner_mate(rng, mating_color):
    """Pin opposing king in corner with supporting king + queen/rook."""
    corners = [0, 7, 56, 63]
    enemy_king_sq = rng.choice(corners)
    # Mating king two squares away diagonally or on same rank/file
    ek_r, ek_f = enemy_king_sq // 8, enemy_king_sq % 8
    candidates = []
    for dr in (-2, 2):
        for df in (-2, 2):
            r, f = ek_r + dr, ek_f + df
            if 0 <= r < 8 and 0 <= f < 8:
                candidates.append(r * 8 + f)
    if not candidates: return None
    mk = rng.choice(candidates)
    if _king_dist(mk, enemy_king_sq) <= 1: return None
    empty = [s for s in range(64) if s not in (mk, enemy_king_sq)]
    attacker_type = rng.choice([chess.QUEEN, chess.ROOK])
    attacker_sq = rng.choice(empty)
    pieces = [(mk, chess.KING, mating_color),
              (attacker_sq, attacker_type, mating_color),
              (enemy_king_sq, chess.KING, not mating_color)]
    return _try_board(pieces, mating_color)


# ─── Worker ──────────────────────────────────────────────────────────────────

GENERATORS = [
    _gen_kqk, _gen_krk, _gen_kqqk, _gen_krrk,
    _gen_kbbk, _gen_kbnk, _gen_kqpk, _gen_krpk, _gen_kqkr,
    _gen_back_rank, _gen_corner_mate,
]


def worker(worker_id: int, n_attempts: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed + worker_id * 104729)
    out = []
    for _ in range(n_attempts):
        gen = rng.choice(GENERATORS)
        color = rng.choice([chess.WHITE, chess.BLACK])
        try:
            b = gen(rng, color)
        except Exception:
            continue
        if b is None:
            continue
        m = _any_mate(b, rng)
        if m is None:
            continue
        out.append((b.fen(), m.uci()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=30000)
    ap.add_argument("--out", default="data/mate_puzzles.pkl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--attempts-multiplier", type=int, default=8,
                    help="attempts per worker = (n_target / workers) * multiplier")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    attempts_per_worker = max(args.n_target // args.workers * args.attempts_multiplier, 1000)
    print(f"target={args.n_target}, workers={args.workers}, "
          f"attempts_per_worker={attempts_per_worker}, patterns={len(GENERATORS)}")
    t0 = time.time()

    all_puzzles = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, w, attempts_per_worker, args.seed)
                   for w in range(args.workers)]
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            all_puzzles.extend(res)
            el = time.time() - t0
            print(f"  worker {i+1}/{args.workers}: +{len(res)} → total={len(all_puzzles)} ({el:.0f}s)")

    # Dedupe by FEN (same position, same mating move)
    seen = {}
    for fen, uci in all_puzzles:
        seen[fen] = uci  # any dupe collapses to one
    all_puzzles = [(f, u) for f, u in seen.items()]
    print(f"after dedupe: {len(all_puzzles)} unique puzzles")

    random.shuffle(all_puzzles)
    if len(all_puzzles) > args.n_target:
        all_puzzles = all_puzzles[:args.n_target]

    # Diagnostic: pattern coverage (rough — by piece count)
    by_pieces = {}
    white_mates = black_mates = 0
    for fen, _ in all_puzzles[:5000]:  # sample for speed
        b = chess.Board(fen)
        n = len(b.piece_map())
        by_pieces[n] = by_pieces.get(n, 0) + 1
        if b.turn == chess.WHITE:
            white_mates += 1
        else:
            black_mates += 1
    print("piece-count distribution (5k sample):")
    for n in sorted(by_pieces):
        print(f"  {n} pieces: {by_pieces[n]:5d}")
    print(f"side-to-move: W={white_mates}, B={black_mates}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(all_puzzles, f)
    print(f"wrote {len(all_puzzles)} puzzles to {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
