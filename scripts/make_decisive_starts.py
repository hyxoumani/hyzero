#!/usr/bin/env python3
"""Build a decisive-start curriculum file for self-play.

Self-play is value-signal starved: ~92.6% of games draw, and ~63.5% of those
draws are by repetition. Uniformly sampling starting positions (FEN per line)
means most games begin balanced, so they tend to shuffle into rule-draws that
carry no decisive ±1 value signal.

This generator biases the start distribution toward decisively-imbalanced
positions (white-absolute material |Δ| ≥ 3) so more games begin with a side
already winning, producing more decisive terminals to train the value head.

It also seeds a near-mate bucket: sparse, trivially-won endgames a few plies
shy of checkmate, so self-play practices CONVERTING won positions into mate.
These are derived from a mate-in-1 puzzle pickle by walking BACK 2-5 plies to
build a conversion runway (a bare mate-in-1 start yields a 1-ply game that the
replay buffer discards, because `min_len = unroll_k + 1 = 6` steps at
src/data/replay_buffer.rs:161 with unroll_k=5 from src/py/training.rs).

Output mix (deterministic given --seed):
  - ~55% positions with |Δ| ≥ 3 (decisively imbalanced)
  - ~25% positions sampled from the original distribution (diversity)
  - ~20% near-mate conversion starts walked back from mate-in-1 puzzles

When the mate-puzzle bucket is unavailable (file missing or python-chess
absent), its share is redistributed proportionally to the imbalanced/original
buckets (so the output stays full and the existing curriculum is unchanged).

Material values match `compute_material_diff` in src/selfplay/game_task.rs:
  P=1, N=3, B=3, R=5, Q=9, K=0.

If the input yields too few imbalanced positions (< MIN_IMBALANCED), the
shortfall is topped up by synthesizing from balanced input FENs: remove ONE
non-king, non-pawn piece so that |Δ| ≥ 3. Synthesized FENs are emitted ONLY
when valid and non-terminal, validated with python-chess. If python-chess is
unavailable, synthesis is skipped entirely (rather than emit unvalidated FENs
that the Rust side would silently fall back from to the standard start,
diluting the curriculum).

Does NOT modify or overwrite the input file.

Usage:
    python3 scripts/make_decisive_starts.py \
        --in data/starting_positions.txt \
        --out data/decisive_starts.txt \
        --mate-puzzles data/mate_puzzles_v2.pkl
    python3 scripts/make_decisive_starts.py --self-test
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys

# Piece values, same units as compute_material_diff in game_task.rs.
# Keyed by lowercase piece letter (color handled by case in the FEN board field).
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

# A position is "decisively imbalanced" when white-absolute |Δ| ≥ this margin.
DECISIVE_MARGIN = 3

# Target fractions of the output by bucket. The near-mate bucket is only filled
# when the mate-puzzle source is available; otherwise its share is redistributed
# to the imbalanced/original buckets in their existing 55:25 proportion.
IMBALANCED_FRACTION = 0.55
NEAR_MATE_FRACTION = 0.20
# The remaining ~0.25 is the original distribution (diversity).

# If the input yields fewer than this many imbalanced positions, synthesize
# more from balanced inputs to reach roughly this floor.
MIN_IMBALANCED = 300

# Pieces eligible for removal during synthesis (never king or pawn).
SYNTH_REMOVABLE = ("n", "b", "r", "q")

# Near-mate walk-back: half-move depths to draw from, mixed roughly uniformly.
WALK_BACK_DEPTHS = (2, 3, 4, 5)

# Each near-mate output must have a legal continuation at least this many plies
# long (no forced-win proof — just legality) so the resulting self-play game can
# exceed the replay buffer's min_len = unroll_k + 1 = 6 steps and survive.
MIN_RUNWAY_PLIES = 6

# Per-FEN walk-back retries before skipping a mate puzzle (different random moves).
WALK_BACK_RETRIES = 6

# Distinct near-mate FENs to build into the pool before sampling with replacement.
# Sized so the pool comfortably covers the output's near-mate share with variety.
NEAR_MATE_POOL_SIZE = 4000


def _has_python_chess() -> bool:
    """Return True if python-chess is importable."""
    try:
        import chess  # noqa: F401
    except ImportError:
        return False
    return True


def compute_material_diff(fen: str) -> int:
    """White-absolute material diff for a FEN, matching game_task.rs.

    Uppercase letters are white pieces, lowercase are black. Returns
    (white material) − (black material) using P1/N3/B3/R5/Q9/K0.
    """
    board_field = fen.split()[0]
    delta = 0
    for ch in board_field:
        if ch == "/" or ch.isdigit():
            continue
        val = PIECE_VALUES.get(ch.lower())
        if val is None:
            continue
        if ch.isupper():
            delta += val
        else:
            delta -= val
    return delta


def is_imbalanced(fen: str, margin: int = DECISIVE_MARGIN) -> bool:
    """True if the white-absolute material |Δ| is at least `margin`."""
    return abs(compute_material_diff(fen)) >= margin


def classify(fens):
    """Split FENs into (imbalanced, balanced) lists preserving input order."""
    imbalanced = []
    balanced = []
    for fen in fens:
        if is_imbalanced(fen):
            imbalanced.append(fen)
        else:
            balanced.append(fen)
    return imbalanced, balanced


def _fen_is_valid_nonterminal(fen: str) -> bool:
    """Validate a FEN with python-chess: parseable, legal, and non-terminal.

    Returns False if python-chess is unavailable (caller must guard).
    """
    try:
        import chess
    except ImportError:
        return False
    try:
        board = chess.Board(fen)
    except ValueError:
        return False
    if not board.is_valid():
        return False
    # Non-terminal: at least one legal move and not already game-over.
    if board.is_game_over(claim_draw=False):
        return False
    if not any(board.legal_moves):
        return False
    return True


def synthesize_imbalanced(balanced_fens, need, rng):
    """Synthesize up to `need` imbalanced FENs from balanced inputs.

    For each balanced FEN, remove ONE non-king, non-pawn piece so the
    resulting |Δ| ≥ DECISIVE_MARGIN. To make white-absolute |Δ| large, we
    remove a piece so the opponent of its owner gains the imbalance —
    concretely we try removing each eligible piece and keep the first removal
    that yields a valid, non-terminal, imbalanced position.

    Requires python-chess for validation; returns [] if unavailable.
    """
    if not _has_python_chess():
        return []
    import chess

    out = []
    pool = list(balanced_fens)
    rng.shuffle(pool)
    for fen in pool:
        if len(out) >= need:
            break
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        # Candidate squares holding eligible (non-king, non-pawn) pieces.
        squares = []
        for sq, piece in board.piece_map().items():
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                squares.append(sq)
        rng.shuffle(squares)
        for sq in squares:
            trial = board.copy(stack=False)
            trial.remove_piece_at(sq)
            trial_fen = trial.fen()
            if is_imbalanced(trial_fen) and _fen_is_valid_nonterminal(trial_fen):
                out.append(trial_fen)
                break
    return out


def _heavy_piece_safe(board, color):
    """True if every heavy piece (Q/R) of `color` is not attacked-and-undefended.

    A heavy piece that is attacked by the opponent and not defended by its own
    side is one defender move from being lost — which would dissolve the win. We
    reject any walk-back ply that leaves the winning side's heavy piece hanging.
    """
    import chess

    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type in (chess.QUEEN, chess.ROOK):
            if board.is_attacked_by(not color, sq) and not board.is_attacked_by(
                color, sq
            ):
                return False
    return True


def _king_only_moves(board, color):
    """Legal moves of `color` that move only its king (keeps the heavy piece put)."""
    king_sq = board.king(color)
    return [m for m in board.legal_moves if m.from_square == king_sq]


def _has_runway(board, plies):
    """True if some legal line of length `plies` stays non-terminal at every step.

    Depth-first existence check (not a forced-win proof): guarantees the position
    can yield a self-play game longer than the replay buffer's min_len so the
    trajectory is not discarded. `plies` is small (6), so the search is cheap.
    """
    if plies == 0:
        return True
    for move in board.legal_moves:
        trial = board.copy(stack=False)
        trial.push(move)
        if trial.is_game_over(claim_draw=False):
            continue
        if _has_runway(trial, plies - 1):
            return True
    return False


def _walk_back_once(fen, plies, rng):
    """Walk back `plies` half-moves from a mate-in-1 FEN; return a FEN or None.

    Alternates (winning-side king-only move, defender random legal move). The
    side to move at the mate FEN is the winning side. After each ply the position
    must stay legal, non-terminal, and keep the winning side's heavy piece safe.
    Returns None if any ply has no valid candidate.
    """
    import chess

    board = chess.Board(fen)
    winning = board.turn  # side delivering mate-in-1 is the winning side
    cur = board.copy()
    for _ in range(plies):
        if cur.turn == winning:
            candidates = _king_only_moves(cur, winning)
        else:
            candidates = list(cur.legal_moves)
        if not candidates:
            return None
        rng.shuffle(candidates)
        advanced = False
        for move in candidates:
            trial = cur.copy()
            trial.push(move)
            if trial.is_game_over(claim_draw=False):
                continue
            if not trial.is_valid():
                continue
            if not _heavy_piece_safe(trial, winning):
                continue
            cur = trial
            advanced = True
            break
        if not advanced:
            return None
    if cur.is_game_over(claim_draw=False):
        return None
    if not _has_runway(cur, MIN_RUNWAY_PLIES):
        return None
    return cur.fen()


def _walk_back(fen, plies, rng, retries=WALK_BACK_RETRIES):
    """Walk back `plies` plies, retrying with fresh random moves before skipping."""
    for _ in range(retries):
        out = _walk_back_once(fen, plies, rng)
        if out is not None:
            return out
    return None


def load_mate_puzzles(path):
    """Load a mate-in-1 puzzle pickle as a list of FEN strings.

    The pickle is list[(fen, uci_move)]; only the FEN is needed here (the mate
    move is implied by side-to-move). Returns [] on any load/format error so the
    near-mate bucket degrades gracefully rather than aborting generation.
    """
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError):
        return []
    fens = []
    for entry in data:
        if isinstance(entry, (tuple, list)) and entry:
            fens.append(entry[0])
        elif isinstance(entry, str):
            fens.append(entry)
    return fens


def build_near_mate_pool(mate_fens, pool_size, rng):
    """Build a pool of distinct near-mate conversion starts from mate-in-1 FENs.

    For each sampled mate FEN, walk back a depth drawn ~uniformly from
    WALK_BACK_DEPTHS, validating each ply. Skips FENs whose walk-back fails.
    Returns (pool, depth_counts). Requires python-chess; returns ([], {}) without.
    """
    if not _has_python_chess():
        return [], {}
    pool = []
    depth_counts = {d: 0 for d in WALK_BACK_DEPTHS}
    order = list(mate_fens)
    rng.shuffle(order)
    for fen in order:
        if len(pool) >= pool_size:
            break
        depth = rng.choice(WALK_BACK_DEPTHS)
        start = _walk_back(fen, depth, rng)
        if start is None:
            continue
        pool.append(start)
        depth_counts[depth] += 1
    return pool, depth_counts


def build_mix(fens, seed, allow_synth=True, mate_fens=None):
    """Build the curriculum mix from input FENs.

    Returns (output_fens, stats_dict). Deterministic given `seed`. When
    `mate_fens` is a non-empty list and python-chess is available, ~20% of the
    output is near-mate conversion starts; otherwise that share is redistributed
    to the imbalanced/original buckets and the near-mate count is 0.
    """
    rng = random.Random(seed)
    imbalanced, balanced = classify(fens)

    near_mate_pool = []
    depth_counts = {}
    if mate_fens:
        near_mate_pool, depth_counts = build_near_mate_pool(
            mate_fens, NEAR_MATE_POOL_SIZE, rng
        )

    synthesized = []
    if allow_synth and len(imbalanced) < MIN_IMBALANCED:
        need = MIN_IMBALANCED - len(imbalanced)
        synthesized = synthesize_imbalanced(balanced, need, rng)
        imbalanced = imbalanced + synthesized

    total = len(fens)
    # Sample with replacement so the mix ratio holds regardless of pool sizes,
    # but only when a pool is non-empty. The near-mate bucket only contributes
    # when its pool is non-empty; otherwise its share is redistributed to the
    # imbalanced/original buckets in their existing 55:25 proportion.
    if near_mate_pool:
        imbalanced_frac = IMBALANCED_FRACTION
        near_mate_frac = NEAR_MATE_FRACTION
    else:
        # Redistribute the near-mate share proportionally (0.55 : 0.25 → split).
        rest = 1.0 - NEAR_MATE_FRACTION  # 0.80
        imbalanced_frac = IMBALANCED_FRACTION / rest
        near_mate_frac = 0.0

    n_imbalanced = int(round(total * imbalanced_frac))
    n_near_mate = int(round(total * near_mate_frac))
    n_balanced = total - n_imbalanced - n_near_mate

    out = []
    near_mate_out = set()
    if imbalanced:
        out.extend(rng.choices(imbalanced, k=n_imbalanced))
    elif balanced:
        # No imbalanced pool at all: fall back to original distribution.
        out.extend(rng.choices(balanced, k=n_imbalanced))
    if n_near_mate > 0 and near_mate_pool:
        near_mate_sample = rng.choices(near_mate_pool, k=n_near_mate)
        near_mate_out = set(near_mate_sample)
        out.extend(near_mate_sample)
    if balanced:
        out.extend(rng.choices(balanced, k=n_balanced))
    elif imbalanced:
        out.extend(rng.choices(imbalanced, k=n_balanced))

    rng.shuffle(out)

    stats = {
        "input_count": total,
        "imbalanced_input": len(imbalanced) - len(synthesized),
        "synthesized": len(synthesized),
        "imbalanced_pool": len(imbalanced),
        "balanced_pool": len(balanced),
        "near_mate_pool": len(near_mate_pool),
        "near_mate_depths": depth_counts,
        "output_count": len(out),
        "output_imbalanced": sum(1 for f in out if is_imbalanced(f)),
        "output_near_mate": sum(1 for f in out if f in near_mate_out),
    }
    return out, stats


def read_fens(path):
    """Read non-empty, stripped FEN lines from `path`."""
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def write_fens(path, fens):
    """Write FENs one per line (no header)."""
    with open(path, "w", encoding="utf-8") as fh:
        for fen in fens:
            fh.write(fen + "\n")


# ── Embedded self-test ─────────────────────────────────────────────
_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# White up a rook (black missing a8 rook): Δ = +5.
_WHITE_UP_ROOK = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"
# Black up a queen (white missing d1 queen): Δ = -9.
_BLACK_UP_QUEEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
# White up exactly a knight: Δ = +3 (decisive margin boundary).
_WHITE_UP_KNIGHT = "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# White up a single pawn: Δ = +1 (below the decisive margin).
_WHITE_UP_PAWN = "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Sparse mate-in-1 positions for side-to-move (KQ-vs-K / KR-vs-K conversions),
# matching the shape of data/mate_puzzles_v2.pkl. Used to exercise walk-back.
_MATE_IN_1_FENS = [
    "4K3/8/k7/3q2q1/8/8/8/8 b - - 0 1",
    "8/8/8/8/8/4K3/2Q5/5k2 w - - 0 1",
    "8/q7/8/8/7K/2k5/8/6q1 b - - 0 1",
]


def self_test():
    """Exercise classify / mix / validate on embedded FENs. Returns 0 on pass."""
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # classify: material diffs.
    check(compute_material_diff(_START_FEN) == 0, "start FEN should be balanced (Δ=0)")
    check(compute_material_diff(_WHITE_UP_ROOK) == 5, "white-up-rook should be Δ=+5")
    check(compute_material_diff(_BLACK_UP_QUEEN) == -9, "black-up-queen should be Δ=-9")
    check(compute_material_diff(_WHITE_UP_KNIGHT) == 3, "white-up-knight should be Δ=+3")
    check(compute_material_diff(_WHITE_UP_PAWN) == 1, "white-up-pawn should be Δ=+1")

    # is_imbalanced: margin boundary is inclusive at 3.
    check(is_imbalanced(_WHITE_UP_KNIGHT), "Δ=+3 must classify as imbalanced")
    check(is_imbalanced(_BLACK_UP_QUEEN), "Δ=-9 must classify as imbalanced")
    check(not is_imbalanced(_START_FEN), "balanced start must not be imbalanced")
    check(not is_imbalanced(_WHITE_UP_PAWN), "Δ=+1 must not be imbalanced")

    imbalanced, balanced = classify(
        [_START_FEN, _WHITE_UP_ROOK, _BLACK_UP_QUEEN, _WHITE_UP_PAWN]
    )
    check(len(imbalanced) == 2, "classify should find 2 imbalanced")
    check(len(balanced) == 2, "classify should find 2 balanced (start + pawn)")

    # mix ratio: with no mate bucket, the near-mate share (0.20) is redistributed,
    # so imbalanced rises to 0.55/0.80 ≈ 0.69 of the output.
    sample = (
        [_WHITE_UP_ROOK, _BLACK_UP_QUEEN, _WHITE_UP_KNIGHT] * 200
        + [_START_FEN, _WHITE_UP_PAWN] * 200
    )
    out, stats = build_mix(sample, seed=12345, allow_synth=False)
    check(stats["output_count"] == len(sample), "output count must equal input count")
    check(stats["output_near_mate"] == 0, "no mate bucket → near-mate output must be 0")
    frac = stats["output_imbalanced"] / max(1, stats["output_count"])
    check(0.62 <= frac <= 0.76, f"output imbalanced fraction {frac:.2f} not ~0.69")

    # determinism: same seed → identical output.
    out2, _ = build_mix(sample, seed=12345, allow_synth=False)
    check(out == out2, "build_mix must be deterministic for a fixed seed")
    out3, _ = build_mix(sample, seed=999, allow_synth=False)
    check(out != out3, "different seeds should produce different output")

    # validate: synthesis path (only if python-chess present).
    if _has_python_chess():
        rng = random.Random(7)
        synth = synthesize_imbalanced([_START_FEN] * 10, need=5, rng=rng)
        check(len(synth) > 0, "synthesis should produce ≥1 FEN from balanced starts")
        for fen in synth:
            check(is_imbalanced(fen), f"synthesized FEN must be imbalanced: {fen}")
            check(
                _fen_is_valid_nonterminal(fen),
                f"synthesized FEN must be valid & non-terminal: {fen}",
            )
        # Top-up path: starts-only input must reach the imbalanced floor.
        out_synth, stats_synth = build_mix([_START_FEN] * 500, seed=3)
        check(
            stats_synth["synthesized"] > 0,
            "balanced-only input should trigger synthesis",
        )

        # Near-mate walk-back: each output must parse, be non-terminal, and have
        # ≥ MIN_RUNWAY_PLIES of legal play (so the self-play game exceeds the
        # replay buffer's min_len = unroll_k + 1 = 6 and is not discarded).
        import chess

        nm_rng = random.Random(11)
        nm_pool, nm_depths = build_near_mate_pool(
            _MATE_IN_1_FENS * 50, pool_size=60, rng=nm_rng
        )
        check(len(nm_pool) > 0, "walk-back should yield ≥1 near-mate start")
        check(
            sum(1 for d in WALK_BACK_DEPTHS if nm_depths.get(d, 0) > 0) >= 2,
            f"walk-back depths should span ≥2 of {WALK_BACK_DEPTHS}: {nm_depths}",
        )
        for fen in nm_pool:
            board = chess.Board(fen)
            check(board.is_valid(), f"near-mate FEN must be valid: {fen}")
            check(
                not board.is_game_over(claim_draw=False),
                f"near-mate FEN must be non-terminal: {fen}",
            )
            check(
                _has_runway(board, MIN_RUNWAY_PLIES),
                f"near-mate FEN must have ≥{MIN_RUNWAY_PLIES} playable plies: {fen}",
            )

        # Near-mate fraction band: with a mate bucket, output is ~20% near-mate.
        nm_sample = (
            [_WHITE_UP_ROOK, _BLACK_UP_QUEEN, _WHITE_UP_KNIGHT] * 200
            + [_START_FEN, _WHITE_UP_PAWN] * 200
        )
        nm_out, nm_stats = build_mix(
            nm_sample, seed=7, allow_synth=False, mate_fens=_MATE_IN_1_FENS * 50
        )
        check(
            nm_stats["output_count"] == len(nm_sample),
            "near-mate mix output count must equal input count",
        )
        nm_frac = nm_stats["output_near_mate"] / max(1, nm_stats["output_count"])
        check(
            0.15 <= nm_frac <= 0.25,
            f"near-mate output fraction {nm_frac:.2f} not in [0.15, 0.25]",
        )
        # Determinism with the mate bucket present.
        nm_out2, _ = build_mix(
            nm_sample, seed=7, allow_synth=False, mate_fens=_MATE_IN_1_FENS * 50
        )
        check(nm_out == nm_out2, "near-mate mix must be deterministic for a fixed seed")
    else:
        print("[self-test] python-chess unavailable — skipping synthesis checks")

    if failures:
        for msg in failures:
            print(f"[self-test] FAIL: {msg}", file=sys.stderr)
        print(f"[self-test] {len(failures)} failure(s)", file=sys.stderr)
        return 1
    print("[self-test] all checks passed")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        default="data/starting_positions.txt",
        help="input starts file (FEN per line). Never modified.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/decisive_starts.txt",
        help="output curriculum file (FEN per line).",
    )
    parser.add_argument(
        "--mate-puzzles",
        dest="mate_path",
        default="data/mate_puzzles_v2.pkl",
        help=(
            "mate-in-1 puzzle pickle (list[(fen, uci)]) for the near-mate bucket. "
            "If missing or unreadable, the near-mate share is redistributed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260610,
        help="deterministic RNG seed (fixed default).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run embedded classify/mix/validate self-test and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the stats summary on stderr.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    fens = read_fens(args.in_path)
    mate_fens = load_mate_puzzles(args.mate_path) if args.mate_path else []
    out, stats = build_mix(fens, seed=args.seed, mate_fens=mate_fens)
    write_fens(args.out_path, out)

    if not args.quiet:
        out_frac = stats["output_imbalanced"] / max(1, stats["output_count"])
        nm_frac = stats["output_near_mate"] / max(1, stats["output_count"])
        print(
            f"[make_decisive_starts] in={args.in_path} ({stats['input_count']} FENs) "
            f"-> out={args.out_path} ({stats['output_count']} FENs)",
            file=sys.stderr,
        )
        print(
            f"[make_decisive_starts] imbalanced_input={stats['imbalanced_input']} "
            f"synthesized={stats['synthesized']} "
            f"output_imbalanced={stats['output_imbalanced']} ({out_frac:.1%})",
            file=sys.stderr,
        )
        print(
            f"[make_decisive_starts] mate_puzzles={len(mate_fens)} "
            f"near_mate_pool={stats['near_mate_pool']} "
            f"depths={stats['near_mate_depths']} "
            f"output_near_mate={stats['output_near_mate']} ({nm_frac:.1%})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
