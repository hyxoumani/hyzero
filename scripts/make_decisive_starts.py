#!/usr/bin/env python3
"""Build a decisive-start curriculum file for self-play.

Self-play is value-signal starved: ~92.6% of games draw, and ~63.5% of those
draws are by repetition. Uniformly sampling starting positions (FEN per line)
means most games begin balanced, so they tend to shuffle into rule-draws that
carry no decisive ±1 value signal.

This generator biases the start distribution toward decisively-imbalanced
positions (white-absolute material |Δ| ≥ 3) so more games begin with a side
already winning, producing more decisive terminals to train the value head.

Output mix (deterministic given --seed):
  - ~70% positions with |Δ| ≥ 3 (decisively imbalanced)
  - ~30% positions sampled from the original distribution (diversity)

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
        --out data/decisive_starts.txt
    python3 scripts/make_decisive_starts.py --self-test
"""
from __future__ import annotations

import argparse
import random
import sys

# Piece values, same units as compute_material_diff in game_task.rs.
# Keyed by lowercase piece letter (color handled by case in the FEN board field).
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

# A position is "decisively imbalanced" when white-absolute |Δ| ≥ this margin.
DECISIVE_MARGIN = 3

# Target fraction of the output drawn from the imbalanced pool.
IMBALANCED_FRACTION = 0.70

# If the input yields fewer than this many imbalanced positions, synthesize
# more from balanced inputs to reach roughly this floor.
MIN_IMBALANCED = 300

# Pieces eligible for removal during synthesis (never king or pawn).
SYNTH_REMOVABLE = ("n", "b", "r", "q")


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


def build_mix(fens, seed, allow_synth=True):
    """Build the curriculum mix from input FENs.

    Returns (output_fens, stats_dict). Deterministic given `seed`.
    """
    rng = random.Random(seed)
    imbalanced, balanced = classify(fens)

    synthesized = []
    if allow_synth and len(imbalanced) < MIN_IMBALANCED:
        need = MIN_IMBALANCED - len(imbalanced)
        synthesized = synthesize_imbalanced(balanced, need, rng)
        imbalanced = imbalanced + synthesized

    total = len(fens)
    # Sample with replacement so the mix ratio holds regardless of pool sizes,
    # but only when a pool is non-empty.
    n_imbalanced = int(round(total * IMBALANCED_FRACTION))
    n_balanced = total - n_imbalanced

    out = []
    if imbalanced:
        out.extend(rng.choices(imbalanced, k=n_imbalanced))
    elif balanced:
        # No imbalanced pool at all: fall back to original distribution.
        out.extend(rng.choices(balanced, k=n_imbalanced))
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
        "output_count": len(out),
        "output_imbalanced": sum(1 for f in out if is_imbalanced(f)),
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

    # mix ratio: a large input should produce ~70% imbalanced output.
    sample = (
        [_WHITE_UP_ROOK, _BLACK_UP_QUEEN, _WHITE_UP_KNIGHT] * 200
        + [_START_FEN, _WHITE_UP_PAWN] * 200
    )
    out, stats = build_mix(sample, seed=12345, allow_synth=False)
    check(stats["output_count"] == len(sample), "output count must equal input count")
    frac = stats["output_imbalanced"] / max(1, stats["output_count"])
    check(0.60 <= frac <= 0.80, f"output imbalanced fraction {frac:.2f} not ~0.70")

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
    out, stats = build_mix(fens, seed=args.seed)
    write_fens(args.out_path, out)

    if not args.quiet:
        out_frac = stats["output_imbalanced"] / max(1, stats["output_count"])
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
