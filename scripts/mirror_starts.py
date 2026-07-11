#!/usr/bin/env python3
"""Mirror-augment a FEN starts file by adding file-reflected positions.

The memorization-radius study showed hyzero generalizes locally around trained
positions (~+20 pts one square off-manifold), and that position VARIETY
amplifies that transfer. File mirroring (a<->h reflection) is a cheap ~2x
variety multiplier: every legal position has a legal mirror image with the same
game-theoretic value, so mirroring doubles the start distribution without any
new supervision.

For each input start this emits the original plus its file-mirror, except:
  - Mirrors are SKIPPED when the original carries any castling rights: a
    horizontal flip swaps king/queenside and does not preserve castling, so the
    reflected FEN would misdescribe the position. (Endgame curricula are
    castling-free, so this only guards odd inputs.)
  - Duplicates are dropped on a board+side-to-move key, including collisions
    against originals and previously emitted mirrors (self-symmetric positions
    mirror onto themselves and are skipped this way).
  - Every mirrored FEN is validated legal with python-chess; illegal mirrors
    (should not occur under horizontal symmetry) are dropped and counted.

To keep the augmented set disjoint from the evaluation probes, any candidate
(original OR mirror) whose key matches a probe position — in EITHER orientation,
i.e. the probe's own key or the probe's mirror key — is dropped and reported.

Does NOT modify the input file.

Usage:
    python3 scripts/mirror_starts.py \
        --in data/curriculum_endgame_10k.txt \
        --out data/curriculum_endgame_mirror16k.txt \
        --probes data/probe_won_starts_120.txt data/probe_holdout_starts_150.txt
    python3 scripts/mirror_starts.py --self-test
"""
from __future__ import annotations

import argparse
import sys

# File reflection a<->h maps board columns and en-passant/coordinate files.
_FILE_MIRROR = str.maketrans("abcdefgh", "hgfedcba")
# Horizontal flip swaps king- and queen-side castling for each color, so a true
# involution must swap the castling letters too (mirrors that carry castling are
# never emitted, but this keeps mirror() self-inverse for the involution test).
_CASTLE_MIRROR = str.maketrans("KQkq", "QKqk")


def _has_python_chess() -> bool:
    """Return True if python-chess is importable."""
    try:
        import chess  # noqa: F401
    except ImportError:
        return False
    return True


def _expand_rank(rank: str) -> str:
    """Expand a FEN rank to 8 single-char cells ('1' per empty square)."""
    out = []
    for ch in rank:
        if ch.isdigit():
            out.append("1" * int(ch))
        else:
            out.append(ch)
    return "".join(out)


def _compress_rank(cells: str) -> str:
    """Compress an 8-cell rank string back to FEN run-length form."""
    out = []
    run = 0
    for ch in cells:
        if ch == "1":
            run += 1
            continue
        if run:
            out.append(str(run))
            run = 0
        out.append(ch)
    if run:
        out.append(str(run))
    return "".join(out)


def mirror_board_field(board_field: str) -> str:
    """File-reflect (a<->h) a FEN board-placement field."""
    ranks = board_field.split("/")
    return "/".join(_compress_rank(_expand_rank(r)[::-1]) for r in ranks)


def _mirror_square(sq: str) -> str:
    """File-reflect an algebraic square/en-passant field ('-' passes through)."""
    if sq == "-":
        return sq
    return sq[0].translate(_FILE_MIRROR) + sq[1:]


def mirror_fen(fen: str) -> str:
    """Return the file-mirror (a<->h reflection) of a FEN.

    Side to move and move clocks are unchanged; the board, castling rights, and
    en-passant square are reflected. This is a pure-string involution:
    ``mirror_fen(mirror_fen(x)) == x``.
    """
    parts = fen.split()
    board, stm, castling, ep = parts[0], parts[1], parts[2], parts[3]
    rest = parts[4:]
    mirrored = [
        mirror_board_field(board),
        stm,
        castling.translate(_CASTLE_MIRROR) if castling != "-" else "-",
        _mirror_square(ep),
    ]
    return " ".join(mirrored + rest)


def position_key(fen: str) -> str:
    """Dedup key: board placement + side to move (ignores clocks/ep/castling)."""
    parts = fen.split()
    return parts[0] + " " + parts[1]


def has_castling_rights(fen: str) -> bool:
    """True if the FEN carries any castling rights (castling field != '-')."""
    return fen.split()[2] != "-"


def is_legal_fen(fen: str) -> bool:
    """Validate a FEN with python-chess: parseable and a legal position.

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
    return board.is_valid()


def build_forbidden_keys(probe_fens):
    """Forbidden dedup keys: each probe position in BOTH orientations.

    A mirrored curriculum start could land on a probe position or on a probe's
    own mirror image; forbidding both orientations keeps the augmented set
    disjoint from the probes regardless of which reflection it collides with.
    """
    forbidden = set()
    for fen in probe_fens:
        forbidden.add(position_key(fen))
        forbidden.add(position_key(mirror_fen(fen)))
    return forbidden


def build_mirrored(starts, forbidden_keys=None):
    """Emit originals + legal file-mirrors, deduped and probe-disjoint.

    Returns (output_fens, stats). Requires python-chess for legality validation.
    """
    forbidden = forbidden_keys or set()
    seen = set()
    out = []
    stats = {
        "input_count": len(starts),
        "originals_kept": 0,
        "mirrors_kept": 0,
        "dup_originals": 0,
        "dup_mirrors": 0,
        "castling_skips": 0,
        "illegal_mirrors": 0,
        "probe_collisions_original": 0,
        "probe_collisions_mirror": 0,
    }

    for fen in starts:
        key = position_key(fen)
        if key in seen:
            stats["dup_originals"] += 1
        elif key in forbidden:
            stats["probe_collisions_original"] += 1
        else:
            out.append(fen)
            seen.add(key)
            stats["originals_kept"] += 1

        if has_castling_rights(fen):
            stats["castling_skips"] += 1
            continue
        mfen = mirror_fen(fen)
        if not is_legal_fen(mfen):
            stats["illegal_mirrors"] += 1
            continue
        mkey = position_key(mfen)
        if mkey in seen:
            stats["dup_mirrors"] += 1
            continue
        if mkey in forbidden:
            stats["probe_collisions_mirror"] += 1
            continue
        out.append(mfen)
        seen.add(mkey)
        stats["mirrors_kept"] += 1

    stats["output_count"] = len(out)
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
# KR-vs-K endgame, castling-free, asymmetric about the vertical axis.
_KR_ENDGAME = "8/R7/8/2k5/8/3R4/8/5K2 w - - 0 1"
# Endgame with an en-passant square, to exercise ep reflection.
_EP_ENDGAME = "8/8/8/2pP4/8/8/k7/4K3 w - c6 0 1"


def self_test():
    """Exercise mirror involution, castling skip, and legality. Returns 0 on pass."""
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # Involution: mirroring twice restores the original FEN (pure-string check).
    # (No legal chess position is file-self-symmetric: a<->h fixes no file, so a
    # lone king cannot map onto itself — hence involution is tested on pairs.)
    for fen in (_START_FEN, _KR_ENDGAME, _EP_ENDGAME):
        check(mirror_fen(mirror_fen(fen)) == fen, f"mirror not involutive: {fen}")

    # File reflection actually flips a<->h (not a no-op) for asymmetric boards.
    mkr = mirror_fen(_KR_ENDGAME)
    check(position_key(mkr) != position_key(_KR_ENDGAME), "asymmetric mirror must differ")
    check(mirror_board_field("8/R7/8/2k5/8/3R4/8/5K2") == "8/7R/8/5k2/8/4R3/8/2K5",
          "board reflection incorrect")
    # En-passant file reflects (c6 -> f6).
    check(mirror_fen(_EP_ENDGAME).split()[3] == "f6", "en-passant file must reflect")
    # Castling letters swap king/queen side; the start's right-set is preserved.
    check(set(mirror_fen(_START_FEN).split()[2]) == set("KQkq"),
          "start castling right-set must be preserved under mirror")

    # Castling skip: an input with castling rights emits only the original.
    out, stats = build_mirrored([_START_FEN])
    check(stats["castling_skips"] == 1, "castling FEN must skip its mirror")
    check(stats["mirrors_kept"] == 0, "no mirror emitted for castling FEN")
    check(out == [_START_FEN], "castling FEN output must be the original only")

    # Dedup vs originals AND emitted mirrors: feeding a start and its own mirror
    # yields exactly two positions (the second input's mirror re-collides).
    out_dup, stats_dup = build_mirrored([_KR_ENDGAME, mkr])
    check(stats_dup["output_count"] == 2, "start + its mirror must yield 2 outputs")
    check(stats_dup["dup_originals"] == 1, "the mirror re-input must dedup as original")
    check(stats_dup["dup_mirrors"] == 1, "its mirror must dedup against the original")

    # Probe disjointness: a start whose mirror equals a probe is dropped.
    forbidden = build_forbidden_keys([_KR_ENDGAME])
    _, stats_p = build_mirrored([mkr], forbidden_keys=forbidden)
    check(stats_p["probe_collisions_original"] == 1,
          "start matching a probe mirror must be dropped")

    if _has_python_chess():
        import chess

        # Legality: an asymmetric endgame and its mirror are both legal.
        out_kr, stats_kr = build_mirrored([_KR_ENDGAME])
        check(stats_kr["mirrors_kept"] == 1, "legal endgame must emit its mirror")
        check(stats_kr["illegal_mirrors"] == 0, "endgame mirror must be legal")
        for fen in out_kr:
            check(chess.Board(fen).is_valid(), f"emitted FEN must be legal: {fen}")
        # The emitted mirror is the file-reflection and legal on its own.
        check(chess.Board(mkr).is_valid(), "reflected endgame FEN must be legal")
    else:
        print("[self-test] python-chess unavailable — skipping legality checks")

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
        default="data/curriculum_endgame_10k.txt",
        help="input starts file (FEN per line). Never modified.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/curriculum_endgame_mirror16k.txt",
        help="output augmented file (FEN per line).",
    )
    parser.add_argument(
        "--probes",
        nargs="*",
        default=[
            "data/probe_won_starts_120.txt",
            "data/probe_holdout_starts_150.txt",
        ],
        help="probe files to stay disjoint from (both orientations checked).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run embedded mirror/legality self-test and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the stats summary on stderr.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not _has_python_chess():
        print(
            "[mirror_starts] python-chess is required to validate mirrored FENs; "
            "install it (pip install chess) and re-run.",
            file=sys.stderr,
        )
        return 2

    starts = read_fens(args.in_path)
    probe_fens = []
    for pf in args.probes:
        probe_fens.extend(read_fens(pf))
    forbidden = build_forbidden_keys(probe_fens)

    out, stats = build_mirrored(starts, forbidden_keys=forbidden)
    write_fens(args.out_path, out)

    if not args.quiet:
        collisions = (
            stats["probe_collisions_original"] + stats["probe_collisions_mirror"]
        )
        print(
            f"[mirror_starts] in={args.in_path} ({stats['input_count']} FENs) "
            f"-> out={args.out_path} ({stats['output_count']} FENs)",
            file=sys.stderr,
        )
        print(
            f"[mirror_starts] originals_kept={stats['originals_kept']} "
            f"mirrors_kept={stats['mirrors_kept']} "
            f"dup_originals={stats['dup_originals']} "
            f"dup_mirrors={stats['dup_mirrors']} "
            f"castling_skips={stats['castling_skips']} "
            f"illegal_mirrors={stats['illegal_mirrors']}",
            file=sys.stderr,
        )
        print(
            f"[mirror_starts] probe_files={len(args.probes)} "
            f"probe_positions={len(probe_fens)} "
            f"probe_collisions_original={stats['probe_collisions_original']} "
            f"probe_collisions_mirror={stats['probe_collisions_mirror']} "
            f"collisions_dropped={collisions}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
