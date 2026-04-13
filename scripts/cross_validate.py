#!/usr/bin/env python3
"""Cross-validate hyzero engine against python-chess."""

import chess
import subprocess
import sys
import random
import argparse
from pathlib import Path

ENGINE_BIN = Path(__file__).parent.parent / "target" / "release" / "perft"

# ANSI color codes (disabled if not a terminal)
_USE_COLOR = sys.stdout.isatty()
_GREEN = "\033[32m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_RESET = "\033[0m" if _USE_COLOR else ""


def _pass(msg: str) -> str:
    return f"{_GREEN}[PASS]{_RESET} {msg}"


def _fail(msg: str) -> str:
    return f"{_RED}[FAIL]{_RESET} {msg}"


def _header(msg: str) -> str:
    return f"\n{_BOLD}=== {msg} ==={_RESET}"


def check_engine_binary() -> bool:
    if not ENGINE_BIN.exists():
        print(
            f"{_RED}Error:{_RESET} Engine binary not found at {ENGINE_BIN}\n"
            "Run: cargo build --release --bin perft",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Perft validation
# ---------------------------------------------------------------------------

PERFT_SUITE = [
    # (name, fen, {depth: expected_count})
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609},
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        {1: 48, 2: 2039, 3: 97862, 4: 4085603},
    ),
    (
        "position3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624, 6: 11030083},
    ),
    (
        "position4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        {1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292},
    ),
    (
        "position5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        {1: 44, 2: 1486, 3: 62379, 4: 2103487},
    ),
    (
        "position6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 10",
        {1: 42, 2: 1892, 3: 76031, 4: 3288373},
    ),
]


def run_perft_validation() -> tuple[int, int]:
    """Run standard perft suite. Returns (passed, total)."""
    print(_header("Perft Validation"))
    passed = 0
    total = 0

    for name, fen, depths in PERFT_SUITE:
        for depth, expected in sorted(depths.items()):
            total += 1
            label = f"{name} d{depth}"
            try:
                result = subprocess.run(
                    [str(ENGINE_BIN), fen, str(depth)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    print(
                        _fail(
                            f"{label}: engine error — {result.stderr.strip()}"
                        )
                    )
                    continue

                actual = int(result.stdout.strip())
                if actual == expected:
                    passed += 1
                    print(_pass(f"{label}: {actual}"))
                else:
                    print(
                        _fail(
                            f"{label}: expected {expected}, got {actual}"
                        )
                    )
            except subprocess.TimeoutExpired:
                print(_fail(f"{label}: TIMEOUT (>30s)"))
            except ValueError:
                print(
                    _fail(
                        f"{label}: could not parse engine output: "
                        f"{result.stdout.strip()!r}"
                    )
                )

    return passed, total


# ---------------------------------------------------------------------------
# Termination validation
# ---------------------------------------------------------------------------

TERMINATION_FENS = [
    # (name, fen)
    # checkmate (3)
    ("fool's mate (checkmate)", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
    ("scholar's mate (checkmate)", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"),
    ("Qg7 mate (checkmate)", "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"),
    # check but not checkmate (1)
    ("queen check, can escape", "4k3/4Q3/8/8/8/8/8/4K3 b - - 0 1"),
    # stalemate (2)
    ("stalemate basic", "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"),
    ("stalemate pawn block", "5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"),
    # insufficient material (3)
    ("K vs K", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("K+B vs K", "4k3/8/8/8/8/8/8/4KB2 w - - 0 1"),
    ("K+N vs K", "4k3/8/8/8/8/8/8/4KN2 w - - 0 1"),
    # not insufficient (2)
    ("K+R vs K (not insuf)", "4k3/8/8/8/8/8/8/4KR2 w - - 0 1"),
    ("K+P vs K (not insuf)", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
    # ongoing (2)
    ("startpos (ongoing)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("midgame (ongoing)", "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
]


def _ref_status(board: chess.Board) -> str:
    """Return the python-chess termination status string for a board."""
    if board.is_checkmate():
        return "checkmate"
    elif board.is_stalemate():
        return "stalemate"
    elif board.has_insufficient_material(chess.WHITE) and board.has_insufficient_material(chess.BLACK):
        return "insufficient_material"
    elif board.is_check():
        return "check"
    else:
        return "ongoing"


def run_termination_tests() -> tuple[int, int]:
    """Compare game termination status against python-chess. Returns (passed, total)."""
    print(_header("Termination Validation"))
    passed = 0
    total = len(TERMINATION_FENS)

    for name, fen in TERMINATION_FENS:
        board = chess.Board(fen)
        ref_status = _ref_status(board)

        try:
            result = subprocess.run(
                [str(ENGINE_BIN), "--status", fen],
                capture_output=True,
                text=True,
                timeout=10,
            )
            engine_status = result.stdout.strip()
        except subprocess.TimeoutExpired:
            engine_status = "TIMEOUT"
        except Exception as e:
            engine_status = f"ERROR: {e}"

        if ref_status == engine_status:
            passed += 1
            print(_pass(f"{name}: {ref_status}"))
        else:
            print(
                _fail(
                    f"{name}: expected {ref_status}, got {engine_status}"
                )
            )

    return passed, total


# ---------------------------------------------------------------------------
# Legal move comparison
# ---------------------------------------------------------------------------

EDGE_CASE_FENS = [
    # Standard start position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Kiwipete (castling, EP, captures)
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # Position 4 (promotions)
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    # En passant discovered check — EP is legal (white gives check)
    "8/8/8/k2pP2R/8/8/8/4K3 w - d6 0 1",
    # En passant discovered check — EP is ILLEGAL (black exposes own king)
    "8/8/8/8/R2pPk2/8/8/4K3 b - e3 0 1",
    # Castling through check (d1 attacked, queenside illegal)
    "r3k2r/8/8/8/3q4/8/8/R3K2R w KQkq - 0 1",
    # Double check position
    "4k3/8/8/8/8/8/r5R1/4K3 w - - 0 1",
    # Promotion with capture
    "r7/P7/8/8/8/8/8/4K1k1 w - - 0 1",
    # Stalemate (black king a8, white queen c7, white king b6 — black has no legal moves)
    "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1",
    # Checkmate (black king a8, white queens a2+b2, white king e1 — black is in check with no escape)
    "k7/8/8/8/8/8/QQ6/4K3 b - - 0 1",
]


def _engine_moves(fen: str) -> tuple[set[str] | None, str]:
    """Return (set_of_uci_moves, error_message). On success error is empty."""
    try:
        result = subprocess.run(
            [str(ENGINE_BIN), "--moves", fen],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None, f"engine error: {result.stderr.strip()}"
        raw = result.stdout.strip()
        moves = set(raw.split("\n")) if raw else set()
        return moves, ""
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"


def run_move_comparison() -> tuple[int, int]:
    """Compare legal move lists for edge-case FENs. Returns (passed, total)."""
    print(_header("Legal Move Comparison"))
    passed = 0
    total = len(EDGE_CASE_FENS)

    for fen in EDGE_CASE_FENS:
        board = chess.Board(fen)
        ref_moves = set(m.uci() for m in board.legal_moves)
        # Build a short label from the FEN (first field pair)
        parts = fen.split()
        label = f"{parts[0]} {parts[1]}"

        eng_moves, err = _engine_moves(fen)
        if eng_moves is None:
            print(_fail(f"{label}: {err}"))
            continue

        if ref_moves == eng_moves:
            passed += 1
            print(_pass(f"{label} ({len(ref_moves)} moves)"))
        else:
            missing = ref_moves - eng_moves
            extra = eng_moves - ref_moves
            details = []
            if missing:
                details.append(f"missing {{{', '.join(sorted(missing))}}}")
            if extra:
                details.append(f"extra {{{', '.join(sorted(extra))}}}")
            print(_fail(f"{label}: {'; '.join(details)}"))

    return passed, total


# ---------------------------------------------------------------------------
# Fuzz testing
# ---------------------------------------------------------------------------


def fuzz_test(num_games: int) -> tuple[int, int]:
    """
    Play num_games random games, comparing legal moves at every position.
    Returns (games_passed, positions_checked).
    """
    print(_header(f"Fuzz Testing ({num_games} games)"))
    games_passed = 0
    positions_checked = 0

    for game_num in range(num_games):
        board = chess.Board()
        move_count = 0
        game_ok = True

        while not board.is_game_over() and move_count < 300:
            fen = board.fen()
            ref_moves = sorted(m.uci() for m in board.legal_moves)

            try:
                result = subprocess.run(
                    [str(ENGINE_BIN), "--moves", fen],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                raw = result.stdout.strip()
                engine_moves = sorted(raw.split("\n")) if raw else []
            except subprocess.TimeoutExpired:
                print(
                    _fail(
                        f"game {game_num + 1}, move {move_count + 1}: TIMEOUT"
                    )
                )
                print(f"  FEN: {fen}")
                return games_passed, positions_checked

            if ref_moves != engine_moves:
                game_ok = False
                print(
                    _fail(
                        f"game {game_num + 1}, move {move_count + 1}"
                    )
                )
                print(f"  FEN: {fen}")
                ref_set, eng_set = set(ref_moves), set(engine_moves)
                missing = ref_set - eng_set
                extra = eng_set - ref_set
                if missing:
                    print(f"  Missing from engine: {sorted(missing)}")
                if extra:
                    print(f"  Extra in engine: {sorted(extra)}")
                return games_passed, positions_checked

            # Compare game termination status
            ref_status = _ref_status(board)
            try:
                status_result = subprocess.run(
                    [str(ENGINE_BIN), "--status", fen],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                engine_status = status_result.stdout.strip()
            except subprocess.TimeoutExpired:
                engine_status = "TIMEOUT"

            if ref_status != engine_status:
                game_ok = False
                print(
                    _fail(
                        f"STATUS MISMATCH at game {game_num + 1}, move {move_count + 1}"
                    )
                )
                print(f"  FEN: {fen}")
                print(f"  Expected: {ref_status}, Got: {engine_status}")
                return games_passed, positions_checked

            positions_checked += 1
            move = random.choice(list(board.legal_moves))
            board.push(move)
            move_count += 1

        if game_ok:
            games_passed += 1

    print(
        _pass(
            f"{games_passed}/{num_games} games, "
            f"{positions_checked} positions checked"
        )
    )
    return games_passed, positions_checked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-validate hyzero engine against python-chess."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--perft", action="store_true", help="Run standard perft suite"
    )
    group.add_argument(
        "--moves",
        action="store_true",
        help="Compare legal moves for edge-case positions",
    )
    group.add_argument(
        "--fuzz",
        type=int,
        metavar="N",
        help="Fuzz N random games comparing legal moves",
    )
    group.add_argument(
        "--termination",
        action="store_true",
        help="Validate game termination detection against python-chess",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run everything (perft + moves + fuzz 5 + termination)",
    )
    args = parser.parse_args()

    # Default: run --all when no args given
    run_all = args.all or not (args.perft or args.moves or args.fuzz or args.termination)

    if not check_engine_binary():
        return 1

    overall_pass = True
    summary_lines: list[str] = []

    if args.perft or run_all:
        p, t = run_perft_validation()
        ok = p == t
        overall_pass = overall_pass and ok
        summary_lines.append(f"Perft: {p}/{t} passed")

    if args.moves or run_all:
        p, t = run_move_comparison()
        ok = p == t
        overall_pass = overall_pass and ok
        summary_lines.append(f"Moves: {p}/{t} passed")

    if args.fuzz is not None:
        n = args.fuzz
        gp, _ = fuzz_test(n)
        ok = gp == n
        overall_pass = overall_pass and ok
        summary_lines.append(f"Fuzz: {gp}/{n} games passed")
    elif run_all:
        # Each game invokes the engine binary per-position (~57s overhead per game).
        n = 5
        gp, _ = fuzz_test(n)
        ok = gp == n
        overall_pass = overall_pass and ok
        summary_lines.append(f"Fuzz: {gp}/{n} games passed")

    if args.termination or run_all:
        p, t = run_termination_tests()
        ok = p == t
        overall_pass = overall_pass and ok
        summary_lines.append(f"Termination: {p}/{t} passed")

    print(_header("Summary"))
    for line in summary_lines:
        print(f"  {line}")
    if overall_pass:
        print(f"\n{_GREEN}Overall: PASS{_RESET}")
    else:
        # Collect lines where passed count does not equal total (e.g. "Perft: 28/30 passed")
        failed_parts = []
        for line in summary_lines:
            # Format is "Label: X/Y passed" or "Fuzz: X/Y games passed"
            try:
                counts = line.split(": ", 1)[1].split()[0]  # "X/Y"
                x, y = counts.split("/")
                if x != y:
                    failed_parts.append(line.split(":")[0])
            except (IndexError, ValueError):
                pass
        desc = ", ".join(failed_parts) if failed_parts else "unknown"
        print(f"\n{_RED}Overall: FAIL ({desc}){_RESET}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
