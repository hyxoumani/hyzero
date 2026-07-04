"""Tests for scripts/kqvk_audit.py game selection and outcome counting.

Builds a small inline PGN fixture — a KQvK checkmate, a KQvK stalemate, a
KRvK checkmate, a non-basic-mate game (must be excluded), and a malformed KQvK
game with an illegal move (must be skipped) — then asserts the audit selects and
counts correctly across both tracked classes and the combined roll-up.

Run with: cd python && pytest tests/test_kqvk_audit.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import chess
import chess.pgn

# Import the audit module directly from the scripts directory.
_AUDIT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "kqvk_audit.py"
_spec = importlib.util.spec_from_file_location("kqvk_audit", _AUDIT_PATH)
kqvk_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kqvk_audit)


def _make_pgn(fen: str, uci_moves: list[str]) -> str:
    """Serialize a game starting from ``fen`` with the given UCI moves to PGN."""
    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.setup(board)
    node = game
    for uci in uci_moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result(claim_draw=True)
    return str(game)


def _write_fixture(path: Path) -> None:
    games = [
        # KQvK -> checkmate (Qh7-h8#).
        _make_pgn("5k2/7Q/5K2/8/8/8/8/8 w - - 0 1", ["h7h8"]),
        # KQvK -> stalemate (Qc2-g6).
        _make_pgn("7k/5K2/8/8/8/8/2Q5/8 w - - 0 1", ["c2g6"]),
        # KRvK -> checkmate (Ra1-a8#).
        _make_pgn("7k/8/6K1/8/8/8/8/R7 w - - 0 1", ["a1a8"]),
        # Non-basic-mate standard-start game — must be excluded from selection.
        _make_pgn(chess.STARTING_FEN, ["e2e4", "e7e5"]),
        # Malformed KQvK game: illegal SAN "Qa1" — must be skipped, not counted.
        '[FEN "5k2/7Q/5K2/8/8/8/8/8 w - - 0 1"]\n[SetUp "1"]\n\n1. Qa1 1-0\n',
    ]
    path.write_text("\n\n".join(games) + "\n", encoding="utf-8")


def test_selects_basic_mates_and_counts_outcomes(tmp_path):
    """Well-formed KQvK/KRvK games are selected; outcomes classified per class."""
    pgn = tmp_path / "sample.pgn"
    _write_fixture(pgn)

    result = kqvk_audit.audit_pgn(str(pgn))

    kq = result["classes"]["KQvK"]
    assert kq["games"] == 2
    assert kq["mates"] == 1
    assert kq["stalemate"] == 1
    assert abs(kq["mate_rate"] - 0.5) < 1e-9

    kr = result["classes"]["KRvK"]
    assert kr["games"] == 1
    assert kr["mates"] == 1
    assert abs(kr["mate_rate"] - 1.0) < 1e-9

    combined = result["combined"]
    assert combined["games"] == 3
    assert combined["mates"] == 2
    assert combined["stalemate"] == 1
    assert combined["insufficient_material"] == 0
    assert combined["repetition"] == 0
    assert combined["other"] == 0
    assert abs(combined["mate_rate"] - (2 / 3)) < 1e-9


def test_parse_material_kq():
    """`parse_material` maps "KQ" to a single-queen strong-side spec."""
    extra = kqvk_audit.parse_material("KQ")
    assert dict(extra) == {chess.QUEEN: 1}


def test_empty_pgn_yields_zero_games(tmp_path):
    """An empty PGN produces zero games and a zero mate_rate (no divide-by-zero)."""
    pgn = tmp_path / "empty.pgn"
    pgn.write_text("", encoding="utf-8")
    result = kqvk_audit.audit_pgn(str(pgn))
    assert result["combined"]["games"] == 0
    assert result["combined"]["mate_rate"] == 0.0
    assert result["valid"] is False
