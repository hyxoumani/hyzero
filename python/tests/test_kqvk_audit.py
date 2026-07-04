"""Tests for scripts/kqvk_audit.py game selection and outcome counting.

Builds a small inline PGN fixture — a KQvK checkmate, a KQvK stalemate, a
non-KQvK game (must be excluded), and a malformed KQvK game with an illegal
move (must be skipped) — then asserts the audit selects and counts correctly.

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
        # Non-KQvK standard-start game — must be excluded from selection.
        _make_pgn(chess.STARTING_FEN, ["e2e4", "e7e5"]),
        # Malformed KQvK game: illegal SAN "Qa1" — must be skipped, not counted.
        '[FEN "5k2/7Q/5K2/8/8/8/8/8 w - - 0 1"]\n[SetUp "1"]\n\n1. Qa1 1-0\n',
    ]
    path.write_text("\n\n".join(games) + "\n", encoding="utf-8")


def test_selects_kqvk_and_counts_outcomes(tmp_path):
    """Only well-formed KQvK games are selected; outcomes classified correctly."""
    pgn = tmp_path / "sample.pgn"
    _write_fixture(pgn)

    result = kqvk_audit.audit_pgn(str(pgn))

    assert result["kqvk_games"] == 2
    assert result["mates"] == 1
    assert result["stalemate"] == 1
    assert result["insufficient_material"] == 0
    assert result["repetition"] == 0
    assert result["other"] == 0
    assert abs(result["mate_rate"] - 0.5) < 1e-9


def test_parse_material_kq():
    """`parse_material` maps "KQ" to a single-queen strong-side spec."""
    extra = kqvk_audit.parse_material("KQ")
    assert dict(extra) == {chess.QUEEN: 1}


def test_empty_pgn_yields_zero_games(tmp_path):
    """An empty PGN produces zero games and a zero mate_rate (no divide-by-zero)."""
    pgn = tmp_path / "empty.pgn"
    pgn.write_text("", encoding="utf-8")
    result = kqvk_audit.audit_pgn(str(pgn))
    assert result["kqvk_games"] == 0
    assert result["mate_rate"] == 0.0
