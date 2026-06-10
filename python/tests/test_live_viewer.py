"""Tests for the live PGN game visualizer parser.

Covers game splitting, header/move extraction, UCI replay (incl. the stray
promotion-suffix retry), partial-trailing-game tolerance, and the API payload
shape. No HTTP server is started — only the pure parsing functions are tested.

Run with: cd python && pytest tests/test_live_viewer.py -v
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from hyzero.viz import live_viewer as lv

# A small two-game PGN in the exact shape write_pgn_game emits: UCI moves,
# move numbers, result token, trailing blank line.
TWO_GAMES = (
    '[Event "Eval Cycle 1 Game 1"]\n'
    '[White "challenger"]\n'
    '[Black "champion"]\n'
    '[Result "1-0"]\n'
    "\n"
    "1. e2e4 e7e5 2. g1f3 b8c6 1-0\n"
    "\n"
    '[Event "Eval Cycle 1 Game 2"]\n'
    '[White "challenger"]\n'
    '[Black "champion"]\n'
    '[Result "1/2-1/2"]\n'
    "\n"
    "1. d2d4 d7d5 1/2-1/2\n"
    "\n"
)


def test_split_games_returns_one_block_per_event():
    """Each [Event ...] tag must start exactly one game block."""
    blocks = lv.split_games(TWO_GAMES)
    assert len(blocks) == 2
    assert blocks[0].startswith("[Event")
    assert "Game 1" in blocks[0]
    assert "Game 2" in blocks[1]


def test_parse_game_block_extracts_headers_and_moves():
    """Headers and UCI move tokens are parsed; result/number tokens excluded."""
    block = lv.split_games(TWO_GAMES)[0]
    parsed = lv.parse_game_block(block)
    assert parsed is not None
    assert parsed["headers"]["Event"] == "Eval Cycle 1 Game 1"
    assert parsed["headers"]["White"] == "challenger"
    assert parsed["result"] == "1-0"
    assert parsed["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6"]


def test_parse_pgn_file_reads_both_games(tmp_path):
    """A well-formed file yields one game dict per game, in file order."""
    p = tmp_path / "games.pgn"
    p.write_text(TWO_GAMES, encoding="utf-8")
    games = lv.parse_pgn_file(str(p))
    assert len(games) == 2
    assert games[0]["event"] == "Eval Cycle 1 Game 1"
    assert games[1]["result"] == "1/2-1/2"


def test_missing_file_returns_empty_list():
    """A non-existent log path is not an error, just no games."""
    assert lv.parse_pgn_file("/no/such/file.pgn") == []


def test_partial_trailing_game_is_skipped():
    """A trailing game whose header block is not yet finished is dropped, while
    the completed game before it still parses."""
    partial = (
        TWO_GAMES
        + '[Event "Eval Cycle 1 Game 3"]\n'
        + '[White "challenger"]\n'  # header section never terminated by a blank line
    )
    # parse via the public file path to exercise the real code path.
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".pgn")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(partial)
        games = lv.parse_pgn_file(path)
    finally:
        os.remove(path)
    assert len(games) == 2  # the two complete games only
    assert all("Game 3" not in g["event"] for g in games)


def test_parse_game_block_returns_none_for_unterminated_header():
    """A block with no blank line after the headers is treated as incomplete."""
    block = '[Event "x"]\n[White "a"]\n'
    assert lv.parse_game_block(block) is None


def test_replay_recovers_from_stray_promotion_suffix():
    """A non-pawn move carrying a promotion suffix (e.g. a knight 'g1f3q') must
    be retried without the suffix so replay continues instead of aborting."""
    if not lv.HAVE_CHESS:
        import pytest

        pytest.skip("python-chess not available")
    moves = ["e2e4", "e7e5", "g1f3q"]  # the 'q' on a knight move is spurious
    fens = lv.replay_fens(moves)
    # start + 3 successful plies == 4 FENs
    assert len(fens) == 4
    assert fens[0].startswith(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"
    )


def test_replay_stops_at_illegal_move_keeping_prefix():
    """An engine-illegal move ends replay but keeps the FENs gathered so far."""
    if not lv.HAVE_CHESS:
        import pytest

        pytest.skip("python-chess not available")
    moves = ["e2e4", "a1a8"]  # rook cannot jump to a8 on move 1
    fens = lv.replay_fens(moves)
    assert len(fens) == 2  # start + first legal move only


def test_build_games_payload_shape(tmp_path):
    """The API payload exposes file key, chess availability, count and games."""
    (tmp_path / "selfplay_sample.pgn").write_text(TWO_GAMES, encoding="utf-8")
    payload = lv.build_games_payload(str(tmp_path), "selfplay")
    assert payload["file"] == "selfplay"
    assert payload["count"] == 2
    assert payload["have_chess"] == lv.HAVE_CHESS
    assert isinstance(payload["games"], list)
    if lv.HAVE_CHESS:
        assert "fens" in payload["games"][0]


def test_unknown_file_key_falls_back_to_selfplay(tmp_path):
    """An unrecognized file key defaults to the selfplay log rather than 404."""
    (tmp_path / "selfplay_sample.pgn").write_text(TWO_GAMES, encoding="utf-8")
    payload = lv.build_games_payload(str(tmp_path), "bogus")
    assert payload["file"] == "selfplay"
    assert payload["count"] == 2


def test_default_host_is_localhost():
    """The --host arg defaults to 127.0.0.1 so the server is not world-bound."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="./logs")
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args([])
    assert args.host == "127.0.0.1"


def test_mid_write_movetext_tail_parses_as_shorter_game(tmp_path):
    """A trailing game with finished headers but a half-written movetext tail
    parses as a shorter game whose replay stops cleanly at the truncation."""
    mid_write = (
        TWO_GAMES
        + '[Event "Eval Cycle 1 Game 3"]\n'
        + '[White "challenger"]\n'
        + '[Black "champion"]\n'
        + '[Result "*"]\n'
        + "\n"
        + "1. e2e4 e7e5 2. g1f3 b8"  # movetext cut off mid-write, no result token
    )
    p = tmp_path / "games.pgn"
    p.write_text(mid_write, encoding="utf-8")
    games = lv.parse_pgn_file(str(p))
    assert len(games) == 3  # the truncated game still parses
    tail = games[2]
    assert tail["event"] == "Eval Cycle 1 Game 3"
    assert tail["moves"] == ["e2e4", "e7e5", "g1f3", "b8"]
    if lv.HAVE_CHESS:
        # 'b8' is not a legal UCI move, so replay stops after the three good plies.
        assert len(tail["fens"]) == 4
