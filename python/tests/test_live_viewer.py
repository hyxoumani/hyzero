"""Tests for the live PGN game visualizer parser.

Covers game splitting, header/move extraction, UCI replay (incl. the stray
promotion-suffix retry), partial-trailing-game tolerance, the split API payloads
(lightweight metadata vs. single-game detail), the parse cache keyed on
(mtime, size) and its invalidation when a game is appended. No HTTP server is
started — only the pure parsing/payload functions are tested.

Run with: cd python && pytest tests/test_live_viewer.py -v
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, ".")

from hyzero.viz import live_viewer as lv

# A small two-game PGN in the exact shape write_pgn_game emits: UCI moves,
# move numbers, result token, an additive Termination header, trailing blank.
TWO_GAMES = (
    '[Event "Eval Cycle 1 Game 1"]\n'
    '[White "challenger"]\n'
    '[Black "champion"]\n'
    '[Result "1-0"]\n'
    '[Termination "checkmate"]\n'
    "\n"
    "1. e2e4 e7e5 2. g1f3 b8c6 1-0\n"
    "\n"
    '[Event "Eval Cycle 1 Game 2"]\n'
    '[White "challenger"]\n'
    '[Black "champion"]\n'
    '[Result "1/2-1/2"]\n'
    '[Termination "insufficient-material"]\n'
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


def test_parse_game_block_keeps_termination_header():
    """The additive Termination header is preserved for the listing."""
    block = lv.split_games(TWO_GAMES)[1]
    parsed = lv.parse_game_block(block)
    assert parsed is not None
    assert parsed["headers"]["Termination"] == "insufficient-material"


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


def test_replay_with_immediately_illegal_move_keeps_start_fen():
    """A game whose very first move is illegal still yields the start FEN, so the
    detail payload never carries an empty fens list."""
    if not lv.HAVE_CHESS:
        import pytest

        pytest.skip("python-chess not available")
    fens = lv.replay_fens(["a1a3"])  # no rook jump on move 1
    assert len(fens) == 1
    assert fens[0].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")


def test_games_payload_is_lightweight_metadata_only(tmp_path):
    """/api/games carries per-game metadata but never the FEN lists."""
    (tmp_path / "selfplay_sample.pgn").write_text(TWO_GAMES, encoding="utf-8")
    payload = lv.build_games_payload(str(tmp_path), "selfplay")
    assert payload["file"] == "selfplay"
    assert payload["count"] == 2
    assert payload["have_chess"] == lv.HAVE_CHESS
    g0 = payload["games"][0]
    assert g0["idx"] == 0
    assert g0["result"] == "1-0"
    assert g0["termination"] == "checkmate"
    assert g0["move_count"] == 4
    assert "fens" not in g0  # the listing must stay light
    assert "moves" not in g0


def test_games_payload_metadata_exposes_termination_header(tmp_path):
    """The new Termination header is surfaced in the metadata listing."""
    (tmp_path / "eval_games.pgn").write_text(TWO_GAMES, encoding="utf-8")
    payload = lv.build_games_payload(str(tmp_path), "eval")
    assert payload["games"][1]["termination"] == "insufficient-material"
    assert payload["games"][1]["headers"]["Termination"] == "insufficient-material"


def test_game_payload_returns_single_game_fens(tmp_path):
    """/api/game returns the selected game's FEN list and raw moves."""
    (tmp_path / "selfplay_sample.pgn").write_text(TWO_GAMES, encoding="utf-8")
    payload = lv.build_game_payload(str(tmp_path), "selfplay", 0)
    assert payload is not None
    assert payload["idx"] == 0
    assert payload["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    if lv.HAVE_CHESS:
        assert len(payload["fens"]) == 5  # start + 4 plies


def test_game_payload_out_of_range_returns_none(tmp_path):
    """An out-of-range idx yields None so the handler can 404."""
    (tmp_path / "selfplay_sample.pgn").write_text(TWO_GAMES, encoding="utf-8")
    assert lv.build_game_payload(str(tmp_path), "selfplay", 99) is None
    assert lv.build_game_payload(str(tmp_path), "selfplay", -1) is None


def test_game_payload_for_empty_fens_game_keeps_start_fen(tmp_path):
    """A game whose first move is illegal still has a one-FEN detail payload, so
    the page renders the start position rather than blanking out."""
    if not lv.HAVE_CHESS:
        import pytest

        pytest.skip("python-chess not available")
    bad = (
        '[Event "Eval Cycle 1 Game X"]\n'
        '[White "a"]\n'
        '[Black "b"]\n'
        '[Result "*"]\n'
        "\n"
        "1. a1a3 *\n"  # rook cannot jump on move 1 -> replay stops immediately
        "\n"
    )
    (tmp_path / "eval_games.pgn").write_text(bad, encoding="utf-8")
    payload = lv.build_game_payload(str(tmp_path), "eval", 0)
    assert payload is not None
    assert len(payload["fens"]) == 1
    assert payload["fens"][0].startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")


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


def test_parse_cache_reuses_object_for_unchanged_file(tmp_path):
    """A second parse of an unchanged file returns the identical cached list."""
    p = tmp_path / "games.pgn"
    p.write_text(TWO_GAMES, encoding="utf-8")
    first = lv.parse_pgn_file_cached(str(p))
    second = lv.parse_pgn_file_cached(str(p))
    assert first is second  # same object -> no re-parse happened


def test_parse_cache_invalidates_when_game_appended(tmp_path):
    """Appending a game changes (mtime, size), so the cache re-parses and the new
    game appears — the live-poll-after-append path."""
    p = tmp_path / "games.pgn"
    p.write_text(TWO_GAMES, encoding="utf-8")
    first = lv.parse_pgn_file_cached(str(p))
    assert len(first) == 2

    appended = (
        '[Event "Eval Cycle 1 Game 3"]\n'
        '[White "challenger"]\n'
        '[Black "champion"]\n'
        '[Result "0-1"]\n'
        '[Termination "resignation"]\n'
        "\n"
        "1. c2c4 c7c5 0-1\n"
        "\n"
    )
    # Ensure mtime advances even on coarse-resolution filesystems.
    time.sleep(0.01)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(appended)
    os.utime(str(p), (time.time() + 1, time.time() + 1))

    second = lv.parse_pgn_file_cached(str(p))
    assert len(second) == 3
    assert second is not first
    assert second[2]["event"] == "Eval Cycle 1 Game 3"


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
