"""Minimal live game visualizer for hyzero PGN logs.

Serves a single static page plus a tiny JSON API so you can watch self-play and
eval games near-live in a browser while a baseline run appends games to the log
files. A partially-written trailing game is tolerated: a trailing block whose
header section is not yet terminated by a blank line is skipped, while a trailing
block with finished headers but movetext still being written parses as a
(shorter) game whose replay simply stops at the truncated tail rather than being
fatal.

Moves in hyzero PGNs are long-algebraic / UCI tokens (e.g. ``a2a4``, ``c2c1n``),
not SAN. When ``python-chess`` is importable we replay each game and return a
per-ply FEN list so the page can render the board with no client-side chess
logic. If ``python-chess`` is missing we fall back to returning the raw UCI move
list (the page then only shows move text, no board).

The API is split so polling stays cheap even when a 4-hour run keeps appending
games. ``/api/games`` returns *lightweight metadata only* (per game: index,
headers, move count, result) and never the FEN lists. ``/api/game`` returns the
FEN list (or raw moves) for a single selected game. Parses are cached keyed on
(path, mtime, size), so a poll against an unchanged file does no re-parsing.

Launch:

    cd python && python -m hyzero.viz.live_viewer --logs-dir ../logs --port 8642

then open http://localhost:8642/ in a browser. The page polls every ~3s, lists
games newest-first, supports ply stepping, and can auto-follow the latest game.

The logs directory is only ever read, never written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:  # python-chess is optional; without it we serve raw move text only.
    import chess  # type: ignore

    HAVE_CHESS = True
except Exception:  # pragma: no cover - exercised only in chess-less envs.
    chess = None  # type: ignore
    HAVE_CHESS = False


# Maps a CLI/API short name to its log filename.
FILE_MAP = {
    "selfplay": "selfplay_sample.pgn",
    "eval": "eval_games.pgn",
}

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_MOVENUM_RE = re.compile(r"^\d+\.(\.\.)?$")
_RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

# Header keys surfaced in the lightweight metadata listing.
_META_HEADERS = ("Event", "White", "Black", "Result", "Termination")


def split_games(text: str) -> List[str]:
    """Split raw PGN text into per-game blocks.

    Each block starts at an ``[Event ...]`` tag. Any preamble before the first
    Event tag (there is none in hyzero logs, but be safe) is dropped.
    """
    parts = text.split("[Event")
    return ["[Event" + p for p in parts[1:]]


def parse_game_block(block: str) -> Optional[Dict[str, Any]]:
    """Parse one PGN block into headers + UCI move tokens.

    Returns ``None`` for a block that has no completed header section yet (a
    partially-written trailing game), so the caller can skip it gracefully.
    """
    lines = block.splitlines()
    headers: Dict[str, str] = {}
    idx = 0
    for idx, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2)
            continue
        if line.strip() == "":
            # Blank line terminates the header section.
            break
    else:
        # No blank line seen at all -> header section not finished yet.
        return None

    if "Event" not in headers:
        return None

    movetext = " ".join(lines[idx + 1 :])
    moves: List[str] = []
    result_in_text: Optional[str] = None
    for tok in movetext.split():
        if _MOVENUM_RE.match(tok):
            continue
        if tok in _RESULT_TOKENS:
            result_in_text = tok
            break
        moves.append(tok)

    return {
        "headers": headers,
        "result": headers.get("Result", result_in_text or "*"),
        "moves": moves,
    }


def _try_push(board: "chess.Board", token: str) -> bool:
    """Push a UCI token, retrying without a stray promotion suffix.

    hyzero occasionally emits a promotion suffix on a non-pawn move (e.g.
    ``f7d8q`` for a knight); strip the trailing piece letter and retry so the
    rest of the game still replays.
    """
    candidates = [token]
    if token and token[-1] in "qrbn":
        candidates.append(token[:-1])
    for cand in candidates:
        try:
            board.push_uci(cand)
            return True
        except Exception:
            continue
    return False


def replay_fens(moves: List[str]) -> List[str]:
    """Replay UCI moves into a per-ply FEN list (start position first).

    Replay stops at the first illegal move and returns the FENs gathered so far,
    so an engine-illegal tail move never breaks the whole game. The list always
    has at least the start position, so callers never see an empty ``fens``.
    """
    board = chess.Board()
    fens = [board.fen()]
    for tok in moves:
        if not _try_push(board, tok):
            break
        fens.append(board.fen())
    return fens


def parse_pgn_file(path: str) -> List[Dict[str, Any]]:
    """Parse a PGN file into a list of game dicts, newest game last.

    Each game dict has ``headers``, ``event``, ``white``, ``black``, ``result``,
    ``moves`` and, when python-chess is available, ``fens`` (replayed per-ply,
    start position first). A missing file yields an empty list. A trailing game
    whose header section is not yet terminated is skipped; one with finished
    headers but a half-written movetext tail parses as a shorter game (replay
    stops at the truncation).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except FileNotFoundError:
        return []

    games: List[Dict[str, Any]] = []
    for block in split_games(text):
        try:
            parsed = parse_game_block(block)
        except Exception:
            # Defensive: never let one malformed block kill the response.
            continue
        if parsed is None:
            continue
        game: Dict[str, Any] = {
            "headers": parsed["headers"],
            "event": parsed["headers"].get("Event", ""),
            "white": parsed["headers"].get("White", ""),
            "black": parsed["headers"].get("Black", ""),
            "result": parsed["result"],
            "moves": parsed["moves"],
        }
        if HAVE_CHESS:
            try:
                game["fens"] = replay_fens(parsed["moves"])
            except Exception:
                game["fens"] = [chess.Board().fen()]
        games.append(game)
    return games


# Cache of parsed files keyed on path -> (mtime, size, games). A poll against an
# unchanged file (same mtime+size) reuses the cached parse instead of re-reading
# and re-replaying every game, which keeps cost flat as the log grows.
_PARSE_CACHE: Dict[str, Tuple[float, int, List[Dict[str, Any]]]] = {}


def _file_stat(path: str) -> Optional[Tuple[float, int]]:
    """Return (mtime, size) for ``path`` or ``None`` if it does not exist."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_mtime, st.st_size)


def parse_pgn_file_cached(path: str) -> List[Dict[str, Any]]:
    """Parse ``path`` via :func:`parse_pgn_file`, caching on (mtime, size).

    Returns the same list shape as :func:`parse_pgn_file`. A missing file yields
    an empty list and is not cached, so it re-checks once the file appears.
    """
    stat = _file_stat(path)
    if stat is None:
        _PARSE_CACHE.pop(path, None)
        return []
    mtime, size = stat
    cached = _PARSE_CACHE.get(path)
    if cached is not None and cached[0] == mtime and cached[1] == size:
        return cached[2]
    games = parse_pgn_file(path)
    _PARSE_CACHE[path] = (mtime, size, games)
    return games


def _resolve_key(key: str) -> str:
    """Map an API file key to a known one, defaulting to ``selfplay``."""
    return key if key in FILE_MAP else "selfplay"


def build_games_payload(logs_dir: str, key: str) -> Dict[str, Any]:
    """Build the *lightweight* metadata payload for ``/api/games``.

    The listing carries only per-game metadata (index, headers, move count,
    result) and never the FEN lists, so polling stays cheap as the log grows.
    The per-game FEN list is fetched separately via :func:`build_game_payload`.
    """
    key = _resolve_key(key)
    path = os.path.join(logs_dir, FILE_MAP[key])
    games = parse_pgn_file_cached(path)
    meta = []
    for i, g in enumerate(games):
        headers = g.get("headers", {})
        meta.append(
            {
                "idx": i,
                "headers": {h: headers[h] for h in _META_HEADERS if h in headers},
                "event": g.get("event", ""),
                "white": g.get("white", ""),
                "black": g.get("black", ""),
                "result": g.get("result", "*"),
                "termination": headers.get("Termination", ""),
                "move_count": len(g.get("moves", [])),
            }
        )
    return {
        "file": key,
        "have_chess": HAVE_CHESS,
        "count": len(games),
        "games": meta,
    }


def build_game_payload(logs_dir: str, key: str, idx: int) -> Optional[Dict[str, Any]]:
    """Build the single-game payload for ``/api/game``.

    Returns the selected game's FEN list (when python-chess is available) plus
    the raw moves, or ``None`` if ``idx`` is out of range for the current file.
    """
    key = _resolve_key(key)
    path = os.path.join(logs_dir, FILE_MAP[key])
    games = parse_pgn_file_cached(path)
    if idx < 0 or idx >= len(games):
        return None
    g = games[idx]
    payload: Dict[str, Any] = {
        "file": key,
        "idx": idx,
        "have_chess": HAVE_CHESS,
        "event": g.get("event", ""),
        "white": g.get("white", ""),
        "black": g.get("black", ""),
        "result": g.get("result", "*"),
        "moves": g.get("moves", []),
    }
    if HAVE_CHESS:
        payload["fens"] = g.get("fens", [])
    return payload


def make_handler(logs_dir: str):
    """Build a request handler bound to ``logs_dir``."""

    class Handler(BaseHTTPRequestHandler):
        # Silence the default per-request stderr logging.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            qs = parse_qs(parsed.query)
            if parsed.path == "/api/games":
                key = (qs.get("file", ["selfplay"]) or ["selfplay"])[0]
                self._send_json(200, build_games_payload(logs_dir, key))
                return
            if parsed.path == "/api/game":
                key = (qs.get("file", ["selfplay"]) or ["selfplay"])[0]
                raw_idx = (qs.get("idx", ["0"]) or ["0"])[0]
                try:
                    idx = int(raw_idx)
                except ValueError:
                    self._send_json(400, {"error": "idx must be an integer"})
                    return
                payload = build_game_payload(logs_dir, key, idx)
                if payload is None:
                    self._send_json(404, {"error": "game index out of range"})
                    return
                self._send_json(200, payload)
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def serve(logs_dir: str, port: int, host: str = "127.0.0.1") -> None:
    """Run the HTTP server until interrupted."""
    handler = make_handler(logs_dir)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(
        f"[live_viewer] serving logs from {os.path.abspath(logs_dir)} "
        f"on http://{host}:{port}/ (python-chess: {'yes' if HAVE_CHESS else 'no'})"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="hyzero live PGN game visualizer")
    ap.add_argument("--logs-dir", default="./logs", help="directory holding the PGN files")
    ap.add_argument("--port", type=int, default=8642, help="HTTP port to listen on")
    ap.add_argument("--host", default="127.0.0.1", help="address to bind (default localhost)")
    args = ap.parse_args(argv)
    serve(args.logs_dir, args.port, args.host)


PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>hyzero live games</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
  #sidebar { width: 320px; border-right: 1px solid #8884; overflow-y: auto; padding: 8px; box-sizing: border-box; }
  #main { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 16px; gap: 12px; }
  h1 { font-size: 15px; margin: 4px 0 8px; }
  .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px; }
  .game { padding: 6px 8px; border: 1px solid #8884; border-radius: 6px; margin-bottom: 6px; cursor: pointer; font-size: 12px; }
  .game.active { outline: 2px solid #4a90d9; }
  .game .res { float: right; font-weight: bold; }
  .game .term { color: #888; font-style: italic; }
  .board { display: grid; grid-template-columns: repeat(8, 56px); grid-template-rows: repeat(8, 56px);
           border: 2px solid #444; }
  .sq { display: flex; align-items: center; justify-content: center; font-size: 40px; line-height: 1; }
  .light { background: #f0d9b5; } .dark { background: #b58863; }
  .sq span { color: #111; }
  button { font-size: 14px; padding: 4px 10px; cursor: pointer; }
  #status { font-size: 12px; opacity: 0.7; }
  #plyinfo { font-size: 13px; min-width: 90px; text-align: center; }
  #nochess { color: #c0392b; font-size: 13px; }
  #errbar { display: none; background: #c0392b; color: #fff; font-size: 12px;
            padding: 6px 10px; white-space: pre-wrap; }
</style>
</head>
<body>
<div id="errbar"></div>
<div id="sidebar">
  <h1>games</h1>
  <div class="controls">
    <label><input type="radio" name="file" value="selfplay" checked /> selfplay</label>
    <label><input type="radio" name="file" value="eval" /> eval</label>
  </div>
  <label class="controls" style="margin:6px 0;">
    <input type="checkbox" id="follow" checked /> auto-follow latest
  </label>
  <div id="status">loading…</div>
  <div id="list"></div>
</div>
<div id="main">
  <div id="nochess" hidden>python-chess not available — board disabled, showing moves only</div>
  <div class="board" id="board"></div>
  <div class="controls">
    <button id="first">⏮</button>
    <button id="prev">◀</button>
    <span id="plyinfo">–</span>
    <button id="next">▶</button>
    <button id="last">⏭</button>
  </div>
  <div id="moves" style="font-size:12px; max-width:520px; word-wrap:break-word;"></div>
</div>
<script>
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
const GLYPHS = {
  K:"♔",Q:"♕",R:"♖",B:"♗",N:"♘",P:"♙",
  k:"♚",q:"♛",r:"♜",b:"♝",n:"♞",p:"♟"
};

// Surface any uncaught JS error in a visible banner so a render failure is
// diagnosable instead of leaving the page silently stuck.
function showError(msg) {
  const bar = document.getElementById("errbar");
  bar.textContent = "JS error: " + msg;
  bar.style.display = "block";
}
window.onerror = function (msg, src, line, col) {
  showError(msg + " (" + (src || "?") + ":" + line + ":" + col + ")");
  return false;
};

let metas = [];        // lightweight metadata, file order (oldest first)
let selected = 0;      // index into display order (newest = 0)
let detail = null;     // currently loaded single-game detail {idx, fens, moves}
let ply = 0;
let haveChess = true;
let userSelected = false;

function metaIdxForDisplay(i) { return metas.length - 1 - i; }
function currentMeta() { return metas[metaIdxForDisplay(selected)]; }

function fenToBoard(fen) {
  const rows = fen.split(" ")[0].split("/");
  const grid = [];
  for (const row of rows) {
    const cells = [];
    for (const ch of row) {
      if (/\d/.test(ch)) { for (let i=0;i<+ch;i++) cells.push(""); }
      else cells.push(ch);
    }
    grid.push(cells);
  }
  return grid;
}

function renderBoard(fen) {
  const board = document.getElementById("board");
  board.innerHTML = "";
  const grid = fenToBoard(fen);
  for (let r=0; r<8; r++) {
    for (let c=0; c<8; c++) {
      const sq = document.createElement("div");
      sq.className = "sq " + (((r+c)%2===0) ? "light" : "dark");
      const piece = (grid[r] && grid[r][c]) || "";
      if (piece) { const s=document.createElement("span"); s.textContent=GLYPHS[piece]||piece; sq.appendChild(s); }
      board.appendChild(sq);
    }
  }
}

function renderCurrent() {
  const plyEl = document.getElementById("plyinfo");
  const movesEl = document.getElementById("moves");
  if (!detail) { plyEl.textContent = "–"; movesEl.textContent = ""; renderBoard(START_FEN); return; }
  const fens = (haveChess && detail.fens) ? detail.fens : null;
  if (fens && fens.length) {
    if (ply > fens.length - 1) ply = fens.length - 1;
    if (ply < 0) ply = 0;
    renderBoard(fens[ply]);
    plyEl.textContent = ply + " / " + (fens.length - 1);
  } else {
    // No replayable FENs (chess-less, or every move was illegal): show start.
    renderBoard(START_FEN);
    plyEl.textContent = haveChess ? "0 / 0" : "–";
  }
  const moves = detail.moves || [];
  movesEl.textContent = moves.map((m,i)=> (i%2===0 ? (i/2+1)+". " : "") + m).join(" ");
}

function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  for (let i=0; i<metas.length; i++) {
    const g = metas[metaIdxForDisplay(i)];
    const d = document.createElement("div");
    d.className = "game" + (i===selected ? " active" : "");
    const term = g.termination ? "<br><span class='term'>"+g.termination+"</span>" : "";
    d.innerHTML = "<span class='res'>"+(g.result||"*")+"</span>#"+(metas.length-i)+" "+
                  (g.event||"") + "<br><small>"+(g.white||"")+" vs "+(g.black||"")+
                  " ("+(g.move_count||0)+" plies)</small>" + term;
    d.onclick = () => { selected=i; ply=0; userSelected=true; renderList(); loadDetail(true); };
    list.appendChild(d);
  }
}

function lastPly() {
  if (haveChess && detail && detail.fens && detail.fens.length) ply = detail.fens.length - 1;
}

async function loadDetail(goLast) {
  const meta = currentMeta();
  if (!meta) { detail = null; renderCurrent(); return; }
  const file = document.querySelector("input[name=file]:checked").value;
  try {
    const r = await fetch("/api/game?file=" + file + "&idx=" + meta.idx);
    if (!r.ok) throw new Error("HTTP " + r.status);
    detail = await r.json();
    if (goLast) lastPly();
    renderCurrent();
  } catch (e) {
    showError("game fetch: " + e);
  }
}

async function poll() {
  const file = document.querySelector("input[name=file]:checked").value;
  try {
    const r = await fetch("/api/games?file=" + file);
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    haveChess = data.have_chess;
    document.getElementById("nochess").hidden = haveChess;
    metas = data.games || [];
    document.getElementById("status").textContent =
      data.count + " games (" + (haveChess ? "board on" : "moves only") + ")";

    const follow = document.getElementById("follow").checked;
    if (follow && !userSelected) selected = 0;
    if (selected > metas.length - 1) selected = Math.max(0, metas.length - 1);
    renderList();

    // Decide whether to (re)fetch the selected game's detail: on first load, on
    // a selection change, or when the followed/loaded game grew (more plies).
    const meta = currentMeta();
    if (!meta) { detail = null; renderCurrent(); return; }
    const followingLatest = follow && !userSelected;
    const stale = !detail || detail.idx !== meta.idx ||
                  (detail.moves && detail.moves.length !== meta.move_count);
    if (stale) loadDetail(followingLatest);
    else renderCurrent();
  } catch (e) {
    document.getElementById("status").textContent = "poll error: " + e;
    showError("poll: " + e);
  }
}

document.getElementById("first").onclick = () => { ply=0; renderCurrent(); };
document.getElementById("prev").onclick  = () => { ply=Math.max(0,ply-1); renderCurrent(); };
document.getElementById("next").onclick  = () => { ply=ply+1; renderCurrent(); };
document.getElementById("last").onclick  = () => { lastPly(); renderCurrent(); };
document.querySelectorAll("input[name=file]").forEach(el =>
  el.onchange = () => { selected=0; ply=0; userSelected=false; detail=null; poll(); });
document.getElementById("follow").onchange = (e) => { if (e.target.checked) userSelected=false; };
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") { ply=Math.max(0,ply-1); renderCurrent(); }
  if (e.key === "ArrowRight") { ply=ply+1; renderCurrent(); }
});

renderBoard(START_FEN);
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
