# Session & Net Play

This page covers the interactive (human-vs-human) play path: the session module
and the `server`/`client` binaries that let two players move pieces over a Unix
socket. This is separate from the self-play training loop.

## Session Module (`src/session/mod.rs`)

`SessionObj` is currently a thin stub. It holds an `Arc<PrecomputedItems>` and is
created via `SessionObj::start_session(precomputed_items)`. It exists as the
initialization hook for per-session state (precomputed move tables, future
MCTS/game context) but carries no behavior beyond holding the precomputed items
today. The actual game state for net play lives in the server binary's
`SharedState`, not in `SessionObj`.

## Server Binary (`src/bin/server.rs`)

`cargo run --bin server` — a two-player game host over a Unix domain socket.

- **Socket**: `/tmp/hyzero.sock` (`SOCKET_PATH`). A stale socket is removed on
  startup and on shutdown.
- **Shared state**: `Arc<Mutex<SharedState>>` holding the `GameBoard`,
  `GameHistory`, and `turn_count`. The board is built from `PrecomputedItems` and
  two `Player`s (White/Black).
- **Connection order**: the first client to connect is White, the second is
  Black. Each gets a `handle_client` task.
- **Turn signaling**: per-client `mpsc::channel<String>` notify channels. White is
  signaled `YOUR_TURN` to start; Black is told `WAIT`.

### Protocol

The server sends newline-terminated text lines; the client parses prefixes:

| Server → Client | Meaning |
|-----------------|---------|
| `COLOR white` / `COLOR black` | color assignment (sent first) |
| `YOUR_TURN` | it is this client's move |
| `WAIT` | wait for the opponent |
| `OK <move>` | move accepted |
| `INVALID <reason>` | rejected (bad command / not your turn / illegal move) |
| `BOARD <bitboard string>` | board snapshot after a move |
| `OPPONENT_MOVED <move>` | the opponent played |
| `GAME_OVER <result>` | terminal (`{:?}` of `GameResult`) |

Client → Server: `MOVE <notation>` (anything else gets `INVALID bad command`).

### Move handling

On `MOVE`, the server checks it is the sender's turn
(`expected_color = turn_count % 2 == 0 ? White : Black`), then calls
`GameBoard::process_move(notation, color, turn)`. On success it records the move
in history, increments `turn_count`, sends `OK` + `BOARD` to the mover and
`OPPONENT_MOVED` + `BOARD` + (`YOUR_TURN` or `GAME_OVER`) to the opponent. On a
non-`Ongoing` `GameResult` it sends `GAME_OVER` to both and breaks the loop.
When both handlers finish, the server prints the move history and removes the
socket.

## Client Binary (`src/bin/client.rs`)

`cargo run --bin client` — a line-oriented terminal client.

- Connects to `/tmp/hyzero.sock`; reads the initial `COLOR` line to learn its
  side.
- In a loop, reads server lines (which may bundle several `\n`-separated
  messages). On `YOUR_TURN` (or after `INVALID`) it prompts on stderr
  (`[White] Your move:`), reads a line from stdin, and sends `MOVE <input>`.
- Prints accepted moves, opponent moves, board strings, `WAIT` notices, and exits
  on `GAME_OVER` or server disconnect.

## Usage

```bash
# Terminal 1
cargo run --bin server
# Terminal 2 (White)
cargo run --bin client
# Terminal 3 (Black)
cargo run --bin client
```

## Gotchas

- **Two clients required**: the server blocks accepting Black until a second
  client connects; White cannot move until Black is present.
- **Turn enforcement is server-side** via `turn_count` parity — a client that
  sends out of turn gets `INVALID not your turn`.
- **Stale socket**: the server removes `/tmp/hyzero.sock` on start and stop; a
  crashed server may leave it behind (a fresh start cleans it up).

## Related

- [Chess Engine](chess-engine.md) — `GameBoard`, `process_move`, `GameResult`
- `src/session/mod.rs`, `src/bin/server.rs`, `src/bin/client.rs`
- `src/game/history.rs` — `GameHistory` move recording
