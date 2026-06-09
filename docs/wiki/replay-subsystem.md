# Replay Subsystem

There are **two distinct "replay" concepts** in hyzero:

1. The **training replay buffer** — an in-memory ring buffer of `GameTrajectory`s
   used to sample training batches (`src/data/replay_buffer.rs`). It is
   **memory-only**; nothing is written to disk. See
   [Self-Play Coordinator](selfplay-coordinator.md) for its sampling behavior.
2. The **per-ply MCTS replay capture** — an opt-in diagnostic that dumps each
   ply's MCTS state to a `.replay` file and lets a developer step through it in a
   TUI. This page is about that.

The capture is off by default (zero overhead when disabled) and exists to
diagnose what the network is thinking — chosen move, root value, per-child
priors / Q-values / visit counts / PUCT — at every ply. Files are bincode blobs,
one per game, viewed with the `replay` binary. Writing to disk only happens when
`HYZERO_REPLAY_DIR` is set.

## Components

- `src/data/replay_record.rs` — `ReplayRecord` (per-ply MCTS dump: `action`,
  `legal_moves`, `child_visits`, `priors`, `q_values`, `root_value`,
  `white_to_move`) and `ReplayFile` (per-game container: `steps`, `game_outcome`,
  `model_version`, `is_draw`, `starting_fen`, `c_puct`). All per-child arrays are
  the same length and indexed identically (position `i` = child reached by
  `legal_moves[i]`).
- `src/selfplay/replay_writer.rs` — `write_replay(replay, dir)` bincode-serializes
  a `ReplayFile` to `<dir>/replay_<unix_secs>_<seq>_v<model_version>.replay`. A
  process-wide `REPLAY_SEQ` AtomicU64 disambiguates files written in the same
  second. Two unit tests cover round-trip and filename format.
- `src/selfplay/game_task.rs` — the game loop populates a `Vec<ReplayRecord>` per
  ply (gated on `config.replay_dir.is_some()`) and calls `write_replay` once at
  game end. Skipped entirely when `replay_dir` is `None`.
- `src/bin/replay.rs` — `cargo run --bin replay -- <file.replay>` TUI viewer
  (crossterm). Rebuilds the board from `starting_fen` by replaying actions
  ply-by-ply (un-flipping Black-POV actions to absolute coordinates) and renders
  the board plus an MCTS table with columns `move | N | N% | P | Q | U | PUCT`.
  Keys: `←/→` step, `Home/End` jump, `q`/`Esc` quit, space ≈ `→`.

## How to use

**Capture during self-play.** Set `HYZERO_REPLAY_DIR` before running `selfplay`
or `run_baseline.sh`:

```bash
mkdir -p replays/run-2026-05-04
HYZERO_REPLAY_DIR=replays/run-2026-05-04 cargo run --release --bin selfplay
# or:
HYZERO_REPLAY_DIR=replays/run-2026-05-04 bash scripts/run_baseline.sh 1800
```

Each completed game writes one file:
`replay_<unix_secs>_<seq>_v<model_version>.replay`.

**View a replay.**

```bash
cargo run --release --bin replay -- replays/run-2026-05-04/replay_1714857600_000003_v42.replay
```

## Adjacent PGN tooling (not `.replay`)

- `scripts/compare_peak_vs_end_games.py` — splits eval-game PGNs into "peak"
  (promotion cycles) vs "end" (locked-in draw cycles). Reads `logs/eval_games.pgn`
  by matching `[Event "Eval Cycle N Game M"]` headers; does **not** read `.replay`
  files.
- `scripts/run_nonstop.sh` — continuous-training driver
  (`while true; do bash scripts/run_baseline.sh 86400; done` with
  `HYZERO_RESUME_FROM=checkpoints/best.pt`). Does not enable replay capture by
  itself; pass `HYZERO_REPLAY_DIR=...` to capture from a non-stop run.

## Gotchas

- **Bincode format is the contract.** `ReplayFile`/`ReplayRecord` derive serde +
  bincode. Any field add/remove/reorder breaks existing files; rev the format
  intentionally.
- **Actions are stored in current-player POV (Black-flipped).** The viewer
  un-flips before rendering UCI; any other consumer must too.
- **Per-ply overhead.** Each ply's MCTS dump (legal moves + visits + priors +
  q-values) is held in memory for the whole game and serialized at game end. Off
  by default for that reason.
- **`HYZERO_REPLAY_DIR` must be creatable.** `write_replay` calls
  `create_dir_all`; a failure logs but is not fatal (the game completes normally).
- **No retention policy.** Every game writes a file; a 24h non-stop run can
  produce thousands. Manage disk manually, or enable capture only for diagnostics.

## Related

- [Self-Play Coordinator](selfplay-coordinator.md) — the in-memory training buffer (distinct)
- [MCTS](mcts.md) — `extract_root_diagnostics` produces the per-child stats stored here
- `src/data/replay_record.rs`, `src/selfplay/replay_writer.rs`, `src/bin/replay.rs`
