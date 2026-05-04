# Replay Subsystem

The replay subsystem captures per-ply MCTS state from completed self-play games and lets a developer step through them interactively in a TUI. It is opt-in (off by default, zero overhead when disabled) and is meant for diagnosing what the network is actually thinking — chosen move, root value, per-child priors / Q-values / visit counts / PUCT — at every ply of a real game. Replays are written as bincode-serialized `ReplayFile` blobs, one file per game, and viewed with the `replay` binary.

## Components

- `src/data/replay_record.rs` — `ReplayRecord` (per-ply MCTS dump: chosen action, legal moves, child visits, priors, q-values, root value, side to move) and `ReplayFile` (the per-game container: steps + outcome + model_version + is_draw + starting_fen + c_puct).
- `src/selfplay/replay_writer.rs` — `write_replay(replay, dir)` serializes a `ReplayFile` with bincode to `<dir>/replay_<unix_secs>_<seq>_v<model_version>.replay`. Process-wide `REPLAY_SEQ` AtomicU64 disambiguates files written in the same second. Two unit tests cover round-trip and filename format.
- `src/selfplay/game_task.rs` — game loop populates a `Vec<ReplayRecord>` per ply (gated on `config.replay_dir.is_some()`), then calls `write_replay` once at game end. Replay capture is wired in but skipped when `replay_dir` is `None`.
- `src/bin/replay.rs` — `cargo run --bin replay -- <file.replay>` TUI viewer (crossterm-based). Loads the file, rebuilds the board from the starting FEN by replaying actions ply-by-ply (un-flipping Black-POV actions to absolute coordinates), and renders the board on the left + an MCTS table on the right with columns `move | N | N% | P | Q | U | PUCT`. Keys: `←/→` step, `Home/End` jump, `q`/`Esc` quit, space alias for `→`.
- `scripts/compare_peak_vs_end_games.py` — text-only PGN analyzer that splits eval-game PGNs into "peak" (cycles with promotions) vs "end" (locked-in draw cycles) and reports differences. Operates on `logs/eval_games.pgn`, not on `.replay` files — adjacent diagnostic tool, not part of the replay binary's path.
- `scripts/run_nonstop.sh` — continuous-training driver (`while true; do bash scripts/run_baseline.sh 86400; done` with `HYZERO_RESUME_FROM=checkpoints/best.pt`). Sets Gumbel + 800/400 sims defaults but does not itself enable replay capture; pass `HYZERO_REPLAY_DIR=...` if you want replays from a non-stop run.

## How to use

**Capture replays during self-play.** Set `HYZERO_REPLAY_DIR` to a directory before running `selfplay` or `run_baseline.sh`:

```bash
mkdir -p replays/run-2026-05-04
HYZERO_REPLAY_DIR=replays/run-2026-05-04 cargo run --release --bin selfplay
# Or via the baseline script:
HYZERO_REPLAY_DIR=replays/run-2026-05-04 bash scripts/run_baseline.sh 1800
```

Each completed game writes one file: `replay_<unix_secs>_<seq>_v<model_version>.replay`. Files are bincode blobs; size is roughly proportional to (plies × legal_moves_per_ply × 16 bytes).

**View a replay.**

```bash
cargo run --release --bin replay -- replays/run-2026-05-04/replay_1714857600_000003_v42.replay
```

**Continuous training (no replay capture by default).**

```bash
nohup bash scripts/run_nonstop.sh > logs/nonstop_outer.log 2>&1 &
# Add HYZERO_REPLAY_DIR=... to capture replays from each 24h block
```

**Compare peak vs end eval games (PGN, not .replay).**

```bash
python3 scripts/compare_peak_vs_end_games.py logs/eval_games.pgn
```

## Gotchas

- **Bincode format is the contract.** `ReplayFile` derives `serde::{Serialize, Deserialize}` with bincode. Any field add/remove/reorder breaks existing files; rev the format intentionally if needed.
- **Actions are stored in current-player POV (Black-flipped).** The viewer un-flips before rendering UCI; downstream consumers must do the same.
- **Replay capture has nonzero overhead per ply.** Each ply's MCTS dump (legal moves + visits + priors + q-values) is held in memory for the whole game and serialized at game end. Off by default for a reason; expect modest memory growth and a per-game write at end.
- **`HYZERO_REPLAY_DIR` must exist or be creatable.** `write_replay` calls `fs::create_dir_all`, but the writing process needs permission. If creation fails, the game completes normally and the error is logged but not fatal.
- **No buffer / no retention policy.** Every game writes a file. A 24h non-stop run can produce thousands of `.replay` files. Manage disk usage manually (or by setting `HYZERO_REPLAY_DIR` only for diagnostic runs).
- **`scripts/compare_peak_vs_end_games.py` reads PGN, not `.replay`.** It pattern-matches on `[Event "Eval Cycle N Game M"]` headers; doesn't validate moves through python-chess (eval games can start from non-standard FENs without `[FEN]` headers).

## Related

- `src/data/replay_record.rs` — `ReplayRecord`, `ReplayFile`
- `src/selfplay/replay_writer.rs` — `write_replay`
- `src/selfplay/game_task.rs` — capture wiring inside the play loop
- `src/bin/replay.rs` — TUI viewer
- `scripts/compare_peak_vs_end_games.py` — eval-PGN analyzer
- `scripts/run_nonstop.sh` — continuous-training driver
- [MCTS & Self-Play](mcts-selfplay.md) — game loop and replay buffer (training-time, distinct from replay capture)
