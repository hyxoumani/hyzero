---
name: debug
description: Watch a live hyzero training run's self-play games and stats. Parses the growing logs/selfplay_sample.pgn plus the newest logs/baseline_*.log into one report — games so far, termination mix, KQvK/KRvK conversion, avg length, in-run mate-rate trend, and recent training loss — and can pretty-print any single game in SAN. Use for "watch the run", "how's the run going", "show me a game", or eyeballing self-play quality mid-training. Strictly read-only, no GPU.
---

# debug

Live-watch a training run *while it runs*. `scripts/diagnostics/watch.py` reads
the append-only self-play PGN and the newest training log and folds them into
one human-readable report. It is strictly READ-ONLY (safe against a live 12h
run) and needs no GPU.

Where `diagnose-engine` probes a *frozen checkpoint* (value ladder, queen-hang
prior, arena conversion), `debug` watches the *running* self-play + training
stream. Pair them: use `debug` to notice a run going sideways, then
`diagnose-engine` for a checkpoint-level root-cause probe.

## Live-file handling

The PGN is split on `[Event` boundaries with the final block always held back as
a *carry* until the next game begins, so a game still being written (a truncated
final game) is never half-counted or allowed to crash the report. `PgnTail`
remembers a byte offset between polls, so `--follow` reads only newly-appended
bytes and resets cleanly if the file is rotated/truncated. Legacy back-rank
promotion tokens (`a1a8q` for a rook) are repaired via the 5→4-char retry shared
with `pgn_quality`.

## Modes

### `--snapshot` (default)

One-shot report on the CURRENT run:

- games so far
- termination mix (counts + pct)
- endgame-class conversion (KQvK / KRvK, per-class + combined mate rate)
- avg game length (plies)
- mate-rate trend: first window of games vs the most recent window (is mate
  conversion improving *within* the run?)
- tail of the newest `logs/baseline_*.log` (last ~5 `[py_training] step` loss
  lines) so training state is visible in the same report

```bash
python3 scripts/diagnostics/watch.py
python3 scripts/diagnostics/watch.py --window 60          # wider trend window
python3 scripts/diagnostics/watch.py --pgn logs/eval_games.pgn
```

### `--game [N|last]`

Pretty-print one game in SAN with move numbers and its termination, so a human
can eyeball play quality.

```bash
python3 scripts/diagnostics/watch.py --game last
python3 scripts/diagnostics/watch.py --game 42
```

### `--follow [--interval S]`

Loop the snapshot every `S` seconds (default 300) until Ctrl-C — for interactive
watching. Uses the incremental `PgnTail`, so it tolerates the PGN growing.

```bash
python3 scripts/diagnostics/watch.py --follow --interval 120
```

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--pgn` | `logs/selfplay_sample.pgn` | self-play PGN to read |
| `--baseline-glob` | `logs/baseline_*.log` | training-log glob (newest is tailed) |
| `--window` | `40` | trend window size in games |
| `--interval` | `300` | `--follow` loop seconds |

## Files

- `scripts/diagnostics/watch.py` — the live watcher (snapshot / game / follow)
- reuses `scripts/diagnostics/pgn_quality.py` for game parsing + legacy repair

## Notes

- Read-only: never writes checkpoints, logs, or caches; safe against a live run.
- CPU-only; no model is loaded (pure log parsing).
- Tested cheaply via `python/tests/test_watch.py` (fixture PGN, no model).
