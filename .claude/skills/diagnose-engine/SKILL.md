---
name: diagnose-engine
description: Diagnostic toolkit for the hyzero endgame-conversion campaign. Probes a frozen checkpoint's value ladder, queen-hang policy prior, and self-play conversion rate, plus a PGN quality/repair report. Use when investigating why the engine fails to convert won endgames (KQvK/KRvK) into mates, or to compare checkpoints on comparable, pinned positions.
---

# diagnose-engine

Packages the campaign's diagnostic probes as standalone scripts so any session
can measure a checkpoint the same way. All Python probes run on CPU or GPU and
load the network heads from env, so they MUST be configured to match how the
checkpoint was trained.

## Head configuration (Python probes)

The value/policy heads are reconstructed from these env vars — set them to match
the training run, or the loaded weights will mismatch the head shape:

| Env | Values | Default | Meaning |
| --- | --- | --- | --- |
| `HYZERO_VALUE_HEAD` | `scalar` \| `categorical` | `scalar` | value-head type (HL-Gauss vs scalar) |
| `HYZERO_MOVES_LEFT_HEAD` | `0` \| `1` | `0` | moves-left head present |

`--device cpu` (default) or `--device cuda`.

## Commands

### value-ladder `<ckpt>`

KQvK value-by-DTZ ladder + correlation. Samples winning K+Q vs K positions
(strong side to move), probes Syzygy DTZ, forward-passes the value head, and
reports mean value per DTZ bucket (`1-2`, `3-5`, `6-10`, `11-15`, `15+`) and the
Pearson/Spearman correlation between DTZ and value.

```bash
HYZERO_VALUE_HEAD=categorical \
python3 scripts/diagnostics/value_ladder.py checkpoints/best.pt \
    --tb data/syzygy --samples 400 --device cpu
```

Healthy: monotone ladder (small DTZ -> value near +1), strong NEGATIVE
correlation (dtz up -> value down). A flat ladder (corr near 0) is the
value-starvation signature.

### hang-test `<ckpt>`

Five PINNED KQvK positions; reports the raw-policy prior mass on moves that HANG
the queen (queen moves the lone enemy king can then capture) and classifies each
top move as `hang` / `mate` / `safe`. The FENs are fixed in the script so runs
are directly comparable across checkpoints.

```bash
python3 scripts/diagnostics/hang_test.py checkpoints/best.pt --device cpu
```

Healthy: `mean_hang_prior_mass` near 0 and `top_move_hang_count` 0. High hang
mass means the policy actively selects queen-losing moves.

### conversion-probe `<ckpt> [device]`

Wraps the arena tool: replays the checkpoint against itself over the 120 fixed
won-endgame starts (both colors -> 240 games) with adjudication OFF, counting
actual checkmates. This is the standalone form of the `run_baseline.sh` probe
block.

```bash
scripts/diagnostics/conversion_probe.sh checkpoints/best.pt cpu
```

Env: `HYZERO_PROBE_STARTS` (default `data/probe_won_starts_120.txt`, copied from
the campaign runs/ path if absent), `HYZERO_PROBE_GAMES` (240),
`HYZERO_PROBE_SIMS` (100). Builds the `arena` binary first. Output JSON:
`{"games", "checkmates", "rate", "checkpoint"}`.

### pgn-quality `<pgn>`

Termination distribution + endgame-class (KQvK/KRvK) conversion rates +
standard-start stats for a PGN log. Includes the legacy 4-char-retry repair:
pre-fix logs appended a spurious `q` to non-pawn back-rank moves (`a7a8q` for a
rook); this report retries such illegal 5-char tokens as their 4-char form so
old corrupted games replay to their true terminal. New (fixed) logs report zero
repairs.

```bash
python3 scripts/diagnostics/pgn_quality.py logs/selfplay_sample.pgn
```

Output JSON keys: `total_games`, `termination`, `endgame` (per-class + combined
`mate_rate`), `std_start`, `repair`.

## Expected output (shape)

Each Python command prints ONE JSON line to stdout (pipe to `jq`). Example
value-ladder shape:

```json
{"checkpoint": "best.pt", "value_head": "categorical", "samples": 400,
 "buckets": {"dtz_1_2": {"n": 22, "mean_value": 0.94}, ...},
 "pearson_dtz_value": -0.31, "spearman_dtz_value": -0.34}
```

## Files

- `scripts/diagnostics/value_ladder.py`    — value-ladder probe
- `scripts/diagnostics/hang_test.py`       — queen-hang prior-mass probe
- `scripts/diagnostics/conversion_probe.sh` — arena conversion probe
- `scripts/diagnostics/pgn_quality.py`     — PGN termination/conversion/repair report

## Notes

- Syzygy tables live in `data/syzygy` (3-4 man: KQvK, KRvK, ...). value-ladder
  needs `KQvK.rtbw/.rtbz`.
- The probes are read-only: they never write checkpoints, logs, or caches.
- Tested cheaply via `python/tests/test_pgn_quality.py` (fixture, no model).
