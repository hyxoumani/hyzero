# Baseline Scoring

`scripts/run_baseline.sh` runs a controlled self-play session, extracts metrics
from the log, computes a single composite score, and writes
`logs/baseline_score.json`. It is the project's headline benchmark used to accept
or reject changes.

```bash
bash scripts/run_baseline.sh 1800     # 30-minute (1800s) controlled run; arg = duration
```

## Run Setup

- **Duration** is the first argument (default 1800s). Knobs come from env with
  defaults: `HYZERO_SIMS=200`, `HYZERO_EVAL_SIMS=100`, `HYZERO_GAMES=9` (1 eval +
  8 self-play slots), `HYZERO_BATCH_SIZE=64`, `HYZERO_GAMES_PER_SIDE=4`,
  `HYZERO_PROMOTION_THRESHOLD=0.55`, `HYZERO_CHAMPION_SCORE_WEIGHT=2.0`,
  `HYZERO_ELO_SCORE_WEIGHT=0.05`, `HYZERO_DEVICE=cuda`.
- **Resume point**: `HYZERO_RESUME_FROM`, default `checkpoints/mate_pretrained.pt`.
  If that default file is missing, the script **auto-builds** it from
  `checkpoints/pretrain_dynamics.pt` (or `best.pt`) plus the Lichess mate puzzles
  via `scripts/pretrain_on_mates.py`. If the puzzles or a base checkpoint are
  missing it warns and falls back to `pretrain_dynamics.pt`, then to random init.
  Rationale: every run starts from a network whose reward head already recognizes
  mating moves, avoiding the bootstrap failure where self-play never generates
  mates.
- **Supervision** (opt-in, default on if the files exist):
  `HYZERO_STARTS_FILE=data/starting_positions.txt`,
  `HYZERO_TABLEBASE_CACHE_PATH=data/syzygy/cache_tb_plus_mates.pkl`,
  `HYZERO_TABLEBASE_FRAC=0.45`. Missing files are warned and disabled (training
  falls back to pure self-play).
- **Extra training env** exported before the run:
  `HYZERO_POLICY_ENTROPY_WEIGHT=0.01`, `HYZERO_LR_SCHEDULE=cosine`,
  `HYZERO_LR_COSINE_T_MAX=7000`, `HYZERO_LR_COSINE_ETA_MIN=1e-5`.
- **Env dump**: the script prints `[env] $(env | grep '^HYZERO_' | sort)` so each
  log records the exact configuration.

## Startup Auto-wipe

Before running, the script does a **full-slate reset** of `checkpoints/`: it
deletes all `model_v*.pt` and `best*.pt` (including `best_v*.pt`), **except** the
resume-from file itself (compared via `realpath`). This makes the ladder start
from scratch each run while preserving the pretrained starting weights. Override
`HYZERO_RESUME_FROM=checkpoints/best.pt` if you want champion continuity instead.

## Metric Extraction

The binary is run directly (`target/release/selfplay`, not `cargo run`, to avoid
stdout buffering) for `DURATION`, then SIGTERM/SIGKILL'd. Metrics are parsed from
the log:

- `games_completed`, `training_steps`, `first_loss`/`last_loss`,
  `last_policy_loss` (from `[py_training]` lines).
- `avg_game_length` from `Game received` lines.
- `eval_cycles` = count of `[eval] … ladder_match` lines.
- `promotions` = count of `[eval] … promoted` lines; `max_champion_version` =
  max `champion_version=` across promotion lines (kept for debugging, **not**
  scored).
- `last_win_rate` = last `win_rate=` on a `ladder_match` line.
- `last_candidate_elo` = last `candidate_elo=` on a `ladder_match` line (falls
  back to 1500.0 for cycles predating the field, or when there are no eval
  cycles).
- `checkpoints`, `errors`.

## Composite Score

```
score = (8.55 − last_policy_loss)
      + (promotions · CHAMPION_SCORE_WEIGHT)
      − (avg_game_length / 100)
      + (last_candidate_elo − 1500.0) · ELO_SCORE_WEIGHT
```

Higher is better. The four terms reward: fast policy learning, promotion **count**
(not the version tag), shorter games, and Elo progress against the archive pool.
The Elo term is **signed** — gains add, regressions subtract — and
`ELO_SCORE_WEIGHT` defaults to 0.05. `8.55` is the assumed initial policy loss
anchor.

## `baseline_score.json` Schema

```json
{
  "score": <float>,
  "timestamp": "<YYYYMMDD_HHMMSS>",
  "git_commit": "<short sha>",
  "duration_s": <int>,
  "metrics": {
    "games_completed": <int>,
    "training_steps": <int>,
    "first_loss": <float>,
    "last_loss": <float>,
    "last_policy_loss": <float>,
    "avg_game_length": <float>,
    "last_win_rate": <float>,
    "last_candidate_elo": <float>,
    "eval_cycles": <int>,
    "promotions": <int>,
    "max_champion_version": <int>,
    "checkpoints": <int>,
    "errors": <int>
  },
  "config": {
    "games_per_side": <int>,
    "promotion_threshold": <float>,
    "champion_score_weight": <float>,
    "eval_sims": <int>,
    "concurrent_games": <int>,
    "batch_size": <int>,
    "simulations": <int>,
    "device": "<str>",
    "resume_from": "<path>",
    "starts_file": "<path>",
    "tablebase_cache": "<path>",
    "tablebase_frac": <float>
  },
  "log_file": "<logs/baseline_*.log>"
}
```

If a prior `baseline_score.json` exists, the script prints whether the new score
improved or regressed (it always overwrites the JSON; the per-run log is kept
under `logs/`). A non-zero `errors` count makes the script exit 1 with a warning.

## Gotchas

- **The Elo term is parsed from `ladder_match` lines.** Renaming the
  `candidate_elo=` field or the `ladder_match` anchor silently breaks scoring —
  see [Elo Ladder Evaluation](elo-ladder-eval.md).
- **Score counts discrete promotions**, not `max_champion_version` (which can jump
  by arbitrary amounts depending on producer/consumer rate ratios).
- **The auto-wipe never deletes the resume-from file**, but it does delete a prior
  `best.pt` unless `HYZERO_RESUME_FROM` points at it.
- Single-run score has a noise floor (binomial eval noise on small game samples +
  step-count jitter); compare medians of reruns for small deltas.

## Related

- [Elo Ladder Evaluation](elo-ladder-eval.md) — source of `candidate_elo` / promotions
- [Champion Pool & Promotion](champion-pool-promotion.md) — `best.pt` / archive files wiped here
- [Testing Procedures](testing.md) — `e2e_test.sh` and the test suite
- `scripts/run_baseline.sh`, `logs/baseline_score.json`
